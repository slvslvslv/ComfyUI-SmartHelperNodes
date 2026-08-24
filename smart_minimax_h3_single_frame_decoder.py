from __future__ import annotations

import copy
import logging
import os
import re
import types

import torch

import comfy.utils
import folder_paths
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.sd import VAE as ComfyVAE

DECODER_BASENAME = "minimax_h3_single_frame_decoder_500k.safetensors"
# Lives on Z: via the same Models\\VAE junction pattern as Krea2 (subdir -> Z:\\Models\\VAE).
DEFAULT_DECODER_NAME = os.path.join("minimax_h3", DECODER_BASENAME)


def resolve_decoder_path(decoder_name: str) -> str:
    names = [decoder_name]
    base = os.path.basename(decoder_name.replace("\\", "/"))
    nested = os.path.join("minimax_h3", base)
    for candidate in (nested, base):
        if candidate not in names:
            names.append(candidate)
    for name in names:
        path = folder_paths.get_full_path("vae", name)
        if path:
            return path
    return folder_paths.get_full_path_or_raise("vae", decoder_name)

_QKV_RE = re.compile(
    r"^decoder\.transformer_blocks\.(\d+)\.attn\.to_([qkv])\.(weight|bias)$"
)
_PREFIXES = ("first_stage_model.", "vae.", "model.")


def _strip_prefix(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, dim_head: int) -> torch.Tensor:
    # ComfyUI MiniMaxH3 Attention views fused qkv as [heads, 3 * dim_head] per token.
    if q.ndim == 1:
        q = q.view(heads, dim_head)
        k = k.view(heads, dim_head)
        v = v.view(heads, dim_head)
        return torch.stack((q, k, v), dim=1).reshape(heads * 3 * dim_head)
    in_features = q.shape[1]
    q = q.view(heads, dim_head, in_features)
    k = k.view(heads, dim_head, in_features)
    v = v.view(heads, dim_head, in_features)
    return torch.stack((q, k, v), dim=1).reshape(heads * 3 * dim_head, in_features)


def _swap_ff_w1_halves(tensor: torch.Tensor) -> torch.Tensor:
    # Diffusers GEGLU stores [value, gate]; ComfyUI MiniMaxH3 FFN reads [gate, value].
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0)


def convert_h3_single_frame_decoder_sd(sd: dict, heads: int, dim_head: int) -> dict:
    """Map the iamkaikai diffusers decoder export onto ComfyUI MiniMaxH3VideoVAE keys."""
    out = {}
    qkv = {}
    already_fused = any(
        _strip_prefix(k).endswith("attn.to_qkv.weight") for k in sd
    )
    for raw_key, tensor in sd.items():
        key = _strip_prefix(raw_key)
        if not already_fused:
            match = _QKV_RE.match(key)
            if match:
                qkv[(int(match.group(1)), match.group(2), match.group(3))] = tensor
                continue
            if key.startswith("decoder.proj_in."):
                key = "decoder.x_embedder." + key[len("decoder.proj_in."):]
            else:
                key = key.replace(".attn.to_out.0.", ".attn.to_out.")
                key = key.replace(".ff.net.0.proj.", ".ff.w1.")
                key = key.replace(".ff.net.2.", ".ff.w2.")
                if ".ff.w1." in key:
                    tensor = _swap_ff_w1_halves(tensor)
        if key.startswith("decoder.") or key.startswith("post_quant_conv."):
            out[key] = tensor
    if already_fused:
        return out
    for block, kind, suffix in list(qkv):
        if kind != "q":
            continue
        try:
            fused = _fuse_qkv(
                qkv[(block, "q", suffix)],
                qkv[(block, "k", suffix)],
                qkv[(block, "v", suffix)],
                heads,
                dim_head,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Incomplete Q/K/V tensors for decoder block {block} {suffix}"
            ) from exc
        out[f"decoder.transformer_blocks.{block}.attn.to_qkv.{suffix}"] = fused
    return out


def _slice_temporal(samples, index: int):
    if getattr(samples, "ndim", 0) != 5 or samples.shape[2] <= 1:
        return samples
    t = int(samples.shape[2])
    idx = t + index if index < 0 else index
    idx = max(0, min(idx, t - 1))
    return samples[:, :, idx:idx + 1].contiguous()


class SmartMiniMaxH3SingleFrameDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        vaes = list(folder_paths.get_filename_list("vae"))
        if DEFAULT_DECODER_NAME not in vaes:
            vaes = [DEFAULT_DECODER_NAME] + vaes
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_name": (vaes, {"default": DEFAULT_DECODER_NAME}),
                "latent_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 4096,
                    "tooltip": "Temporal latent slice to decode independently. 0 = first token (HF default). -1 = last token.",
                }),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "overlay"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = (
        "Overlay the experimental MiniMax H3 single-frame decoder onto an official H3 video VAE. "
        "Encoder stays official. Decode is one untiled temporal slice via decoder(post_quant_conv(z)), "
        "matching iamkaikai/MiniMax-H3-Single-Frame-VAE-500K. Not for video."
    )

    @classmethod
    def IS_CHANGED(cls, vae, decoder_name, latent_index=0):
        try:
            path = resolve_decoder_path(decoder_name)
            mtime = os.path.getmtime(path) if path and os.path.exists(path) else 0
        except FileNotFoundError:
            mtime = 0
        return f"{decoder_name}:{mtime}:{latent_index}"

    def overlay(self, vae, decoder_name, latent_index=0):
        model = vae.first_stage_model
        if not isinstance(model, MiniMaxH3VideoVAE):
            raise ValueError(
                "H3 still decoder requires the official MiniMax H3 video VAE on the VAE input"
            )

        path = resolve_decoder_path(decoder_name)
        raw = comfy.utils.load_torch_file(path, safe_load=True)
        attn = model.decoder.transformer_blocks[0].attn
        converted = convert_h3_single_frame_decoder_sd(raw, attn.heads, attn.dim_head)
        del raw

        decoder_sd = {
            key[len("decoder."):]: tensor
            for key, tensor in converted.items()
            if key.startswith("decoder.")
        }
        pqc_sd = {
            key[len("post_quant_conv."):]: tensor
            for key, tensor in converted.items()
            if key.startswith("post_quant_conv.")
        }
        leftover = [
            key for key in converted
            if not key.startswith("decoder.") and not key.startswith("post_quant_conv.")
        ]
        if leftover:
            logging.warning(
                "SmartMiniMaxH3SingleFrameDecoder leftover keys: %s", leftover[:12]
            )
        if len(decoder_sd) + len(pqc_sd) < 400:
            raise RuntimeError(
                f"H3 still decoder matched {len(decoder_sd) + len(pqc_sd)} keys in {decoder_name}; "
                "expected the iamkaikai MiniMax H3 single-frame decoder export"
            )

        # copy.copy shares nn.Module._modules; isolate decoder/post_quant_conv so the cached official VAE stays intact.
        cloned_model = copy.copy(model)
        cloned_model._modules = model._modules.copy()
        cloned_model._parameters = model._parameters.copy()
        cloned_model._buffers = model._buffers.copy()
        cloned_model.tiling = False
        cloned_model.decoder = copy.deepcopy(model.decoder)
        cloned_model.post_quant_conv = copy.deepcopy(model.post_quant_conv)

        missing_d, unexpected_d = cloned_model.decoder.load_state_dict(decoder_sd, strict=False)
        missing_p, unexpected_p = cloned_model.post_quant_conv.load_state_dict(pqc_sd, strict=False)
        if missing_d:
            logging.warning(
                "SmartMiniMaxH3SingleFrameDecoder missing decoder keys: %s", list(missing_d)[:12]
            )
        if len(missing_d) > 20:
            raise RuntimeError(
                f"H3 still decoder is missing {len(missing_d)} decoder keys; "
                "key conversion likely does not match this ComfyUI MiniMaxH3VideoVAE"
            )
        if unexpected_d or unexpected_p:
            logging.warning(
                "SmartMiniMaxH3SingleFrameDecoder unexpected keys: decoder=%s post_quant_conv=%s",
                list(unexpected_d)[:8],
                list(unexpected_p)[:8],
            )
        if missing_p:
            raise RuntimeError(
                f"H3 still decoder failed to load post_quant_conv: missing {list(missing_p)}"
            )

        index = int(latent_index)

        def still_decode(self, z, **kwargs):
            # HF: one temporal slice, denorm, decoder(post_quant_conv(z)), last pixel frame, ImageNet denorm.
            z = _slice_temporal(z, index)
            latents_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(z)
            latents_std = self.latents_std.view(1, -1, 1, 1, 1).to(z)
            z = z * latents_std + latents_mean
            dec = self._decode_pixels(z)
            dec = dec[:, :, -1:, :, :]
            dec = dec.float()
            dec.mul_(self.pixel_std.to(dec)).add_(self.pixel_mean.to(dec)).clamp_(0.0, 1.0).mul_(2.0).sub_(1.0)
            return dec

        cloned_model.decode = types.MethodType(still_decode, cloned_model)
        cloned_model.decode_tiled = types.MethodType(still_decode, cloned_model)

        out = copy.copy(vae)
        out.first_stage_model = cloned_model
        out.size = None
        out.handles_tiling = False
        # Fresh patcher: clone() shares backup with the official VAE and would restore original decoder weights on GPU load.
        out.patcher = vae.patcher.__class__(
            cloned_model,
            vae.patcher.load_device,
            vae.patcher.offload_device,
            0,
            weight_inplace_update=getattr(vae.patcher, "weight_inplace_update", False),
        )
        logging.info(
            "SmartMiniMaxH3SingleFrameDecoder: loaded %s decoder / %s post_quant_conv keys from %s",
            len(decoder_sd) - len(unexpected_d),
            len(pqc_sd) - len(unexpected_p),
            decoder_name,
        )

        def decode(self, samples_in, vae_options={}, _index=index):
            return ComfyVAE.decode(self, _slice_temporal(samples_in, _index), vae_options)

        out.decode = types.MethodType(decode, out)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SmartMiniMaxH3SingleFrameDecoder": SmartMiniMaxH3SingleFrameDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartMiniMaxH3SingleFrameDecoder": "Smart MiniMax H3 Single-Frame Decoder",
}

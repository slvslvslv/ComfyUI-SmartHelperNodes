import math
import torch
import torch.nn.functional as F
import comfy.clip_vision
import comfy.model_management as mm
import comfy.utils
import node_helpers


def _normalize_image_output(decoded):
    """Convert decoded VAE output to ComfyUI's standard IMAGE shape."""
    if decoded.ndim == 5 and decoded.shape[0] == 1:
        return decoded[0]
    return decoded


def _find_coprime_skip(n):
    """Find a shift_skip value > n//2 that is coprime with n, ensuring full position coverage."""
    candidate = n // 2 + 1
    while candidate < n:
        if math.gcd(candidate, n) == 1:
            return candidate
        candidate += 1
    return 1


class MobiusLatentShiftWrapper:
    """
    Mobius latent shift adapted for WAN FLF conditioning.

    Rolls the noisy latent each step so the model sees varying temporal RoPE positions.
    Optional concat-conditioning rolling can be enabled experimentally when desired.
    """

    def __init__(self, shift_skip, latent_frames, roll_concat_conditioning=False):
        self.shift_skip = shift_skip
        self.latent_frames = latent_frames
        self.shift_idx = 0
        self.active_shift = 0
        self.last_sigma = None
        self.roll_concat_conditioning = roll_concat_conditioning

    def __call__(self, apply_model_func, args_dict):
        sigma = args_dict["timestep"]
 
        if self.last_sigma is None or not torch.equal(sigma, self.last_sigma):
            self.last_sigma = sigma.clone()
            self.active_shift = self.shift_idx
            self.shift_idx = (self.shift_idx + self.shift_skip) % self.latent_frames
 
        if self.active_shift == 0:
            return apply_model_func(args_dict["input"], args_dict["timestep"], **args_dict["c"])
 
        shift = -self.active_shift
        x = torch.roll(args_dict["input"], shift, dims=2)
        c = dict(args_dict["c"])
 
        # Rolling FLF concat conditioning makes the fixed-frame constraints move
        # through time, which can destabilize the tail of the clip.
        if self.roll_concat_conditioning and "c_concat" in c and c["c_concat"] is not None:
            c_concat = c["c_concat"]
            if hasattr(c_concat, "cond"):
                rolled = torch.roll(c_concat.cond, shift, dims=2)
                c["c_concat"] = c_concat.__class__(rolled)
            else:
                c["c_concat"] = torch.roll(c_concat, shift, dims=2)
 
        output = apply_model_func(x, args_dict["timestep"], **c)
        return torch.roll(output, -shift, dims=2)


class LoopBoundaryBlend:
    """
    Post-CFG callback that blends end-frame predictions toward start-frame predictions.

    For a seamless loop (start == end image), the denoised predictions at the temporal
    boundaries should converge. This corrects the known WAN end-frame noise artifact
    by gradually replacing the last K frames' predictions with the first K frames'.
    """

    def __init__(self, blend_frames, blend_strength):
        self.blend_frames = blend_frames
        self.blend_strength = blend_strength

    def __call__(self, args):
        denoised = args["denoised"]
        T = denoised.shape[2]
        K = min(self.blend_frames, T // 4)

        if K <= 0 or self.blend_strength <= 0:
            return denoised

        ramp = torch.linspace(0.0, self.blend_strength, K, device=denoised.device, dtype=denoised.dtype)
        ramp = ramp.view(1, 1, K, 1, 1)

        result = denoised.clone()
        result[:, :, -K:] = denoised[:, :, -K:] * (1.0 - ramp) + denoised[:, :, :K] * ramp
        return result


class SmartMobiusWanLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "motion_amplitude": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05,
                     "tooltip": "1.0 = official FLF, 2.0 = max structural repulsion boost"},
                ),
                "enable_latent_shift": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Mobius-style circular latent rolling each denoising step"},
                ),
                "shift_skip": (
                    "INT",
                    {"default": 0, "min": 0, "max": 100, "step": 1,
                     "tooltip": "Roll stride per step. 0 = auto (coprime with latent frame count)"},
                ),
                "enable_loop_blend": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Blend end-frame predictions toward start-frame predictions after CFG"},
                ),
                "blend_frames": (
                    "INT",
                    {"default": 5, "min": 1, "max": 20, "step": 1,
                     "tooltip": "Number of end frames to correct via boundary blending"},
                ),
                "blend_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "0 = no correction, 1 = fully replace end with start predictions"},
                ),
                "enable_circular_rope": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "Compress temporal RoPE so last frame is closer to first"},
                ),
                "roll_concat_conditioning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Experimental. Roll concat_mask/concat_latent_image together with the latent cycle."
                    },
                ),
                "circular_rope_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                     "tooltip": "0 = no change, 1 = maximum temporal compression"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model_high", "model_low", "positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "SmartHelperNodes/Wan"
    DESCRIPTION = (
        "Mobius seamless loop for WAN 2.2 dual-pass (high/low noise). "
        "Three independent mechanisms: (1) Latent shift — cycles frame positions across "
        "denoising steps to distribute RoPE bias. (2) Loop blend — corrects end-frame noise "
        "by blending toward start-frame predictions. (3) Circular RoPE — compresses temporal "
        "positions so the last frame is closer to the first in attention space."
    )

    def execute(
        self,
        model_high,
        model_low,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        start_image=None,
        end_image=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        motion_amplitude=1.0,
        enable_latent_shift=True,
        shift_skip=0,
        enable_loop_blend=True,
        blend_frames=5,
        blend_strength=0.5,
        enable_circular_rope=True,
        roll_concat_conditioning=False,
        circular_rope_strength=0.5,
    ):
        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1

        latent = torch.zeros(
            [batch_size, vae.latent_channels, latent_frames, height // spacial_scale, width // spacial_scale],
            device=mm.intermediate_device(),
        )

        # --- Image preprocessing ---
        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        # --- Build reference image sequence (gray fill + start/end frames) ---
        official_image = torch.ones((length, height, width, 3), device=mm.intermediate_device()) * 0.5
        mask = torch.ones(
            (1, 1, latent_frames * 4, height // spacial_scale, width // spacial_scale),
            device=mm.intermediate_device(),
        )

        if start_image is not None:
            official_image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0
        if end_image is not None:
            official_image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        official_latent = vae.encode(official_image[:, :, :, :3])

        # --- Inverse structural repulsion (PainterFLF2V algorithm) ---
        if length > 2 and motion_amplitude > 1.001 and start_image is not None and end_image is not None:
            start_l = official_latent[:, :, 0:1]
            end_l = official_latent[:, :, -1:]
            t = torch.linspace(0.0, 1.0, official_latent.shape[2], device=official_latent.device).view(1, 1, -1, 1, 1)
            linear_latent = start_l * (1 - t) + end_l * t

            diff = official_latent - linear_latent
            h, w = diff.shape[-2], diff.shape[-1]
            low_freq_diff = F.interpolate(
                diff.view(-1, vae.latent_channels, h, w), size=(max(1, h // 8), max(1, w // 8)), mode="area"
            )
            low_freq_diff = F.interpolate(low_freq_diff, size=(h, w), mode="bilinear")
            low_freq_diff = low_freq_diff.view_as(diff)
            high_freq_diff = diff - low_freq_diff

            boost_scale = (motion_amplitude - 1.0) * 4.0
            concat_latent_image = official_latent + (high_freq_diff * boost_scale)
        else:
            concat_latent_image = official_latent

        # --- Mask reshape: [1,1,T*4,H,W] -> [1,4,T,H,W] ---
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)

        # --- Inject conditioning ---
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # --- CLIP vision merge ---
        clip_vision_output = None
        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image
        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat(
                    [clip_vision_output.penultimate_hidden_states,
                     clip_vision_end_image.penultimate_hidden_states],
                    dim=-2,
                )
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # --- Model patching (shared state across both models) ---
        patched_high = model_high.clone()
        patched_low = model_low.clone()

        if enable_latent_shift:
            effective_skip = shift_skip if shift_skip > 0 else _find_coprime_skip(latent_frames)
            if math.gcd(effective_skip, latent_frames) != 1:
                effective_skip = _find_coprime_skip(latent_frames)
            wrapper = MobiusLatentShiftWrapper(
                effective_skip,
                latent_frames,
                roll_concat_conditioning=roll_concat_conditioning,
            )
            patched_high.set_model_unet_function_wrapper(wrapper)
            patched_low.set_model_unet_function_wrapper(wrapper)

        if enable_loop_blend and blend_strength > 0 and blend_frames > 0:
            blend_fn = LoopBoundaryBlend(blend_frames, blend_strength)
            patched_high.set_model_sampler_post_cfg_function(blend_fn)
            patched_low.set_model_sampler_post_cfg_function(blend_fn)

        if enable_circular_rope and circular_rope_strength > 0.0:
            scale_t = 1.0 - (circular_rope_strength / latent_frames)
            for m in (patched_high, patched_low):
                to = m.model_options.get("transformer_options", {})
                existing_rope = to.get("rope_options", {})
                existing_rope["scale_t"] = scale_t
                to["rope_options"] = existing_rope
                m.model_options["transformer_options"] = to

        out_latent = {"samples": latent}
        return (patched_high, patched_low, positive, negative, out_latent)


class SmartMobiusVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {
                "enable_frame_invariance": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Decode boundary region with wrapped context to fix 3D VAE seam artifacts"},
                ),
                "boundary_frames": (
                    "INT",
                    {"default": 3, "min": 2, "max": 5, "step": 1,
                     "tooltip": "Latent frames to wrap at each boundary side"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "SmartHelperNodes/Wan"
    DESCRIPTION = (
        "VAE decode with optional frame-invariance for seamless loops. "
        "Wraps boundary latent frames and decodes them with full temporal context "
        "to eliminate 3D VAE seam artifacts at the loop point."
    )

    def execute(self, samples, vae, enable_frame_invariance=True, boundary_frames=3):
        latent = samples["samples"]

        if not enable_frame_invariance or latent.shape[2] < boundary_frames * 2 + 1:
            decoded = _normalize_image_output(vae.decode(latent))
            return (decoded,)

        k = boundary_frames

        main_pixels = vae.decode(latent)

        boundary_latent = torch.cat([latent[:, :, -k:], latent[:, :, :k]], dim=2)
        boundary_pixels = vae.decode(boundary_latent)
        frame_dim = 1 if boundary_pixels.ndim == 5 else 0

        left_pixel_count = 1 + (k - 1) * 4
        tail_slice = [slice(None)] * boundary_pixels.ndim
        tail_slice[frame_dim] = slice(0, left_pixel_count)
        tail_good = boundary_pixels[tuple(tail_slice)]

        head_slice = [slice(None)] * boundary_pixels.ndim
        head_slice[frame_dim] = slice(left_pixel_count, None)
        head_good = boundary_pixels[tuple(head_slice)]

        result = main_pixels.clone()
        frame_dim = 1 if result.ndim == 5 else 0

        tail_slice = [slice(None)] * result.ndim
        tail_slice[frame_dim] = slice(-tail_good.shape[frame_dim], None)
        result[tuple(tail_slice)] = tail_good

        head_slice = [slice(None)] * result.ndim
        head_slice[frame_dim] = slice(0, head_good.shape[frame_dim])
        result[tuple(head_slice)] = head_good

        return (_normalize_image_output(result),)


NODE_CLASS_MAPPINGS = {
    "SmartMobiusWanLoop": SmartMobiusWanLoop,
    "SmartMobiusVAEDecode": SmartMobiusVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartMobiusWanLoop": "Smart Mobius WAN Loop",
    "SmartMobiusVAEDecode": "Smart Mobius VAE Decode",
}

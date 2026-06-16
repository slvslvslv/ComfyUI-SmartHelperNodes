import torch
import torch.nn.functional as F
import comfy.clip_vision
import comfy.model_management
import comfy.utils
import node_helpers


class SmartPainterFLF2V:
    """
    Smart variation of PainterFLF2V with dual high/low noise outputs,
    bidirectional motion amplitude, configurable mask fade/spread,
    temporal latent smoothing, end-frame noise-burst fix,
    per-expert end-frame strength, and end-frame offset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "motion_amplitude": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": (
                        "DISABLED at 1.0 — outputs official vanilla behavior (gray-fill between keyframes).\n"
                        "\n"
                        "Controls how much the model deviates from a smooth start-to-end transition.\n"
                        "\n"
                        "Lowering toward 0.0: blends the conditioning latent toward a linear interpolation "
                        "between start and end frames — suppresses camera drift/reinterpretation, "
                        "making the output increasingly static.\n"
                        "At 0.0 it is pure linear crossfade.\n"
                        "\n"
                        "Raising above 1.0: amplifies high-frequency structural differences (the 'anti-ghost' signal), "
                        "boosting subject motion and removing slow-motion feel, but may increase camera instability.\n"
                        "\n"
                        "For static camera try 0.3-0.7.\n"
                        "For dynamic motion try 1.2-1.8."
                    ),
                }),
                "mask_fade_frames": ("INT", {
                    "default": 0, "min": 0, "max": 40, "step": 1,
                    "tooltip": (
                        "DISABLED at 0 — mask uses hard edges only (official behavior).\n"
                        "\n"
                        "Number of extra pixel frames after the start keyframe (and before the end keyframe) "
                        "where the mask gradually transitions instead of jumping from 0 to 1.\n"
                        "\n"
                        "Raising this extends the 'influence zone' of your keyframes deeper into the video, "
                        "giving the model a gradual ramp from anchored to free.\n"
                        "\n"
                        "Higher values = wider gradient = more frames stay partially anchored = "
                        "more stable camera but less creative freedom for the model in the middle.\n"
                        "\n"
                        "For static camera try 8-16.\n"
                        "For dynamic scenes keep at 0-4."
                    ),
                }),
                "mask_fade_min": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Only active when mask_fade_frames > 0.\n"
                        "\n"
                        "Mask value at the keyframe edge of the fade ramp (closest to start/end image).\n"
                        "0.0 = fully anchored (model must follow the conditioning exactly).\n"
                        "\n"
                        "Raising this loosens the anchor even at the boundary — the model gets partial freedom "
                        "right next to keyframes.\n"
                        "Useful if you want soft transitions rather than hard locks.\n"
                        "\n"
                        "For maximum camera stability keep at 0.0."
                    ),
                }),
                "mask_fade_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Only active when mask_fade_frames > 0.\n"
                        "\n"
                        "Mask value at the far edge of the fade ramp (farthest from the keyframe).\n"
                        "1.0 = fully free (model can generate whatever it wants beyond the fade).\n"
                        "\n"
                        "Lowering this caps how free the model gets even in the middle of the video — "
                        "e.g. 0.6 means the model is never more than 60%% free, always partially anchored.\n"
                        "\n"
                        "Lower values = tighter overall constraint = more stable but less dynamic.\n"
                        "\n"
                        "For static camera try 0.5-0.7.\n"
                        "For full motion keep at 1.0."
                    ),
                }),
                "high_noise_end_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "DISABLED at 1.0 — end frame has full influence for the high-noise expert.\n"
                        "\n"
                        "Controls how strongly the end frame conditions the HIGH-noise model step.\n"
                        "Affects both the mask (anchoring strength) and the pixel image "
                        "(blends end frame toward neutral gray).\n"
                        "\n"
                        "The high-noise expert establishes coarse structure — giving it less end-frame "
                        "constraint lets it find a more natural motion path, reducing the end-frame noise burst.\n"
                        "\n"
                        "Lowering toward 0.0: weakens end-frame conditioning for the high-noise step.\n"
                        "At 0.0 the high-noise expert completely ignores the end frame.\n"
                        "\n"
                        "For two-step WAN, try 0.6-0.8 to reduce end noise while still guiding toward the target."
                    ),
                }),
                "low_noise_end_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "DISABLED at 1.0 — end frame has full influence for the low-noise expert.\n"
                        "\n"
                        "Controls how strongly the end frame conditions the LOW-noise model step.\n"
                        "Affects both the mask (anchoring strength) and the pixel image "
                        "(blends end frame toward neutral gray).\n"
                        "\n"
                        "The low-noise expert refines details — keeping it tightly anchored to the "
                        "end frame ensures the final result actually arrives at the target.\n"
                        "\n"
                        "Lowering toward 0.0: weakens end-frame conditioning for the low-noise step.\n"
                        "At 0.0 the low-noise expert completely ignores the end frame.\n"
                        "\n"
                        "Usually keep at 1.0 (full strength) so the refiner locks in the end frame."
                    ),
                }),
                "end_anchor_extra": ("INT", {
                    "default": 3, "min": 0, "max": 12, "step": 1,
                    "tooltip": (
                        "DISABLED at 0 — end frame uses official mask (only the end frame pixel slot is anchored, "
                        "leaving the last latent block 75%% free — this causes the end-frame noise burst).\n"
                        "\n"
                        "Fixes a WAN FLF2V design asymmetry: the start frame gets +3 extra mask slots "
                        "to fill a full latent block, but the end frame does not.\n"
                        "\n"
                        "At 3 (recommended): mirrors the start-frame treatment, anchoring the full last latent block "
                        "and blending the 3 pre-end pixel frames toward the end image for smoother VAE encoding.\n"
                        "\n"
                        "Higher values extend anchoring + blending even deeper before the end frame.\n"
                        "\n"
                        "This eliminates most of the noise burst visible in the last 3-7 frames during early denoising."
                    ),
                }),
                "end_frame_offset": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 4,
                    "tooltip": (
                        "DISABLED at 0 — end frame is placed at the very last frame (official behavior).\n"
                        "\n"
                        "Shifts the end frame forward by this many pixel frames. "
                        "The model anchors to the end image at the shifted position, then continues "
                        "generating freely for the remaining tail frames.\n"
                        "\n"
                        "Snapped to multiples of 4 internally to align with VAE latent block boundaries.\n"
                        "\n"
                        "Example: length=81, offset=8 → end image is placed at frame 72, "
                        "frames 73-80 are free continuation.\n"
                        "\n"
                        "Use this to make the model 'pass through' a target pose and keep moving, "
                        "rather than decelerating into a hard stop at the end frame.\n"
                        "\n"
                        "The tail frames have no target — expect some drift. "
                        "Keep offset moderate (8-16) for best quality."
                    ),
                }),
                "temporal_smooth_sigma": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "DISABLED at 0.0 — no temporal smoothing applied.\n"
                        "\n"
                        "Gaussian sigma for temporal smoothing of the conditioning latent.\n"
                        "\n"
                        "Higher values widen the blur kernel, averaging each frame with more distant neighbors — "
                        "strongly stabilizes camera and reduces jitter but dampens subject motion.\n"
                        "\n"
                        "Lower values only blend adjacent frames, removing high-freq temporal noise "
                        "while preserving most motion dynamics.\n"
                        "\n"
                        "Start around 1.0-2.0 for static camera."
                    ),
                }),
                "temporal_smooth_kernel": ("INT", {
                    "default": 5, "min": 1, "max": 21, "step": 2,
                    "tooltip": (
                        "Only active when temporal_smooth_sigma > 0.\n"
                        "At 1: smoothing is a no-op (single-sample kernel).\n"
                        "\n"
                        "Size of the temporal Gaussian smoothing window (odd values work best; even rounded up).\n"
                        "\n"
                        "Raising this allows the Gaussian to reach more distant frames — needed when sigma is high, "
                        "otherwise the kernel gets clipped and the smoothing is weaker than expected.\n"
                        "\n"
                        "Lowering this limits the reach even if sigma is large, capping the blur range.\n"
                        "\n"
                        "Rule of thumb: kernel >= 2*sigma + 1.\n"
                        "For sigma 1.0 use 5.\n"
                        "For sigma 3.0 use 9-11."
                    ),
                }),
            },
            "optional": {
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_reference_image": ("CLIP_VISION_OUTPUT", {
                    "tooltip": (
                        "Optional third CLIP Vision slot for an off-screen reference image "
                        "(subject identity, clothing, object that isn't in start/end frames).\n"
                        "\n"
                        "Merged with clip_vision_start_image and clip_vision_end_image by "
                        "concatenating their token sequences along the token axis. "
                        "Feeds the model's img_emb cross-attention pathway — the only non-keyframe "
                        "image-guidance channel available on stock WAN 2.2 base checkpoints "
                        "(no VACE / no ref_conv required).\n"
                        "\n"
                        "Provides semantic / style / identity guidance. NOT pixel-faithful — "
                        "good for 'keep the jacket this color', weak for 'reproduce this exact logo'.\n"
                        "\n"
                        "Leave unconnected to keep the original pipeline unchanged."
                    ),
                }),
                "cv_reference_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": (
                        "Only active when clip_vision_reference_image is connected.\n"
                        "\n"
                        "Scales the reference image's CLIP token magnitudes before they are merged "
                        "with the start/end CLIP tokens. Higher values make the reference dominate "
                        "the cross-attention, lower values let start/end guidance win.\n"
                        "\n"
                        "0.0 = reference disabled (same as leaving clip_vision_reference_image "
                        "unconnected).\n"
                        "1.0 = equal weight with start/end (one reference token = one start/end token).\n"
                        "2.0-3.0 = reference boosted (useful if the reference subject is being "
                        "ignored by the model).\n"
                        "\n"
                        "Defaults to 1.0."
                    ),
                }),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "initial_reference_image": ("IMAGE", {
                    "tooltip": (
                        "Optional VAE-encoded reference injected as a reference_latent conditioning key.\n"
                        "\n"
                        "NOTE: Only effective on WAN checkpoints that ship with a ref_conv layer "
                        "(Fun Control, Animate, SCAIL, HuMo, VACE-merged). "
                        "Silently ignored by stock WAN 2.2 FLF2V / I2V / T2V and most custom merges. "
                        "For a reference-image channel that works on any base WAN 2.2 checkpoint, "
                        "use clip_vision_reference_image above instead.\n"
                        "\n"
                        "Encoded via the VAE and appended to the conditioning's reference_latents list "
                        "(a zero latent is appended on the negative side). "
                        "Acts as a persistent identity/style anchor across the whole clip, "
                        "independent of start_image and end_image.\n"
                        "\n"
                        "Leave unconnected to keep the original pipeline unchanged."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "SmartHelperNodes/Wan"
    DESCRIPTION = (
        "Smart Painter FLF2V: First-Last-Frame video conditioning for two-step WAN workflows.\n"
        "Outputs separate positive_high and positive_low conditioning with independent "
        "end-frame strength per expert model.\n"
        "Features: bidirectional motion amplitude, mask fade/spread, temporal smoothing, "
        "end-frame noise-burst fix, and end-frame offset for continuation beyond target."
    )

    def execute(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        motion_amplitude=1.0,
        mask_fade_frames=0,
        mask_fade_min=0.0,
        mask_fade_max=1.0,
        high_noise_end_strength=1.0,
        low_noise_end_strength=1.0,
        end_anchor_extra=3,
        end_frame_offset=0,
        temporal_smooth_sigma=0.0,
        temporal_smooth_kernel=5,
        start_image=None,
        end_image=None,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        clip_vision_reference_image=None,
        cv_reference_strength=1.0,
        initial_reference_image=None,
    ):
        end_frame_offset = (end_frame_offset // 4) * 4
        end_frame_offset = min(end_frame_offset, length - 2)
        end_lat_offset = end_frame_offset // 4

        spacial_scale = vae.spacial_compression_encode()
        latent_frames = ((length - 1) // 4) + 1
        device = comfy.model_management.intermediate_device()

        latent = torch.zeros(
            [batch_size, vae.latent_channels, latent_frames, height // spacial_scale, width // spacial_scale],
            device=device,
        )

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        needs_separate = (
            end_image is not None
            and abs(high_noise_end_strength - low_noise_end_strength) > 0.001
        )

        image_high = self._build_pixel_image(
            length, height, width, device,
            start_image, end_image, high_noise_end_strength,
            end_anchor_extra, end_frame_offset,
        )
        if needs_separate:
            image_low = self._build_pixel_image(
                length, height, width, device,
                start_image, end_image, low_noise_end_strength,
                end_anchor_extra, end_frame_offset,
            )
        else:
            image_low = image_high

        latent_high = vae.encode(image_high[:, :, :, :3])
        if needs_separate:
            latent_low = vae.encode(image_low[:, :, :, :3])
        else:
            latent_low = latent_high

        concat_high = self._apply_amplitude_to_latent(
            latent_high, start_image, end_image, length,
            motion_amplitude, vae, device, end_lat_offset,
        )
        if needs_separate:
            concat_low = self._apply_amplitude_to_latent(
                latent_low, start_image, end_image, length,
                motion_amplitude, vae, device, end_lat_offset,
            )
        else:
            concat_low = concat_high

        if temporal_smooth_sigma > 0.0:
            if concat_high.shape[2] > 1:
                concat_high = self._temporal_smooth(
                    concat_high, temporal_smooth_sigma, temporal_smooth_kernel,
                    start_image, end_image, end_lat_offset,
                )
            if needs_separate and concat_low.shape[2] > 1:
                concat_low = self._temporal_smooth(
                    concat_low, temporal_smooth_sigma, temporal_smooth_kernel,
                    start_image, end_image, end_lat_offset,
                )
            elif not needs_separate:
                concat_low = concat_high

        mask_high = self._build_mask(
            latent_frames, latent.shape[-2], latent.shape[-1],
            start_image, end_image, length,
            mask_fade_frames, mask_fade_min, mask_fade_max,
            high_noise_end_strength, end_anchor_extra,
            end_frame_offset, device,
        )
        mask_high = mask_high.view(
            1, mask_high.shape[2] // 4, 4, mask_high.shape[3], mask_high.shape[4],
        ).transpose(1, 2)

        if abs(high_noise_end_strength - low_noise_end_strength) > 0.001:
            mask_low = self._build_mask(
                latent_frames, latent.shape[-2], latent.shape[-1],
                start_image, end_image, length,
                mask_fade_frames, mask_fade_min, mask_fade_max,
                low_noise_end_strength, end_anchor_extra,
                end_frame_offset, device,
            )
            mask_low = mask_low.view(
                1, mask_low.shape[2] // 4, 4, mask_low.shape[3], mask_low.shape[4],
            ).transpose(1, 2)
        else:
            mask_low = mask_high

        positive_high = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_high, "concat_mask": mask_high},
        )
        positive_low = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_low, "concat_mask": mask_low},
        )
        negative_out = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_high, "concat_mask": mask_high},
        )

        scaled_reference = self._scale_clip_vision(
            clip_vision_reference_image, cv_reference_strength,
        )
        clip_vision_output = self._merge_clip_vision(
            clip_vision_start_image, clip_vision_end_image, scaled_reference,
        )
        if clip_vision_output is not None:
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"clip_vision_output": clip_vision_output},
            )
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"clip_vision_output": clip_vision_output},
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, {"clip_vision_output": clip_vision_output},
            )

        if initial_reference_image is not None:
            ref_img = comfy.utils.common_upscale(
                initial_reference_image[:1].movedim(-1, 1),
                width, height, "bilinear", "center",
            ).movedim(1, -1)
            ref_latent = vae.encode(ref_img[:, :, :, :3])
            neg_ref_latent = torch.zeros_like(ref_latent)
            positive_high = node_helpers.conditioning_set_values(
                positive_high, {"reference_latents": [ref_latent]}, append=True,
            )
            positive_low = node_helpers.conditioning_set_values(
                positive_low, {"reference_latents": [ref_latent]}, append=True,
            )
            negative_out = node_helpers.conditioning_set_values(
                negative_out, {"reference_latents": [neg_ref_latent]}, append=True,
            )

        return (positive_high, positive_low, negative_out, {"samples": latent})

    # ------------------------------------------------------------------
    # Pixel image builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_pixel_image(length, height, width, device,
                           start_image, end_image, end_strength,
                           end_anchor_extra, end_frame_offset):
        image = torch.ones((length, height, width, 3), device=device) * 0.5
        if start_image is not None:
            image[:start_image.shape[0]] = start_image
        if end_image is not None:
            n_end = end_image.shape[0]
            end_stop = length - end_frame_offset
            end_start = max(end_stop - n_end, 0)

            if end_strength < 1.0:
                blended = 0.5 + (end_image - 0.5) * end_strength
                image[end_start:end_stop] = blended[:end_stop - end_start]
            else:
                image[end_start:end_stop] = end_image[:end_stop - end_start]

            if end_anchor_extra > 0:
                ref = image[end_start]
                for i in range(1, end_anchor_extra + 1):
                    idx = end_start - i
                    if idx >= 0:
                        alpha = 1.0 - (i / (end_anchor_extra + 1.0))
                        image[idx] = image[idx] * (1.0 - alpha) + ref * alpha
        return image

    # ------------------------------------------------------------------
    # Motion amplitude wrapper
    # ------------------------------------------------------------------
    def _apply_amplitude_to_latent(self, encoded_latent, start_image, end_image,
                                   length, amplitude, vae, device, end_lat_offset=0):
        T = encoded_latent.shape[2]
        end_idx = T - end_lat_offset

        if start_image is not None and end_image is not None and end_idx > 1:
            start_l = encoded_latent[:, :, 0:1]
            end_l = encoded_latent[:, :, end_idx - 1:end_idx]
            t = torch.linspace(0.0, 1.0, end_idx, device=device).view(1, 1, -1, 1, 1)
            linear = start_l * (1.0 - t) + end_l * t

            active = encoded_latent[:, :, :end_idx]
            processed = self._apply_motion_amplitude(
                active, linear, amplitude, vae.latent_channels,
                start_image, end_image, length,
            )
            if end_lat_offset > 0:
                result = encoded_latent.clone()
                result[:, :, :end_idx] = processed
                return result
            return processed
        return encoded_latent

    # ------------------------------------------------------------------
    # Motion amplitude: bidirectional
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_motion_amplitude(official_latent, linear_latent, amplitude, latent_ch,
                                start_image, end_image, length):
        has_both = start_image is not None and end_image is not None
        if not has_both or length <= 2:
            return official_latent

        if amplitude < 0.999:
            stabilize = 1.0 - amplitude
            return official_latent * (1.0 - stabilize) + linear_latent * stabilize

        if amplitude > 1.001:
            diff = official_latent - linear_latent
            h, w = diff.shape[-2], diff.shape[-1]

            low_freq = F.interpolate(
                diff.reshape(-1, latent_ch, h, w),
                size=(max(1, h // 8), max(1, w // 8)), mode="area",
            )
            low_freq = F.interpolate(low_freq, size=(h, w), mode="bilinear", align_corners=False)
            low_freq = low_freq.view_as(diff)

            high_freq = diff - low_freq
            boost = (amplitude - 1.0) * 4.0
            return official_latent + high_freq * boost

        return official_latent

    # ------------------------------------------------------------------
    # Temporal Gaussian smoothing (preserves keyframe latents)
    # ------------------------------------------------------------------
    @staticmethod
    def _temporal_smooth(latent, sigma, kernel_size, start_image, end_image,
                         end_lat_offset=0):
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if kernel_size < 3:
            return latent
        half = kernel_size // 2
        T = latent.shape[2]
        if T <= 1:
            return latent

        coords = torch.arange(kernel_size, device=latent.device, dtype=latent.dtype) - half
        kernel = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1, 1, 1)

        B, C, _, H, W = latent.shape
        inp = latent.reshape(1, B * C, T, H, W)
        inp = F.pad(inp, (0, 0, 0, 0, half, half), mode="replicate")
        weight = kernel.expand(B * C, 1, kernel_size, 1, 1)
        smoothed = F.conv3d(inp, weight, groups=B * C)
        smoothed = smoothed.reshape(B, C, T, H, W)

        if start_image is not None:
            n = min(start_image.shape[0], T)
            lat_n = max(1, (n - 1) // 4 + 1)
            smoothed[:, :, :lat_n] = latent[:, :, :lat_n]
        if end_image is not None:
            end_pos = T - end_lat_offset
            n = min(end_image.shape[0], end_pos)
            lat_n = max(1, (n - 1) // 4 + 1)
            preserve_start = max(end_pos - lat_n, 0)
            smoothed[:, :, preserve_start:end_pos] = latent[:, :, preserve_start:end_pos]

        return smoothed

    # ------------------------------------------------------------------
    # Mask builder with fade/spread, end-anchor fix, and offset
    # ------------------------------------------------------------------
    @staticmethod
    def _build_mask(latent_frames, lat_h, lat_w, start_image, end_image, length,
                    fade_frames, fade_min, fade_max,
                    end_frame_strength, end_anchor_extra,
                    end_frame_offset, device):
        total_pixel_frames = latent_frames * 4
        mask = torch.ones((1, 1, total_pixel_frames, lat_h, lat_w), device=device)

        n_start = start_image.shape[0] if start_image is not None else 0
        n_end = end_image.shape[0] if end_image is not None else 0

        if n_start > 0:
            mask[:, :, :n_start + 3] = 0.0

        if n_end > 0:
            end_mask_value = 1.0 - end_frame_strength
            end_anchor = n_end + end_anchor_extra
            anchor_right = total_pixel_frames - end_frame_offset
            anchor_left = max(anchor_right - end_anchor, 0)
            mask[:, :, anchor_left:anchor_right] = end_mask_value

        if fade_frames > 0:
            if n_start > 0:
                fade_start_idx = n_start + 3
                fade_end_idx = min(fade_start_idx + fade_frames, total_pixel_frames)
                n_fade = fade_end_idx - fade_start_idx
                if n_fade > 0:
                    ramp = torch.linspace(fade_min, fade_max, n_fade, device=device)
                    mask[:, :, fade_start_idx:fade_end_idx] = ramp.view(1, 1, -1, 1, 1)

            if n_end > 0:
                end_anchor = n_end + end_anchor_extra
                anchor_right = total_pixel_frames - end_frame_offset
                fade_end_idx = anchor_right - end_anchor
                fade_start_idx = max(fade_end_idx - fade_frames, 0)
                if n_start > 0:
                    fade_start_idx = max(fade_start_idx, n_start + 3 + fade_frames)
                n_fade = fade_end_idx - fade_start_idx
                if n_fade > 0:
                    ramp = torch.linspace(fade_max, fade_min, n_fade, device=device)
                    mask[:, :, fade_start_idx:fade_end_idx] = ramp.view(1, 1, -1, 1, 1)

        return mask

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_clip_vision(*outputs):
        valid = [o for o in outputs if o is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]
        states = torch.cat([o.penultimate_hidden_states for o in valid], dim=-2)
        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = states
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _scale_clip_vision(output, strength):
        if output is None or strength <= 0.0:
            return None
        if abs(strength - 1.0) < 1e-6:
            return output
        scaled = comfy.clip_vision.Output()
        scaled.penultimate_hidden_states = output.penultimate_hidden_states * strength
        for attr in ("image_embeds", "last_hidden_state", "mm_projected", "image_sizes"):
            if hasattr(output, attr):
                setattr(scaled, attr, getattr(output, attr))
        return scaled


NODE_CLASS_MAPPINGS = {
    "SmartPainterFLF2V": SmartPainterFLF2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartPainterFLF2V": "Smart Painter FLF2V",
}

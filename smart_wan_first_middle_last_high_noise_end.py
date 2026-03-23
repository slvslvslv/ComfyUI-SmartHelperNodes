import comfy
import comfy.clip_vision
import comfy.model_management
import comfy.utils
import node_helpers
import torch
import torch.nn.functional as F


class SmartWanFirstMiddleLastFrameToVideoHighNoiseEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16, "display": "number"}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16, "display": "number"}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4, "display": "number"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "display": "number"}),
            },
            "optional": {
                "mode": (["NORMAL", "SINGLE_PERSON"], {"default": "NORMAL"}),
                "start_image": ("IMAGE",),
                "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "middle_frame_ratio": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01, "display": "slider"},
                ),
                "high_noise_mid_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "high_noise_end_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "low_noise_start_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "low_noise_mid_strength": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "low_noise_end_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "structural_repulsion_boost": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "round": 0.01, "display": "slider"},
                ),
                "clip_vision_start_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_middle_image": ("CLIP_VISION_OUTPUT",),
                "clip_vision_end_image": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive_high", "positive_low", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "SmartHelperNodes/Wan"
    DESCRIPTION = "Standalone SmartHelper copy of Wan First-Middle-Last Frame to Video with separate high-noise end-frame strength."

    def execute(
        self,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        mode="NORMAL",
        start_image=None,
        middle_image=None,
        end_image=None,
        middle_frame_ratio=0.5,
        high_noise_mid_strength=0.8,
        high_noise_end_strength=1.0,
        low_noise_start_strength=1.0,
        low_noise_mid_strength=0.2,
        low_noise_end_strength=1.0,
        structural_repulsion_boost=1.0,
        clip_vision_start_image=None,
        clip_vision_middle_image=None,
        clip_vision_end_image=None,
    ):
        spacial_scale = vae.spacial_compression_encode()
        latent_channels = vae.latent_channels
        latent_t = ((length - 1) // 4) + 1

        device = comfy.model_management.intermediate_device()

        latent = torch.zeros(
            [batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale],
            device=device,
        )

        if start_image is not None:
            start_image = comfy.utils.common_upscale(
                start_image[:length].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)

        if middle_image is not None:
            middle_image = comfy.utils.common_upscale(
                middle_image[:1].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)

        if end_image is not None:
            end_image = comfy.utils.common_upscale(
                end_image[-length:].movedim(-1, 1),
                width,
                height,
                "bilinear",
                "center",
            ).movedim(1, -1)

        image = torch.ones((length, height, width, 3), device=device) * 0.5
        mask_base = torch.ones(
            (1, 1, latent_t * 4, latent.shape[-2], latent.shape[-1]),
            device=device,
        )

        middle_idx = self._calculate_aligned_position(middle_frame_ratio, length)
        middle_idx = max(4, min(middle_idx, length - 5))

        mask_high_noise = mask_base.clone()
        mask_low_noise = mask_base.clone()

        if start_image is not None:
            image[: start_image.shape[0]] = start_image
            mask_high_noise[:, :, : start_image.shape[0] + 3] = 0.0

            low_start_mask_value = 1.0 - low_noise_start_strength
            mask_low_noise[:, :, : start_image.shape[0] + 3] = low_start_mask_value

        if middle_image is not None:
            image[middle_idx : middle_idx + 1] = middle_image

            start_range = max(0, middle_idx)
            end_range = min(length, middle_idx + 4)

            high_noise_mask_value = 1.0 - high_noise_mid_strength
            mask_high_noise[:, :, start_range:end_range] = high_noise_mask_value

            low_middle_mask_value = 1.0 - low_noise_mid_strength
            mask_low_noise[:, :, start_range:end_range] = low_middle_mask_value

        if end_image is not None:
            image[-end_image.shape[0] :] = end_image

            high_noise_end_mask_value = 1.0 - high_noise_end_strength
            mask_high_noise[:, :, -end_image.shape[0] :] = high_noise_end_mask_value

            low_end_mask_value = 1.0 - low_noise_end_strength
            mask_low_noise[:, :, -end_image.shape[0] :] = low_end_mask_value

        # Build a separate high-noise image when the end-frame influence is reduced.
        # The mask alone is not enough because the model still sees end-frame pixels in
        # concat_latent_image; blend toward neutral gray instead of hard-disabling it.
        needs_separate_high_latent = (end_image is not None and high_noise_end_strength < 1.0)
        if needs_separate_high_latent:
            image_high = image.clone()
            image_high[-end_image.shape[0] :] = self._blend_with_neutral(
                end_image,
                high_noise_end_strength,
            )
            concat_latent_image_high = vae.encode(image_high[:, :, :, :3])
        else:
            concat_latent_image_high = None

        concat_latent_image = vae.encode(image[:, :, :, :3])

        if concat_latent_image_high is None:
            concat_latent_image_high = concat_latent_image

        if structural_repulsion_boost > 1.001 and length > 4:
            mask_h, mask_w = mask_high_noise.shape[-2], mask_high_noise.shape[-1]
            boost_factor = structural_repulsion_boost - 1.0

            def create_spatial_gradient(img1, img2):
                if img1 is None or img2 is None:
                    return None

                motion_diff = torch.abs(img2[0] - img1[0]).mean(dim=-1, keepdim=False)
                motion_diff_4d = motion_diff.unsqueeze(0).unsqueeze(0)
                motion_diff_scaled = F.interpolate(
                    motion_diff_4d,
                    size=(mask_h, mask_w),
                    mode="bilinear",
                    align_corners=False,
                )

                motion_normalized = (motion_diff_scaled - motion_diff_scaled.min()) / (
                    motion_diff_scaled.max() - motion_diff_scaled.min() + 1e-8
                )

                spatial_gradient = 1.0 - motion_normalized * boost_factor * 2.5
                spatial_gradient = torch.clamp(spatial_gradient, 0.02, 1.0)
                return spatial_gradient[0, 0]

            if start_image is not None and middle_image is not None:
                start_img = start_image[0:1].to(device)
                mid_img = middle_image[0:1].to(device)

                spatial_gradient_1 = create_spatial_gradient(start_img, mid_img)

                if spatial_gradient_1 is not None:
                    start_end = start_image.shape[0] + 3
                    mid_protect_start = max(start_end, middle_idx - 4)
                    transition_end = min(mid_protect_start, length)

                    for frame_idx in range(start_end, transition_end):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient_1

            if middle_image is not None and end_image is not None:
                mid_img = middle_image[0:1].to(device)
                end_img = end_image[-1:].to(device)

                spatial_gradient_2 = create_spatial_gradient(mid_img, end_img)

                if spatial_gradient_2 is not None:
                    mid_protect_end = middle_idx + 5
                    transition_start = mid_protect_end
                    end_start = length - end_image.shape[0]

                    for frame_idx in range(transition_start, end_start):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient_2

            if start_image is not None and end_image is not None and middle_image is None:
                start_img = start_image[0:1].to(device)
                end_img = end_image[-1:].to(device)

                spatial_gradient = create_spatial_gradient(start_img, end_img)

                if spatial_gradient is not None:
                    start_end = start_image.shape[0] + 3
                    end_start = length - end_image.shape[0]

                    for frame_idx in range(start_end, end_start):
                        current_mask = mask_high_noise[:, :, frame_idx, :, :]
                        mask_high_noise[:, :, frame_idx, :, :] = current_mask * spatial_gradient

        needs_separate_low_latent = (
            mode == "SINGLE_PERSON"
            or low_noise_start_strength == 0.0
            or low_noise_mid_strength == 0.0
            or (end_image is not None and low_noise_end_strength < 1.0)
        )

        if needs_separate_low_latent:
            image_low_only = torch.ones((length, height, width, 3), device=device) * 0.5

            if mode == "SINGLE_PERSON":
                if start_image is not None:
                    image_low_only[: start_image.shape[0]] = start_image
            else:
                if start_image is not None and low_noise_start_strength > 0.0:
                    image_low_only[: start_image.shape[0]] = start_image
                if middle_image is not None and low_noise_mid_strength > 0.0:
                    image_low_only[middle_idx : middle_idx + 1] = middle_image
                if end_image is not None and low_noise_end_strength > 0.0:
                    image_low_only[-end_image.shape[0] :] = self._blend_with_neutral(
                        end_image,
                        low_noise_end_strength,
                    )

            concat_latent_image_low = vae.encode(image_low_only[:, :, :, :3])
        else:
            concat_latent_image_low = concat_latent_image

        mask_high_reshaped = mask_high_noise.view(
            1,
            mask_high_noise.shape[2] // 4,
            4,
            mask_high_noise.shape[3],
            mask_high_noise.shape[4],
        ).transpose(1, 2)

        mask_low_reshaped = mask_low_noise.view(
            1,
            mask_low_noise.shape[2] // 4,
            4,
            mask_low_noise.shape[3],
            mask_low_noise.shape[4],
        ).transpose(1, 2)

        positive_high_noise = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped},
        )

        positive_low_noise = node_helpers.conditioning_set_values(
            positive,
            {"concat_latent_image": concat_latent_image_low, "concat_mask": mask_low_reshaped},
        )

        negative_out = node_helpers.conditioning_set_values(
            negative,
            {"concat_latent_image": concat_latent_image_high, "concat_mask": mask_high_reshaped},
        )

        clip_vision_output = self._merge_clip_vision_outputs(
            clip_vision_start_image,
            clip_vision_middle_image,
            clip_vision_end_image,
        )

        if clip_vision_output is not None:
            positive_low_noise = node_helpers.conditioning_set_values(
                positive_low_noise,
                {"clip_vision_output": clip_vision_output},
            )

            negative_out = node_helpers.conditioning_set_values(
                negative_out,
                {"clip_vision_output": clip_vision_output},
            )

        out_latent = {"samples": latent}
        return (positive_high_noise, positive_low_noise, negative_out, out_latent)

    @classmethod
    def _calculate_aligned_position(cls, ratio, total_frames):
        desired_idx = int(total_frames * ratio)
        latent_idx = desired_idx // 4
        aligned_idx = latent_idx * 4
        aligned_idx = max(0, min(aligned_idx, total_frames - 1))
        return aligned_idx

    @classmethod
    def _merge_clip_vision_outputs(cls, *outputs):
        valid_outputs = [output for output in outputs if output is not None]

        if not valid_outputs:
            return None

        if len(valid_outputs) == 1:
            return valid_outputs[0]

        all_states = [output.penultimate_hidden_states for output in valid_outputs]
        combined_states = torch.cat(all_states, dim=-2)

        result = comfy.clip_vision.Output()
        result.penultimate_hidden_states = combined_states
        return result

    @staticmethod
    def _blend_with_neutral(image, strength):
        strength = max(0.0, min(float(strength), 1.0))
        return 0.5 + (image - 0.5) * strength


NODE_CLASS_MAPPINGS = {
    "SmartWanFirstMiddleLastFrameToVideoHighNoiseEnd": SmartWanFirstMiddleLastFrameToVideoHighNoiseEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartWanFirstMiddleLastFrameToVideoHighNoiseEnd": "Smart Wan First-Middle-Last Frame to Video (High Noise End)",
}

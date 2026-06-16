import math
import torch
import comfy.utils
from nodes import node_helpers

MAX_IMAGES = 6
_MAX_CHOICES = [str(i) for i in range(MAX_IMAGES + 1)]


def _optional_images():
    return {f"image{i}": ("IMAGE",) for i in range(1, MAX_IMAGES + 1)}


def _collect_images(kwargs):
    return [kwargs.get(f"image{i}") for i in range(1, MAX_IMAGES + 1)]


class SmartTextEncodeEditAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "use_vl_encoding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VL image feeding: prepend Picture N vision tokens, pass downscaled images to clip.tokenize, and apply the edit-style llama_template. Turn off to behave like plain text encode + reference_latents.",
                }),
                "vl_megapixels": ("FLOAT", {
                    "default": 0.50,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target megapixels for Vision-Language model. Set to 0 to disable VL image feeding. Recommended: 0.2-1.0 MP. Qwen2.5-VL trained range: 0.2-1.0 MP"
                }),
                "max_images_allowed": (_MAX_CHOICES, {
                    "default": "6",
                    "tooltip": f"Maximum number of images to process. Images are processed in order: image1..image{MAX_IMAGES}",
                }),
            },
            "optional": {
                "vae": ("VAE",),
                **_optional_images(),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "SmartHelperNodes"

    def encode(self, clip, prompt, use_vl_encoding=True, vl_megapixels=0.50, max_images_allowed="6", vae=None, **kwargs):
        max_images_allowed = int(max_images_allowed)

        ref_latents = []
        images = _collect_images(kwargs)
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        vl_disabled = (not use_vl_encoding) or vl_megapixels <= 0

        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None:
                samples = image.movedim(-1, 1)

                if not vl_disabled:
                    total = int(vl_megapixels * 1_000_000)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))

        tokens = clip.tokenize(
            image_prompt + prompt,
            images=images_vl if not vl_disabled and len(images_vl) > 0 else None,
            llama_template=llama_template if not vl_disabled else None
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)

        return (conditioning,)


class SmartTextEncodeEditAdvancedDual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "use_vl_encoding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VL image feeding: prepend Picture N vision tokens, pass downscaled images to clip.tokenize, and apply the edit-style llama_template. Turn off to behave like plain text encode + reference_latents.",
                }),
                "vl_megapixels": ("FLOAT", {
                    "default": 0.50,
                    "min": 0.0,
                    "max": 4.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target megapixels for Vision-Language model. Set to 0 to disable VL image feeding. Recommended: 0.2-1.0 MP. Qwen2.5-VL trained range: 0.2-1.0 MP"
                }),
                "max_images_allowed": (_MAX_CHOICES, {
                    "default": "6",
                    "tooltip": f"Maximum number of images to process. Images are processed in order: image1..image{MAX_IMAGES}",
                }),
            },
            "optional": {
                "vae": ("VAE",),
                **_optional_images(),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "SmartHelperNodes"

    def encode(self, clip, positive, negative, use_vl_encoding=True, vl_megapixels=0.50, max_images_allowed="6", vae=None, **kwargs):
        max_images_allowed = int(max_images_allowed)

        ref_latents = []
        images = _collect_images(kwargs)
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        vl_disabled = (not use_vl_encoding) or vl_megapixels <= 0

        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None:
                samples = image.movedim(-1, 1)

                if not vl_disabled:
                    total = int(vl_megapixels * 1_000_000)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))

        tokens_positive = clip.tokenize(
            image_prompt + positive,
            images=images_vl if not vl_disabled and len(images_vl) > 0 else None,
            llama_template=llama_template if not vl_disabled else None
        )
        conditioning_positive = clip.encode_from_tokens_scheduled(tokens_positive)

        tokens_negative = clip.tokenize(
            negative,
            images=None,
            llama_template=None
        )
        conditioning_negative = clip.encode_from_tokens_scheduled(tokens_negative)

        if len(ref_latents) > 0:
            conditioning_positive = node_helpers.conditioning_set_values(conditioning_positive, {"reference_latents": ref_latents}, append=True)
            conditioning_negative = node_helpers.conditioning_set_values(conditioning_negative, {"reference_latents": ref_latents}, append=True)

        return (conditioning_positive, conditioning_negative)


NODE_CLASS_MAPPINGS = {
    "SmartTextEncodeEditAdvanced": SmartTextEncodeEditAdvanced,
    "SmartTextEncodeEditAdvancedDual": SmartTextEncodeEditAdvancedDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartTextEncodeEditAdvanced": "Smart TextEncodeEditAdvanced (6 images)",
    "SmartTextEncodeEditAdvancedDual": "Smart TextEncodeEditAdvanced Dual (6 images)",
}

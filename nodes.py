from __future__ import annotations

import os
import importlib
import importlib.util
import torch
import folder_paths
import comfy
import comfy.sd # Added import for CLIP type access
import json
import sys
import re # Added for regex operations
import random # Added for random choice
import time # Added for IS_CHANGED
import platform
import inspect
from inspect import cleandoc
from typing import Any, Dict
from server import PromptServer
from comfy_execution.graph_utils import is_link

# Custom Nodes for ComfyUI
# Note: All nodes in this file should use the "Smart" prefix in their names
# Category: SmartHelperNodes

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

# Helper function at module level
def format_lora_string(lora_name, strength, blocks_type, existing_string=""):
    # Convert "-" to "none" for blocks_type
    blocks = "none" if blocks_type == "-" else blocks_type
    new_lora_string = f"{lora_name}, str:{strength}, blocks:({blocks})"
    if existing_string:
        return f"{existing_string}\n{new_lora_string}"
    return new_lora_string

# Helper function to format objects with binary data handling
def format_object(obj, indent=0, max_depth=5, path=""):
    if max_depth <= 0:
        return " " * indent + "[max depth reached]"
    
    # Handle None
    if obj is None:
        return " " * indent + "None"
    
    # Get type and handle binary data
    obj_type = type(obj)
    
    # Check for binary data (bytes, bytearray, memoryview)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        size_bytes = sys.getsizeof(obj)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= 1:
            return f"{' ' * indent}[...binary data ({size_mb:.2f} MB)...]"
        else:
            size_kb = size_bytes / 1024
            return f"{' ' * indent}[...binary data ({size_kb:.2f} KB)...]"
    
    # Handle strings
    if isinstance(obj, str):
        if len(obj) > 1000:  # Truncate long strings
            return f"{' ' * indent}\"{obj[:1000]}...\""
        return f"{' ' * indent}\"{obj}\""
    
    # Handle basic types
    if isinstance(obj, (int, float, bool)):
        return f"{' ' * indent}{obj}"
    
    # Handle dictionaries
    if isinstance(obj, dict):
        if not obj:
            return f"{' ' * indent}{{}}"
        
        result = [f"{' ' * indent}{{"]
        for k, v in obj.items():
            key_str = repr(k) if isinstance(k, str) else str(k)
            next_path = f"{path}.{key_str}" if path else key_str
            
            # Format value with increased indentation
            value_str = format_object(v, indent + 2, max_depth - 1, next_path)
            
            # Remove the indentation from value_str as we'll add it manually
            if value_str.startswith(" " * (indent + 2)):
                value_str = value_str[(indent + 2):]
                
            result.append(f"{' ' * (indent + 2)}{key_str}: {value_str}")
        
        result.append(f"{' ' * indent}}}")
        return "\n".join(result)
    
    # Handle lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        if not obj:
            return f"{' ' * indent}[]" if isinstance(obj, list) else \
                   f"{' ' * indent}()" if isinstance(obj, tuple) else \
                   f"{' ' * indent}set()"
        
        brackets = "[]" if isinstance(obj, list) else "()" if isinstance(obj, tuple) else "set([])"
        result = [f"{' ' * indent}{brackets[0]}"]
        
        for i, item in enumerate(obj):
            next_path = f"{path}[{i}]" if path else f"[{i}]"
            item_str = format_object(item, indent + 2, max_depth - 1, next_path)
            
            # Remove the indentation from item_str as we'll add it manually
            if item_str.startswith(" " * (indent + 2)):
                item_str = item_str[(indent + 2):]
                
            result.append(f"{' ' * (indent + 2)}{item_str},")
        
        result.append(f"{' ' * indent}{brackets[1]}")
        return "\n".join(result)
    
    # For other objects, try to get dict representation or use str()
    try:
        if hasattr(obj, '__dict__') and obj.__dict__:
            result = [f"{' ' * indent}{obj.__class__.__name__} {{"]
            for k, v in obj.__dict__.items():
                # Skip private attributes
                if k.startswith('_'):
                    continue
                    
                next_path = f"{path}.{k}" if path else k
                
                # Format value with increased indentation
                value_str = format_object(v, indent + 2, max_depth - 1, next_path)
                
                # Remove the indentation from value_str as we'll add it manually
                if value_str.startswith(" " * (indent + 2)):
                    value_str = value_str[(indent + 2):]
                    
                result.append(f"{' ' * (indent + 2)}{k}: {value_str}")
            
            result.append(f"{' ' * indent}}}")
            return "\n".join(result)
    except:
        pass
    
    # Fallback to simple string representation
    return f"{' ' * indent}{str(obj)}"

class SmartHVLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora": (folder_paths.get_filename_list("loras"),
                         {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
                                         "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
                "blocks_type": (["none", "all", "single_blocks", "double_blocks"], {"default": "double_blocks"}),
            },
            "optional": {
                "prev_lora": ("HYVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "lora_string": ("STRING", {"forceInput": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("HYVIDLORA", "STRING")
    RETURN_NAMES = ("lora", "lora_string")
    FUNCTION = "getlorapath"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Select a LoRA model and output both the LORA data and a formatted string containing the LoRA name, strength, and blocks"

    def getlorapath(self, lora, strength, blocks_type, prev_lora=None, fuse_lora=False, lora_string=""):
        loras_list = []
        lora_name = lora.split(".")[0]
        final_lora_string = format_lora_string(lora_name, strength, blocks_type, lora_string)

        lora_data = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora_name,
            "fuse_lora": fuse_lora,
            "blocks": None if blocks_type == "none" else blocks_type
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora_data)
        return (loras_list, final_lora_string)

class SmartHVLoraStack:
    max_lora_num = 50  # Maximum number of LoRA slots

    @classmethod
    def INPUT_TYPES(cls):
        # Get the list of available LoRA filenames once
        available_loras = folder_paths.get_filename_list("loras")
        # Build the input dictionary
        inputs = {
            "required": {
                "lora_count": ("INT", {"default": 1, "min": 0, "max": cls.max_lora_num, "step": 1}),
                "blocks_type": (["none", "all", "single_blocks", "double_blocks"], {"default": "double_blocks"}),
            },
            "optional": {
                "prev_lora": ("HYVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "lora_string": ("STRING", {"forceInput": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

        # Define input fields for each possible LoRA slot
        for i in range(1, cls.max_lora_num + 1):
            inputs["required"][f"lora_{i}_enabled"] = ("BOOLEAN", {"default": True})
            inputs["required"][f"lora_{i}_name"] = (["None"] + available_loras, {"default": "None"})
            inputs["required"][f"lora_{i}_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}
            )

        return inputs

    RETURN_TYPES = ("HYVIDLORA", "STRING")
    RETURN_NAMES = ("lora", "lora_string")
    FUNCTION = "stack_loras"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Stack multiple LoRAs together with individual strength controls and enable/disable switches"

    def stack_loras(self, lora_count, blocks_type, lora_string="", prev_lora=None, unique_id=None, **kwargs):
        loras_list = []
        final_lora_string = lora_string if lora_string else ""  # Ensure empty string if None

        # If there was a previous LoRA stack provided, start with that.
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        # Process only the first lora_count entries.
        for i in range(1, lora_count + 1):
            curr_enabled = kwargs.get(f"lora_{i}_enabled", False)
            curr_lora = kwargs.get(f"lora_{i}_name")
            curr_strength = kwargs.get(f"lora_{i}_strength", 1.0)

            # Only process this LoRA if it is enabled and a valid name is selected.
            if curr_enabled and curr_lora and curr_lora != "None":
                # Optionally, adjust the lora_string (using your existing helper function)
                curr_lora_name = curr_lora.split(".")[0]
                final_lora_string = format_lora_string(curr_lora_name, curr_strength, blocks_type, final_lora_string)

                loras_list.append({
                    "path": folder_paths.get_full_path("loras", curr_lora),
                    "strength": curr_strength,
                    "name": curr_lora_name,
                    "fuse_lora": False,
                    "blocks": None if blocks_type == "none" else blocks_type
                })

        return (loras_list, final_lora_string)

class SmartFormatString:
    max_param_num = 5  # Maximum number of parameters

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "string_format": ("STRING", {
                    "default": "Value is: %1", 
                    "multiline": True,  
                    "tooltip": "Format string with:\n%1-%5 for parameters\n%H for hours (00-23)\n%M for minutes (00-59)\n%S for seconds (00-59)\n%y for year (00-99)\n%m for month (01-12)\n%d for day (01-31)"
                }),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

        # Define input fields for each possible parameter slot
        for i in range(1, cls.max_param_num + 1):
            inputs["optional"][f"param_{i}"] = (any_type, )

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_string",)
    FUNCTION = "format_string"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Format a string by replacing %1, %2, etc. with provided parameter values"

    def format_string(self, string_format, unique_id=None, **kwargs):
        from datetime import datetime
        
        result = string_format
        now = datetime.now()
        
        # Replace time-based placeholders
        time_placeholders = {
            '%H': f"{now.hour:02d}",
            '%M': f"{now.minute:02d}",
            '%S': f"{now.second:02d}",
            '%y': f"{now.year % 100:02d}",
            '%m': f"{now.month:02d}",
            '%d': f"{now.day:02d}"
        }
        
        for placeholder, value in time_placeholders.items():
            result = result.replace(placeholder, value)
        
        # Process numbered parameters in reverse order to handle double-digit numbers correctly
        for i in range(self.max_param_num, 0, -1):
            param_value = kwargs.get(f"param_{i}", "")
            str_value = str(param_value) if param_value is not None else ""
            result = result.replace(f"%{i}", str_value)
            
        return (result,)
        
    @classmethod
    def IS_CHANGED(cls, string_format, unique_id=None, **kwargs):
        # Return a value that's always different to force re-evaluation when time-based placeholders are used
        import time
        if any(x in string_format for x in ['%H', '%M', '%S', '%y', '%m', '%d']):
            return time.time()
        return 0

class SmartFormatString10(SmartFormatString):
    max_param_num = 10  # Override with 10 parameters


SMART_BUS_TYPE = "SMART_BUS"
SMART_BUS_SLOT_LABELS = (
    "any 1",
    "any 2",
    "any 3",
    "any 4",
    "any 5",
    "any 6",
    "any 7",
    "any 8",
    "any 9",
    "any 10",
)


def _normalize_smart_bus(bus, size=10):
    values = [None] * size
    labels = list(SMART_BUS_SLOT_LABELS[:size])

    if isinstance(bus, dict):
        source_values = bus.get("values", [])
        source_labels = bus.get("labels", [])
    elif isinstance(bus, (list, tuple)):
        source_values = bus
        source_labels = []
    else:
        source_values = []
        source_labels = []

    for index, value in enumerate(source_values[:size]):
        values[index] = value

    for index, label in enumerate(source_labels[:size]):
        if isinstance(label, str) and label.strip():
            labels[index] = label.strip()

    return {
        "values": values,
        "labels": labels,
    }


def _get_smart_bus_node(unique_id=None, extra_pnginfo=None):
    if not unique_id or not extra_pnginfo:
        return None

    try:
        info = extra_pnginfo[0] if isinstance(extra_pnginfo, (list, tuple)) else extra_pnginfo
        workflow = info.get("workflow")
        node_value = unique_id[0] if isinstance(unique_id, (list, tuple)) else unique_id
        node_id = str(node_value)
    except (IndexError, AttributeError, TypeError):
        return None

    if not workflow:
        return None

    return next((node for node in workflow.get("nodes", []) if str(node.get("id")) == node_id), None)


def _get_workflow_node_by_id(node_id=None, extra_pnginfo=None):
    if node_id is None or not extra_pnginfo:
        return None

    try:
        info = extra_pnginfo[0] if isinstance(extra_pnginfo, (list, tuple)) else extra_pnginfo
        workflow = info.get("workflow")
    except (IndexError, AttributeError, TypeError):
        return None

    if not workflow:
        return None

    node_id = str(node_id)
    return next((node for node in workflow.get("nodes", []) if str(node.get("id")) == node_id), None)


def _get_smart_bus_labels(unique_id=None, extra_pnginfo=None, size=10):
    labels = list(SMART_BUS_SLOT_LABELS[:size])
    node = _get_smart_bus_node(unique_id, extra_pnginfo)
    if not node:
        return labels

    properties = node.get("properties", {})
    for index, default_label in enumerate(labels):
        value = properties.get(default_label)
        if isinstance(value, str) and value.strip():
            labels[index] = value.strip()

    return labels


def _smart_bus_param_is_linked(prompt, unique_id, param_key):
    """True if param_* has a wire in the prompt; False if absent/null (keep bus slot)."""
    if prompt is None or unique_id is None:
        return None
    node_value = unique_id[0] if isinstance(unique_id, (list, tuple)) else unique_id
    node_id = str(node_value)
    node = prompt.get(node_id)
    if not isinstance(node, dict):
        return None
    raw = (node.get("inputs") or {}).get(param_key)
    if raw is None:
        return False
    return is_link(raw)


class SmartBusIn:
    max_param_num = 10

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {},
            "optional": {
                "bus": (SMART_BUS_TYPE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            },
        }

        for i in range(1, cls.max_param_num + 1):
            inputs["optional"][f"param_{i}"] = (any_type,)

        return inputs

    RETURN_TYPES = (SMART_BUS_TYPE,)
    RETURN_NAMES = ("bus",)
    FUNCTION = "build_bus"
    CATEGORY = "SmartHelperNodes/Bus"
    DESCRIPTION = "Merge into a Smart Bus: only wired param slots replace values; unwired slots keep the incoming bus."

    def build_bus(self, bus=None, unique_id=None, extra_pnginfo=None, prompt=None, **kwargs):
        bus_data = _normalize_smart_bus(bus, self.max_param_num)
        values = bus_data["values"]

        for i in range(1, self.max_param_num + 1):
            key = f"param_{i}"
            linked = _smart_bus_param_is_linked(prompt, unique_id, key)
            if linked is True:
                if key in kwargs:
                    values[i - 1] = kwargs[key]
            elif linked is False:
                pass
            else:
                if key in kwargs:
                    values[i - 1] = kwargs[key]

        labels = _get_smart_bus_labels(unique_id, extra_pnginfo, self.max_param_num)

        return ({
            "values": values,
            "labels": labels,
        },)


class SmartBusOut:
    max_param_num = 10

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bus": (SMART_BUS_TYPE,),
            }
        }

    RETURN_TYPES = (SMART_BUS_TYPE,) + (any_type,) * max_param_num
    RETURN_NAMES = ("bus",) + SMART_BUS_SLOT_LABELS
    FUNCTION = "expand_bus"
    CATEGORY = "SmartHelperNodes/Bus"
    DESCRIPTION = "Expand a Smart Bus back into its 10 stored passthrough values."

    def expand_bus(self, bus):
        bus_data = _normalize_smart_bus(bus, self.max_param_num)
        values = bus_data["values"]
        return (bus_data, *values)


class SmartBusIn5(SmartBusIn):
    max_param_num = 5
    DESCRIPTION = "Merge into a Smart Bus: only wired param slots replace values; unwired slots keep the incoming bus."


class SmartBusOut5(SmartBusOut):
    max_param_num = 5
    RETURN_TYPES = (SMART_BUS_TYPE,) + (any_type,) * max_param_num
    RETURN_NAMES = ("bus",) + SMART_BUS_SLOT_LABELS[:max_param_num]
    DESCRIPTION = "Expand a Smart Bus back into its 5 stored passthrough values."

class SmartSaveText:
    """
    A node that saves text to a file, creating directories as needed and appending to existing files.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "filename": ("STRING", {"default": "output.txt"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_text"
    CATEGORY = "SmartHelperNodes"
    OUTPUT_NODE = True

    def save_text(self, text, filename):
        # Convert relative path to absolute path within ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        full_path = os.path.join(output_dir, filename)
        
        # Create directories if they don't exist
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Check if we need a leading newline (file exists, non-empty, doesn't end with newline)
        needs_newline = False
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            with open(full_path, "rb") as f:
                f.seek(-1, 2)  # Seek to last byte from end
                needs_newline = f.read(1) != b'\n'
        
        # Append to file
        with open(full_path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write('\n')
            f.write(text)
            
        return (text,)

class SmartSaveAnimatedGIF:
    """
    A node that saves a series of images as an animated GIF file with specified FPS.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Series of images to save as animated GIF"}),
                "filename": ("STRING", {"default": "animation.gif", "tooltip": "Output filename for the GIF"}),
                "fps": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 60.0, "step": 0.1, "tooltip": "Frames per second for the animation"}),
                "loop": ("INT", {"default": 0, "min": 0, "max": 65535, "tooltip": "Number of loops (0 = infinite loop)"}),
                "optimize": ("BOOLEAN", {"default": True, "tooltip": "Optimize the GIF file size"}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "Quality of the GIF (higher = better quality, larger file)"}),
                "preserve_transparency": ("BOOLEAN", {"default": True, "tooltip": "Preserve alpha channel as GIF transparency (binary)"}),
                "alpha_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Alpha values below this are transparent (0 = only fully transparent, 1 = all transparent)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_animated_gif"
    CATEGORY = "SmartHelperNodes"
    OUTPUT_NODE = True
    DESCRIPTION = "Save a series of images as an animated GIF with customizable FPS, loop count, quality settings, and transparency preservation"

    def save_animated_gif(self, images, filename, fps, loop, optimize, quality, preserve_transparency, alpha_threshold):
        try:
            from PIL import Image
            import numpy as np
            import torch
        except ImportError as e:
            raise RuntimeError(f"Required libraries not available: {e}. Please install Pillow.")

        # Ensure filename has .gif extension
        if not filename.lower().endswith('.gif'):
            if filename.endswith('.'):
                filename = filename + 'gif'
            else:
                filename = filename + '.gif'

        # Convert relative path to absolute path within ComfyUI's output directory
        output_dir = folder_paths.get_output_directory()
        full_path = os.path.join(output_dir, filename)
        
        # Create directories if they don't exist
        directory = os.path.dirname(full_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Convert images to PIL format
        pil_frames = []
        has_transparent_pixel = False
        used_colors = set()
        alpha_thresh = alpha_threshold * 255

        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # Handle batch of images
            batch_size = images.shape[0]
        else:
            # If not a batch tensor, treat as list
            batch_size = len(images)

        for i in range(batch_size):
            if isinstance(images, torch.Tensor):
                img = images[i]
            else:
                img = images[i]

            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
                if img.dim() == 3:
                    if img.shape[0] in [1, 3, 4]:  # CHW
                        img = img.permute(1, 2, 0)
                if img.max() <= 1.0:
                    img = img * 255
                img = img.round().clamp(0, 255).byte().numpy()
                mode = 'RGBA' if img.shape[-1] == 4 else 'RGB' if img.shape[-1] == 3 else 'L' if img.shape[-1] == 1 else None
                if mode is None:
                    raise ValueError(f"Unsupported tensor channel count: {img.shape[-1]}")
                if mode == 'L':
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
                    mode = 'RGB'
                pil_img = Image.fromarray(img, mode)
            elif isinstance(img, np.ndarray):
                if img.max() <= 1.0:
                    img = img * 255
                img = np.round(img).clip(0, 255).astype(np.uint8)
                mode = 'RGBA' if img.shape[-1] == 4 else 'RGB' if img.shape[-1] == 3 else 'L' if img.shape[-1] == 1 else None
                if mode is None:
                    raise ValueError(f"Unsupported array channel count: {img.shape[-1]}")
                if mode == 'L':
                    img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
                    mode = 'RGB'
                pil_img = Image.fromarray(img, mode)
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Convert to RGBA for consistency if transparency might be present
            if pil_img.mode not in ['RGB', 'RGBA']:
                pil_img = pil_img.convert('RGB')

            if preserve_transparency and pil_img.mode == 'RGBA':
                data = np.array(pil_img)
                alpha = data[:, :, 3]
                if np.any(alpha < alpha_thresh):
                    has_transparent_pixel = True
                opaque_mask = alpha >= alpha_thresh
                opaque_rgb = data[opaque_mask, :3]
                for color in opaque_rgb:
                    used_colors.add(tuple(map(int, color)))

            pil_frames.append(pil_img)

        if not pil_frames:
            raise ValueError("No valid images provided")

        if not preserve_transparency or not has_transparent_pixel:
            # Composite to RGB with white background
            for i in range(len(pil_frames)):
                if pil_frames[i].mode == 'RGBA':
                    background = Image.new('RGB', pil_frames[i].size, (255, 255, 255))
                    background.paste(pil_frames[i], mask=pil_frames[i].getchannel('A'))
                    pil_frames[i] = background
            pil_images = pil_frames
        else:
            # Find global key_color not used in any opaque pixel
            key_color = None
            attempts = 0
            while attempts < 10000 and key_color is None:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                if (r, g, b) not in used_colors:
                    key_color = (r, g, b)
                attempts += 1

            if key_color is None:
                key_color = (255, 0, 255)
                # Replace occurrences of key_color in opaque areas
                for i in range(len(pil_frames)):
                    if pil_frames[i].mode == 'RGBA':
                        data = np.array(pil_frames[i])
                        alpha = data[:, :, 3]
                        match = (data[:, :, 0] == 255) & (data[:, :, 1] == 0) & (data[:, :, 2] == 255) & (alpha >= alpha_thresh)
                        data[match, 0] = 254
                        pil_frames[i] = Image.fromarray(data, 'RGBA')

            # Create RGB frames with key_color in transparent areas
            rgb_frames = []
            for frame in pil_frames:
                if frame.mode == 'RGBA':
                    data = np.array(frame)
                    transparent_mask = data[:, :, 3] < alpha_thresh
                    rgb_data = data[:, :, :3]
                    rgb_data[transparent_mask] = key_color
                    frame_rgb = Image.fromarray(rgb_data, 'RGB')
                else:
                    frame_rgb = frame.convert('RGB')
                rgb_frames.append(frame_rgb)

            # Create large image for global palette
            widths, heights = zip(*(f.size for f in rgb_frames))
            max_width = max(widths)
            total_height = sum(heights) + 32  # Extra for key block
            large_img = Image.new('RGB', (max_width, total_height))
            y = 0
            for f in rgb_frames:
                large_img.paste(f, (0, y))
                y += f.height
            key_block = Image.new('RGB', (32, 32), key_color)
            large_img.paste(key_block, (0, y))

            # Quantize large image
            large_quant = large_img.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE)
            global_palette = large_quant.getpalette()[:256*3]

            # Find trans_index
            trans_index = -1
            for i in range(256):
                if (global_palette[i*3] == key_color[0] and
                    global_palette[i*3 + 1] == key_color[1] and
                    global_palette[i*3 + 2] == key_color[2]):
                    trans_index = i
                    break

            if trans_index == -1:
                # Fallback to per-frame palettes
                pil_images = []
                for frame_rgb in rgb_frames:
                    frame_p = frame_rgb.quantize(colors=256)
                    palette = frame_p.getpalette()[:256*3]
                    for j in range(256):
                        if (palette[j*3] == key_color[0] and
                            palette[j*3 + 1] == key_color[1] and
                            palette[j*3 + 2] == key_color[2]):
                            frame_p.info['transparency'] = j
                            frame_p.info['disposal'] = 2
                            break
                        pil_images.append(frame_p)
                    if pil_images:
                        pil_images[0].info['background'] = pil_images[0].info.get('transparency', 0)
            else:
                # Use global palette
                palette_img = Image.new('P', (1, 1))
                palette_img.putpalette(global_palette)
                pil_images = []
                for frame_rgb in rgb_frames:
                    frame_p = frame_rgb.quantize(palette=palette_img)
                    frame_p.info['transparency'] = trans_index
                    frame_p.info['disposal'] = 2
                    pil_images.append(frame_p)
                if pil_images:
                    pil_images[0].info['background'] = trans_index

        # Calculate duration per frame in milliseconds
        duration = int(1000 / fps)
        
        # Save as animated GIF
        pil_images[0].save(
            full_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=loop,
            optimize=optimize,
            quality=quality
        )
        
        return (full_path,)

class SmartRemoveComments:
    """
    A node that removes all comments starting with // anywhere in the line
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remove_comments"
    CATEGORY = "SmartHelperNodes"

    def remove_comments(self, text):
        # Split text into lines, remove comment portions, and rejoin
        lines = text.splitlines()
        processed_lines = []
        for line in lines:
            # Find the position of // if it exists
            comment_pos = line.find('//')
            if comment_pos != -1:
                # Keep only the part before the comment
                line = line[:comment_pos].rstrip()
            # Only add non-empty lines after comment removal
            if line.strip():
                processed_lines.append(line)
        return ("\n".join(processed_lines),)

class SmartLoadLoRA:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "optional": {
                "model2": ("MODEL", {"tooltip": "Optional second diffusion model; the same LoRA is applied to both models (UNet only on this input)."}),
                "lora_string": ("STRING", {"forceInput": True, "default": ""})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "MODEL")
    RETURN_NAMES = ("model", "clip", "lora_string", "model2")
    FUNCTION = "load_lora"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Load a LoRA model (just like the core LoRA loader), but also outputs the model name and strength to a formatted string. Optional second model applies the same UNet LoRA."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, model2=None, lora_string=""):
        if strength_model == 0 and strength_clip == 0:
            m2_out = model2 if model2 is not None else model
            return (model, clip, lora_string, m2_out)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        if model2 is not None:
            model2_lora, _ = comfy.sd.load_lora_for_models(model2, None, lora, strength_model, 0)
        else:
            model2_lora = model_lora

        # Create formatted string output
        lora_name_without_ext = lora_name.split(".")[0]
        new_lora_string = f"{lora_name_without_ext}, model_str:{strength_model}, clip_str:{strength_clip}"
        if lora_string:
            final_lora_string = f"{lora_string}\n{new_lora_string}"
        else:
            final_lora_string = new_lora_string

        return (model_lora, clip_lora, final_lora_string, model2_lora)


class SmartLoadDoubleLoRA:
    def __init__(self):
        self.loaded_lora_1 = None
        self.loaded_lora_2 = None

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model1": ("MODEL", {"tooltip": "First diffusion model."}),
                "model2": ("MODEL", {"tooltip": "Second diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model both LoRAs will be applied to."}),
                "lora_name_1": (lora_list, {"tooltip": "LoRA applied to model1 and CLIP."}),
                "strength_model_1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip_1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "lora_name_2": (lora_list, {"tooltip": "LoRA applied to model2 and CLIP."}),
                "strength_model_2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip_2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "lora_string": ("STRING", {"forceInput": True, "default": ""})
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model1", "model2", "clip", "lora_string")
    FUNCTION = "load_double_lora"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Apply two different LoRAs: lora1 to model1, lora2 to model2, and both to CLIP sequentially."

    def _get_lora(self, cache_attr, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)
        cached = getattr(self, cache_attr)
        if cached is not None and cached[0] == lora_path:
            return cached[1]
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        setattr(self, cache_attr, (lora_path, lora))
        return lora

    def load_double_lora(self, model1, model2, clip,
                         lora_name_1, strength_model_1, strength_clip_1,
                         lora_name_2, strength_model_2, strength_clip_2,
                         lora_string=""):
        # LoRA 1 → model1 + clip
        if strength_model_1 == 0 and strength_clip_1 == 0:
            m1_out, clip_out = model1, clip
        else:
            lora1 = self._get_lora("loaded_lora_1", lora_name_1)
            m1_out, clip_out = comfy.sd.load_lora_for_models(model1, clip, lora1, strength_model_1, strength_clip_1)

        # LoRA 2 → model2 + clip (already patched by lora1)
        if strength_model_2 == 0 and strength_clip_2 == 0:
            m2_out = model2
        else:
            lora2 = self._get_lora("loaded_lora_2", lora_name_2)
            m2_out, clip_out = comfy.sd.load_lora_for_models(model2, clip_out, lora2, strength_model_2, strength_clip_2)

        # Build lora_string
        parts = []
        if lora_string:
            parts.append(lora_string)
        n1 = lora_name_1.split(".")[0]
        parts.append(f"{n1}, model_str:{strength_model_1}, clip_str:{strength_clip_1}")
        n2 = lora_name_2.split(".")[0]
        parts.append(f"{n2}, model_str:{strength_model_2}, clip_str:{strength_clip_2}")

        return (m1_out, m2_out, clip_out, "\n".join(parts))


class SmartPrompt:
    """
    A node that provides a multiline text input, removes comments, supports random selection syntax { A | B | C }, 
    and optionally encodes it using a CLIP model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "tooltip": "Enter your prompt text. Lines with // comments will be cleaned up. Use { A | B | C } for random selection."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP model for encoding the text."})
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("text", "conditioning")
    FUNCTION = "process_prompt"
    CATEGORY = "SmartHelperNodes"

    @staticmethod
    def parse_random_syntax(text):
        """Parses the { A | B | C } syntax and selects a random element."""
        def replace_match(match):
            # Extract content inside braces and split by '|'
            options = [opt.strip() for opt in match.group(1).split('|')]
            # Return a randomly chosen option
            return random.choice(options)

        # Regex to find { content } where content contains at least one |
        # Example: {cat|dog|bird}
        pattern = r'\{([^}]+?\|[^}]+?)\}'
        
        # Repeatedly apply substitution until no more matches are found
        # This handles multiple occurrences, e.g., "{a|b} and {c|d}"
        processed_text = text
        # Limit iterations to prevent potential infinite loops with malformed input, though unlikely with this pattern
        iterations = 0
        max_iterations = 100 # Safety limit
        while re.search(pattern, processed_text) and iterations < max_iterations:
             # Replace one occurrence at a time
             processed_text = re.sub(pattern, replace_match, processed_text, count=1) 
             iterations += 1
             
        if iterations >= max_iterations:
             print(f"Warning: Max iterations reached during random syntax processing in SmartPrompt. Input text: {text}")

        return processed_text

    def process_prompt(self, text, clip=None):
        # 1. Remove comments
        lines = text.splitlines()
        processed_lines = []
        for line in lines:
            comment_pos = line.find('//')
            if comment_pos != -1:
                line = line[:comment_pos].rstrip()
            # Keep lines that are not empty after comment removal
            if line.strip():
                processed_lines.append(line)
        
        comment_cleaned_text = "\n".join(processed_lines)
        
        # 2. Process random syntax { A | B | C }
        processed_text = self.parse_random_syntax(comment_cleaned_text)
        
        # 3. Encode if CLIP is provided
        conditioning = None
        if clip is not None:
            tokens = clip.tokenize(processed_text)
            # Use encode_from_tokens_scheduled if available, otherwise fall back to encode
            if hasattr(clip, 'encode_from_tokens_scheduled'):
                 conditioning = clip.encode_from_tokens_scheduled(tokens)
            elif hasattr(clip, 'encode'):
                 # Standard CLIPTextEncode uses encode
                 conditioning, _ = clip.encode_token_weights(tokens) # Assuming encode returns pooled_output as second value
            else:
                 print(f"Warning: Provided CLIP object for SmartPrompt node is missing a recognized encoding method (encode or encode_from_tokens_scheduled).")
                 # Return empty conditioning or handle appropriately
                 # For now, returning None which might cause downstream issues if conditioning is expected.
                 # A more robust solution might involve returning a default "empty" conditioning object if possible.

        return (processed_text, conditioning)

    @classmethod
    def IS_CHANGED(cls, text, clip=None, **kwargs):
        # Check if the random syntax { A | B } is present in the input text
        # but ignore patterns that have a comment // before the opening { on the same line
        lines = text.splitlines()
        pattern = r'\{([^}]+?\|[^}]+?)\}'

        for line in lines:
            # Skip lines that have // before any { on that line
            comment_pos = line.find('//')
            brace_pos = line.find('{')

            # If there's a { and either no // or the // comes after the {
            if brace_pos != -1 and (comment_pos == -1 or comment_pos > brace_pos):
                if re.search(pattern, line):
                    return time.time()

        # Hash the input text to detect changes. Add clip object's potential changes if needed.
        # Note: Hashing the clip object itself might be complex/unreliable.
        # Relying on text changes is standard for prompt nodes.
        return hash(text)

class SmartModelOrLoraToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model": ("WANVIDEOMODEL", {"default": None}),
                "lora": ("WANVIDLORA", {"default": None}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_string",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Converts a model and/or LORA object to a formatted string"

    def process(self, model=None, lora=None):
        output_lines = []
        
        # Process model if provided
        if model is not None:
            try:
                # Access the model name from the pipeline dictionary in the nested structure
                if hasattr(model, 'model') and hasattr(model.model, 'pipeline') and isinstance(model.model.pipeline, dict) and 'model_name' in model.model.pipeline:
                    model_name = model.model.pipeline['model_name']
                    # Remove .safetensors extension if present
                    if model_name.endswith('.safetensors'):
                        model_name = model_name[:-12]  # Remove the last 12 characters ('.safetensors')
                    output_lines.append(f"MODEL: {model_name}")
                else:
                    output_lines.append("MODEL: unknown")
            except Exception as e:
                output_lines.append(f"MODEL: unknown ({str(e)})")
        
        # Process lora if provided
        if lora is not None:
            for lora_item in lora:
                name = lora_item["name"]
                strength = lora_item["strength"]
                output_lines.append(f"LORA: {name} : {strength:.2f}")
            
        return ("\n".join(output_lines),)

class SmartShowAnything:
    """
    A node that displays any type of input as text and returns it unchanged.
    Supports recursive object unfolding and proper handling of binary data.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, 
                "optional": {"anything": (any_type, {})},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
               }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Displays any type of content as text in the UI and passes it through unchanged. Also supports recursive object unfolding."

    def log_input(self, unique_id=None, extra_pnginfo=None, **kwargs):
        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if type(val) is str:
                        values.append(val)
                    elif type(val) is list:
                        values = val
                    else:
                        # Use the new format_object function for recursive object formatting
                        formatted_val = format_object(val, max_depth=5)
                        values.append(formatted_val)
                except Exception as e:
                    values.append(f"Error formatting object: {str(e)}")
                    pass

        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
            print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
        else:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]
        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],), }
        else:
            return {"ui": {"text": values}, "result": (values,), }

class SmartSetInt:
    """
    A node that sets the output to a specific integer value.
    When bypassed, it passes through the input value instead.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", {"default": 0, "tooltip": "Input value (used when node is bypassed)"}),
                "value": ("INT", {"default": 0, "tooltip": "Output value (used when node is active)"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "set_value"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Sets the output to a specific integer value. When bypassed, passes through the input value."

    def set_value(self, input, value):
        return (value,)

    @classmethod
    def check_lazy_status(cls, input, value):
        # This indicates which inputs are needed
        # When bypassed, ComfyUI will pass through the input
        return ["value"]

class SmartSetFloat:
    """
    A node that sets the output to a specific float value.
    When bypassed, it passes through the input value instead.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("FLOAT", {"default": 0.0, "tooltip": "Input value (used when node is bypassed)"}),
                "value": ("FLOAT", {"default": 0.0, "tooltip": "Output value (used when node is active)"}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "set_value"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Sets the output to a specific float value. When bypassed, passes through the input value."

    def set_value(self, input, value):
        return (value,)

    @classmethod
    def check_lazy_status(cls, input, value):
        # This indicates which inputs are needed
        # When bypassed, ComfyUI will pass through the input
        return ["value"]

class SmartModelInfo:
    """
    A node that displays information about a MODEL object and passes it through.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to inspect"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "get_model_info"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Displays information about a MODEL object including config details"
    OUTPUT_NODE = True

    def get_model_info(self, model):
        info_lines = []
        
        # Try to get model config info
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'model_config'):
                config = model.model.model_config
                info_lines.append("=== Model Config ===")
                
                # Get model type/class name
                if hasattr(config, '__class__'):
                    info_lines.append(f"Type: {config.__class__.__name__}")
                
                # Get unet config if available
                if hasattr(config, 'unet_config'):
                    unet_config = config.unet_config
                    if isinstance(unet_config, dict):
                        info_lines.append("\n=== UNet Config ===")
                        for key, value in unet_config.items():
                            if isinstance(value, (int, float, str, bool)):
                                info_lines.append(f"{key}: {value}")
                
                # Check for latent format
                if hasattr(config, 'latent_format'):
                    latent_format = config.latent_format
                    if hasattr(latent_format, '__class__'):
                        info_lines.append(f"\nLatent Format: {latent_format.__class__.__name__}")
        except Exception as e:
            info_lines.append(f"Error reading model config: {str(e)}")
        
        # Check model_options
        try:
            if hasattr(model, 'model_options'):
                options = model.model_options
                if options and isinstance(options, dict):
                    info_lines.append("\n=== Model Options ===")
                    for key, value in options.items():
                        if key != 'transformer_options' and isinstance(value, (int, float, str, bool)):
                            info_lines.append(f"{key}: {value}")
        except Exception as e:
            info_lines.append(f"Error reading model options: {str(e)}")
        
        # Check attachments (where custom data might be stored)
        try:
            if hasattr(model, 'attachments'):
                attachments = model.attachments
                if attachments and isinstance(attachments, dict):
                    info_lines.append("\n=== Attachments ===")
                    for key, value in attachments.items():
                        info_lines.append(f"{key}: {value}")
        except Exception as e:
            info_lines.append(f"Error reading attachments: {str(e)}")
        
        if not info_lines:
            info_lines.append("No model information available")
        
        info_text = "\n".join(info_lines)
        return (model, info_text)

class SmartHWMonitor:
    """
    A node that monitors hardware metrics including CPU/GPU temperature and RAM usage.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refresh_rate": ("INT", {"default": 1, "min": 1, "max": 60, "step": 1, "tooltip": "Refresh rate in seconds for hardware monitoring"}),
                "include_cpu_temp": ("BOOLEAN", {"default": True, "tooltip": "Include CPU temperature in output"}),
                "include_gpu_temp": ("BOOLEAN", {"default": True, "tooltip": "Include GPU temperature in output"}),
                "include_ram": ("BOOLEAN", {"default": True, "tooltip": "Include RAM usage in output"}),
                "include_cpu_usage": ("BOOLEAN", {"default": True, "tooltip": "Include CPU usage percentage in output"}),
                "include_vram": ("BOOLEAN", {"default": True, "tooltip": "Include VRAM usage in output"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("hw_info",)
    FUNCTION = "get_hardware_info"
    CATEGORY = "SmartHelperNodes"
    OUTPUT_NODE = True
    DESCRIPTION = "Monitor hardware metrics including CPU/GPU temperature and RAM usage"

    def get_hardware_info(self, refresh_rate, include_cpu_temp, include_gpu_temp, include_ram, include_cpu_usage, include_vram):
        hw_info_lines = []
        hw_info_lines.append(f"=== Hardware Monitor ({time.strftime('%H:%M:%S')}) ===")
        
        # CPU Usage
        if include_cpu_usage:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                hw_info_lines.append(f"CPU Usage: {cpu_percent:.1f}%")
            except ImportError:
                hw_info_lines.append("CPU Usage: psutil not available")
            except Exception as e:
                hw_info_lines.append(f"CPU Usage: Error - {str(e)}")

        # CPU Temperature
        if include_cpu_temp:
            try:
                import psutil
                cpu_temp = None
                
                # Method 1: Try psutil sensors (Linux/some Windows)
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    # Try different sensor names for CPU temperature
                    for sensor_name in ['coretemp', 'cpu_thermal', 'k10temp', 'acpi']:
                        if sensor_name in temps and temps[sensor_name]:
                            cpu_temp = temps[sensor_name][0].current
                            break
                
                # Method 2: Try Windows WMI Win32_TemperatureProbe
                if cpu_temp is None and platform.system() == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run([
                            'wmic', 'path', 'Win32_TemperatureProbe', 'get', 'CurrentReading', '/format:value'
                        ], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if 'CurrentReading=' in line:
                                    temp_value = line.split('=')[1].strip()
                                    if temp_value.isdigit():
                                        # Win32_TemperatureProbe returns tenths of Kelvin
                                        temp_celsius = (int(temp_value) / 10.0) - 273.15
                                        # Only accept reasonable temperature values
                                        if 0 <= temp_celsius <= 120:
                                            cpu_temp = temp_celsius
                                            break
                    except Exception:
                        pass
                
                # Method 3: Try Windows WMI MSAcpi_ThermalZoneTemperature
                if cpu_temp is None and platform.system() == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run([
                            'wmic', '/namespace:\\\\root\\wmi', 'path', 'MSAcpi_ThermalZoneTemperature', 
                            'get', 'CurrentTemperature', '/format:value'
                        ], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if 'CurrentTemperature=' in line:
                                    temp_value = line.split('=')[1].strip()
                                    if temp_value.isdigit():
                                        # Convert from tenths of Kelvin to Celsius
                                        temp_celsius = (int(temp_value) / 10.0) - 273.15
                                        # Only accept reasonable temperature values
                                        if 0 <= temp_celsius <= 120:
                                            cpu_temp = temp_celsius
                                            break
                    except Exception:
                        pass
                
                # Method 4: Try Windows Performance Counters
                if cpu_temp is None and platform.system() == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run([
                            'typeperf', '\\Thermal Zone Information(*)\\Temperature', '-sc', '1'
                        ], capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            lines = result.stdout.split('\n')
                            for line in lines:
                                if 'Temperature' in line and ',' in line:
                                    parts = line.split(',')
                                    if len(parts) >= 2:
                                        temp_str = parts[-1].strip().replace('"', '')
                                        try:
                                            temp_value = float(temp_str)
                                            temp_celsius = temp_value - 273.15  # Convert Kelvin to Celsius
                                            # Only accept reasonable temperature values
                                            if 0 <= temp_celsius <= 120:
                                                cpu_temp = temp_celsius
                                                break
                                        except ValueError:
                                            continue
                    except Exception:
                        pass
                
                # Method 5: Try reading thermal zones on Linux
                if cpu_temp is None and platform.system() == "Linux":
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp_millicelsius = int(f.read().strip())
                            cpu_temp = temp_millicelsius / 1000.0
                    except (FileNotFoundError, ValueError, PermissionError):
                        pass
                
                # Method 6: Try alternative thermal zones on Linux
                if cpu_temp is None and platform.system() == "Linux":
                    try:
                        import glob
                        thermal_zones = glob.glob('/sys/class/thermal/thermal_zone*/temp')
                        for zone_file in thermal_zones:
                            try:
                                with open(zone_file, 'r') as f:
                                    temp_millicelsius = int(f.read().strip())
                                    potential_temp = temp_millicelsius / 1000.0
                                    # Filter reasonable CPU temperatures (0-120°C)
                                    if 0 <= potential_temp <= 120:
                                        cpu_temp = potential_temp
                                        break
                            except (ValueError, PermissionError):
                                continue
                    except Exception:
                        pass
                
                # Method 7: Try Windows PowerShell WMI query
                if cpu_temp is None and platform.system() == "Windows":
                    try:
                        import subprocess
                        result = subprocess.run([
                            'powershell', '-Command', 
                            'Get-WmiObject -Namespace "root\\wmi" -Class MSAcpi_ThermalZoneTemperature | Select-Object CurrentTemperature'
                        ], capture_output=True, text=True, timeout=15)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                line = line.strip()
                                if line.isdigit():
                                    temp_celsius = (int(line) / 10.0) - 273.15
                                    # Only accept reasonable temperature values
                                    if 0 <= temp_celsius <= 120:
                                        cpu_temp = temp_celsius
                                        break
                    except Exception:
                        pass
                
                # Only show temperature if we found a real sensor reading
                if cpu_temp is not None:
                    hw_info_lines.append(f"CPU Temperature: {cpu_temp:.1f}°C")
                else:
                    # Provide helpful instructions based on platform
                    if platform.system() == "Windows":
                        hw_info_lines.append("CPU Temperature: Not available")
                        hw_info_lines.append("To enable CPU temperature monitoring:")
                        hw_info_lines.append("• Install OpenHardwareMonitor (ohm.openhardwaremonitor.org)")
                        hw_info_lines.append("• OR install HWiNFO64 with shared memory enabled")
                        hw_info_lines.append("• OR run ComfyUI as Administrator")
                        hw_info_lines.append("• OR install motherboard-specific monitoring software")
                    elif platform.system() == "Linux":
                        hw_info_lines.append("CPU Temperature: Not available")
                        hw_info_lines.append("To enable CPU temperature monitoring:")
                        hw_info_lines.append("• Install lm-sensors: sudo apt install lm-sensors")
                        hw_info_lines.append("• Run: sudo sensors-detect")
                        hw_info_lines.append("• Load sensor modules: sudo modprobe coretemp")
                        hw_info_lines.append("• Install psutil: pip install psutil")
                    elif platform.system() == "Darwin":
                        hw_info_lines.append("CPU Temperature: Not available")
                        hw_info_lines.append("To enable CPU temperature monitoring:")
                        hw_info_lines.append("• Install TG Pro or Macs Fan Control")
                        hw_info_lines.append("• OR install iStat Menus")
                        hw_info_lines.append("• macOS restricts direct sensor access")
                    else:
                        hw_info_lines.append("CPU Temperature: Not available on this platform")
                # If no real temperature found, don't show anything (no estimates)
                        
            except ImportError:
                # Don't show error message, just skip temperature
                pass
            except Exception as e:
                # Don't show error message, just skip temperature
                pass

        # GPU Temperature
        if include_gpu_temp:
            gpu_temp_found = False
            
            # Try NVIDIA GPU first
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    temps = result.stdout.strip().split('\n')
                    for i, temp in enumerate(temps):
                        if temp.strip():
                            hw_info_lines.append(f"GPU {i} Temperature: {temp.strip()}°C")
                            gpu_temp_found = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
            except Exception as e:
                pass
            
            # Try AMD GPU if NVIDIA not found
            if not gpu_temp_found:
                try:
                    import psutil
                    if hasattr(psutil, 'sensors_temperatures'):
                        temps = psutil.sensors_temperatures()
                        for sensor_name in ['amdgpu', 'radeon']:
                            if sensor_name in temps and temps[sensor_name]:
                                for sensor in temps[sensor_name]:
                                    if 'edge' in sensor.label.lower() or 'junction' in sensor.label.lower():
                                        hw_info_lines.append(f"GPU Temperature: {sensor.current:.1f}°C")
                                        gpu_temp_found = True
                                        break
                                if gpu_temp_found:
                                    break
                except Exception:
                    pass
            
            if not gpu_temp_found:
                hw_info_lines.append("GPU Temperature: No compatible GPU found")

        # RAM Usage
        if include_ram:
            try:
                import psutil
                memory = psutil.virtual_memory()
                ram_used_gb = memory.used / (1024**3)
                ram_total_gb = memory.total / (1024**3)
                ram_percent = memory.percent
                hw_info_lines.append(f"RAM Usage: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.1f}%)")
            except ImportError:
                hw_info_lines.append("RAM Usage: psutil not available")
            except Exception as e:
                hw_info_lines.append(f"RAM Usage: Error - {str(e)}")
        
        # System info
        hw_info_lines.append(f"System: {platform.system()} {platform.release()}")
        
        # VRAM Usage
        if include_vram:
            vram_found = False
            
            # Try NVIDIA GPU VRAM
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():
                            parts = line.strip().split(', ')
                            if len(parts) == 2:
                                used_mb = int(parts[0])
                                total_mb = int(parts[1])
                                used_gb = used_mb / 1024
                                total_gb = total_mb / 1024
                                percent = (used_mb / total_mb) * 100
                                hw_info_lines.append(f"GPU {i} VRAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({percent:.1f}%)")
                                vram_found = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
            except Exception:
                pass
            
            # Try AMD GPU VRAM (if available)
            if not vram_found:
                try:
                    # Try reading from /sys filesystem on Linux for AMD
                    if platform.system() == "Linux":
                        import glob
                        amd_cards = glob.glob('/sys/class/drm/card*/device/mem_info_vram_*')
                        for card_path in amd_cards:
                            try:
                                with open(card_path, 'r') as f:
                                    vram_info = f.read().strip()
                                    # Parse AMD VRAM info if format is known
                                    hw_info_lines.append(f"AMD VRAM: {vram_info}")
                                    vram_found = True
                                    break
                            except:
                                continue
                except Exception:
                    pass
            
            if not vram_found:
                hw_info_lines.append("VRAM: No compatible GPU found")

        return ("\n".join(hw_info_lines),)

    @classmethod
    def IS_CHANGED(cls, refresh_rate, include_cpu_temp, include_gpu_temp, include_ram, include_cpu_usage, include_vram):
        # Force refresh based on refresh_rate
        import time
        return int(time.time() / refresh_rate)


_SMARTHELPER_GGUF_PKG = "SmartHelperGGUFPkg"


def _get_unet_gguf_filenames():
    try:
        return folder_paths.get_filename_list("unet_gguf")
    except KeyError:
        return []


def _ensure_comfyui_gguf_package():
    if _SMARTHELPER_GGUF_PKG in sys.modules:
        return
    root = None
    for custom_root in folder_paths.get_folder_paths("custom_nodes"):
        for name in ("ComfyUI-GGUF", "comfyui-gguf"):
            candidate = os.path.join(custom_root, name)
            if os.path.isfile(os.path.join(candidate, "__init__.py")):
                root = candidate
                break
        if root:
            break
    if not root:
        raise RuntimeError(
            "ComfyUI-GGUF is required for GGUF mode. Install from https://github.com/city96/ComfyUI-GGUF"
        )
    init_py = os.path.join(root, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        _SMARTHELPER_GGUF_PKG,
        init_py,
        submodule_search_locations=[root],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_SMARTHELPER_GGUF_PKG] = mod
    spec.loader.exec_module(mod)


def _load_gguf_model(unet_name: str):
    _ensure_comfyui_gguf_package()
    nodes_mod = importlib.import_module(f"{_SMARTHELPER_GGUF_PKG}.nodes")
    return nodes_mod.UnetLoaderGGUF().load_unet(unet_name)[0]


def _load_safetensors_diffusion_model(unet_name: str, weight_dtype: str):
    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    return comfy.sd.load_diffusion_model(unet_path, model_options=model_options)


class SmartModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gguf": ("BOOLEAN", {"default": False}),
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "gguf_name": (_get_unet_gguf_filenames(),),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "model_name")
    FUNCTION = "load"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Load a diffusion model from .safetensors (Load Diffusion Model) or .gguf (Unet Loader GGUF)."

    def load(self, gguf, unet_name, weight_dtype, gguf_name):
        if gguf:
            model = _load_gguf_model(gguf_name)
            return (model, gguf_name)
        model = _load_safetensors_diffusion_model(unet_name, weight_dtype)
        return (model, unet_name)


class SmartDualModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gguf": ("BOOLEAN", {"default": False}),
                "unet_name_1": (folder_paths.get_filename_list("diffusion_models"),),
                "unet_name_2": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "gguf_name_1": (_get_unet_gguf_filenames(),),
                "gguf_name_2": (_get_unet_gguf_filenames(),),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "STRING")
    RETURN_NAMES = ("model_1", "model_2", "model_names")
    FUNCTION = "load"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Load two diffusion models (safetensors or GGUF) with a combined name summary."

    def load(self, gguf, unet_name_1, unet_name_2, weight_dtype, gguf_name_1, gguf_name_2):
        if gguf:
            m1 = _load_gguf_model(gguf_name_1)
            m2 = _load_gguf_model(gguf_name_2)
            n1, n2 = gguf_name_1, gguf_name_2
        else:
            m1 = _load_safetensors_diffusion_model(unet_name_1, weight_dtype)
            m2 = _load_safetensors_diffusion_model(unet_name_2, weight_dtype)
            n1, n2 = unet_name_1, unet_name_2
        text = f"Model #1: {n1}\nModel #2: {n2}"
        return (m1, m2, text)


class SmartGetParametersAsString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["full_params", "node_status"], {"default": "full_params"}),
            },
            "optional": {
                "source": (any_type, {"tooltip": "Connect to any node output to read that node's parameters"}),
                "params_str": ("STRING", {"forceInput": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("params_str",)
    FUNCTION = "get_parameters"
    CATEGORY = "SmartHelperNodes"
    OUTPUT_NODE = True
    DESCRIPTION = "Connect to any node's output to read all its widget parameter names and values as a multi-line string."

    def get_parameters(self, mode, unique_id=None, prompt=None, extra_pnginfo=None, source=None, params_str=""):
        def _append_value(new_value):
            if params_str:
                return (f"{params_str}\n-------------------------------------\n{new_value}",)
            return (new_value,)

        if prompt is None or unique_id is None:
            return _append_value("")

        node_id = str(unique_id[0] if isinstance(unique_id, (list, tuple)) else unique_id)
        my_node = prompt.get(node_id)
        if not my_node or not isinstance(my_node, dict):
            return _append_value("")

        source_raw = (my_node.get("inputs") or {}).get("source")
        if source_raw is None or not is_link(source_raw):
            return _append_value("No source node connected.")

        source_node_id = str(source_raw[0])
        source_node = prompt.get(source_node_id)
        if not source_node or not isinstance(source_node, dict):
            return _append_value(f"Could not find source node {source_node_id}.")

        node_title = source_node.get("class_type", "Unknown")
        node_status = "ON"
        workflow_node = _get_workflow_node_by_id(source_node_id, extra_pnginfo)
        if workflow_node:
            node_title = workflow_node.get("title") or node_title
            node_mode = workflow_node.get("mode", 0)
            if node_mode == 2:
                node_status = "muted"
            elif node_mode == 4:
                node_status = "bypassed"

        if mode == "node_status":
            return _append_value(f"{node_title}: {node_status}")

        if node_status != "ON":
            return _append_value(f"NODE: {node_title}, {node_status}")

        source_inputs = source_node.get("inputs", {})
        lines = [f"NODE: {node_title}"]

        for param_name, param_value in source_inputs.items():
            if is_link(param_value):
                lines.append(f"{param_name}: <linked>")
            else:
                lines.append(f"{param_name}: {param_value}")

        return _append_value("\n".join(lines))


class SmartBypassSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        dynamic_inputs = {
            "input1": (any_type, {"lazy": True, "tooltip": "Value to pass through when control 1 is active."}),
            "control1": (any_type, {"lazy": True, "tooltip": "Connect any node output here. If that source node is active, input 1 is selected."}),
        }

        stack = inspect.stack()
        if len(stack) > 1 and stack[1].function == "get_input_info":
            class AllContainer:
                def __contains__(self, item):
                    return True

                def __getitem__(self, key):
                    if str(key).startswith("control"):
                        return any_type, {"lazy": True}
                    return any_type, {"lazy": True}

            dynamic_inputs = AllContainer()

        return {
            "required": {},
            "optional": dynamic_inputs,
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_type, "INT")
    RETURN_NAMES = ("value", "selected_index")
    FUNCTION = "switch"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Selects the first input whose matching control source node is connected and not muted or bypassed."

    @staticmethod
    def _normalize_unique_id(unique_id):
        if unique_id is None:
            return None
        if isinstance(unique_id, (list, tuple)):
            if not unique_id:
                return None
            return str(unique_id[0])
        return str(unique_id)

    @classmethod
    def _get_candidate_indices(cls, node_inputs):
        indices = set()
        for key in (node_inputs or {}):
            if key.startswith("input") or key.startswith("control"):
                suffix = key[5:] if key.startswith("input") else key[7:]
                if suffix.isdigit():
                    indices.add(int(suffix))
        return sorted(indices)

    @classmethod
    def _get_selected_index(cls, prompt, unique_id, extra_pnginfo):
        if prompt is None:
            return 0

        node_id = cls._normalize_unique_id(unique_id)
        if node_id is None:
            return 0

        switch_node = prompt.get(node_id)
        if not isinstance(switch_node, dict):
            return 0

        node_inputs = switch_node.get("inputs") or {}
        for index in cls._get_candidate_indices(node_inputs):
            control_key = f"control{index}"
            control_raw = node_inputs.get(control_key)
            if control_raw is None or not is_link(control_raw):
                continue

            control_node_id = str(control_raw[0])
            workflow_node = _get_workflow_node_by_id(control_node_id, extra_pnginfo)
            if workflow_node:
                node_mode = workflow_node.get("mode", 0)
                if node_mode in (2, 4):
                    continue

            input_key = f"input{index}"
            input_raw = node_inputs.get(input_key)
            if input_raw is None or not is_link(input_raw):
                continue

            return index

        return 0

    def check_lazy_status(self, *args, **kwargs):
        selected_index = self._get_selected_index(
            kwargs.get("prompt"),
            kwargs.get("unique_id"),
            kwargs.get("extra_pnginfo"),
        )
        if selected_index <= 0:
            return []
        return [f"input{selected_index}"]

    def switch(self, unique_id=None, prompt=None, extra_pnginfo=None, **kwargs):
        selected_index = self._get_selected_index(prompt, unique_id, extra_pnginfo)
        if selected_index <= 0:
            return (None, 0)

        selected_key = f"input{selected_index}"
        return (kwargs.get(selected_key), selected_index)


NODE_CLASS_MAPPINGS = {
    "SmartHVLoraSelect": SmartHVLoraSelect,
    "SmartHVLoraStack": SmartHVLoraStack,
    "SmartFormatString": SmartFormatString,
    "SmartFormatString10": SmartFormatString10,
    "SmartSaveText": SmartSaveText,
    "SmartSaveAnimatedGIF": SmartSaveAnimatedGIF,
    "SmartRemoveComments": SmartRemoveComments,
    "SmartLoadLoRA": SmartLoadLoRA,
    "SmartLoadDoubleLoRA": SmartLoadDoubleLoRA,
    "SmartPrompt": SmartPrompt,
    "SmartModelOrLoraToString": SmartModelOrLoraToString,
    "SmartShowAnything": SmartShowAnything,
    "SmartHWMonitor": SmartHWMonitor,
    "SmartSetInt": SmartSetInt,
    "SmartSetFloat": SmartSetFloat,
    "SmartBusIn": SmartBusIn,
    "SmartBusOut": SmartBusOut,
    "SmartBusIn5": SmartBusIn5,
    "SmartBusOut5": SmartBusOut5,
    "SmartModelInfo": SmartModelInfo,
    "SmartModelLoader": SmartModelLoader,
    "SmartDualModelLoader": SmartDualModelLoader,
    "SmartGetParametersAsString": SmartGetParametersAsString,
    "SmartBypassSwitcher": SmartBypassSwitcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartHVLoraSelect": "Smart HunyuanVideo Lora Select",
    "SmartHVLoraStack": "Smart HunyuanVideo Lora Stack",
    "SmartFormatString": "Smart Format String",
    "SmartFormatString10": "Smart Format String (10 params)",
    "SmartSaveText": "Smart Save Text File",
    "SmartSaveAnimatedGIF": "Smart Save Animated GIF",
    "SmartRemoveComments": "Smart Remove Comments",
    "SmartLoadLoRA": "Smart Load LoRA",
    "SmartLoadDoubleLoRA": "Smart Load Double LoRA",
    "SmartPrompt": "Smart Prompt",
    "SmartModelOrLoraToString": "Smart Model or Lora to String",
    "SmartShowAnything": "Smart Show Anything",
    "SmartHWMonitor": "Smart Hardware Monitor",
    "SmartSetInt": "Smart Set Int",
    "SmartSetFloat": "Smart Set Float",
    "SmartBusIn": "Smart Bus In",
    "SmartBusOut": "Smart Bus Out",
    "SmartBusIn5": "Smart Bus In 5",
    "SmartBusOut5": "Smart Bus Out 5",
    "SmartModelInfo": "Smart Model Info",
    "SmartModelLoader": "Smart Model Loader",
    "SmartDualModelLoader": "Smart Dual Model Loader",
    "SmartGetParametersAsString": "Smart Get Parameters As String",
    "SmartBypassSwitcher": "Bypass Switcher",
}

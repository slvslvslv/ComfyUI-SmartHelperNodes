import os
import folder_paths
import comfy
import comfy.sd # Added import for CLIP type access
import json
import sys
import re # Added for regex operations
import random # Added for random choice
import time # Added for IS_CHANGED

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
        
        # Process numbered parameters as before
        for i in range(1, self.max_param_num + 1):
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
            
        # Append to file (or create if doesn't exist)
        with open(full_path, "a+", encoding="utf-8") as f:
            # Add newline if file is not empty and doesn't end with one
            if f.tell() != 0:
                f.seek(0)
                content = f.read()
                if content and not content.endswith('\n'):
                    f.write('\n')
            # Write the new text
            f.write(text)
            
        return (text,)

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
                "lora_string": ("STRING", {"forceInput": True, "default": ""})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "lora_string")
    FUNCTION = "load_lora"
    CATEGORY = "SmartHelperNodes"
    DESCRIPTION = "Load a LoRA model (just like the core LoRA loader), but also outputs the model name and strength to a formatted string."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, lora_string=""):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, lora_string)

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
        
        # Create formatted string output
        lora_name_without_ext = lora_name.split(".")[0]
        new_lora_string = f"{lora_name_without_ext}, model_str:{strength_model}, clip_str:{strength_clip}"
        if lora_string:
            final_lora_string = f"{lora_string}\n{new_lora_string}"
        else:
            final_lora_string = new_lora_string
            
        return (model_lora, clip_lora, final_lora_string)

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
        pattern = r'\{([^}]+?\|[^}]+?)\}'
        # Use time.time() to force re-execution if the random syntax is detected
        # Otherwise, rely on the hash of the text for standard change detection.
        if re.search(pattern, text):
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

NODE_CLASS_MAPPINGS = {
    "SmartHVLoraSelect": SmartHVLoraSelect,
    "SmartHVLoraStack": SmartHVLoraStack,
    "SmartFormatString": SmartFormatString,
    "SmartFormatString10": SmartFormatString10,
    "SmartSaveText": SmartSaveText,
    "SmartRemoveComments": SmartRemoveComments,
    "SmartLoadLoRA": SmartLoadLoRA,
    "SmartPrompt": SmartPrompt,
    "SmartModelOrLoraToString": SmartModelOrLoraToString,
    "SmartShowAnything": SmartShowAnything,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmartHVLoraSelect": "Smart HunyuanVideo Lora Select",
    "SmartHVLoraStack": "Smart HunyuanVideo Lora Stack",
    "SmartFormatString": "Smart Format String",
    "SmartFormatString10": "Smart Format String (10 params)",
    "SmartSaveText": "Smart Save Text File",
    "SmartRemoveComments": "Smart Remove Comments",
    "SmartLoadLoRA": "Smart Load LoRA",
    "SmartPrompt": "Smart Prompt",
    "SmartModelOrLoraToString": "Smart Model or Lora to String",
    "SmartShowAnything": "Smart Show Anything",
}

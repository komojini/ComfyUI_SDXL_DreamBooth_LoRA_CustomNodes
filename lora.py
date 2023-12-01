import os
import sys
from pathlib import Path
import shutil
import torch
import comfy.sd
import comfy.utils
import folder_paths
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from .s3_utils import download_file


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

class XLDB_LoRA:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        self.loaded_lora = None
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "None")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (file_list, ),
                
            }
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"


    def load_lora(self, model, clip, lora_name):
        """
            The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
            For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.

            Arguments:
                - model (`MODEL`): The model object
                - clip (`CLIP`): The clip object
                - lora_name (`string`): The name of the lora file

            Returns: `tuple`:
                - First value is a `MODEL` object
                - Secound value is a `CLIP` object
        """
        strength_model = 1.0
        strength_clip = 1.0

        if lora_name == "None":
            return model, clip
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                del self.loaded_lora
        
        if lora is None:
            if lora_path and "checkpoint" in lora_path:
                
                lora = self.load_checkpoint_lora(model, clip, lora_path)
                self.loaded_lora = (lora_path, lora)
            else:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)

        return model_lora, clip_lora
    
    
    def load_checkpoint_lora(self, model, clip, lora_path):
        lora_path = Path(lora_path).parent
        lora_path = os.path.join(lora_path, "pytorch_lora_weights.bin")
        lora_path = str(lora_path)
        lora_weights = torch.load(lora_path)
        return lora_weights


class S3Bucket_Load_LoRA:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        # file_list = folder_paths.get_filename_list("loras")
        # file_list.insert(0, "None")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": ("STRING", {
                    "multiline": False,
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
               "strength_clip": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                }),    
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"


    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        """
            The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
            For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.

            Arguments:
                - model (`MODEL`): The model object
                - clip (`CLIP`): The clip object
                - lora_name (`string`): The name of the lora file

            Returns: `tuple`:
                - First value is a `MODEL` object
                - Secound value is a `CLIP` object
        """

        if lora_name == "None":
            return model, clip
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if not lora_path:
            new_lora_path = Path("tmp") / "loras"
            if os.path.exists(new_lora_path):
                shutil.rmtree(new_lora_path, ignore_errors=False, onerror=None)
                os.makedirs(new_lora_path, exist_ok=True)
            folder_paths.add_model_folder_path("loras", new_lora_path)
            lora_path = new_lora_path / lora_name
            if not os.path.exists(lora_path):
                os.makedirs(Path(lora_path).parent)
            download_file(bucket_path=lora_name, file_path=lora_path)

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                del self.loaded_lora
        
        lora_path = str(lora_path)
        
        if lora is None:
            if lora_path and "checkpoint" in lora_path:
                
                lora = self.load_checkpoint_lora(model, clip, lora_path)
                self.loaded_lora = (lora_path, lora)
            else:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)

        return model_lora, clip_lora
    
    
    def load_checkpoint_lora(self, model, clip, lora_path):
        lora_path = Path(lora_path).parent
        lora_path = os.path.join(lora_path, "pytorch_lora_weights.bin")
        lora_path = str(lora_path)
        lora_weights = torch.load(lora_path)
        return lora_weights 
    

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "XLDB_LoRA": XLDB_LoRA,
    "S3Bucket_Load_LoRA": S3Bucket_Load_LoRA,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "XLDB_LoRA": "XLDB LoRA",
    "S3Bucket_Load_LoRA": "S3 Bucket Load LoRA",
}

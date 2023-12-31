import os
import sys
from pathlib import Path
import shutil
import torch
import uuid
import comfy.sd
import comfy.utils
import folder_paths
from .s3_utils import download_file_from_s3_bucket, download_file_from_url


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
            },
        }
        
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"


    def load_lora(self, model, clip, lora_name, **bucket_creds):
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
        self.lora_name_map = {}

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
                "remote_lora_path_or_url": ("STRING", {
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
            },
            "optional": {
                "BUCKET_ENDPOINT_URL": ("STRING", {"default": ""}),
                "BUCKET_ACCESS_KEY_ID": ("STRING", {"default": ""}),
                "BUCKET_SECRET_ACCESS_KEY": ("STRING", {"default": ""}),
                "BUCKET_NAME": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    def get_local_lora_name(self, lora_name: str):
        if lora_name in self.lora_name_map:
            return self.lora_name_map[lora_name]
        if lora_name.endswith(".safetensors") or "checkpoint" in lora_name:
            self.lora_name_map[lora_name] = lora_name
        else:
            self.lora_name_map[lora_name] = f"{uuid.uuid4()}.safetensors"
        return self.lora_name_map[lora_name]

    def load_lora(self, model, clip, remote_lora_path_or_url, strength_model, strength_clip, **bucket_creds):
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
        for key, value in bucket_creds.items():
            if value:
                os.environ[key] = value

        if remote_lora_path_or_url == "None" or not remote_lora_path_or_url:
            return model, clip

        local_lora_path = folder_paths.get_full_path("loras", remote_lora_path_or_url)
        lora = None
        if not local_lora_path:
            new_lora_dir = Path("/tmp") / "loras"
            lora_url_or_path = remote_lora_path_or_url
            
            if os.path.exists(new_lora_dir):
                shutil.rmtree(new_lora_dir, ignore_errors=False, onerror=None)
            
            os.makedirs(new_lora_dir, exist_ok=True)
            folder_paths.add_model_folder_path("loras", new_lora_dir)
            
            local_lora_name = self.get_local_lora_name(remote_lora_path_or_url)
        
            local_lora_path = new_lora_dir / local_lora_name
            if not os.path.exists(local_lora_path.parent):
                os.makedirs(local_lora_path.parent, exist_ok=True)
            
            local_lora_path = str(local_lora_path)
            if "drive.google" in lora_url_or_path:
                local_lora_path = download_file_from_url(url=lora_url_or_path, download_path=local_lora_path)

                # import gdown
                # gdown.download(lora_url_or_path, local_lora_path, quiet=False)
            else:
                local_lora_path = download_file_from_s3_bucket(bucket_file_path=lora_url_or_path, download_path=local_lora_path)

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == local_lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None
        
        local_lora_path = str(local_lora_path)
        print(f"Downloaded LoRA path: {local_lora_path}")
        
        if lora is None:
            if local_lora_path and "checkpoint" in local_lora_path:
                
                lora = self.load_checkpoint_lora(model, clip, local_lora_path)
                self.loaded_lora = (local_lora_path, lora)
            else:
                lora = comfy.utils.load_torch_file(local_lora_path, safe_load=True)
                self.loaded_lora = (local_lora_path, lora)

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

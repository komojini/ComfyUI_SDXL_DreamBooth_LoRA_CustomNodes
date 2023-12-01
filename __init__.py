from .lora import *
from dotenv import load_dotenv

load_dotenv()

LIVE_NODE_CLASS_MAPPINGS = {
    "XL DreamBooth LoRA": XLDB_LoRA,
    "S3 Bucket LoRA": S3Bucket_Load_LoRA,
}

LIVE_NODE_DISPLAY_NAME_MAPPINGS = {
    "XL DreamBooth LoRA": "XL DreamBooth LoRA",
    "S3 Bucket LoRA": "S3 Bucket LoRA"
}
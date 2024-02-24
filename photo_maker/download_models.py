import os
import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline

current_directory =os.path.dirname(os.path.abspath(__file__))

hf_hub_download(repo_id="TencentARC/PhotoMaker", local_dir=f'{current_directory}/checkpoints', filename="photomaker-v1.bin", repo_type="model")

PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    'SG161222/RealVisXL_V3.0',
    torch_dtype=torch.bfloat16,
    cache_dir=f'{current_directory}/checkpoints',
    use_safetensors=True,
    variant="fp16"
)

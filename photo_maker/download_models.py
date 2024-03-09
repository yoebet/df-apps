import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline

hf_hub_download(repo_id="TencentARC/PhotoMaker",
                filename="photomaker-v1.bin", repo_type="model")

PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    'SG161222/RealVisXL_V3.0',
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="fp16"
)

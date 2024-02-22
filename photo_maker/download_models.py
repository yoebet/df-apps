import os
import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline

if not os.path.exists('checkpoints/photomaker-v1.bin'):
    hf_hub_download(repo_id="TencentARC/PhotoMaker", local_dir='checkpoints', filename="photomaker-v1.bin", repo_type="model")
    print('photomaker-v1.bin 下载成功')
else:
    print('photomaker-v1.bin 已经存在')

PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    'SG161222/RealVisXL_V3.0',
    torch_dtype=torch.bfloat16,
    cache_dir='checkpoints',
    use_safetensors=True,
    variant="fp16"
)

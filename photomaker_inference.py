import torch
import numpy as np
import random
import os
import sys

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from huggingface_hub import hf_hub_download
import spaces
import gradio as gr

from photomaker import PhotoMakerStableDiffusionXLPipeline

from utils.style_template import styles
from utils.aspect_ratio_template import aspect_ratios

# global variable
base_model_path = 'SG161222/RealVisXL_V3.0'
try:
    if torch.cuda.is_available():
        device = "cuda"
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"
ASPECT_RATIO_LABELS = list(aspect_ratios)
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_LABELS[0]

# download PhotoMaker checkpoint to cache
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker",destination_path='./checkpoints', filename="photomaker-v1.bin", repo_type="model")

if device == "mps":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.bfloat16

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch_dtype,
    use_safetensors=True, 
    variant="fp16",
    local_files_only=True
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)
pipe.id_encoder.to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
pipe.fuse_lora()


def inference(image_urls:list, prompt, negative_prompt, aspect_ratio_name, style_name, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed):
    # check the trigger word
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    input_ids = pipe.tokenizer.encode(prompt)
    if image_token_id not in input_ids:
        raise gr.Error(f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt! Please refer to step 2️⃣")

    if input_ids.count(image_token_id) > 1:
        raise gr.Error(f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!")

    # determine output dimensions by the aspect ratio
    output_w, output_h = aspect_ratios[aspect_ratio_name]
    print(f"[Debug] Generate image using aspect ratio [{aspect_ratio_name}] => {output_w} x {output_h}")

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)



    input_id_images = []
    for img in image_urls:
        input_id_images.append(load_image(img))
    
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(start_merge_step)
    images = pipe(
        prompt=prompt,
        width=output_w,
        height=output_h,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
    ).images
    return images

def apply_style(style_name: str, positive: str, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


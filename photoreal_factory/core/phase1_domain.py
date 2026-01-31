import cv2
import numpy as np
import torch
from PIL import Image

def run_domain_shift(img_bgr, pipe, params: dict):
    # BGR -> RGB -> PIL
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Seed固定 (Configから取得、なければ固定値)
    seed = params.get("seed", 2026)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # ControlNetを使う場合、control_imageが必要
    # ※SDXL TurboやRefinerのみの場合は不要だが、
    # 3DCGの形を維持するならControlNet Tile推奨
    
    result = pipe(
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        image=pil_image,           # Img2Img入力
        control_image=pil_image,   # ControlNet入力 (形状維持)
        strength=params["strength"], # 書き換え強度
        guidance_scale=params.get("guidance_scale", 7.5),
        controlnet_conditioning_scale=params.get("control_scale", 0.6), # 形状維持の強さ
        num_inference_steps=params["steps"],
        generator=generator,       # ★最重要: シード固定
        output_type="pil"
    ).images[0]

    return result
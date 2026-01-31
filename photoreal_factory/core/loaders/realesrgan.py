import torch
from realesrgan import RealESRGANer

def load_realesrgan():
    return RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        tile=1024,          # ← 96GB前提
        tile_pad=32,
        pre_pad=0,
        half=True           # fp16固定
    )

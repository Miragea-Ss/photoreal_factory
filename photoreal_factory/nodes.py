import torch
import numpy as np
import cv2
import os
import json
import folder_paths
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .core.vram_pool import VRAM_POOL
from .core.phase1_domain import run_domain_shift
from .core.phase2_upscale import run_upscale
from .core.phase3_physics import apply_film_physics


class PhotorealFactoryLive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                "scale_factor": ("INT", {"default": 2, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "PhotorealFactory"

    def process(self, image, upscale_model, scale_factor):
        upscale_path = folder_paths.get_full_path("upscale_models", upscale_model)
        upscaler = VRAM_POOL.load("upscaler", {"model_path": upscale_path})

        results = []
        for i in range(image.shape[0]):
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            up = run_upscale(img, upscaler, scale_factor)
            up = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            results.append(torch.from_numpy(up.astype(np.float32) / 255.0))

        return (torch.stack(results),)


class PhotorealFolderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "C:\\Images\\Input"}),
                "image_index": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "load_image"
    CATEGORY = "PhotorealFactory/IO"

    def load_image(self, folder_path, image_index):
        files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        path = os.path.join(folder_path, files[image_index % len(files)])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (torch.from_numpy(img).unsqueeze(0), Path(path).stem)


class PhotorealImageSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_folder": ("STRING", {"default": "C:\\Images\\Output"}),
                "filename": ("STRING", {"default": "output"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "PhotorealFactory/IO"

    def save(self, image, output_folder, filename):
        os.makedirs(output_folder, exist_ok=True)
        img = (image[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_folder, f"{filename}.png"))
        return {}
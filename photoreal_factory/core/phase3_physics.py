# core/phase3_physics.py
import numpy as np
import cv2

def apply_film_physics(img_rgb: np.ndarray, params: dict):
    img = img_rgb.astype(np.float32) / 255.0
    luma = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]

    # 輝度依存グレイン（暗部ほど強い）
    strength = params.get("grain_power", 0.05)
    noise = np.random.normal(0, strength, img.shape)
    noise *= (1.0 - luma[...,None])

    # エッジ保護
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_mask = (edges == 0)[...,None]

    img = img + noise * edge_mask
    img = np.clip(img, 0.0, 1.0)

    return (img * 255).astype(np.uint8)

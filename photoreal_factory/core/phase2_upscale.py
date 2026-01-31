import torch
import numpy as np
import cv2

def run_upscale(img_bgr, model, scale):
    """
    Spandrelモデルを使った高速アップスケール (Blackwell最適化)
    Input: BGR image (numpy)
    Output: BGR image (numpy)
    """
    # 1. 前処理 (BGR -> RGB -> Tensor)
    # 0-1に正規化
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # [H, W, C] -> [1, C, H, W]
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    
    # GPUへ転送 & FP16化
    img_t = img_t.to("cuda").half()

    # 2. 推論 (RTX 6000のVRAMがあればTiling不要で一発処理)
    with torch.no_grad():
        output_t = model(img_t)

    # 3. 後処理 (Tensor -> RGB -> BGR)
    # [1, C, H, W] -> [H, W, C]
    output_rgb = output_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    
    # 0-1 -> 0-255
    output_rgb = np.clip(output_rgb, 0, 1) * 255.0
    output_rgb = output_rgb.round().astype(np.uint8)

    # RGB -> BGR
    return cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
# core/optical_grain.py
import cv2
import numpy as np

class OpticalFlowGrainLock:
    def __init__(self, grain_power=0.06):
        self.prev_gray = None
        self.prev_grain = None
        self.grain_power = grain_power

    def _init_grain(self, shape):
        h, w = shape
        return np.random.normal(0, self.grain_power, (h, w)).astype(np.float32)

    def apply(self, frame_rgb: np.ndarray):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            # 初回フレーム
            grain = self._init_grain(gray.shape)
        else:
            # Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            h, w = gray.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)

            grain = cv2.remap(
                self.prev_grain,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

        # 輝度依存マスク（暗部強）
        luma = gray.astype(np.float32) / 255.0
        grain = grain * (1.0 - luma)

        self.prev_gray = gray
        self.prev_grain = grain

        # RGBに注入
        out = frame_rgb.astype(np.float32)
        for c in range(3):
            out[..., c] += grain * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)

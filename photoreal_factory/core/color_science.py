import numpy as np
import cv2


# ===============================
# sRGB → Linear
# ===============================
def linearize_srgb(img_8bit: np.ndarray) -> np.ndarray:
    img = img_8bit.astype(np.float32) / 255.0
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4
    )


# ===============================
# Log Curves
# ===============================
def apply_log_curve(linear: np.ndarray, curve="ARRI_LOGC") -> np.ndarray:
    if curve == "ARRI_LOGC":
        a = 5.555556
        b = 0.052272
        c = 0.247190
        d = 0.385537
        e = 5.367655

        return np.clip(
            (a * np.log10(b + c * linear) + d) / e,
            0.0,
            1.0
        )

    raise ValueError(f"Unknown log curve: {curve}")


def to_log_image(img_rgb_8bit: np.ndarray, curve="ARRI_LOGC") -> np.ndarray:
    linear = linearize_srgb(img_rgb_8bit)
    log_img = apply_log_curve(linear, curve)
    return (log_img * 65535.0).astype(np.uint16)


# ==========================================================
# EXR / ACEScg 追加（← ここが「追記」部分）
# ==========================================================

# Linear sRGB → ACEScg Matrix
ACESCG_MAT = np.array([
    [1.45143932, -0.23651075, -0.21492857],
    [-0.07655377,  1.17622970, -0.09967593],
    [0.00831615, -0.00603245,  0.99771630],
], dtype=np.float32)


def linear_to_acescg(linear_rgb: np.ndarray) -> np.ndarray:
    """
    linear sRGB → ACEScg
    input/output: float32, range unrestricted
    """
    h, w, _ = linear_rgb.shape
    reshaped = linear_rgb.reshape(-1, 3)
    aces = reshaped @ ACESCG_MAT.T
    return aces.reshape(h, w, 3)


def to_exr_acescg(img_rgb_8bit: np.ndarray) -> np.ndarray:
    """
    8bit sRGB → Linear → ACEScg (32bit float)
    EXR保存専用
    """
    linear = linearize_srgb(img_rgb_8bit)
    aces = linear_to_acescg(linear)
    return aces.astype(np.float32)

# core/phase4_log.py
from core.color_science import to_log_image

def convert_output(img_rgb, output_cfg: dict):
    if output_cfg["mode"] == "LOG":
        return to_log_image(
            img_rgb,
            curve=output_cfg.get("log_curve", "ARRI_LOGC")
        )
    return img_rgb

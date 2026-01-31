from pathlib import Path
import cv2
import imageio.v3 as iio
import numpy as np

from core.color_science import (
    to_log_image,
    to_exr_acescg
)


# ===============================
# Image Writer
# ===============================
def write_image(
    img_rgb: np.ndarray,
    out_path: Path,
    mode: str,
    log_curve: str = "ARRI_LOGC"
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "PNG":
        cv2.imwrite(
            str(out_path.with_suffix(".png")),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        )

    elif mode == "LOG":
        log_img = to_log_image(img_rgb, curve=log_curve)
        iio.imwrite(
            out_path.with_suffix(".tiff"),
            log_img
        )

    elif mode == "EXR":
        exr_img = to_exr_acescg(img_rgb)
        iio.imwrite(
            out_path.with_suffix(".exr"),
            exr_img
        )

    else:
        raise ValueError(f"[IO] Unknown output mode: {mode}")


# ===============================
# Image Sequence Writer
# ===============================
def write_sequence(
    img_rgb: np.ndarray,
    out_dir: Path,
    frame_idx: int,
    mode: str,
    log_curve: str = "ARRI_LOGC"
):
    name = f"frame_{frame_idx:06d}"
    write_image(
        img_rgb,
        out_dir / name,
        mode,
        log_curve
    )


# ===============================
# Video Writer (Final Encode)
# ===============================
def write_video(frames, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        fps,
        (w, h)
    )

    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    writer.release()

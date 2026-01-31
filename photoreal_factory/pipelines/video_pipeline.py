# pipelines/video_pipeline.py
from pathlib import Path
import cv2
import numpy as np
import torch

from core.vram_pool import VRAM_POOL
from core.phase1_domain import run_domain_shift
from core.phase2_upscale import run_upscale
from core.phase3_physics import apply_film_physics
from core.phase4_log import convert_output


def run_video_pipeline(input_path: Path, output_path: Path, profile: dict):
    input_path = Path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ===============================
    # Load models (VRAM常駐)
    # ===============================
    upscaler = VRAM_POOL.load(
        "upscaler",
        profile["loaders"]["upscale"]
    )

    img2img = VRAM_POOL.load(
        "img2img",
        profile["loaders"]["img2img"]
    )

    # ===============================
    # Video IO
    # ===============================
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * profile["upscale"]["scale"])
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * profile["upscale"]["scale"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
        True
    )

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] Frames: {total}")

    # ===============================
    # 固定seed（時間一貫性）
    # ===============================
    seed = profile.get("video", {}).get("seed", 123456)
    generator = torch.Generator("cuda").manual_seed(seed)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        print(f"[VIDEO] {idx}/{total}")

        try:
            # ---- Phase 2: Upscale ----
            up = run_upscale(
                frame,
                upscaler,
                profile["upscale"]["scale"]
            )

            # ---- Phase 1: Domain Shift ----
            result = img2img(
                image=run_domain_shift(
                    up,
                    img2img,
                    profile["img2img"]
                ),
                generator=generator
            ).images[0]

            frame_rgb = np.array(result)

            # ---- Phase 3: Film Physics ----
            phys = apply_film_physics(
                frame_rgb,
                profile["film_physics"]
            )

            # ---- Phase 4: Output ----
            out = convert_output(
                phys,
                profile["output"]
            )

            if profile["output"]["mode"] == "LOG":
                # LOG動画は一旦RGBに戻して格納
                out = (out.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

            writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"[VIDEO ERROR] frame {idx}: {e}")
            torch.cuda.empty_cache()
            continue

    cap.release()
    writer.release()

    print("[VIDEO] Complete")

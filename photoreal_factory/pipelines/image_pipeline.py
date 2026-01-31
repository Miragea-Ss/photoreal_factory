# pipelines/image_pipeline.py
from pathlib import Path
import cv2
import numpy as np
import torch

from core.vram_pool import VRAM_POOL
from core.phase1_domain import run_domain_shift
from core.phase2_upscale import run_upscale
from core.phase3_physics import apply_film_physics
from core.phase4_log import convert_output


def run_image_pipeline(input_dir: Path, output_dir: Path, profile: dict):
    input_dir = Path(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===============================
    # Load models (VRAM常駐)
    # ===============================
    upscaler = VRAM_POOL.load(
        "upscaler",
        profile["loaders"]["upscale"]
    )

    img2img_pipe = VRAM_POOL.load(
        "img2img",
        profile["loaders"]["img2img"]
    )

    # ===============================
    # Collect files (量産前提)
    # ===============================
    files = []
    for ext in profile["io"]["extensions"]:
        files.extend(input_dir.rglob(ext))

    if not files:
        raise RuntimeError("[FACTORY] No images found")

    total = len(files)
    print(f"[FACTORY] Image jobs: {total}")

    # ===============================
    # Main loop（1000枚でも止まらない）
    # ===============================
    for idx, path in enumerate(sorted(files), 1):
        print(f"[IMAGE] {idx}/{total} {path.name}")

        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            # ---- Phase 2: Upscale ----
            up = run_upscale(
                img,
                upscaler,
                profile["upscale"]["scale"]
            )

            # ---- Phase 1: Domain Shift ----
            photoreal = run_domain_shift(
                up,
                img2img_pipe,
                profile["img2img"]
            )

            photoreal_np = np.array(photoreal)

            # ---- Phase 3: Film Physics ----
            phys = apply_film_physics(
                photoreal_np,
                profile["film_physics"]
            )

            # ---- Phase 4: Output Color ----
            out = convert_output(
                phys,
                profile["output"]
            )

            # ---- Save ----
            stem = path.stem
            if profile["output"]["mode"] == "LOG":
                save_path = output_dir / f"{stem}_cinema_log.tiff"
                from imageio.v3 import imwrite
                imwrite(save_path, out)
            else:
                save_path = output_dir / f"{stem}_cinema.png"
                cv2.imwrite(
                    str(save_path),
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                )

        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")
            torch.cuda.empty_cache()
            continue

    print("[FACTORY] Image pipeline complete")

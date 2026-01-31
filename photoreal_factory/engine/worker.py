# engine/worker.py
from pathlib import Path


def run_job(input_path: str, output_path: str, profile: dict):
    """
    工場ジョブディスパッチャ
    - input / output は文字列パス
    - profile は profiles 側で完全定義された dict
    """

    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if not input_dir.exists():
        raise RuntimeError(f"[FACTORY] Input path not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    job_type = profile["type"]

    # ===============================
    # Image pipeline
    # ===============================
    if job_type == "image":
        from pipelines.image_pipeline import run_image_pipeline

        run_image_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            profile=profile
        )

    # ===============================
    # Video pipeline
    # ===============================
    elif job_type == "video":
        from pipelines.video_pipeline import run_video_pipeline

        run_video_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            profile=profile
        )

    # ===============================
    # Future extension point
    # ===============================
    else:
        raise RuntimeError(f"[FACTORY] Unknown job type: {job_type}")

from pathlib import Path
import cv2
import imageio.v3 as iio


# ===============================
# Image Loader
# ===============================
def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"[IO] Failed to load image: {path}")
    return img


# ===============================
# Image Sequence Loader
# ===============================
def load_image_sequence(dir_path: Path, extensions):
    files = []
    for ext in extensions:
        files.extend(dir_path.glob(ext))

    if not files:
        raise RuntimeError(f"[IO] No input files in {dir_path}")

    return sorted(files)


# ===============================
# Video Loader (Generator)
# ===============================
def load_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[IO] Cannot open video: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        yield idx, frame

    cap.release()

def apply_film_physics(img_rgb: np.ndarray, params: dict):
    img = img_rgb.astype(np.float32) / 255.0

    # 周波数分離
    low = cv2.GaussianBlur(img, (0, 0), sigmaX=4)
    high = img - low

    # 輝度依存ノイズ
    luma = 0.2126*img[...,0] + 0.7152*img[...,1] + 0.0722*img[...,2]
    grain = np.random.normal(0, params["grain_power"], img.shape)
    grain *= (1.0 - luma[...,None])

    # 高周波のみ注入
    img = low + high * 1.1 + grain * 0.7
    img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)

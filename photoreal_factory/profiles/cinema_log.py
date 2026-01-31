# profiles/cinema_log.py

# 工場の稼働設定書 (Blackwell GPU用)
PROFILE = {
    "type": "image",  # ジョブの種類
    
    # 対象ファイルの拡張子
    "io": {
        "extensions": ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    },

    # AIモデルの読み込み設定
    "loaders": {
        "upscale": {
            # ★重要: ここに .pth ファイルがあるか確認してください
            "model_path": r"L:\Tools\photoreal_factory\models\RealESRGAN_x4plus.pth"
        },
        "img2img": {
            # 初回自動ダウンロード
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0"
        }
    },

    # Phase 2: アップスケール倍率
    "upscale": {
        "scale": 4
    },

    # Phase 1: 3DCG -> 実写変換設定
    "img2img": {
        "prompt": "cinematic raw photo, intricate skin texture, 8k uhd, dslr, film grain, masterpiece",
        "negative_prompt": "3d render, cgi, anime, cartoon, plastic skin, blur, low quality, distorted",
        
        "strength": 0.35,        # 書き換え強度 (0.3〜0.5推奨)
        "steps": 25,             # 画質ステップ数
        "guidance_scale": 7.5,
        "control_scale": 0.6,    # 形を維持する強さ
        "seed": 2026             # ★Seed固定（結果を一定にする）
    },

    # Phase 3: フィルム物理シミュレーション
    "film_physics": {
        "grain_power": 0.08,     # 粒子の強さ (0.05-0.15)
        "protect_edges": True,   # エッジを保護するか
        "seed": 2026             # 空間ノイズ固定
    },

    # Phase 4: 出力設定
    "output": {
        "mode": "PNG" 
    }
}
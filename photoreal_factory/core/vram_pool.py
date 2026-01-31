import torch
import gc
import spandrel # ★ここが最新ライブラリ

class VRAMPool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VRAMPool, cls).__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def load(self, key: str, config: dict):
        if key in self._models:
            return self._models[key]

        print(f"[VRAM_POOL] Loading new model: {key}")
        model = self._instantiate_model(key, config)
        self._models[key] = model
        return model

    def _instantiate_model(self, key, config):
        # 1. アップスケーラー (Spandrel使用)
        if "upscale" in key or "upscaler" in key:
            path = config["model_path"]
            print(f"  -> Loading Upscaler from: {path}")
            
            # Spandrelがモデル構造(RealESRGAN, HAT, SwinIRなど)を自動判定してロード
            model_descriptor = spandrel.ModelLoader().load_from_file(path)
            model = model_descriptor.model.to("cuda")
            model.eval() # 推論モード
            # ハーフ精度 (RTX 6000 BlackwellならFP16で爆速)
            model = model.half()
            return model

        # 2. 生成AI (SDXL)
        elif "img2img" in key:
            from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
            
            print("  -> Loading SDXL ControlNet Pipeline...")
            cnet = ControlNetModel.from_pretrained(
                "xinsir/controlnet-tile-sdxl-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=cnet,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
            return pipe

        else:
            raise ValueError(f"Unknown key: {key}")

    def unload_all(self):
        self._models.clear()
        gc.collect()
        torch.cuda.empty_cache()

VRAM_POOL = VRAMPool()
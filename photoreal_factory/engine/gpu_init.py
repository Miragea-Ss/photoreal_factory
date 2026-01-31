import torch
import os

def init_gpu():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # 再現性優先
    torch.backends.cudnn.deterministic = True

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:512,"
        "garbage_collection_threshold:0.8"
    )

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()

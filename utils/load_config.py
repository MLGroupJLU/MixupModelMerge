import torch

if torch.cuda.is_available():
    # cache_dir = "/mnt/data/yule/.cache"
    cache_dir = "/root/autodl-fs"
else:
    cache_dir = "/root/autodl-fs"
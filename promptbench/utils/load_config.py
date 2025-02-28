import torch

if torch.cuda.is_available():
    # cache_dir = "/mnt/data/yule/.cache"
    cache_dir = "/home/wangxu/data_zy"
else:
    cache_dir = "/Users/yule/.cache"
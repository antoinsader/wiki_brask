import os
import psutil
import torch


def log_resource_usage(step: int, use_cuda: bool) -> None:
    """Print CPU RAM and GPU VRAM usage at the given step."""
    process   = psutil.Process(os.getpid())
    cpu_gb    = process.memory_info().rss / 1024 ** 3

    if use_cuda:
        vram_used_gb  = torch.cuda.memory_allocated() / 1024 ** 3
        vram_reserved_gb = torch.cuda.memory_reserved()  / 1024 ** 3
        print(f"  [step {step}] CPU RAM: {cpu_gb:.2f} GB | "
              f"VRAM allocated: {vram_used_gb:.2f} GB  reserved: {vram_reserved_gb:.2f} GB")
    else:
        print(f"  [step {step}] CPU RAM: {cpu_gb:.2f} GB")

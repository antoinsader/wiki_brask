import os
import psutil
import torch
import ctypes
import numpy as np

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


def drop_mmap_pages(mm: np.memmap, start_row: int, end_row: int):
    """
    Advise the OS to drop pages for written rows from the page cache.
    Only works on Linux. Silently no-ops on other platforms.
    """
    try:
        row_bytes  = mm.strides[0]
        offset     = start_row * row_bytes
        length     = (end_row - start_row) * row_bytes
        # MADV_DONTNEED = 4 on Linux
        ctypes.cdll.LoadLibrary("libc.so.6").madvise(
            ctypes.c_void_p(mm.ctypes.data + offset),
            ctypes.c_size_t(length),
            ctypes.c_int(4)
        )
    except Exception:
        pass


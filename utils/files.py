import pickle
import os
import numpy as np
import torch
from tqdm import tqdm



def cache_array(ar, filename):
    with open(filename, 'wb') as f:
        pickle.dump(ar, f)
    print(f"Array chached in file {filename}")


def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def save_tensor(tensor: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tensor.cpu().numpy())
    print(f"Tensor cached in file {path}")



def read_tensor(path: str, mmap: bool = False) -> torch.Tensor:
    print(f"Reading from {path}")
    if mmap:
        arr = np.load(path, mmap_mode='r')
        return torch.from_numpy(arr)
    else:
        arr = np.load(path)
        return torch.from_numpy(arr.copy())

def scan_text_file_lines(fp, scan_head_ids= False):
    total = 0
    head_ids = set()
    with open(fp, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"scanning {fp} lines"):
            if scan_head_ids:
                head_ids.add(line.split("\t", 1)[0])
            total +=1
    return total, head_ids


def init_mmap(path: str, shape: tuple, dtype: str) -> np.memmap:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.lib.format.open_memmap(
        path,
        mode='w+',
        dtype=dtype,
        shape=shape
    )

import torch
import pickle
import numpy as np
from transformers import  BertTokenizerFast

def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)
def read_tensor(path: str, mmap: bool = False) -> torch.Tensor:
    print(f"Reading from {path}")
    if mmap:
        arr = np.load(path, mmap_mode='r')
        return torch.from_numpy(arr)
    else:
        arr = np.load(path)
        return torch.from_numpy(arr.copy())

ids_all   = read_cached_array("../data/minimized/description_embeddings_ids.pkl")
mask_all  = read_tensor("../data/minimized/description_embeddings_all_masks.npz.npy")

golden_triples = read_cached_array("../data/minimized/golden_triples.pkl")


total_real_tokens = 0
total_positives   = 0

real_lens = mask_all.sum(dim=1).int()  # (N,)
id_to_real_len = dict(zip(ids_all, real_lens.tolist()))


for entity_id, triples in golden_triples.items():
    real_length = id_to_real_len[entity_id]
    total_real_tokens += real_length
    seen_head_spans = set()
    seen_tail_spans = set()
    for (hs, he), _, (ts, te) in triples:
        if (hs, he) not in seen_head_spans and hs < real_length:
            seen_head_spans.add((hs, he))
            total_positives += 2  # start + end token

        if (ts, te) not in seen_tail_spans and ts < real_length:
            seen_tail_spans.add((ts, te))
            total_positives += 2

del  ids_all, mask_all, golden_triples

descriptions = read_cached_array("../data/minimized/descriptions.pkl")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
enc = tokenizer(
    list(descriptions.values()),
    truncation=False,        # you want real lengths, not truncated
    padding=False,
    return_attention_mask=False,
    return_token_type_ids=False,
)

lengths = [len(inp) for inp in enc['input_ids']]

lengths_arr  = np.array(lengths)

total_negatives = total_real_tokens - total_positives
ratio = total_negatives / max(total_positives, 1)
print(f"Real imbalance ratio (Suggested pos_weight): {ratio:.1f}, Negatives: {total_negatives}, Positives: {total_positives}")

print("Description lengths information: ")
print(f"Count:       {len(lengths_arr)}")
print(f"Min:         {lengths_arr.min()}")
print(f"Max:         {lengths_arr.max()}")
print(f"Mean:        {lengths_arr.mean():.1f}")
print(f"Median:      {np.median(lengths_arr):.1f}")
print(f"Std:         {lengths_arr.std():.1f}")
print(f"80th pct:    {np.percentile(lengths_arr, 80):.1f}")
print(f"85th pct:    {np.percentile(lengths_arr, 85):.1f}")
print(f"90th pct:    {np.percentile(lengths_arr, 90):.1f}")
print(f"95th pct:    {np.percentile(lengths_arr, 95):.1f}")
print(f"99th pct:    {np.percentile(lengths_arr, 99):.1f}")
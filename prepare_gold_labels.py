# so my goal here is to generate gold labels for 
# (forward_head_start, forward_head_end, backward_tail_start, backward_tail_end) each is a tensor with shape (B, L, 1) having 0,1 values
# (forward_tail_start, forward_tail_end, backward_head_start, backward_tail_end) each is a tensor with shape (B, R, S, L, 1) where S is maximum number of heads for forward, maximum number of tails for backward


import json
import os
import pickle
from sympy import EX
import torch
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, shared_memory
from tqdm import tqdm

from operations.tokenizer import tokenize

from utils.files import cache_array, init_mmap, read_cached_array, read_tensor, save_tensor
from utils.settings import settings
from utils.chunking import  chunk_list
from utils.pre_processed_data import data_loader
from utils.helpers import create_aliases_patterns_map

# For multiprocessing
_ALIASES_PATTERNS_MAP = None
_ALIASES_DICT = None
_DESCRIPTIONS = None

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def share_dict(d: dict) -> tuple[shared_memory.SharedMemory, int]:
    """Serialize dictionary and store it in shared memory. to not being serialized for each process"""
    encoded = pickle.dumps(d, protocol=5)
    size = len(encoded)
    shm = shared_memory.SharedMemory(create=True, size=size)
    shm.buf[:size] = encoded
    return shm, size


def init_worker_discover_aliases(
        aliases_patterns_map, 
        shm_aliases_name, 
        shm_aliases_size, 
        shm_descriptions_name, 
        shm_descriptions_size):
    """Initializer for worker processes in discover_aliases. Loads shared memory data into global variables."""
    global   _ALIASES_PATTERNS_MAP, _ALIASES_DICT, _DESCRIPTIONS
    _ALIASES_PATTERNS_MAP = aliases_patterns_map

    shm_aliases = shared_memory.SharedMemory(name=shm_aliases_name)

    _ALIASES_DICT = pickle.loads(bytes(shm_aliases.buf[:shm_aliases_size]))
    shm_aliases.close()

    shm_descriptions = shared_memory.SharedMemory(name=shm_descriptions_name)
    _DESCRIPTIONS = pickle.loads(bytes(shm_descriptions.buf[:shm_descriptions_size]))
    shm_descriptions.close()


def chunk_description_discover_aliases_spans(
        process_idx: int,
        triples_chunk: list, 
        max_descriptions_length: int
    ) -> tuple[dict, dict]:
    """Discover triple_spans for each triple in the chunk, triple_spans is a dict {head_entity_id: [(head_start_idx, head_end_idx, rel_idx, tail_start_idx, tail_end_idx), ... for each triple with this head]} 
    Parameters:
    ----------
    triples_chunk:  list[(head_entity_id, relation_id, tail_entity_id), ..]

    Returns:
    ----------
    tuple: [triple_spans: dict, not_found_triples: dict]
        triple_spans: dict {head_entity_id: [(head_start_idx, head_end_idx, rel_idx, tail_start_idx, tail_end_idx), ... for each triple with this head]}
        not_found_triples: dict {head_entity_id: [(relation_id, tail_entity_id), ... for each triple with this head that its head aliases or tail aliases were not found in the description]}
    """
    not_found_triples = defaultdict(list)
    triples_head_ids = set([t[0] for t in triples_chunk ])

    descriptions = {k:  _DESCRIPTIONS[k] for k in triples_head_ids if k in _DESCRIPTIONS}
    description_idx_map = {entity_id: idx for idx, entity_id in enumerate(descriptions.keys())}


    print(f"[PROCESS_{process_idx}]: Tokenizing")
    enc = tokenize(list(descriptions.values()), max_descriptions_length)
    triple_spans = defaultdict(list)  # h -> [(head_spans, rel_idx, tail_spans), ...]
    print(f"[PROCESS_{process_idx}]: extracting")



    triples_by_head = defaultdict(list)
    for h, r, t in triples_chunk:
        if h in description_idx_map:
            triples_by_head[h].append((r, t))

    for h, related_triples in tqdm(triples_by_head.items(), desc=f"[PROCESS_{process_idx}] Extracting aliases from triples"):
        description_text = descriptions[h]
        description_idx = description_idx_map[h]

        # Compute head spans once per head entity
        head_spans_found = set()
        if h not in _ALIASES_DICT:
            not_found_triples[h].extend(related_triples)
            continue
        for pattern in [_ALIASES_PATTERNS_MAP[als_str] for als_str in _ALIASES_DICT[h]]:
            for match in pattern.finditer(description_text):
                char_start, char_end = match.span()
                if char_end == 0 or char_start >= char_end:
                    continue
                ts = enc.char_to_token(description_idx, char_start)
                te = enc.char_to_token(description_idx, char_end - 1)
                if ts is not None and te is not None:
                    head_spans_found.add((ts, te))

        # Cache tail spans per tail entity to avoid re-running patterns for the same (h, t)
        tail_spans_cache = {}
        for r, t in related_triples:
            tail_spans = set()
            if t not in _ALIASES_DICT:
                not_found_triples[h].append((r, t))
                continue
            if t not in tail_spans_cache:

                for pattern in [_ALIASES_PATTERNS_MAP[als_str] for als_str in _ALIASES_DICT[t]]:
                    for match in pattern.finditer(description_text):
                        char_start, char_end = match.span()
                        if char_end == 0 or char_start >= char_end:
                            continue
                        ts = enc.char_to_token(description_idx, char_start)
                        te = enc.char_to_token(description_idx, char_end - 1)
                        if ts is not None and te is not None:
                            tail_spans.add((ts, te))
                tail_spans_cache[t] = tail_spans
            if len(tail_spans_cache[t]) == 0:
                not_found_triples[h].append((r, t))
                continue

            for h_span in head_spans_found:
                for t_span in tail_spans_cache[t]:
                    triple_spans[h].append((h_span, r, t_span))
 

    return triple_spans, not_found_triples

def fix_relations(new_relations_ids):
    old_relations_dict = data_loader.get_relations(True)
    old_rel2idx = data_loader.get_rel2idx(True)
    old_semantic_relations = data_loader.get_semantic_relation_embeddings()
    old_trane_relations = data_loader.get_trane_relation_embeddings()

    new_relations_dict = {r: old_relations_dict[r] for r in new_relations_ids if r in old_relations_dict}
    new_rel2idx = {r: idx for idx, r in enumerate(new_relations_ids)}
    new_semantic_relations = np.array([old_semantic_relations[old_rel2idx[r]] for r in new_relations_ids if r in old_rel2idx])
    new_trane_relations = np.array([old_trane_relations[old_rel2idx[r]] for r in new_relations_ids if r in old_rel2idx])

    return new_relations_dict, new_rel2idx, new_trane_relations, new_semantic_relations


def main(use_minimized):
    max_descriptions_length = 256
    chunks_n = 16 if use_cuda else 4
    print("loadiung dictionaries...")
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)


    descriptions = data_loader.get_descriptions(minimized=use_minimized)

    shm_desc, shm_desc_size = share_dict(descriptions)
    shm_aliases, shm_aliases_size = share_dict(aliases_dict)

    len_descriptions = len(descriptions)
    triples = data_loader.get_triples_train(minimized=use_minimized)



    print(f"Distributing to {chunks_n} processes")
    try:
        with Pool(
            processes=chunks_n,  
            initializer=init_worker_discover_aliases, 
            initargs=(aliases_pattern_map, shm_aliases.name, shm_aliases_size, shm_desc.name, shm_desc_size)
        ) as pool:
            args = [(idx, t_chunk, max_descriptions_length) for idx, t_chunk in enumerate(chunk_list(triples, chunks_n))]
            results_chunks = pool.starmap(chunk_description_discover_aliases_spans, args)
    except Exception as ex:
        print(f"Error during multiprocessing: {ex}")
        return False
    finally:
        shm_desc.close(); shm_desc.unlink()
        shm_aliases.close(); shm_aliases.unlink()

    results_all  = defaultdict(list)
    not_found_triples_all = defaultdict(list)
    for res_chunk, not_found_triples in results_chunks:
        for h_id, gold_triple in res_chunk.items():
            results_all[h_id].extend(gold_triple)
        for h_id, not_found in not_found_triples.items():
            not_found_triples_all[h_id].extend(not_found)


    print(f"  {len(results_all)}/{len(descriptions)} entities has golden triples")


    triples_without_gold = []
    new_triples = []
    new_descriptions = {}
    new_relations_ids = set()
    not_found_set = {h: set(pairs) for h, pairs in not_found_triples_all.items()}
    for h,r,t in triples:
        if h in results_all and (r, t) not in not_found_set.get(h, set()):
            new_triples.append((h,r,t))
            new_descriptions[h] = descriptions[h]
            new_relations_ids.add(r)
        else:
            triples_without_gold.append((h,r,t))

    new_relations_dict, new_rel2idx, new_trane_relations, new_semantic_relations = fix_relations(new_relations_ids)



    print(f"  {len(triples_without_gold)}/{len(triples)} triples don't have gold triples")
    print(f"Descriptions number was reduced from {len_descriptions} to {len(new_descriptions)}")
    print(f"Triples number was reduced from {len(triples)} to {len(new_triples)}")
    del aliases_dict, aliases_pattern_map, descriptions, triples




    descriptions_embeddings_ids = read_cached_array(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_IDS)
    description_embeddings = read_tensor(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL, mmap=True)
    print("Creating mask")
    mask = np.array([id_ in set(new_descriptions.keys())
                for id_ in descriptions_embeddings_ids])
    N_new = int(mask.sum())
    B, L, H = description_embeddings.shape
    B, L, H = int(B), int(L), int(H)  
    tmp_path = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL + "_tmp.npy"
    print(f"Starting to do filtering on description embeddings, new size will be {B} -->> {N_new} ")
    description_embeddings_new = init_mmap(tmp_path, (N_new, L, H), "float32")

    src_indices = np.where(mask)[0]
    chunk_size  = 1000
    for start in range(0, len(src_indices), chunk_size):
        end      = min(start + chunk_size, len(src_indices))
        dst_rows = slice(start, end)
        src_rows = src_indices[start:end]
        description_embeddings_new[dst_rows] = description_embeddings[src_rows]

    description_embeddings_new.flush()

    os.replace(tmp_path, settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL)

    print(f"Reducing description embeddings mean and masks")
    description_embeddings_mean  = read_tensor(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN,          mmap=True)
    description_embeddings_masks = read_tensor(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDING_ALL_MASKS,      mmap=True)

    H_mean = description_embeddings_mean.shape[1]
    L_mask = description_embeddings_masks.shape[1]

    # Same chunked-mmap pattern as description_embeddings_all above:
    # never materialise the full filtered tensor in RAM — write directly to a
    # tmp file in chunks, then atomically replace the original.
    tmp_mean_path  = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN          + "_tmp.npy"
    tmp_masks_path = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDING_ALL_MASKS      + "_tmp.npy"

    mean_new  = init_mmap(tmp_mean_path,  (N_new, H_mean), "float32")
    masks_new = init_mmap(tmp_masks_path, (N_new, L_mask), "int64")

    for start in range(0, len(src_indices), chunk_size):
        end      = min(start + chunk_size, len(src_indices))
        dst_rows = slice(start, end)
        src_rows = src_indices[start:end]
        mean_new[dst_rows]  = description_embeddings_mean[src_rows].numpy()
        masks_new[dst_rows] = description_embeddings_masks[src_rows].numpy()

    mean_new.flush()
    masks_new.flush()
    # del before os.replace so the mmap file handles are closed (required on Windows)
    del mean_new, masks_new, description_embeddings_mean, description_embeddings_masks

    descriptions_embeddings_ids_filtered = [id_ for id_, m in zip(descriptions_embeddings_ids, mask) if m]

    print("Saving..")

    os.replace(tmp_mean_path,  settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN)
    os.replace(tmp_masks_path, settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDING_ALL_MASKS)
    cache_array(descriptions_embeddings_ids_filtered, settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_IDS)


    cache_array(results_all, settings.MINIMIZED_FILES.GOLD_TRIPLES)
    cache_array(new_triples, settings.MINIMIZED_FILES.TRIPLES_TRAIN)
    cache_array(new_descriptions, settings.MINIMIZED_FILES.DESCRIPTIONS)
    cache_array(new_relations_dict, settings.MINIMIZED_FILES.RELATIONS)
    cache_array(new_rel2idx, settings.MINIMIZED_FILES.REL2IDX)
    cache_array(new_trane_relations, settings.MINIMIZED_FILES.TRANSE_MODEL_RESULTS)

    new_semantic_relations = torch.from_numpy(new_semantic_relations)
    save_tensor(new_semantic_relations, settings.MINIMIZED_FILES.RELATIONS_EMBEDDINGS)

if __name__=="__main__":
    main(True)
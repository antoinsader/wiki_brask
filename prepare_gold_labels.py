# so my goal here is to generate gold labels for 
# (forward_head_start, forward_head_end, backward_tail_start, backward_tail_end) each is a tensor with shape (B, L, 1) having 0,1 values
# (forward_tail_start, forward_tail_end, backward_head_start, backward_tail_end) each is a tensor with shape (B, R, S, L, 1) where S is maximum number of heads for forward, maximum number of tails for backward

import torch
import re
from collections import defaultdict
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from operations.tokenizer import tokenize

from utils.files import cache_array, read_cached_array, read_tensor
from utils.settings import settings
from utils.chunking import  chunk_list
from utils.pre_processed_data import data_loader
from utils.helpers import create_aliases_patterns_map

# For multiprocessing
_ALIASES_PATTERNS_MAP = None
_ALIASES_DICT = None

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




def init_worker_discover_aliases(aliases_patterns_map, aliases_dict: dict):
    global   _ALIASES_PATTERNS_MAP, _ALIASES_DICT
    _ALIASES_PATTERNS_MAP = aliases_patterns_map
    _ALIASES_DICT = aliases_dict


def chunk_description_discover_aliases_spans(process_idx: int,triples_chunk: list,descriptions_chunk: dict, max_descriptions_length: int):
    """Discover triple_spans for each triple in the chunk, triple_spans is a dict {head_entity_id: [(head_start_idx, head_end_idx, rel_idx, tail_start_idx, tail_end_idx), ... for each triple with this head]} 
    Parameters:
    ----------
    triples_chunk:  list[(head_entity_id, relation_id, tail_entity_id), ..]

    Returns:
    ----------
    head_spans: dict {entity_id: [(head_start, head_end)... for each head]}
    tail_spans: dict {entity_id: [(tail_start, tail_end)... for each tail]}
    """
    not_found_triples = defaultdict(list)
    triples_head_ids = set([t[0] for t in triples_chunk ])
    descriptions = {k:v for k,v in descriptions_chunk.items() if k in triples_head_ids}
    descriptions_ids = list(descriptions.keys())
    descriptions_texts = list(descriptions.values())
    print(f"[PROCESS_{process_idx}]: Tokenizing")
    enc = tokenize(descriptions_texts, max_descriptions_length)
    triple_spans = defaultdict(list)  # h -> [(head_spans, rel_idx, tail_spans), ...]
    print(f"[PROCESS_{process_idx}]: extracting")

    description_idx_map = {entity_id: idx for idx, entity_id in enumerate(descriptions_ids)}

    triples_by_head = defaultdict(list)
    for h, r, t in triples_chunk:
        if h in description_idx_map:
            triples_by_head[h].append((r, t))

    for h, related_triples in tqdm(triples_by_head.items(), desc=f"[PROCESS_{process_idx}] Extracting aliases from triples"):
        description_idx = description_idx_map[h]
        description_text = descriptions_texts[description_idx]

        # Compute head spans once per head entity
        head_spans_found = set()
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



def main(use_minimized):
    max_descriptions_length = 128
    chunks_n = 16 if use_cuda else 4
    print("loadiung dictionaries...")
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)

    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    len_descriptions = len(descriptions)
    triples = data_loader.get_triples_train(minimized=use_minimized)

    print("chunking...")
    _triples_chunks = chunk_list(triples, chunks_n=chunks_n)
    descriptions_chunks = []
    triples_chunks = []

    for t_chunk in _triples_chunks:
        t_chunk_ids = [t[0] for t in t_chunk]
        desc_chunk = {k:v for k, v in descriptions.items() if k in t_chunk_ids}
        descriptions_chunks.append(desc_chunk)
        triples_chunks.append(t_chunk)

    del  _triples_chunks


    print(f"Distributing to {chunks_n} processes")
    with Pool(
        processes=chunks_n,  
        initializer=init_worker_discover_aliases, 
        initargs=(aliases_pattern_map, aliases_dict)) as pool:

        args = [(idx, t_chunk, d_chunk, max_descriptions_length) for idx, (t_chunk,d_chunk) in enumerate(zip(triples_chunks, descriptions_chunks))]
        results_chunks = pool.starmap(chunk_description_discover_aliases_spans, args)
        results_all  = defaultdict(list)
        not_found_triples_all = defaultdict(list)
        for res_chunk, not_found_triples in results_chunks:
            for h_id, gold_triple in res_chunk.items():
                results_all[h_id].extend(gold_triple)
                not_found_triples_all[h_id].extend(not_found_triples[h_id])
        
        print(f"  {len(results_all)}/{len(descriptions)} entities has golden triples")


        triples_without_gold = []
        new_triples = []
        new_descriptions = {}
        for h,r,t in triples:
            if h in results_all and (r, t) not in not_found_triples_all[h]:
                new_triples.append((h,r,t))
                new_descriptions[h] = descriptions[h]
            else:
                triples_without_gold.append((h,r,t))


        print(f"  {len(triples_without_gold)}/{len(triples)} triples don't have gold triples")
        print(f"Descriptions number was reduced from {len_descriptions} to {len(new_descriptions)}")
        print(f"Triples number was reduced from {len(triples)} to {len(new_triples)}")
        del aliases_dict, aliases_pattern_map, descriptions, triples

        description_embeddings = read_tensor(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL)
        description_embeddings_mean = read_tensor(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN)
        descriptions_embeddings_ids = read_cached_array(settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_IDS)



        
        mask  = np.array([id_ in list(new_descriptions.keys()) for id_ in descriptions_embeddings_ids])
        description_embeddings_filtered        = description_embeddings[mask]
        description_embeddings_mean_filtered   = description_embeddings_mean[mask]
        descriptions_embeddings_ids_filtered   = [id_ for id_, m in zip(descriptions_embeddings_ids, mask) if m]



        print(f"new descriptions embeddings shape: {description_embeddings_filtered.shape}, new description meean embeddings shape: {description_embeddings_mean_filtered.shape}, new description embeddings ids: {len(descriptions_embeddings_ids_filtered)}")

        cache_array(results_all, settings.MINIMIZED_FILES.GOLD_TRIPLES)

        cache_array(new_triples, settings.MINIMIZED_FILES.TRIPLES_TRAIN)
        cache_array(new_descriptions, settings.MINIMIZED_FILES.DESCRIPTIONS)


if __name__=="__main__":
    main(True)
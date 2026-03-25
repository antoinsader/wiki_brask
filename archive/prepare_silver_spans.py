


from collections import defaultdict
from multiprocessing import Pool

import torch
from tqdm import tqdm
import re
from transformers import BertTokenizerFast



from utils.files import cache_array
from utils.settings import settings
from utils.chunking import chunk_dict
from utils.pre_processed_data import data_loader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
NUM_WORKERS = 4 if use_cuda else 0

# For multiprocessing
_TOKENIZER =None
_DESCS_HEADS_ALIASES = None
_DESCS_TAILS_ALIASES = None
_ALIASES_PATTERNS_MAP = None


def init_worker(tokenizer: BertTokenizerFast, descriptions_heads_aliases, descriptions_tails_aliases, aliases_patterns_map):
    global _TOKENIZER, _DESCS_HEADS_ALIASES, _DESCS_TAILS_ALIASES, _ALIASES_PATTERNS_MAP
    _TOKENIZER = tokenizer
    _DESCS_HEADS_ALIASES = descriptions_heads_aliases
    _DESCS_TAILS_ALIASES = descriptions_tails_aliases
    _ALIASES_PATTERNS_MAP = aliases_patterns_map


def process_descriptions_chunk(description_chunk: dict, max_descriptions_length: int):
    """Extract head and tail start and end spans for a chunk of descriptions, 
    returns a tuple of (silver_spans_head_start, silver_spans_head_end, silver_spans_tail_start, silver_spans_tail_end, all_sentences_tokens, descriptions_ids)"""

    L = max_descriptions_length

    descriptions_ids = list(description_chunk.keys())
    descriptions_texts = list(description_chunk.values())
    
    chunk_size =  len(descriptions_ids)
    
    silver_spans_head_start = torch.zeros(chunk_size, L)
    silver_spans_head_end = torch.zeros(chunk_size, L)
    silver_spans_tail_start = torch.zeros(chunk_size, L)
    silver_spans_tail_end = torch.zeros(chunk_size, L)

    desc_texts = list(description_chunk.values())
    enc = _TOKENIZER(
        desc_texts,
        return_offsets_mapping=False,
        add_special_tokens = False,
        return_attention_mask=False,
        return_token_type_ids = False,
        padding="max_length",
        truncation=True,
        max_length = L
    )

    tokens = [encoding.tokens for encoding in enc.encodings]

    for desc_idx, desc_id in tqdm(enumerate(descriptions_ids), total=len(descriptions_ids), desc="Extracting silver"):
        description = descriptions_texts[desc_idx]

        description_heads_aliases = _DESCS_HEADS_ALIASES[desc_id]
        description_tails_aliases = _DESCS_TAILS_ALIASES[desc_id]

        for als_str in  description_heads_aliases:
            pattern = _ALIASES_PATTERNS_MAP[als_str]
            for match in pattern.finditer(description):
                char_start, char_end = match.span()
                actual_char_end = char_end - 1
                token_start_idx = enc.char_to_token(desc_idx, char_start)
                token_end_idx = enc.char_to_token(desc_idx, actual_char_end)
                if token_start_idx is not None and token_end_idx is not None:
                    silver_spans_head_start[desc_idx, token_start_idx] = 1
                    silver_spans_head_end[desc_idx, token_end_idx] = 1

        for als_str in description_tails_aliases:
            pattern = _ALIASES_PATTERNS_MAP[als_str]
            for match in pattern.finditer(description):
                char_start, char_end = match.span()
                actual_char_end = char_end - 1
                token_start_idx = enc.char_to_token(desc_idx, char_start)
                token_end_idx = enc.char_to_token(desc_idx, actual_char_end)
                if token_start_idx is not None and token_end_idx is not None:
                    silver_spans_tail_start[desc_idx, token_start_idx] = 1
                    silver_spans_tail_end[desc_idx, token_end_idx] = 1

    return (silver_spans_head_start, silver_spans_head_end, silver_spans_tail_start, silver_spans_tail_end, tokens, descriptions_ids)


def create_aliases_patterns_map(aliases : dict) -> dict[str, re.Pattern]:
    """Creates a map of alias strings to regex patterns that match them."""
    patterns_map = {}
    for als_lst in tqdm(aliases.values(), desc="creating aliases patterns map"):
        for als_str in als_lst:
            escaped = re.escape(als_str)
            flexible = escaped.replace(r'\ ', r'\s+')
            pattern = rf"(?<!\w){flexible}(?!\w)"
            patterns_map[als_str] = re.compile(pattern, flags=re.IGNORECASE)
    del aliases
    return patterns_map

def create_description_heads_tails_map_aliases(descriptions, triples, aliases):

    heads_to_aliases_map = defaultdict(list)
    tails_to_aliases_map = defaultdict(list)

    for d_id in tqdm(descriptions.keys(), desc="extracting heaads from descriptions"):
        if d_id in triples and d_id in aliases:
            for als in aliases[d_id]:
                heads_to_aliases_map[d_id].append(als)
            for _, _, t in triples[d_id]:
                if t in aliases:
                    for als in aliases[t]:
                        tails_to_aliases_map[d_id].append(als)

    return heads_to_aliases_map, tails_to_aliases_map





def main(use_minimized: bool):
    max_descriptions_length = 256
    aliases_dict = data_loader.get_aliases(minimized=use_minimized)
    
    aliases_pattern_map = create_aliases_patterns_map(aliases_dict)


    descriptions = data_loader.get_descriptions(minimized=use_minimized)
    triples_dict = data_loader.get_triples_train(minimized=use_minimized)
    descriptions_heads_aliases, descriptions_tails_aliases = create_description_heads_tails_map_aliases(descriptions, triples_dict, aliases_dict)

    del triples_dict, aliases_dict

    descriptions_chunks = chunk_dict(descriptions, chunks_n=NUM_WORKERS)

    tokenizer :BertTokenizerFast = BertTokenizerFast.from_pretrained('bert-base-cased')

    print("Starting multiprocessing pool for silver spans extraction...")
    with Pool(processes=NUM_WORKERS, 
              initializer=init_worker, 
              initargs=(tokenizer, descriptions_heads_aliases, descriptions_tails_aliases, aliases_pattern_map) 
              ) as pool:
        # Use starmap to pass multiple arguments to process_descriptions_chunk
        args = [(chunk, max_descriptions_length) for chunk in descriptions_chunks]
        results_chunks = pool.starmap(process_descriptions_chunk, args)
        silver_spans_head_start_ar = []
        silver_spans_head_end_ar = []
        silver_spans_tail_start_ar = []
        silver_spans_tail_end_ar = []
        sentences_tokens = []
        desc_ids = []
        for batch in results_chunks:
            b_ss_h_s, b_ss_h_e, b_ss_t_s, b_ss_t_e, b_tokens, b_desc_ids = batch
            silver_spans_head_start_ar.extend(b_ss_h_s)
            silver_spans_head_end_ar.extend(b_ss_h_e)
            silver_spans_tail_start_ar.extend(b_ss_t_s)
            silver_spans_tail_end_ar.extend(b_ss_t_e)
            sentences_tokens.extend(b_tokens)
            desc_ids.extend(b_desc_ids)

        silver_spans_result = {
            "head_start": torch.stack(silver_spans_head_start_ar, dim=0),
            "head_end": torch.stack(silver_spans_head_end_ar, dim=0),
            "tail_start": torch.stack(silver_spans_tail_start_ar, dim=0),
            "tail_end": torch.stack(silver_spans_tail_end_ar, dim=0),
            "sentences_tokens": sentences_tokens,
            "desc_ids": desc_ids
        }

        out_path = settings.MINIMIZED_FILES.SILVER_SPANS if use_minimized else settings.PREPROCESSED_FILES.SILVER_SPANS
        print(f"Caching silver spans result to {out_path}...")
        cache_array(silver_spans_result, out_path)


def filter_descriptions(use_minimized: bool):
    descriptions_all = data_loader.get_descriptions(minimized=use_minimized)
    silver_spans_result = data_loader.get_silver_spans(minimized=use_minimized)
    ss_h_s = silver_spans_result["head_start"]
    ss_h_e = silver_spans_result["head_end"]
    ss_t_s = silver_spans_result["tail_start"]
    ss_t_e = silver_spans_result["tail_end"]
    descs_ids = silver_spans_result["desc_ids"]
    sentence_tokens = silver_spans_result["sentences_tokens"]

    mask = (
        ss_h_s.any(dim=1) |
        ss_h_e.any(dim=1) |
        ss_t_s.any(dim=1) |
        ss_t_e.any(dim=1)
    )
    cleaned_desc_dict=  {
        k: descriptions_all[k]
        for k, keep in zip(descs_ids, mask)
        if keep.item()
    }
    silver_spans_obj = {
        "head_start": ss_h_s[mask],
        "head_end": ss_h_e[mask],
        "tail_start": ss_t_s[mask],
        "tail_end": ss_t_e[mask],
        "sentences_tokens": [sentence_tokens[i] for i, keep in enumerate(mask) if keep],
        "desc_ids": [desc_id for desc_id, keep in zip(descs_ids, mask) if keep]
    }

    out_desc_path = settings.MINIMIZED_FILES.DESCRIPTIONS if use_minimized else settings.PREPROCESSED_FILES.DESCRIPTIONS
    out_silver_spans_path = settings.MINIMIZED_FILES.SILVER_SPANS if use_minimized else settings.PREPROCESSED_FILES.SILVER_SPANS
    print(f"we cleared  {len(descriptions_all) - len(cleaned_desc_dict)}/{len(descriptions_all)} descriptions, now we have {len(cleaned_desc_dict)} descriptions")

    cache_array(cleaned_desc_dict, out_desc_path)
    cache_array(silver_spans_obj, out_silver_spans_path)

if __name__ == "__main__":
    answer = input("Extract silver spans from minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)

    answer = input("Do you want to filter descriptions to keep only those having silver spans? [Y/n]: ").strip().lower()
    if answer == "y":
        answer = input("We will overwrite the descriptions file with the filtered one. Are you sure? [Y/n]: ").strip().lower()
        if answer == "y":
            filter_descriptions(use_minimized)
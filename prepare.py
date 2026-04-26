from collections import defaultdict
import multiprocessing
import random
import torch
import os
import pandas as pd
from transformers import BertModel, BertTokenizerFast, AutoModel

from io import StringIO

from joblib import Parallel, delayed
from tqdm import tqdm


from utils.settings import settings
from utils.files import cache_array, read_cached_array, scan_text_file_lines, save_tensor
from utils.pre_processed_data import data_loader, check_minimized_files
from utils.helpers import  get_strange_chars, ask_factor, timed_input
from operations.embedding import get_rel_embs, save_descriptions_embedding

from operations.normalizer import Normalizer

use_cuda = torch.cuda.is_available()
device_str = "cuda" if use_cuda else "cpu"
device = torch.device(device_str)
num_workers = 4 if use_cuda else 0
descriptions_max_length = 256

rel_embs_batch_size = 8192 if use_cuda else 32


def choose_random_ids(ids_list, n):
    """Choose n random ids from the list of ids that have triples, aliases and descriptions."""
    tr_raw_fp = settings.RAW_FILES.TRIPLES_TRAIN

    _ids_has_everything = ids_list


    triple_heads = set()
    t_rs = set()
    with open(tr_raw_fp, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Choosing ids"):
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triple_heads.add(parts[0])
                t_rs.add((parts[0], parts[1]))



    _ids_has_everything = [i for i in _ids_has_everything if i in triple_heads]

    aliases_all = data_loader.get_aliases(minimized=False)
    aliases_ids = set(aliases_all.keys())
    _ids_has_everything = [i for i in _ids_has_everything if i in aliases_ids]
    del aliases_all, aliases_ids

    descriptions_all = data_loader.get_descriptions(minimized=False)
    descriptions_ids = set(descriptions_all.keys())
    _ids_has_everything = [i for i in _ids_has_everything if i in descriptions_ids]
    del descriptions_all, descriptions_ids

    rels_all = data_loader.get_relations(minimized=False)
    head_ids_not_exists_in_rels = [ h for h,r in t_rs if r not in rels_all ]
    _ids_has_everything = [i for i in _ids_has_everything if i not in head_ids_not_exists_in_rels]
    del rels_all, head_ids_not_exists_in_rels

    if len(_ids_has_everything) < n:
        return _ids_has_everything
    return set(random.sample(_ids_has_everything, n))


def _find_byte_boundaries(filepath, n_parts):
    size = os.path.getsize(filepath)
    boundaries = [0]
    with open(filepath, 'rb') as f:
        for i in range(1, n_parts):
            f.seek(i * size // n_parts)
            f.readline()  # advance past the partial line at the split point
            boundaries.append(f.tell())
    boundaries.append(size)
    return boundaries


def _filter_partition(filepath, start, end, minimized_ids):
    with open(filepath, 'rb') as f:
        f.seek(start)
        raw = f.read(end - start).decode('utf-8', errors='replace')
    chunk = pd.read_csv(
        StringIO(raw), sep='\t', header=None,
        names=['head', 'relation', 'tail'], dtype=str,
        on_bad_lines='skip', usecols=[0, 1, 2],
    )
    chunk.dropna(subset=['head', 'relation', 'tail'], inplace=True)
    filtered = chunk[chunk['head'].isin(minimized_ids)]
    triples = list(zip(filtered['head'], filtered['relation'], filtered['tail']))
    return triples, set(filtered['relation']), set(filtered['tail'])


def minimmizing_triples(minimized_triples_ids, raw_fp, min_files):
    n_workers = multiprocessing.cpu_count()
    boundaries = _find_byte_boundaries(raw_fp, n_workers)

    results = Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_filter_partition)(raw_fp, boundaries[i], boundaries[i + 1], minimized_triples_ids)
        for i in tqdm(range(n_workers), desc="Filtering triples (parallel)")
    )

    triples_min, relation_ids_min, tails_entity_ids_min = [], set(), set()
    for triples, rels, tails in results:
        triples_min.extend(triples)
        relation_ids_min.update(rels)
        tails_entity_ids_min.update(tails)

    cache_array(triples_min, min_files.TRIPLES_TRAIN)
    print(f"\t Triple heads minimization finished: {len(triples_min):,} -> {min_files.TRIPLES_TRAIN}")
    del triples_min
    return relation_ids_min, tails_entity_ids_min

def minimize(minimized_triples_ids):

    min_files = settings.MINIMIZED_FILES
    raw_fp = settings.RAW_FILES.TRIPLES_TRAIN
    print("minimizing triples")
    relation_ids_min, tails_entity_ids_min = minimmizing_triples(
        minimized_triples_ids,
        raw_fp=raw_fp,
        min_files=min_files
    )




    print("minimizing aliases")
    aliases_all = data_loader.get_aliases(minimized=False)
    aliases_min = {}
    for ent_id, als_lst in aliases_all.items():
        if ent_id not in minimized_triples_ids and ent_id not in tails_entity_ids_min:
            continue
        _lst = [als for als in als_lst if als.strip() != ""]
        if _lst:
            aliases_min[ent_id] = _lst

    cache_array(aliases_min, min_files.ALIASES)
    print(f"\t Aliases minimization finished: {len(aliases_min):,} -> {min_files.ALIASES}")
    del aliases_all, aliases_min, tails_entity_ids_min
    

    print("minimizing relations")
    relations_all = data_loader.get_relations(minimized=False)
    relations_min = {k:v for k, v in relations_all.items() if k in relation_ids_min}
    cache_array(relations_min, min_files.RELATIONS)
    print(f"\t Relations minimization finished: {len(relations_min):,} -> {min_files.RELATIONS}")
    del relations_all, relations_min, relation_ids_min


    print("minimizing descriptions")
    descriptions_all = data_loader.get_descriptions(minimized=False)
    descriptions_min = {k: v for k,v in descriptions_all.items() if k in minimized_triples_ids}
    cache_array(descriptions_min, min_files.DESCRIPTIONS)
    print(f"\t Descriptions minimization finished  : {len(descriptions_min):,} -> {min_files.DESCRIPTIONS}")
    del descriptions_all, descriptions_min





    return True


def get_minimized_ids() -> tuple[list, int]:
    print("Scanning raw files..")
    raw = settings.RAW_FILES
    full_files = settings.PREPROCESSED_FILES
    raw.validate()

    if os.path.exists(full_files.TRIPLES_TRAIN):
        full_triples = read_cached_array(full_files.TRIPLES_TRAIN)
        triples_ids = [t[0] for t in full_triples]
        del full_triples
        total_train_triples, all_triples_ids = len(triples_ids), triples_ids
    else:
        print("\nScanning size of triples train...")
        total_train_triples, all_triples_ids = scan_text_file_lines(raw.TRIPLES_TRAIN, scan_head_ids=True)
        print("\nScanning size of descriptions train...")

    if os.path.exists(full_files.DESCRIPTIONS):
        all_descriptions = read_cached_array(full_files.DESCRIPTIONS)
        total_descriptions = len(all_descriptions)
        del all_descriptions
    else:
        total_descriptions = scan_text_file_lines(raw.DESCRIPTIONS)


    print(f"\t Total train triples: {total_train_triples}")
    print(f"\t Total dsecriptions: {total_descriptions}")
    while True:
        factor = ask_factor(f"How much percentage of the triples do you want to keep? type a decimal number between 0 and 1 (1 being all of the dataset) \n")
        minimized_n_triples = max(1, int(total_train_triples * factor))
        print(f"After minimization with factor {factor}:")
        print(f"\t Total train triples: {minimized_n_triples} ({minimized_n_triples / total_train_triples * 100})% of original")
        answer = input("\n Proceed? [y]/[n] abort / [c] change factor: ").strip().lower()
        if answer == "y":
            break
        if answer== "n":
            print("Aborted")
            return [], 0
    return all_triples_ids, minimized_n_triples


def normalize():
    if not check_minimized_files():
        return None

    out_descs = settings.MINIMIZED_FILES.DESCRIPTIONS
    out_aliases = settings.MINIMIZED_FILES.ALIASES

    strange_chars = get_strange_chars()

    normalizer = Normalizer(strange_chars)
    descriptions = data_loader.get_descriptions(minimized=True)
    print(f"Normalizing {len(descriptions)} descriptions")
    norm_descriptions = {}
    for d_id, d_txt in tqdm(descriptions.items(), desc="normalizing descriptions"):
        norm_descriptions[d_id] =  normalizer(d_txt)

    del descriptions
    cache_array(norm_descriptions, out_descs)
    print(f"Normalized descriptions saved {len(norm_descriptions)} -> {out_descs} ")
    del norm_descriptions, normalizer

    normalizer = Normalizer(strange_chars, lowercasing=True)
    aliases = data_loader.get_aliases(minimized=True)
    print(f"Normalizing {len(aliases)} aliases")
    norm_aliases = defaultdict(set)
    total_items = sum(len(lst) for lst in aliases.values())
    with tqdm(total=total_items, desc="Normalizing") as pbar:
        for als_id, als_lst in aliases.items():
            for als_str in als_lst:
                norm = normalizer(als_str)
                if norm.strip() != "":
                    norm_aliases[als_id].add(norm)
            pbar.update(len(als_lst))
    del aliases
    norm_aliases = {k: list(v) for k, v in norm_aliases.items()}
    cache_array(norm_aliases, out_aliases)
    print(f"Normalized aliases saved {len(norm_aliases)} -> {out_aliases} ")
    del norm_aliases
    return True


def embed_relations():
    relations = data_loader.get_relations(minimized=True)
    triples = data_loader.get_triples_train(minimized=True)
    relation_ids = set(t[1] for t in triples)
    relations = {r: rels for r, rels in relations.items() if r in relation_ids}
    del triples, relation_ids
    if not relations:
        print("Relations not found")
        return False

    out_path = settings.MINIMIZED_FILES.RELATIONS_EMBEDDINGS


    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    bert_model = BertModel.from_pretrained("bert-base-cased")
    if use_cuda:
        bert_model = bert_model.half()
    bert_model = bert_model.to(device)

    rel_embs = get_rel_embs(relations,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
        batch_size=rel_embs_batch_size,
        use_cuda=use_cuda,
        device=device)
    
    print(f"Relatioon embeddings has out with shape {rel_embs.shape} (Should be ({len(relations)}, 786))")
    save_tensor(rel_embs, out_path)
    print(f"Sved in {out_path}")

    return True


def embed_descriptions():
    descriptions = data_loader.get_descriptions(minimized=True)
    if not descriptions:
        print("Descriptions not found")
        return False

    out_mean_embs = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_MEAN
    out_all_embs = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL
    out_all_masks = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDING_ALL_MASKS
    out_ids = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_IDS
    sentences = list(descriptions.values())

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased')

    success = save_descriptions_embedding(
        tokenizer=tokenizer, 
        model=model, 
        sentences= sentences, 
        device=device, 
        use_cuda=use_cuda, 
        out_all_embs=out_all_embs,
        out_mean_embs =out_mean_embs,
        out_all_masks=out_all_masks,
        max_length=descriptions_max_length,
    )

    if success:
        cache_array(list(descriptions.keys()), out_ids)
        print(f"Saved -> {out_mean_embs},  {out_all_embs},  {out_all_masks}, {out_ids}")

    return success

def main():
    answer = timed_input("Do you want to perform minimization on dictionaries? [Y/n]").lower().strip()
    if answer == "y":
        all_triples_ids, minmized_n_triples = get_minimized_ids()
        if minmized_n_triples == 0:
            return

        minimized_triples_ids = choose_random_ids(all_triples_ids, minmized_n_triples)
        minimize(minimized_triples_ids)
        print("Finished minimization.. ")
    else:
        print("Skipping minimization")

    answer = timed_input("Do you want to perform normalization on aliases and descriptions? [Y/n]").lower().strip()
    if answer == "y":
        print(f"starting to normalize descriptions and aliases")
        res = normalize()
        if not res:
            print(f"Aborting for normalization error..")
        print(f"Normalization finished..")
    else:
        print("Skipping normalization")


    answer = timed_input("Do you want to perform relation embeddings? [Y/n]").lower().strip()
    if answer == "y":
        print(f"Starting to embed relations..")
        res = embed_relations()
        if not res:
            print("Aborting for relations error..")
        print(f"Embedding relations is finished.")
    else:
        print("skipping relation embeddings")

    answer = timed_input("Do you want to perform description embeddings? [Y/n]").lower().strip()
    if answer == "y":
        print(f"Starting to embed descriptions..")
        res = embed_descriptions()
        if not res:
            print("Aborting for descriptions embedding error..")
        print(f"Embedding descriptions is finished.")

    else:
        print("Skipping description embeddings")

if __name__ == "__main__":
    main()

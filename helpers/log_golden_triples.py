

GOLD_TRIPLES_PATH = "../data/minimized/golden_triples.pkl"
DESCRIPTIONS_PATH    = "../data/minimized/descriptions.pkl"
RELATIONS_PATH = "../data/minimized/relations.pkl"
TRIPLES_PATH = "../data/minimized/triples_train.pkl"
ALIASES_PATH = "../data/minimized/aliases.pkl"
DESCRIPTIONS_MAX_LENGTH = 128


from collections import defaultdict
import os
import pickle

from tqdm import tqdm
import sys
sys.path.insert(0, '..')


from operations.tokenizer import tokenize


def read_cached_array(filename):
    with open(filename, 'rb', buffering=16*1024*1024) as f:
        return pickle.load(f)

def main():
    if not (os.path.exists(DESCRIPTIONS_PATH) and os.path.exists(RELATIONS_PATH) and os.path.exists(TRIPLES_PATH)):
        print("One or more required files are missing. Please run prepare.py first to generate the necessary files.")
        return
    if not os.path.exists(GOLD_TRIPLES_PATH):
        print("Gold triples file is missing. Please run prepare_gold_labels.py first to generate the gold triples.")
        return

    print("Loading tokenizer..")
    
    descriptions = read_cached_array(DESCRIPTIONS_PATH)
    descriptions_ids = [d_id for d_id in descriptions.keys()]
    descriptions_texts = [d_txt for d_txt in descriptions.values()]
    print(f"Tokenizing {len(descriptions_ids)} descriptions...")
    enc = tokenize(descriptions_texts, DESCRIPTIONS_MAX_LENGTH)
    
    tokens = [encoding.tokens for encoding in enc.encodings]
    _triples = read_cached_array(TRIPLES_PATH)
    _aliases  = read_cached_array(ALIASES_PATH)
    head_to_triples = defaultdict(list)
    for h_id, r_id, t_id in _triples:
        head_to_triples[h_id].append(( _aliases[h_id], r_id, _aliases[t_id]))
    del _triples, _aliases


    golden_triples = read_cached_array(GOLD_TRIPLES_PATH)
    relations = read_cached_array(RELATIONS_PATH)
    lines = []
    print(f"Logging golden triples for {len(descriptions_ids)} descriptions...")
    for d_idx, description_id in tqdm(enumerate(descriptions_ids), desc="Logging golden triples", total=len(descriptions_ids)):
        
        lines.append(f"\n************NEW DESECRIPTION*************")
        lines.append(f"Description id: {description_id}")
        lines.append(f"Description text: ")
        lines.append(f"{descriptions[description_id]}")
        lines.append(f"{descriptions_texts[d_idx]}")
    
        lines.append(f"Description tokens: ")
        lines.append(f"{tokens[d_idx]}")


        lines.append("-------------------------------")
        all_ts = head_to_triples[description_id]
        lines.append(f"ORIGINAL TRIPLES ({len(all_ts)}): ")
        lines.append(all_ts)
        lines.append("Details: ")
        for idx, (h_als, r_id, t_als) in enumerate(all_ts):
            lines.append(f"\tOriginal triple number {idx + 1}: ")
            lines.append(f"\t\tHead aliases: {h_als}")
            lines.append(f"\t\tRelation: {relations[r_id]} ({r_id})")
            lines.append(f"\t\tTail aliases: {t_als}")
        if not description_id in golden_triples:
            lines.append(f"!!!!!! No golden triples for description {description_id}")
            continue
        lines.append("-------------------------------")
        lines.append(f"Golden Triples ({len(golden_triples[description_id])}): ")

        for t_idx, (h_spans, r_id, t_spans) in enumerate(golden_triples[description_id]):
            lines.append(f"\tHead ")
            lines.append(f"\tTriple {t_idx + 1}: ")
            lines.append(f"\t\tHead: {tokens[d_idx][h_spans[0]:h_spans[1]+1]}  ")
            lines.append(f"\t\tRelation: {relations[r_id]} ({r_id})")
            lines.append(f"\t\tTail: {tokens[d_idx][t_spans[0]:t_spans[1]+1]}  ")
        
    with open("./golden_triples_log.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line} \n")
    print("Golden triples logged in golden_triples_log.txt")



if __name__=="__main__":
    main()
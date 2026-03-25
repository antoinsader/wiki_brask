import torch
import sys
sys.path.insert(0, '..')

from train import build_sk_from_gold, build_gold_tail_labels
import torch.nn.functional as F




B = 3
L = 15
H = 2

golden_triples = [
    [((0, 1), 'relation1', (2, 3)), ((4, 5), 'relation2', (6, 7))],
    [((0, 2), 'relation1', (3, 4))],
    [((1, 3), 'relation2', (5, 6)), ((7, 8), 'relation1', (9, 10)), ((11, 12), 'relation3', (13, 14))]
]

rel2idx = {
    "relation1": 0,
    "relation2": 1,
    "relation3": 2
}
num_relations = len(rel2idx)

X = torch.randn(B, L, H)  # Example input embeddings
mask = torch.ones(B, L, dtype=torch.bool)  # Example mask (all tokens valid)


sk_embs, sk_mask, unique_subjects_batch = build_sk_from_gold(golden_triples, X, mask)

gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
    triples_batch= golden_triples,
    unique_subjects_batch =unique_subjects_batch ,
    mask=mask,
    num_relations=num_relations,
    rel2idx=rel2idx
)


for b in range(B):
    print(f"Sentence {b}:")
    # print(f"head spans: {[ hs for hs, _, _ in golden_triples[b]  ]} ")
    # print(X[b])
    # print("SK Embeddings:", sk_embs[b])
    # print("Unique Subjects:", unique_subjects_batch[b])
    print(f"tail spans (relation, subject_idx, tail spans, ): {[ (rel2idx[r], unique_subjects_batch[b].index(head), tail) for head, r, tail in golden_triples[b]  ]} ")

    
    print("froward tail start indices:")
    print((gold_fts[b] == 1).nonzero(as_tuple=False))
    print("froward tail end labels:")
    print((gold_fte[b] == 1).nonzero(as_tuple=False))



"""
Plan:
- BraskModel: EntityExtractor(head/ tail), RelationAttention (semantic/transe), FuseExtractor
- Training stages:
    1. Train EntityExtractor (fwd_head/ bwd_tail) only. BCE vs gold triples.
    2. Full model, teacher forcing ratio = 1.0 (gold sk passed in).
    3. Full model, teacher forcing ratio decay from 1.0 to 0.0 over epochs.

Current problems:
1- sk construction: currently we are building sk with R dimension because I thought this would be the correct way as I am comparing in the end with gold_tail_labels which has R.
This is wrong because in paper the sk is built from the subject spans only, regarding the relation.
So I need to change the sk construction to be (B, S, H). Then make sure from FuseExtractor and the compute_loss.

2- Reconsider how max_subjects is build
3- I need to write the function of build_sk

4- Check the notes in BraskModel.


"""


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from models import EntityExtractor
from utils.files import read_cached_array, read_tensor
from utils.settings import settings
from utils.pre_processed_data import data_loader






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
NUM_WORKERS = 4 if use_cuda else 0

CHECKPOINTS_DIR = "checkpoints/"

LEARNING_RATE_STAGE_1 = 1e-4
LEARNING_RATE_STAGE_2 = 1e-5


MODEL_OUTPUT_KEYS = {
    "FORWARD_HEAD_START": "fwd_head_start",
    "FORWARD_HEAD_END": "fwd_head_end",
    "FORWARD_TAIL_START": "fwd_tail_start",
    "FORWARD_TAIL_END": "fwd_tail_end",
    "BACKWARD_TAIL_START": "bwd_tail_start",
    "BACKWARD_TAIL_END": "bwd_tail_end",
    "BACKWARD_HEAD_START": "bwd_head_start",
    "BACKWARD_HEAD_END": "bwd_head_end",
}



class BraskDataset(Dataset):
    """The item would represent one entity with its description embedding and golden triples"""


    BATCH_KEYS = {
        "EMBS": "embs",
        "EMBS_MASKS": "embs_mask",
        "MEAN_EMBS": "mean_embs",
        "GOLDEN_TRIPLES": "golden_triples",
        "ENTITY_ID": "entity_id",

    }

    def __init__(
        self,
        description_embs,
        description_embs_ids,
        description_embs_masks,
        description_mean_embs,
        golden_triples
    ):
        self.embs = description_embs
        self.ids = description_embs_ids
        self.embs_masks = description_embs_masks
        self.mean_embs = description_mean_embs
        self.golden_triples = golden_triples


    def __len__(self):
        return len(self.ids)

    def __getitem__(self,idx):
        entity_id = self.ids[idx]
        return {
            self.BATCH_KEYS["EMBS"]: torch.tensor(self.embs[idx], dtype=torch.float32),
            self.BATCH_KEYS["EMBS_MASKS"]: torch.tensor(self.embs_masks[idx], dtype=torch.float32),
            self.BATCH_KEYS["MEAN_EMBS"]: torch.tensor(self.mean_embs[idx], dtype=torch.float32),
            self.BATCH_KEYS["GOLDEN_TRIPLES"]: self.golden_triples.get(entity_id, []),
            self.BATCH_KEYS["ENTITY_ID"]: entity_id,
        }

def collate_fn(batch):
    return {
        BraskDataset.BATCH_KEYS["EMBS"]: torch.stack([b[BraskDataset.BATCH_KEYS["EMBS"]] for b in batch], dim=0), #(B, L, H)
        BraskDataset.BATCH_KEYS["MEAN_EMBS"]: torch.stack([b[BraskDataset.BATCH_KEYS["MEAN_EMBS"]] for b in batch], dim=0), #(B, L)
        BraskDataset.BATCH_KEYS["EMBS_MASKS"]: torch.stack([b[BraskDataset.BATCH_KEYS["EMBS_MASKS"]] for b in batch], dim=0), #(B, H)
        BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]: [b[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]] for b in batch], # B list [list]
        BraskDataset.BATCH_KEYS["ENTITY_ID"]: [b[BraskDataset.BATCH_KEYS["ENTITY_ID"]] for b in batch], # B list[str]
    }


def build_gold_entity_labels(triples_batch, mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build binary labels for head and tail entity spans from golden triples. Used for computing BCE loss"""


    B, L = mask.shape
    fwd_head_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    fwd_head_end = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_start = torch.zeros((B, L), dtype=torch.float32, device=device)
    bwd_tail_end = torch.zeros((B, L), dtype=torch.float32, device=device)

    for b in range(B):
        for (hs, he), _, (ts, te) in triples_batch[b]:
            if hs < L:
                fwd_head_start[b, hs] = 1.0
            if he < L:
                fwd_head_end[b, he] = 1.0
            if ts < L:
                bwd_tail_start[b, ts] = 1.0
            if te < L:
                bwd_tail_end[b, te] = 1.0

    return fwd_head_start, fwd_head_end, bwd_tail_start, bwd_tail_end

def get_max_subjects(triples_batch, rel2idx):
    max_s = 1
    for triples in triples_batch:
        span_slot = defaultdict(int)
        for (hs, he), r, _ in triples:
            r_idx = rel2idx[r]
            key= (hs, he, r_idx)
            span_slot[key] += 1
            max_s = max(max_s, span_slot[key])
    return max_s


def build_gold_tail_labels(triples_batch, mask, max_subjects, num_relations, rel2idx):
    """Returns (B,R,S, L) for forward tail and backward head."""


    B, L = mask.shape
    R, S = num_relations, max_subjects


    gold_fwd_tail_start = torch.zeros(B, R, S, L, dtype=torch.float32, device=device)
    gold_fwd_tail_end = torch.zeros(B, R, S, L, dtype=torch.float32, device=device)
    gold_bwd_head_start = torch.zeros(B, R, S, L, dtype=torch.float32, device=device)
    gold_bwd_head_end = torch.zeros(B, R, S, L, dtype=torch.float32, device=device)

    for b, triples in enumerate(triples_batch):
        span_slot = defaultdict(int)
        for (hs, he), r, (ts, te) in triples:
            r_idx = rel2idx[r]
            key = (hs, he, r_idx)
            slot = span_slot[key]

            if slot >= S:
                continue
            if ts < L:
                gold_fwd_tail_start[b, r_idx, slot, ts] = 1.0
            if te < L:
                gold_fwd_tail_end[b, r_idx, slot, te] = 1.0
            if hs < L:
                gold_bwd_head_start[b, r_idx, slot, hs] = 1.0
            if he < L:
                gold_bwd_head_end[b, r_idx, slot, he] = 1.0
            span_slot[key] += 1
    return gold_fwd_tail_start, gold_fwd_tail_end, gold_bwd_head_start, gold_bwd_head_end



def build_sk_from_gold(triples_batch, mask, num_relations, max_subjects, rel2idx):
    """Build subject vector (sk) tensor from gold head spans for teacher forcing during training."""
    B, L = mask.shape
    R, S = num_relations, max_subjects

    sk = torch.zeros(B, R, S, 2, dtype=torch.long, device=device)
    sk_mask = torch.zeros(B, R, S, dtype=torch.float32, device=device)



    for b, triples in enumerate(triples_batch):
        slot = defaultdict(int)
        for (hs, he) , r, _ in triples:
            r_idx = rel2idx[r]
            s = slot[(hs, he, r_idx)]
            if s >= S:
                continue
            sk[b, r_idx, s, 0] = hs
            sk[b, r_idx, s, 1] = he
            sk_mask[b, r_idx, s] = 1.0
            slot[(hs, he, r_idx)] += 1
    return sk, sk_mask


# =============== LOSS ================

def masked_bce(pred_logits, gold, mask):
    """Binary cross entropy loss with masking"""
    loss = F.binary_cross_entropy_with_logits(pred_logits, gold, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def masked_bce_4d(pred_logits, gold, mask):
    """Binary cross entropy loss with masking for 4D tensors (for relation attention loss)"""
    loss = F.binary_cross_entropy_with_logits(pred_logits, gold, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)

def stage1_loss(
    fwd_head_start_logits,
    fwd_head_end_logits,
    bwd_tail_start_logits,
    bwd_tail_end_logits,
    
    golden_head_start_labels,
    golden_head_end_labels,
    golden_tail_start_labels,
    golden_tail_end_labels,

    token_mask):
    """Loss for stage 1 training (entity extractor only). From paper L_sub + L_obj"""
    L_fwd = masked_bce(fwd_head_start_logits, golden_head_start_labels, token_mask) + masked_bce(fwd_head_end_logits, golden_head_end_labels, token_mask)
    L_bwd = masked_bce(bwd_tail_start_logits, golden_tail_start_labels, token_mask) + masked_bce(bwd_tail_end_logits, golden_tail_end_labels, token_mask)
    return L_fwd + L_bwd


def brask_loss(outputs,gold_labels,token_mask):
    """Loss for the whole model."""


    _fwd_head_start_loss = masked_bce(
        outputs[MODEL_OUTPUT_KEYS["FORWARD_HEAD_START"]],
        gold_labels["fwd_head_start"],
        token_mask
    )

    _fwd_head_end_loss = masked_bce(
        outputs[MODEL_OUTPUT_KEYS["FORWARD_HEAD_END"]],
        gold_labels["fwd_head_end"],
        token_mask
    )

    _bwd_tail_start_loss = masked_bce(
        outputs[MODEL_OUTPUT_KEYS["BACKWARD_TAIL_START"]],
        gold_labels["bwd_tail_start"],
        token_mask
    )

    _bwd_tail_end_loss = masked_bce(
        outputs[MODEL_OUTPUT_KEYS["BACKWARD_TAIL_END"]],
        gold_labels["bwd_tail_end"],
        token_mask
    )

    L_f_subject = _fwd_head_start_loss + _fwd_head_end_loss
    L_b_subject = _bwd_tail_start_loss + _bwd_tail_end_loss


    mask_exp = token_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)



    _fwd_tail_start_loss = masked_bce_4d(
        outputs[MODEL_OUTPUT_KEYS["FORWARD_TAIL_START"]],
        gold_labels["fwd_tail_start"],
        mask_exp
    )

    _fwd_tail_end_loss = masked_bce_4d(
        outputs[MODEL_OUTPUT_KEYS["FORWARD_TAIL_END"]],
        gold_labels["fwd_tail_end"],
        mask_exp
    )

    _bwd_head_start_loss = masked_bce_4d(
        outputs[MODEL_OUTPUT_KEYS["BACKWARD_HEAD_START"]],
        gold_labels["bwd_head_start"],
        mask_exp
    )
    _bwd_head_end_loss = masked_bce_4d(
        outputs[MODEL_OUTPUT_KEYS["BACKWARD_HEAD_END"]],
        gold_labels["bwd_head_end"],
        mask_exp
    )
    L_f_obj = _fwd_tail_start_loss + _fwd_tail_end_loss
    L_b_obj = _bwd_head_start_loss + _bwd_head_end_loss

    total = L_f_subject + L_b_subject + L_f_obj + L_b_obj
    return total, {
        "L_f_subject": L_f_subject.item(),
        "L_b_subject": L_b_subject.item(),
        "L_f_obj": L_f_obj.item(),
        "L_b_obj": L_b_obj.item(),
    }


class RelationAttention(nn.Module):
    """In paper 3.3.2. Semantic relation guidance: returning fine-grained sentence representatio
    We introduce MLP, 
    We introduce attention_emb_dim
    """



    def __init__(self, hidden_dim, rel_dim, attention_dim=256):
        """rel_dim is the dim of relation embeddings (might be 768 if bert, or other for transe)"""
        super().__init__()
        self.w_r = nn.Linear(rel_dim, attention_dim)
        self.w_g = nn.Linear(hidden_dim, attention_dim)
        self.w_x = nn.Linear(hidden_dim , attention_dim)

        self.V  = nn.Linear(attention_dim, 1)


    def forward(self, X, relation_embedding, tokens_mean_embedding):
        """
        args:
            X: (B, L, H) token embeddings
            relation_embedding: (R, H) H can be hidden_dim or transe_rel_dim
            tokens_mean_embedding: (B, H)
        returns:
            c: (B, R, H) context-aware token representations for each relation
            a: (B, R, L) attention weights for each token and relation
        """
        wx_xi = self.w_x(X) #(B, L, attention_dim)
        wr_rj = self.w_r(relation_embedding) #(R, attention_dim)
        wg_hg = self.w_g(tokens_mean_embedding) #(B, attention_dim)

        x_exp = wx_xi.unsqueeze(1) #(B, 1, L, attention_dim)
        r_exp = wr_rj.unsqueeze(0).unsqueeze(2) #(1, R, 1, attention_dim)
        g_exp = wg_hg.unsqueeze(1).unsqueeze(2) #(B, 1, 1, attention_dim)


        z = torch.tanh(x_exp + r_exp + g_exp) #(B, R, L, attention_dim)
        e = self.V(z).squeeze(-1) #(B, R, L)

        a = torch.softmax(e, dim=-1) #(B, R, L)
        a_exp = a.unsqueeze(-1) #(B, R, L, 1)

        x_exp = X.unsqueeze(1) #(B, 1, L, H)
        c = (a_exp * x_exp).sum(dim=2) #(B, R, H)

        return c,a

class FuseExtractor(nn.Module):
    """3.3.3. Extraction of objects, 3.4. Backward triple extraction"""

    """
    fuse subject representations  and fine-grained sentence expression into ith token representation
    Hik = Ws Sk + Wx xi
    Hij = cj + xi
    Hijk = Hij + Hik

    we feed the special representation into fully connected neural network to obtain start and end probabilities
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.w_s = nn.Linear(hidden_dim , hidden_dim )
        self.w_x = nn.Linear(hidden_dim, hidden_dim)



    def forward(self, X: torch.Tensor, c: torch.Tensor, sk: torch.Tensor, sk_mask: torch.Tensor) -> torch.Tensor:
        """
        args:
            X Tensor for token embeddings (B, L, H)
            c Tensor Relation context (B, R, H)
            sk tensor with shape: (B, max_num_subjects, H)
            sk_mask tensor with shape: (B, max_num_subjects) with 1 for valid subject representations and 0 for padded ones

        Returns:
            h_ijk tensor with shape (B, R, max_num_subjects, L, H)
        """
        gating = False
        R  = c.shape[1]

        X_exp = X.unsqueeze(1) #(B, 1, L, H)
        X_exp = X_exp.expand(-1, R, -1, -1) # (B, R, L, H)

        c_exp = c.unsqueeze(2) # (B, R, 1, H)

        w_x  = self.w_x(X) #(B, L, H)
        w_sk = self.w_s(sk) #(B, max_num_subjects, H)
        w_sk = sk_mask.unsqueeze(-1) * w_sk #(B, max_num_subjects, H) with padded subjects zeroed out

        h_ik = w_sk.unsqueeze(2) + w_x.unsqueeze(1) #(B, max_num_subjects, 1, H) + (B, 1, L, H) -> (B, max_num_subjects, L, H)



        h_ij = c_exp + X_exp # (B, R, 1, H) + (B, R, L, H) -> (B, R, L, H)

        # if gating:
        #     g = torch.sigmoid(W([x, c]))
        #     h_ij = g * X_exp + (1 - g) * c_exp

        # I need (B, R, L, H)
        h_ijk = h_ik.unsqueeze(1) + h_ij.unsqueeze(2) # (B, 1, SUBJECT, L, H) + (B, R, 1, L, H ) -> (B, R, SUBJECT, L, H)


        return h_ijk

def sk_to_embeddings(sk, sk_mask, X):
    B, R, S, _ = sk.shape
    H = X.shape[2]
    sk_embs  = torch.zeros(B, R, S, H, device=X.device)
    for b in range(B):
        for r in range(R):
            for s in range(S):
                if sk_mask[b, r, s] == 0.0:
                    continue

                start = sk[b,r,s,0]
                end = sk[b,r,s,1]
                sk_embs[b, r, s] = (X[b, start] + X[b, end]) / 2.0
    return sk_embs # (B, R, S, H)



class BraskModel(torch.nn.Module):
    def __init__(self, hidden_dim, transe_rel_dim):
        super(BraskModel, self).__init__()


        # ! in the algorithm , I have to do fusion (token emb, relation emb, entity emb (subject, object)):
        # ! the way that the paper do is fusion is attention fusion (score =f(x_i, r, global_context)), we might want in the future do other fusion like gating which would be g = sigmoid(W[x_i, r]); h_i = g * x_i + (1-g) * r


        # ! I have to do relation smart pruning:
        # !   1- pre computed consine similarity between entity description and relation embedding
        # !   2- Not necessarily BERT cosine similarity, might be sentence-transformers
        # !   3- Consider for each description top-k relations + positive relation from ground truth


        # ! After prediction, I might do filtering of the result spans and keep only spans that matches an alias
        # ! I should not do this during training, my objective is to improve precision.

        # ? Can I prevent EntityExtractor to put positive value for [PAD] during training?



        self.fwd_head_predictor = EntityExtractor(hidden_dim)
        self.bwd_tail_predictor = EntityExtractor(hidden_dim)


        self.fwd_relation_attention = RelationAttention(hidden_dim, rel_dim = hidden_dim)
        self.bwd_relation_attention = RelationAttention(hidden_dim, rel_dim = transe_rel_dim)



        self.fwd_fuse_extractor = FuseExtractor(hidden_dim)
        self.bwd_fuse_extractor = FuseExtractor(hidden_dim)

        self.fwd_tail_predictor = EntityExtractor(hidden_dim)
        self.bwd_head_predictor = EntityExtractor(hidden_dim)


        self.inference_threshold = 0.5

    def forward(
        self,
        X,
        X_mean,
        mask,
        sk,
        sk_mask,
        semantic_rel_emb,
        transe_rel_emb
    ):

        # B: batch size, L: sequence length, H: hidden dimension
        description_embs = X #(B, L, H)
        description_embs_mask = mask #(B, L)
        description_mean_embs = X_mean #(B, H)


        fwd_head_start_logits, fwd_head_end_logits = self.fwd_head_predictor(description_embs) # (B, L, 1) , (B, L, 1)
        bwd_tail_start_logits, bwd_tail_end_logits = self.bwd_tail_predictor(description_embs) # (B, L, 1) , (B, L, 1)

        forward_c, forward_a = self.fwd_relation_attention(description_embs, semantic_rel_emb, description_mean_embs) # (B, R, H) , (B, R, L)
        backward_c, backward_a = self.bwd_relation_attention(description_embs, transe_rel_emb, description_mean_embs) # (B, R, H) , (B, R, L)

        sk_embs = sk_to_embeddings(sk, sk_mask, description_embs) # (B, R, S, H)

        forward_hijk = self.fwd_fuse_extractor(description_embs, forward_c, sk_embs, sk_mask) # (B, R, max_num_subjects, L, H)
        backward_hijk = self.bwd_fuse_extractor(description_embs, backward_c, sk_embs, sk_mask) # (B, R, max_num_subjects, L, H)


        B, R, S, L, H = forward_hijk.shape
        forward_tails_start_logits, forward_tail_end_logits = self.fwd_tail_predictor(forward_hijk) # (B, R, S, L, 1)
        backward_head_start_logits, backward_head_end_logits = self.bwd_head_predictor(backward_hijk) # (B, R, S, L, 1)

        return {
                MODEL_OUTPUT_KEYS["FORWARD_HEAD_START"]: fwd_head_start_logits,
                MODEL_OUTPUT_KEYS["FORWARD_HEAD_END"]: fwd_head_end_logits,
                MODEL_OUTPUT_KEYS["BACKWARD_TAIL_START"]: bwd_tail_start_logits,
                MODEL_OUTPUT_KEYS["BACKWARD_TAIL_END"]: bwd_tail_end_logits,
                MODEL_OUTPUT_KEYS["FORWARD_TAIL_START"]: forward_tails_start_logits,
                MODEL_OUTPUT_KEYS["FORWARD_TAIL_END"]: forward_tail_end_logits,
                MODEL_OUTPUT_KEYS["BACKWARD_HEAD_START"]: backward_head_start_logits,
                MODEL_OUTPUT_KEYS["BACKWARD_HEAD_END"]: backward_head_end_logits,
        }

    def extract_sk(self,
                   X,
                   mask,
                   num_relations,
                   max_subjects):
        pass


def set_stage(model:BraskModel, stage:int):
    """Freeze/unfreeze model parameters according to training stage"""

    # freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False


    if stage == 1:
        # unfreeze entity extractor parameters only
        for param in model.fwd_head_predictor.parameters():
            param.requires_grad = True
        for param in model.bwd_tail_predictor.parameters():
            param.requires_grad = True
    elif stage in (2,3):
        # unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True


def get_optimizer(model:BraskModel, stage:int):
    """Get optimizer with parameters according to training stage"""

    lr = LEARNING_RATE_STAGE_1 if stage == 1 else LEARNING_RATE_STAGE_2
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr)



def run_epoch_stage1(
    model : BraskModel,
    dataloader,
    optimizer,
    max_length
):


    model.train()
    total_loss, n_batches = 0.0, 0


    for batch in tqdm(dataloader, desc="Stage 1 epoch"):
        X = batch[BraskDataset.BATCH_KEYS["EMBS"]].to(device) #(B, L, H)
        mask = batch[BraskDataset.BATCH_KEYS["EMBS_MASKS"]].to(device) #(B, L)
        golden_triples = batch[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]] # B lists  of triples


        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)

        fwd_head_start, fwd_head_end = model.fwd_head_predictor(X) # (B, L, 1) , (B, L, 1)
        bwd_tail_start, bwd_tail_end = model.bwd_tail_predictor(X) # (B, L, 1) , (B, L, 1)

        loss = stage1_loss(
            fwd_head_start_logits=fwd_head_start,
            fwd_head_end_logits=fwd_head_end,
            bwd_tail_start_logits=bwd_tail_start,
            bwd_tail_end_logits=bwd_tail_end,
            golden_head_start_labels=gold_fhs,
            golden_head_end_labels=gold_fhe,
            golden_tail_start_labels=gold_bts,
            golden_tail_end_labels=gold_bte,
            token_mask=mask
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss  / (n_batches + 1e-8)



def run_epoch_stage_2(
    model:BraskModel,
    dataloader,
    optimizer,
    rel2idx,
    num_relations,
    teacher_forcing_ratio,
    semantic_rel_emb,
    transe_rel_emb
):


    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in tqdm(dataloader, desc=f"Stage 2 (tf={teacher_forcing_ratio:.2f})"):
        X = batch[BraskDataset.BATCH_KEYS["EMBS"]].to(device) #(B, L, H)
        X_mean = batch[BraskDataset.BATCH_KEYS["MEAN_EMBS"]].to(device) #(B, H)
        mask = batch[BraskDataset.BATCH_KEYS["EMBS_MASKS"]].to(device) #(B, L)
        golden_triples = batch[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]] # B lists  of triples

        max_subjects_in_batch = get_max_subjects(golden_triples, rel2idx)

        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
        gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
            triples_batch=golden_triples,
            mask=mask,
            max_subjects=max_subjects_in_batch,
            num_relations=num_relations,
            rel2idx=rel2idx
        )

        gold_labels = {
            "fwd_head_start": gold_fhs,
            "fwd_head_end": gold_fhe,
            "bwd_tail_start": gold_bts,
            "bwd_tail_end": gold_bte,
            "fwd_tail_start": gold_fts,
            "fwd_tail_end": gold_fte,
            "bwd_head_start": gold_bhs,
            "bwd_head_end": gold_bhe,
        }

        use_gold = (torch.rand(1).item() < teacher_forcing_ratio)
        if use_gold:
            sk, sk_mask = build_sk_from_gold(golden_triples, mask, num_relations, max_subjects_in_batch, rel2idx)
        else:
            sk, sk_mask = model.extract_sk(X, mask, num_relations, max_subjects_in_batch)


        outputs = model(
            X,
            X_mean,
            mask,
            sk,
            sk_mask,
            semantic_rel_emb,
            transe_rel_emb
        )

        loss, components = brask_loss(outputs, gold_labels, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()
        n_batches += 1
    return total_loss / (n_batches + 1e-8)

@torch.no_grad()
def evaluate(
    model: BraskModel,
    dataloader,
    rel2idx,
    num_relations,
    semantic_rel_emb,
    transe_rel_emb,
    stage
):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in tqdm(dataloader, desc="Evaluation"):
        X = batch[BraskDataset.BATCH_KEYS["EMBS"]].to(device) #(B, L, H)
        X_mean = batch[BraskDataset.BATCH_KEYS["MEAN_EMBS"]].to(device) #(B, H)
        mask = batch[BraskDataset.BATCH_KEYS["EMBS_MASKS"]].to(device) #(B, L)
        golden_triples = batch[BraskDataset.BATCH_KEYS["GOLDEN_TRIPLES"]] # B lists  of triples


        gold_fhs, gold_fhe, gold_bts, gold_bte = build_gold_entity_labels(golden_triples, mask)
 
        if stage == 1:
            fwd_head_start, fwd_head_end = model.fwd_head_predictor(X)
            bwd_tail_start, bwd_tail_end = model.bwd_tail_predictor(X)
            loss = stage1_loss(
                fwd_head_start, fwd_head_end,
                bwd_tail_start, bwd_tail_end,
                gold_fhs, gold_fhe, gold_bts, gold_bte, mask
            )
        else:
            max_subjects_in_batch = get_max_subjects(golden_triples, rel2idx)
            gold_fts, gold_fte, gold_bhs, gold_bhe = build_gold_tail_labels(
                triples_batch=golden_triples,
                mask=mask,
                max_subjects=max_subjects_in_batch,
                num_relations=num_relations,
                rel2idx=rel2idx
            )
            gold_labels = {
                "fwd_head_start": gold_fhs, "fwd_head_end": gold_fhe,
                "bwd_tail_start": gold_bts, "bwd_tail_end": gold_bte,
                "fwd_tail_start": gold_fts, "fwd_tail_end": gold_fte,
                "bwd_head_start": gold_bhs, "bwd_head_end": gold_bhe,
            }
            # At eval always use model predictions (no teacher forcing)
            sk, sk_mask = model.extract_sk(X, mask, num_relations, max_subjects_in_batch)
            outputs = model.forward(X, X_mean, mask, sk, sk_mask, semantic_rel_emb, transe_rel_emb)
            loss, _ = brask_loss(outputs, gold_labels, mask)

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


def main():
    batch_size      = 16
    stage1_epochs   = 10
    stage2_epochs   = 10
    stage3_epochs   = 20   # teacher forcing decays over these
    val_split       = 0.1


    out_checkpoint_stage1_best  = os.path.join(CHECKPOINTS_DIR, "brask_stage1_best.pt")
    out_checkpoint_stage2_best  = os.path.join(CHECKPOINTS_DIR, "brask_stage2_best.pt")
    out_checkpoint_stage3_best  = os.path.join(CHECKPOINTS_DIR, "brask_stage3_best.pt")


    rel2idx = data_loader.get_rel2idx(minimized=True)
    description_embs_all, description_embs_ids, description_embs_masks = \
        data_loader.get_description_embeddings_all()
    description_embs_mean = data_loader.get_description_embeddings_mean()
    golden_triples        = data_loader.get_golden_triples()
    semantic_rel_emb      = data_loader.get_semantic_relation_embeddings().to(device)  # (R, H)
    transe_rel_emb        = data_loader.get_trane_relation_embeddings().to(device)    # (R, H)

    assert semantic_rel_emb.shape[0] == transe_rel_emb.shape[0] == len(rel2idx), f"Number of relations mismatch between semantic and transe embeddings and rel2idx: {semantic_rel_emb.shape[0]} vs {transe_rel_emb.shape[0]} vs {len(rel2idx)}"

    transe_rel_dim = transe_rel_emb.shape[1]
    num_relations = semantic_rel_emb.shape[0]

    max_length = description_embs_all.shape[1]
    H          = description_embs_all.shape[2]


    N     = len(description_embs_ids)
    n_val = int(N * val_split)
    ids_train = description_embs_ids[n_val:]
    ids_val   = description_embs_ids[:n_val]

    def make_dataset(ids):
        idx = [description_embs_ids.index(i) for i in ids]
        return BraskDataset(
            description_embs=description_embs_all[idx],
            description_embs_ids=ids,
            description_embs_masks=description_embs_masks[idx],
            description_mean_embs=description_embs_mean[idx],
            golden_triples=golden_triples
        )

    train_loader = DataLoader(make_dataset(ids_train), batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(make_dataset(ids_val),   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = BraskModel(hidden_dim=H, transe_rel_dim=transe_rel_dim).to(device)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")


    # ════════════════════════════════════════
    # STAGE 1 — Entity extractors only
    # ════════════════════════════════════════


    print("\n── Stage 1: Entity extractors ──")
    set_stage(model, stage=1)
    optimizer = get_optimizer(model, stage=1)

    for epoch in range(stage1_epochs):
        train_loss = run_epoch_stage1(model, train_loader, optimizer, max_length)
        val_loss   = evaluate(
                model= model,
                dataloader = val_loader,
                rel2idx=rel2idx,
                num_relations = num_relations,
                semantic_rel_emb=semantic_rel_emb,
                transe_rel_emb=transe_rel_emb,
                stage=1
        )

        print(f"  [S1] Epoch {epoch+1}/{stage1_epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_checkpoint_stage1_best)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")


        # ════════════════════════════════════════
    # STAGE 2 — Full model, teacher forcing = 1.0
    # ════════════════════════════════════════
    print("\n── Stage 2: Full model (teacher forcing = 1.0) ──")
    set_stage(model, stage=2)
    optimizer     = get_optimizer(model, stage=2)
    best_val_loss = float("inf")

    for epoch in range(stage2_epochs):
        train_loss = run_epoch_stage_2(
            model = model,
            dataloader = train_loader,
            optimizer=optimizer,
            rel2idx=rel2idx,
            num_relations = num_relations,
            teacher_forcing_ratio = 1.0,
            semantic_rel_emb=semantic_rel_emb,
            transe_rel_emb=transe_rel_emb
        )
        val_loss   = evaluate(
                model= model,
                dataloader = val_loader,
                rel2idx=rel2idx,
                num_relations = num_relations,
                semantic_rel_emb=semantic_rel_emb,
                transe_rel_emb=transe_rel_emb,
                stage=2
        )


        print(f"  [S2] Epoch {epoch+1}/{stage2_epochs} — train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_checkpoint_stage2_best)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")


# ════════════════════════════════════════
    # STAGE 3 — Full model, teacher forcing decays 1.0 → 0.0
    # ════════════════════════════════════════
    print("\n── Stage 3: Full model (teacher forcing decay) ──")
    set_stage(model, stage=3)
    optimizer     = get_optimizer(model, stage=3)
    best_val_loss = float("inf")

    for epoch in range(stage3_epochs):
        tf_ratio = max(0.0, 1.0 - epoch / stage3_epochs)   # linear decay
        train_loss = run_epoch_stage_2(
            model = model,
            dataloader = train_loader,
            optimizer=optimizer,
            rel2idx=rel2idx,
            num_relations = num_relations,
            teacher_forcing_ratio = tf_ratio,
            semantic_rel_emb=semantic_rel_emb,
            transe_rel_emb=transe_rel_emb
        )
        val_loss = evaluate(
                model= model,
                dataloader = val_loader,
                rel2idx=rel2idx,
                num_relations = num_relations,
                semantic_rel_emb=semantic_rel_emb,
                transe_rel_emb=transe_rel_emb,
                stage=3
        )
        print(f"  [S3] Epoch {epoch+1}/{stage3_epochs}  tf={tf_ratio:.2f} — "
              f"train: {train_loss:.4f}  val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_checkpoint_stage3_best)
            print(f"    ✓ Saved (val={best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best checkpoints saved in {CHECKPOINTS_DIR}")

if __name__ == "__main__":
    main()
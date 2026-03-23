#### THIS FILE IS FOR TEST PURPOSES ONLY.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from utils.pre_processed_data import data_loader
from models.EntityExtractor import EntityExtractor



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MiniBrask(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fwd_head_entity_extractor = EntityExtractor(hidden_dim)
        self.bwd_tail_entity_extractor = EntityExtractor(hidden_dim)

    def forward(self, description_embeddings):
        fwd_head_start_logits, fwd_head_end_logits = self.fwd_head_entity_extractor(description_embeddings)
        bwd_tail_start_logits, bwd_tail_end_logits = self.bwd_tail_entity_extractor(description_embeddings)
        return fwd_head_start_logits, fwd_head_end_logits, bwd_tail_start_logits, bwd_tail_end_logits


# ? This for just staged and we will not use this loss for the whole BraskModel 
def compute_loss(
    fwd_head_start_logits: torch.Tensor, #(B, L)
    fwd_head_end_logits: torch.Tensor, #(B,L)
    bwd_tail_start_logits: torch.Tensor, #(B,L)
    bwd_tail_end_logits: torch.Tensor, #(B,L)
    gold_triples_per_sentence: list, # [b] -> [(hs, he), r, (ts, te)]
    token_mask: torch.Tensor, # (B, L) 0 if pad
    max_length: int # L
)-> torch.Tensor:
    B = len(gold_triples_per_sentence)
    L = max_length
    assert fwd_head_start_logits.shape ==  (B, max_length), f"Invalid fwd_head_start_logits.shape: {fwd_head_start_logits.shape}. SHOULD be {(B, max_length)}"
    assert fwd_head_end_logits.shape ==   (B, max_length), f"Invalid fwd_head_end_logits.shape: {fwd_head_end_logits.shape}. SHOULD be {(B, max_length)}"
    assert bwd_tail_start_logits.shape ==     (B, max_length), f"Invalid bwd_tail_start_logits.shape: {bwd_tail_start_logits.shape}. SHOULD be {(B, max_length)}"
    assert bwd_tail_end_logits.shape == (B, max_length), f"Invalid bwd_tail_end_logits.shape: {bwd_tail_end_logits.shape}. SHOULD be {(B, max_length)}"

    assert token_mask.shape == (B, max_length)
    device = fwd_head_start_logits.device
    fwd_loss_head = torch.tensor(0.0, device=device)
    bwd_loss_tail = torch.tensor(0.0, device=device)
    n_triples = 0
    def masked_bce(logits, gold, mask):
        loss = F.binary_cross_entropy_with_logits(
            logits, gold, reduction="none"
        )
        return (loss * mask).sum()  / (mask.sum() + 1e-8)

    for b, triples in enumerate(gold_triples_per_sentence):
        mask = token_mask[b]
        
        for (head_start, head_end), r, (tail_start, tail_end) in triples:
            gold_h_start = torch.zeros(max_length, device=device)
            gold_h_end = torch.zeros(max_length, device=device)
            gold_h_start[head_start] = 1.0
            gold_h_end[head_end] = 1.0

            gold_t_start = torch.zeros(max_length, device=device)
            gold_t_end   = torch.zeros(max_length, device=device)
            gold_t_start[tail_start] = 1.0
            gold_t_end[tail_end]     = 1.0

            fwd_loss_head = fwd_loss_head +  masked_bce(fwd_head_start_logits[b], gold_h_start, mask) +  masked_bce(fwd_head_end_logits[b], gold_h_end, mask)
            bwd_loss_tail = bwd_loss_tail + masked_bce(bwd_tail_start_logits[b], gold_t_start, mask) +  masked_bce(bwd_tail_end_logits[b], gold_t_end, mask)

            n_triples += 1

            #! here usually I include for the object which I will not do here

    return (fwd_loss_head + bwd_loss_tail) / max(n_triples, 1)


class _EntityExtractorDst(Dataset):
    def __init__(self, description_embeddings, desecription_embeddings_ids, description_embs_masks,  golden_triples):
        self.description_embeddings = description_embeddings
        self.desecription_embeddings_ids = desecription_embeddings_ids
        self.description_embs_masks = description_embs_masks
        self.golden_triples = golden_triples
    def __getitem__(self, index):
        description_embedding = self.description_embeddings[index]
        description_id = self.desecription_embeddings_ids[index]
        mask = self.description_embs_masks[index]
        golden_triples = self.golden_triples[description_id]

        return description_embedding, mask, golden_triples


    def __len__(self):
        return len(self.description_embeddings)

def collate_fn(batch):
    description_embeddings = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    golden_triples = [item[2] for item in batch]
    return description_embeddings, masks, golden_triples


def _extract_spans(start_probs: torch.Tensor, end_probs: torch.Tensor, threshold:float=0.5):
    start_positions = (start_probs >= threshold).nonzero(as_tuple=False).squeeze(-1)
    end_positions = (end_probs >= threshold).nonzero(as_tuple=False).squeeze(-1)
    
    if start_positions.dim() == 0 or end_positions.dim() == 0:
        return set()

    spans = set()
    consumed_ends = set()
    for s in start_positions:
        s = s.item()
        valid_ends = [e.item() for e in end_positions if e.item() >= s and e.item() not in consumed_ends]
        if valid_ends:
            e = min(valid_ends)
            spans.add((s, e))
            consumed_ends.add(e)
    return spans


def _eval_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate(model, val_dataloader, device, max_length, threshold=0.5):
    model.eval()
    fwd_tp = fwd_fp = fwd_fn = 0
    bwd_tp = bwd_fp = bwd_fn = 0
    with torch.no_grad():
        for batch in val_dataloader:
            X, mask, golden_triples = batch
            X = X.to(device)
            mask = mask.to(device)
            fwd_start_logits, fwd_end_logits, bwd_start_logits, bwd_end_logits = model(X)
            fwd_start_probs = torch.sigmoid(fwd_start_logits).squeeze(-1) # (B, L)
            fwd_end_probs = torch.sigmoid(fwd_end_logits).squeeze(-1) # (B, L)
            bwd_start_probs = torch.sigmoid(bwd_start_logits).squeeze(-1) # (B, L)
            bwd_end_probs = torch.sigmoid(bwd_end_logits).squeeze(-1) # (B, L)
            for b, b_golden_triples in enumerate(golden_triples):
                gold_head_spans = set()
                gold_tail_spans = set()
                for (hs, he), _, (ts, te) in b_golden_triples:
                    gold_head_spans.add((hs, he))
                    gold_tail_spans.add((ts, te))
                fwd_pred_spans = _extract_spans(fwd_start_probs[b], fwd_end_probs[b], threshold)
                bwd_pred_spans = _extract_spans(bwd_start_probs[b], bwd_end_probs[b], threshold)
                # True positives
                fwd_tp += len(fwd_pred_spans & gold_head_spans)
                bwd_tp += len(bwd_pred_spans & gold_tail_spans)
                # False positives
                fwd_fp += len(fwd_pred_spans - gold_head_spans)
                bwd_fp += len(bwd_pred_spans - gold_tail_spans)
                # False negatives
                fwd_fn += len(gold_head_spans - fwd_pred_spans)
                bwd_fn += len(gold_tail_spans - bwd_pred_spans)

    fwd_metrics = _eval_metrics(fwd_tp, fwd_fp, fwd_fn)
    bwd_metrics = _eval_metrics(bwd_tp, bwd_fp, bwd_fn)
    return {"forward": fwd_metrics, "backward": bwd_metrics, "fwd_tp": fwd_tp, "bwd_tp": bwd_tp} 

def test_train_alone():
    num_epochs  = 10
    val_split = 0.1
    LEARNING_RATE =   1e-4

    description_embs_all, description_embs_ids, description_embs_masks = data_loader.get_description_embeddings_all()
    golden_triples = data_loader.get_golden_triples()

    N = len(description_embs_all)
    L = description_embs_all.shape[1]
    val_size = int(N * val_split)
    train_size = N - val_size
    full_dataset = _EntityExtractorDst(description_embs_all, description_embs_ids, description_embs_masks,  golden_triples)

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    H = description_embs_all.shape[-1]
    model = _MiniBrask(hidden_dim=H).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader)):
            X = batch[0].to(device) # description embeddings
            mask = batch[1].to(device) # mask
            golden_triples = batch[2] # golden triples
            fwd_head_start_logits, fwd_head_end_logits, bwd_tail_start_logits, bwd_tail_end_logits = model(X) # (B, L, 1)
            loss  = compute_loss(
                fwd_head_start_logits= fwd_head_start_logits.squeeze(-1), 
                fwd_head_end_logits=fwd_head_end_logits.squeeze(-1), 
                bwd_tail_start_logits=bwd_tail_start_logits.squeeze(-1), 
                bwd_tail_end_logits=bwd_tail_end_logits.squeeze(-1), 
                gold_triples_per_sentence=golden_triples, 
                token_mask=mask, 
                max_length=L
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        avg_loss = epoch_loss / max(batch_count, 1)

        # Evaluate:
        metrics = evaluate(model, val_loader, device, max_length=description_embs_all.shape[1], threshold=0.5)
        head_f1 = metrics["forward"]["f1"]
        tail_f1 = metrics["backward"]["f1"]
        avg_f1 = (head_f1 + tail_f1) / 2
        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"| loss: {avg_loss:.4f} "
            f"| forward P={metrics['forward']['precision']:.4f} "
            f"R={metrics['forward']['recall']:.4f} "
            f"F1={head_f1:.4f}, fwd_tp={metrics['fwd_tp']} "
            f"| backward P={metrics['backward']['precision']:.4f} "
            f"R={metrics['backward']['recall']:.4f} "
            f"F1={tail_f1:.4f}, bwd_tp={metrics['bwd_tp']}"
        )
        if avg_f1 > best_f1:
            best_f1    = avg_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "entity_extractor_best.pt")
            print(f"  -> Saved checkpoint (avg F1={best_f1:.4f})")

    print(f"\nBest model at epoch {best_epoch} with avg F1={best_f1:.4f}")

import numpy as np
import torch
from tqdm import tqdm

from helpers.resource_monitor import log_resource_usage
from utils.files import init_mmap

def get_rel_embs(relations_dict, bert_tokenizer, bert_model, batch_size, use_cuda, device):
    """Compute one averaged BERT embedding per relation.

    All aliases for every relation are flattened into a single list and
    processed in chunks of ``BATCH_SIZE``.  Per-alias embeddings are
    accumulated into per-relation sums via ``scatter_add_``, then divided
    by their alias counts to produce the final averages.

    Each alias embedding is the attention-mask mean-pool of the element-wise
    average of BERT's last two hidden layers.

    Args:
        relations_dict (dict[str, list[str]]): Mapping from relation ID to its
            list of text aliases.  Relations with no aliases fall back to the
            relation ID itself.
        bert_tokenizer (BertTokenizerFast): Pre-loaded BERT tokenizer.
        bert_model (BertModel): Pre-loaded BERT model (already on ``device``).

    Returns:
        torch.Tensor: Float32 tensor of shape ``(n_relations, 768)`` where each
            row is the averaged embedding for the corresponding relation.
    """
    rel_ids = list(relations_dict.keys())
    n_rels = len(rel_ids)

    all_aliases = []
    rel_indices = []
    for i, rel_id in enumerate(rel_ids):
        aliases = relations_dict[rel_id]
        if not aliases:
            aliases = [rel_id]
        for alias in aliases:
            all_aliases.append(alias)
            rel_indices.append(i)

    # Sort by character length so each batch has similarly-sized sequences,
    # minimising padding tokens and reducing total BERT computation.
    order = sorted(range(len(all_aliases)), key=lambda i: len(all_aliases[i]))
    all_aliases = [all_aliases[i] for i in order]
    rel_indices = [rel_indices[i] for i in order]

    rel_idx_tensor = torch.tensor(rel_indices, dtype=torch.int64, device=device)

    sums = torch.zeros(n_rels, 768, dtype=torch.float32, device=device)
    counts = torch.zeros(n_rels, dtype=torch.float32, device=device)

    if use_cuda:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu")

    bert_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(all_aliases), batch_size), desc="embedding relations"):
            chunk = all_aliases[start:start + batch_size]
            rel_idx_chunk = rel_idx_tensor[start:start + batch_size]

            encoded = bert_tokenizer(
                chunk,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            with autocast_ctx:
                out = bert_model(**encoded, output_hidden_states=True)

            # average last 2 hidden layers
            hidden = torch.stack(out.hidden_states[-2:], dim=0).mean(dim=0)  # (chunk, seq, 768)

            # attention-mask mean-pool over sequence dimension
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # (chunk, seq, 1)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (chunk, 768)
            emb = emb.float()

            sums.scatter_add_(0, rel_idx_chunk.unsqueeze(1).expand_as(emb), emb)
            ones = torch.ones(len(chunk), dtype=torch.float32, device=device)
            counts.scatter_add_(0, rel_idx_chunk, ones)

    return sums / counts.clamp(min=1).unsqueeze(1)


def save_descriptions_embedding(tokenizer, model, sentences: list[str], device, use_cuda, out_all_embs: str, out_mean_embs: str, out_all_masks: str, max_length: int=256) -> bool:
    """Embed each sentence using BERT, saving (N,L,H) all_embs, (N,H) mean_embs, (N,L) masks.

    All three outputs are written directly to memory-mapped .npy files so that
    no full-dataset tensor ever lives in CPU RAM (which would be tens of GB for
    a large corpus). Tokenization is done upfront for the whole corpus (fast
    with BertTokenizerFast), then padding is applied per-batch to the batch's
    own max length (dynamic padding). The caller pre-sorts sentences by text
    length so that each batch is similarly sized, minimising padding waste.
    """
    model = model.to(device)
    model.eval()
    batch_size = 512 if use_cuda else 32

    N = len(sentences)
    H = model.config.hidden_size

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if use_cuda else torch.autocast(device_type="cpu")
    )

    # ── Step 1: Allocate all outputs as memory-mapped files ───────────────────
    # Using mmap means each batch is written straight to disk page-by-page.
    # The alternative — torch.zeros(N, max_length, H) in CPU RAM — would
    # allocate N×L×H×2 bytes (all_embs) + N×H×4 (mean) + N×L×8 (masks)
    # upfront, which is tens of GB for a large corpus.
    mm_all_embs  = init_mmap(out_all_embs,  shape=(N, max_length, H), dtype="float16")
    mm_mean_embs = init_mmap(out_mean_embs, shape=(N, H),             dtype="float32")
    mm_all_masks = init_mmap(out_all_masks, shape=(N, max_length),    dtype="int64")

    # ── Step 2: Pre-tokenize all sentences (no padding yet) ───────────────────
    # BertTokenizerFast processes the full corpus in seconds. Storing token IDs
    # without padding keeps this structure small. We pad per-batch in step 3 so
    # each batch only pads to its own longest sequence (dynamic padding), which
    # the caller enables by pre-sorting sentences by text length before passing.
    print("Pre-tokenizing...")
    all_enc = tokenizer(sentences, padding=False, truncation=True, max_length=max_length)

    # ── Step 3: Embed one batch at a time, writing results immediately ────────
    with torch.no_grad():
        for batch_num, start in enumerate(tqdm(range(0, N, batch_size), desc="Embedding sentences")):
            end = min(start + batch_size, N)

            batch_enc = {k: all_enc[k][start:end] for k in all_enc}
            enc = tokenizer.pad(batch_enc, padding=True, return_tensors="pt")
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with autocast_ctx:
                out  = model(input_ids=input_ids, attention_mask=attention_mask)
                embs = out.last_hidden_state  # (B, L, H), L = this batch's max len

            attention_mask_exp = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            mean_embs = (
                (embs.float() * attention_mask_exp).sum(dim=1)
                / attention_mask_exp.sum(dim=1).clamp(min=1)
            )  # (B, H)

            B, L, _ = embs.shape
            mm_all_embs[start:end, :L]  = embs.cpu().half().numpy()
            mm_mean_embs[start:end]      = mean_embs.cpu().numpy()
            mm_all_masks[start:end, :L]  = attention_mask.cpu().numpy()

            # Flush dirty mmap pages to disk every 50 batches so the OS page
            # cache does not silently accumulate gigabytes of unwritten data.
            if batch_num % 50 == 0:
                mm_all_embs.flush()
                mm_mean_embs.flush()
                mm_all_masks.flush()

            if batch_num % 100 == 0:
                log_resource_usage(batch_num, use_cuda)

    # ── Step 3: Final flush — data is already on disk, nothing to aggregate ──
    mm_all_embs.flush()
    mm_mean_embs.flush()
    mm_all_masks.flush()
    del mm_all_embs, mm_mean_embs, mm_all_masks

    return True


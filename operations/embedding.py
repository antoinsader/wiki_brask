import torch
from tqdm import tqdm


from utils.files import save_tensor, init_mmap

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
    """Embed each sentence using BERT, and save into tensor files(mean_embs: Tensor[N, D], all_embs: Tensor[N, L, D], all_masks: Tensor[N, L] )"""

    model  = model.to(device)
    model.eval()
    batch_size = 128 if use_cuda else 32

    N = len(sentences)
    H = model.config.hidden_size

    # float16 halves the on-disk size vs float32, which prevents bus errors
    # when N is large (e.g. 40K sentences × 256 × 768 × 2 bytes ≈ 19 GB).
    mm_all_embs = init_mmap(out_all_embs, shape=(N, max_length, H), dtype="float16")

    if use_cuda:
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    else:
        autocast_ctx = torch.autocast(device_type="cpu")

    final_mean_embs = torch.zeros(N, H, dtype=torch.float32)
    final_all_masks = torch.zeros(N, max_length, dtype=torch.int64)
    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), batch_size), desc="Embedding sentences"):
            end = min(start + batch_size, len(sentences))
            chunk = sentences[start: end]
            enc = tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with autocast_ctx:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                embs = out.last_hidden_state  # (B, L, H)

            attention_mask_exp = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            sum_embs = (embs.float() * attention_mask_exp).sum(dim=1)  # (B, H)
            token_counts = attention_mask_exp.sum(dim=1).clamp(min=1)  # (B, 1)
            mean_embs = sum_embs / token_counts  # (B, H)

            final_mean_embs[start:end] = mean_embs.cpu()
            mm_all_embs[start:end] = embs.cpu().half().numpy()
            final_all_masks[start:end] = attention_mask.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    mm_all_embs.flush()
    del mm_all_embs

    save_tensor(final_mean_embs, out_mean_embs)
    del final_mean_embs
    save_tensor(final_all_masks, out_all_masks)

    return True


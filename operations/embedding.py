from multiprocessing import Pool

import torch
from tqdm import tqdm

from utils.chunking import chunk_list
from utils.files import cache_array, save_tensor



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


def description_embedding_chunk_worker(process_idx: int, sentences_chunk: list[str], tokenizer, model, device, use_cuda, max_length: int):

    N = len(sentences_chunk)
    H = model.config.hidden_size
    chunk_batch_size = 256 if use_cuda else 32

    mean_embs = torch.zeros(N, H, dtype=torch.float32)
    all_embs = torch.zeros(N, max_length, H, dtype=torch.float32)
    all_masks = torch.zeros(N, max_length, dtype=torch.int64)

    for start in tqdm(range(0, N, chunk_batch_size), desc=f"[Process-{process_idx}] Embedding sentences"):
        end = min(start + chunk_batch_size, N)
        chunk_batch = sentences_chunk[start:end]
        enc = tokenizer(
            chunk_batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            embs = out.last_hidden_state  # (B, L, H)
        attention_mask_exp = attention_mask.unsqueeze(-1)  # (B, L, 1)
        sum_embs = (embs * attention_mask_exp).sum(dim=1)  # (B, H)
        token_counts = attention_mask_exp.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_embs_batch = sum_embs / token_counts  # (B, H)

        mean_embs[start:end] = mean_embs_batch.cpu()
        all_embs[start:end] = embs.cpu()
        all_masks[start:end] = attention_mask.cpu()  # (B, L)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return mean_embs, all_embs, all_masks




def get_descriptions_embedding(tokenizer, model, sentences: list[str], device, use_cuda, desc_ids: list[str], out_mean_embs: str, out_all_embs: str, out_all_masks: str, out_ids: str,  max_length: int=128) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed each sentence using BERT, returns (mean_embs: Tensor[N, D], all_embs: Tensor[N, L, D], all_masks: Tensor[N, L] )"""

    model  = model.to(device)
    model.eval()
    num_workers = 16 if use_cuda else 1

    sentences_chunks = chunk_list(sentences, chunks_n=num_workers)

    print(f"Distributing embedding work to {num_workers} workers")
    with Pool(
        processes=num_workers,
    ) as pool:
        worker_args = [(i, chunk, tokenizer, model, device, use_cuda, max_length) for i, chunk in enumerate(sentences_chunks)]
        results = pool.starmap(description_embedding_chunk_worker, worker_args)
        results_mean_embs = [res[0] for res in results]
        all_means_embs = torch.cat(results_mean_embs, dim=0)
        print(f"mean_embs shape: {all_means_embs.shape} should be (N, H) ({len(sentences)}, 768)")
        save_tensor(all_means_embs, out_mean_embs)
        del results_mean_embs
        results_all_embs = [res[1] for res in results]
        all_all_embs = torch.cat(results_all_embs, dim=0)
        print(f"all_embs shape: {all_all_embs.shape} should be (N, L, H) ({len(sentences)}, {max_length}, 768)")
        save_tensor(all_all_embs, out_all_embs)
        del results_all_embs
        results_all_masks = [res[2] for res in results]
        all_all_masks = torch.cat(results_all_masks, dim=0)
        save_tensor(all_all_masks, out_all_masks)
        del results_all_masks
        cache_array(desc_ids, out_ids)
        
        return True
    return False


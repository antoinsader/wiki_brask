


### Entities:

Whenever we refer:
- B: batch_size
- L: description_max_length (128)
- H: encoder hidden dim (768 for BERT)
- T: dim for relation after training on TransE
- R: number of relations
- S: number of subjects
- BCE: Binary Cross Entropy loss. The loss function used in the training, it do binary cross entropy with logits calculating loss from the prediction logits and the gold labels.

**Entities**
- descriptions_embs_all: Tensor of shape (B, L, H), vector of H for each `description` inside the batch
- description_embs_masks: Tensor of shape (B, L) having 0 if the token is [PAD]
- description_embs_mean Tensor of shape (B, H) having mean-pooled embedding for each `description`
- descriptions_embs_ids: list of `entity_id` to map `description_embs_all` and `description_embs_mean` and `description_embs_masks`
- golden_triples: dict. pre-processed spans of the tokens for the ground truth. key is `entity_id`, value is a list of tuples (`header_spans`, `relation_id`, `tail_spans`), where `head_spans` and `tail_spans` are tuples `(start_index, end_index)`.
- semantic_rel_emb: Tensor of shape (R, H). Representing relation embeddings using BERT encoder (prepared in prepare.py)
- transe_rel_emb: Tensor of shape (R, T). Representing relation embeddings learned by the transE algorith (prepared in train_transe.py)

### General guideline:

We do the training through 3 stages:

1- Stage 1: entity extractors, we freeze all grads except the layers predicting `forward head` and `backward tail`
2- Stage 2: Unfreeze everything, train with teacher forcing -> `sk` is built from gold labels.
3- Stage 3: Unfreeze everything, train without teacher forcing -> `sk` is built based on model predictions

### Pipeline:

1- Split `descriptions` into `training dataloader` and `validating dataloader` (depending on val_split).

2- **Stage-1**:

- Prepare optimizer
- Loop through `stage1_epochs`, loop through `training dataloader`
- Build gold entity labels: `gold_fhs, gold_fhe, gold_bts, gold_bte` extracted from golden_triples. Those are binary labels (in paper binary tagging system), representing for each position if it is a start or end of an entity.
- Predicting `fwd_head_start`, `fwd_head_end`, `bwd_tail_start`, `bwd_tail_end` using model's `EntityExtractor`. Those are logits, each with shape (B, L, 1)

- Calculate `stage1_loss` based on the logits.
- stage1_loss is calculating `L = L_fwd + L_bwd`. Where `L_fwd` is the sum of `BCE(forward_head_start_logits, golden_head_start_labels) + BCE(forward_head_end_logits, golden_head_end_labels)`, `L_bwd = BCE(backward_tail_start_logits, golden_backward_tail_start_labels)  + BCE(backward_tail_end_logits, golden_backward_tail_end_labels)  `
- loss.backward and calculate total loss across the batches
- Run evaluation of the `evaluate dataloader`
- Save as checkpoint the epoch that scored the best evaluation score.

3- **Stage-2**:

- Prepare optimizer
- Loop through `stage2_epochs`, loop through `training dataloader`
- Build gold entity labels: `gold_fhs, gold_fhe, gold_bts, gold_bte` extracted from golden_triples.
- Build gold tail labels: `gold_fts, gold_fte, gold_bhs, gold_bhe` extracted from golden_triples. Each of these tensor is in shape (B, R, S, L). `gold_fts` is the binary labels for golden `forward tail start`, an item inside these golden tensors can have `0 or 1` value, 1 meaning that for the sentence `b`  for the relation `r` for the subject `s`, the token `l` is a `head start` token.

- 
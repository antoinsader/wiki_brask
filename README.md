
# Wiki BRASK

A PyTorch implementation of the [BRASK](https://www.sciencedirect.com/science/article/abs/pii/S0957417423004062) algorithm — **Bidirectional Relation-guided Attention network with Semantics and Knowledge** — applied to the Wikidata5m dataset for relational triple extraction.

BRASK is used to extract structured knowledge triples `(head entity, relation, tail entity)` directly from natural language descriptions. It does this by learning to locate entity spans in text guided by relation embeddings, running both a forward pass (head → relation → tail) and a backward pass (tail → relation → head) jointly.

This implementation uses PyTorch with DDP support for distributed training and follows a staged circular training approach: entity extraction, relation-guided extraction, backward extraction, and full fine-tuning.

---
## Build the model

### Create environment

1- Create your python env and activate it, download requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

You can use the script for downloading the Wikidata5m:

```bash
python download_dataset.py
```
This downloads the necessary files and extract the files into `/data/raw/` -the default raw folder-.
> The script will locate old existing raw files and asks if you want to override them.

Wikidata5m raw files:
`wikidata5m_text.txt`: containing `description` for each `entity_id`   
`wikidata5m_alias.txt`: containing list of `aliases` for each `entity_id`   
`triples.txt`: containing list of `aliases` for each `entity_id`   

---


## Pipeline

### 1. Pre-processing — `prepare.py`

The **interactive script** would walks through the data **pre-processing stages**. 

Run the script:
```bash
python prepare.py
```

> Before each stage, the terminal would ask if you want to perform the action (default is `y`). 
> All stages operate on the **minimized** dataset (created in stage 1).

**pre-processing stages:**

1. **Minimization** — (a) prompt for a decimal factor `0 < minimization-fraction < 1` of the `training triples` you want to keep. (b) parse the raw files into pickle files. (c) create minimized versions and save it.
2. **Normalization** — (1) clean `descriptions` (remove non-English characters, Unicode NFKC, collapse spaces). (2) clean aliases (same + lowercasing + deduplication). (3) overwrite the minimized files in place.
3. **Relation embeddings** — (1) computes one 768-dim BERT embedding per relation by mean-pooling across all its aliases (average of last two hidden layers, attention-mask pooled, batched with mixed precision), this represents `set R in paper`.
4. **Description embeddings** — encodes every description with BERT, producing (1) per-token description embeddings `(N, L, H)` (2) mean-pooled description embeddings `(N, H)`.

**Output:**

| File | Content |
|---|---|
| `minimized/triples_train.pkl` | Flat list of `(head_id, relation_id, tail_id)` tuples |
| `minimized/aliases.pkl` | Dictionary of `{entity_id: [alias_str, ...]}` |
| `minimized/relations.pkl` | Dictionary `{relation_id: [alias_str, ...]}` |
| `minimized/descriptions.pkl` | Dictionary `{entity_id: description_text}` |
| `minimized/relation_embeddings.npz` | Tensor `(n_relations, H)` |
| `minimized/description_embeddings_all.pt` | Tensor `(B, L, H)` — per-token |
| `minimized/description_embeddings_mean.pt` | Tensor `(B, H)` — mean-pooled |
| `minimized/description_embeddings_ids.pkl` | List of entity IDs matching the embedding rows |

---

### 2. Train TransE — `train_transe.py`

```bash
python train_transe.py
# or with DDP:
torchrun --nproc_per_node=<N_GPUS> train_transe.py
```

Trains the TransE knowledge graph embedding model (Bordes et al. 2013) on the minimized `triples`. 
For each training triple `(h, r, t)`, a `negative triple` is generated on-the-fly by corrupting either the head or tail with a random entity. 
The model minimises the margin-based loss `mean(MARGIN + ||h+r-t||₁ − ||h'+r−t'||₁)` with `Adam` + `CosineAnnealingLR`. Embedding tables are L1-normalised after every batch. 

> Support multi-GPU via PyTorch DDP (detected automatically from the `LOCAL_RANK` environment variable set by `torchrun`).

**Output:** `minimized/transe_rel_embs.npz` — relation embedding matrix of shape `(n_relations, TRANSE_EMB_DIM)`.

---

### 3. Build Gold Labels — `prepare_gold_labels.py`

```bash
python prepare_gold_labels.py
```

We loop through the training triples, and for each triple, we extract the spans (token index) of the `head aliases` and `tail aliases`

The output of golden labels would be a dictionary having `entity_id` as a key, a list of triple spans as a value.
Thus, for each  `entity_id` we will have a list of tuples, each tuple having `(head_spans, relation_id, tail_spans)`. `head_spans` and `tail_spans` are tuples `(start_index, end_index)`.

> Processing is parallelised across triples chunks via `multiprocessing.Pool`.

> The script will remove triples from `training triples` where a corresponding golden triple was not found (might be because the truncated description does not include this triple).

> The script will remove `description entity` if no golden triples for this entity was found. 

> Currently, the script will extract entities spans based on regex patterns of the aliases. so if the triple tail is `herman (given name)`, this tail will not be recorded if the description does not contain `herman (given name)`

> Current regex pattern for detecting aliases inside descriptions is: `rf"(?<!\w){  re.escape(<ALIAS NAME>).replace(r'\ ', r'\s+')  }(?!\w)"`, you can modify this inside `utils/helpers.py: create_aliases_patterns_map`

> IMPORTANT NOTICE: If the description is "Rome is the capital city of Italy, Italy is a beautiful country", and the original triples are [(Rome, capital of, Italy)], the golden triples for this description would be [(  (0,0), capital of, (6,6) ), (0,0), capital of, (7,7) ]. Which means the object `Italy` because it was found 2 times in description, it will be two times in the golden triples. Keeping this behavior currently allowing the `entity extractor` to be evaluated correctly. 


**Output:** gold label files containing per-triple head and tail span positions, mapped to token indices, saved to the minimized data directory. Also overwriting `descriptions`, `description embeddings files `, `triples` to have only those entities that have `golden triples`

> If you want to take a look at what the golden triples generated, you can run the script: 

```bash

cd helpers
python log_golden_triples.py

```

This script will create a file `helpers/golden_triples_log.txt`, containing each `description` tokens and the extracted golden triples for this `entity`. 

> `Train Triples` are tuples of `(head entity, relation, tail entity)`

The log will show details for each entity -> (1) tokens (2) train triples (showing the aliases depending on `head_id` and `tail_id`) aliases (3) golden triples  (showing the tokens depending on the extracted golden triple spans )

Example one entry from the log:

```golden_triples.log

Description id: Q5473840 
Description text:  
Fosterella spectabilis is a bromeliad species in the genus Fosterella. This species is endemic to Bolivia. 
Fosterella spectabilis is a bromeliad species in the genus Fosterella. This species is endemic to Bolivia. 
Description tokens:  
['[CLS]', 'foster', '##ella', 'spec', '##ta', '##bilis', 'is', 'a', 'bro', '##mel', '##iad', 'species', 'in', 'the', 'genus', 'foster', '##ella', '.', 'this', 'species', 'is', 'endemic', 'to', 'bolivia', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', ....] 
------------------------------- 
ORIGINAL TRIPLES (3):  
	Original triple number 1:  
		Head aliases: ['fosterella spectabilis'] 
		Relation: ['taxon rank', 'taxonomic rank', 'rank', 'type of taxon'] (P105) 
		Tail aliases: ['mayr species', 'barcode species', 'nominal species', 'morphological species', 'biological species', 'species (taxonomy)', 'ecological species', 'species (biology)', 'cohesion species', 'phylogenetic species concept', 'phylogenetic species', 'reproductive species', 'species delimitation', 'organism mnemonic', 'genetic species', 'spp.', 'biospecies', 'cladistic species', 'morphospecies', 'isolation species', 'speceis', 'undiscovered species', 'phenetic species', 'species', 'species pluralis', 'genetic similarity species', 'typological species', 'animal species', 'species and speciation', 'species (biological)', 'vavilovian species', 'evolutionary species', 'theory of species', 'recognition species'] 
	Original triple number 2:  
		Head aliases: ['fosterella spectabilis'] 
		Relation: ['parent taxon', 'taxon parent', 'higher taxon'] (P171) 
		Tail aliases: ['fosterella'] 
	Original triple number 3:  
		Head aliases: ['fosterella spectabilis'] 
		Relation: ['instance of', 'is a', 'is an', 'has class', 'has type', 'is a particular', 'is a specific', 'is an individual', 'is a unique', 'is an example of', 'member of', 'unique individual of', 'distinct member of', 'non-type member of', 'unsubclassable example of', 'uninstantiable instance of', 'unsubclassifiable member of', 'not a type but is instance of', 'unsubtypable particular', 'unitary element of class', 'distinct element of', 'distinct individual member of', 'occurrence of', 'rdf:type', 'type'] (P31) 
		Tail aliases: ['fish taxa', 'taxon', 'taxum', 'polytypic taxon', 'taxonomic group', 'supertaxon', 'subtaxon', 'taxxon', 'taxa'] 
------------------------------- 
Golden Triples (4):  
	Head  
	Triple 1:  
		Head: ['foster', '##ella', 'spec', '##ta', '##bilis']   
		Relation: ['taxon rank', 'taxonomic rank', 'rank', 'type of taxon'] (P105) 
		Tail: ['species']   
	Head  
	Triple 2:  
		Head: ['foster', '##ella', 'spec', '##ta', '##bilis']   
		Relation: ['taxon rank', 'taxonomic rank', 'rank', 'type of taxon'] (P105) 
		Tail: ['species']   
	Head  
	Triple 3:  
		Head: ['foster', '##ella', 'spec', '##ta', '##bilis']   
		Relation: ['parent taxon', 'taxon parent', 'higher taxon'] (P171) 
		Tail: ['foster', '##ella']   
	Head  
	Triple 4:  
		Head: ['foster', '##ella', 'spec', '##ta', '##bilis']   
		Relation: ['parent taxon', 'taxon parent', 'higher taxon'] (P171) 
		Tail: ['foster', '##ella']   


```

---

### 4. Train BRASK — `train.py`

> **Work in progress.** The model architecture and forward pass are implemented; the loss function and training loop are still being developed.

```bash
python train.py
```

Implements the full BRASK model as a PyTorch `nn.Module` with the following components:

- **`EntityExtractor`** — two linear heads predicting token-level start and end probabilities for entity spans, returning both probabilities and raw logits for BCE loss.
- **`RelationAttention`** — attention mechanism that produces relation-conditioned sentence representations; used twice in parallel: once with BERT semantic relation embeddings, once with TransE relation embeddings.
- **`FuseExtractor`** — fuses subject representations `s_k`, relation context `c_j`, and token embeddings `x_i` into `h_ijk` for object span prediction.
- **`BraskModel`** — full forward + backward pipeline: predict head spans → extract subject representations → compute relation attention → fuse → predict tail spans; symmetric backward path for tail-first extraction.
- **`entity_extractor_loss`** — masked BCE loss for staged training of the entity extractor alone (in `models/EntityExtractor.py`).

**Planned / not yet implemented:**
- Full `brask_loss` combining forward and backward terms with subject-slot masking.
- Span post-processing (overlapping spans, invalid spans where end < start).
- Gating fusion alternative in `FuseExtractor`.
- Relation smart pruning (top-k by cosine similarity pre-computed per description).
- Post-prediction alias filtering to improve precision.
- Circular staged training loop.

---

## Archive

The first version of the training code is preserved in [`/archive`](archive/). It was a monolithic pipeline that ran end-to-end but required large GPU RAM and did not scale well. The current codebase is a full refactor of that work with cleaner separation of concerns, DDP support, and a corrected labeling strategy. [changelog_mar2026](docs/changelog_mar2026) includes what I am refactoring now.



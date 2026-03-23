
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler


from train_transe import NUM_EPOCHS, NUM_WORKERS
from utils.files import read_cached_array, read_tensor
from utils.settings import settings
from utils.pre_processed_data import check_preprocessed_files, data_loader, check_minimized_files
from models.EntityExtractor import EntityExtractor


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




def check_training_files(use_minimized=False):
    if use_minimized and not check_minimized_files():
        return False

    if not use_minimized and not check_preprocessed_files():
        return False

    files_paths = settings.MINIMIZED_FILES if use_minimized else settings.PREPROCESSED_FILES
    important_files = [files_paths.SILVER_SPANS, files_paths.TRANSE_MODEL_RESULTS, files_paths.DESCRIPTION_EMBEDDINGS_ALL, files_paths.DESCRIPTION_EMBEDDINGS_MEAN, files_paths.DESCRIPTION_EMBEDDINGS_IDS]
    missing = [p for p in important_files if not os.path.isfile(p)]
    print(f"missing: {missing}")
    if missing:
        print("The following important files are missing. Run the appropriate scripts to generate them before training.")
        for p in missing:
            print(f"  Missing: {p}")
        return False

    return True


class BraskDataset(Dataset):
    """Dataset for training the Brask model. 
    Each item is a tuple of (description_embedding: Tensor[L,D], description_mean_embeddings: Tensor[D], description_id: str)"""


    def __init__(self, 
                 description_embeddings: torch.Tensor,
                 description_mean_embeddings: torch.Tensor,
                 description_embs_masks: torch.Tensor,
                 description_embeddings_ids: list[str],
                 golden_triples: dict,
                 ):

        self.N = description_embeddings.shape[0]
        self.description_embeddings = description_embeddings
        self.description_mean_embeddings = description_mean_embeddings
        self.description_embs_masks = description_embs_masks
        self.description_embeddings_ids = description_embeddings_ids
        self.golden_triples = golden_triples


    def __getitem__(self, index):
        description_embedding = self.description_embeddings[index]
        description_id = self.desecription_embeddings_ids[index]
        description_emb_mask = self.description_embs_masks[index]
        description_emb_mean = self.description_mean_embeddings[index]
        golden_triples = self.golden_triples[description_id]

        return description_id, description_embedding, description_emb_mask, description_emb_mean, golden_triples

    def __len__(self):
        return self.N


def collate_fn(batch):
    description_id = [ item[0] for item in batch]
    description_embeddings = torch.stack([item[1] for item in batch], dim=0)
    masks = torch.stack([item[2] for item in batch], dim=0)
    description_emb_mean = torch.stack([item[3] for item in batch], dim=0)
    golden_triples = [item[4] for item in batch]
    return description_id, description_embeddings, masks, description_emb_mean, golden_triples

    

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


def extract_sk(description_embeddings: torch.Tensor, 
               start_probs: torch.Tensor, 
               end_probs: torch.Tensor, 
               start_threshold: float, 
               end_threshold: float, 
               max_span_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract subject representations s_k and padding it.
    args:
        description_embeddings: (B, L, H)
        start_probs: (B, L) with values between 0 and 1
        end_probs: (B, L) with values between 0 and 1
        start_threshold: scalar between 0 and 1
        end_threshold: scalar between 0 and 1
        max_span_length: maximum length of subject spans
    returns:
        forward_s_k: (B, max_num_subjects, H) with padded subject representations zeroed out
        mask: (B, max_num_subjects) with 1 for valid subject representations and 0 for padded ones
    """
    

    #! Here I should do the idea of weighting the subjects using aliases dictionary
    B = description_embeddings.shape[0]
    H = description_embeddings.shape[2]
    s_k = []
    for b in range(B):
        x_emb = description_embeddings[b]
        start_idx = (start_probs[b].squeeze(-1) >= start_threshold).nonzero(as_tuple=False).squeeze(-1)
        start_idx = torch.sort(start_idx).values
        end_idx  = (end_probs[b].squeeze(-1) >= end_threshold).nonzero(as_tuple=False).squeeze(-1)
        consumed_ends = set()
        spans = []
        for s in start_idx:
            end_mask = ( (end_idx >= s) & (end_idx < s+max_span_length))
            valid_ends = end_idx[end_mask]
            valid_ends = [e.item() for e in valid_ends if e.item() not in consumed_ends]
            if len(valid_ends) == 0:
                continue
            e = min(valid_ends)
            spans.append((s.item(), e))
            consumed_ends.add(e)
        s_k_list = []
        for (s, e)  in spans:
            span_emb = (x_emb[s] + x_emb[e]) / 2
            s_k_list.append(span_emb)
        if s_k_list:
            s_k_list = torch.stack(s_k_list, dim=0)
        if len(s_k_list) == 0:
            s_k_list = torch.zeros(1, H, device=x_emb.device)
        s_k.append(s_k_list)
    max_num_subjects = max([s.shape[0] for s in s_k])
    padded_sk = []
    mask = []
    for s in s_k:
        K = s.shape[0]
        if K < max_num_subjects:
            pad = torch.zeros(max_num_subjects - K, s.shape[1], device=s.device)
            s_padded  = torch.cat([s, pad], dim=0)
            m = torch.cat([torch.ones(K, device=s.device), torch.zeros(max_num_subjects - K, device=s.device)], dim=0)
        else:
            s_padded = s
            m=torch.ones(K, device=s.device)

        padded_sk.append(s_padded)
        mask.append(m)
    padded_sk = torch.stack(padded_sk, dim=0) #(B, max_num_subjects, H)
    mask = torch.stack(mask, dim=0) #(B, max_num_subjects)
    return padded_sk, mask


class BraskModel(torch.nn.Module):
    def __init__(self, hidden_dim, trane_rel_dim):
        super(BraskModel, self).__init__()


        # ! I should be careful about the span reconstructions because (the model predicts start and end independently):
        # ! Multiple possible spans
        # ! Overlapping spans
        # ! Invalid spans (end < start)

        # ! for loss I can do with 3 parameters instead of (Entity loss, forward loss, backward loss) do alpha with starting as 1 and tune after.

        # ! in the algorithm , I have to do fusion (token emb, relation emb, entity emb (subject, object)):
        # ! thie way the paper do the confusion is attention fusion (score =f(x_i, r, global_context)), we might want in the future do other fusion like gating which would be g = sigmoid(W[x_i, r]); h_i = g * x_i + (1-g) * r


        # ! I can do relations smart pruning which is doing pre-processed cosine similarity between description and the relations, maybe not bert but something else (I am thinking sentemce transformers but not neccessarily, )
        # ! After I can use it to decide which relations (top-k) to consider .
        # ! +  positive relations from ground truth triples !!!.



        # After prediction:
        #     Generate candidate spans
        #     Keep only spans that:
        #     match aliases (exact or fuzzy)
        #     This improves precision a lot
        # DO NOT: force model to only predict aliases

        # Circular traning:
        #    1- train entity extractor first
        #    2- frozen encoder - train relation extraction and forward extractor (goal : learn relation conditioning without noise )
        # 3-  add backward extractor
        # 4- final fine tuning 


        

        # for predicted head and predicted tail:
        # entity representation = avg(X[start] + X[end]) / 2

        self.forward_head_predict = EntityExtractor(hidden_dim)
        self.backward_tail_predict = EntityExtractor(hidden_dim)
        self.semantic_relation_attention = RelationAttention(hidden_dim, rel_dim=hidden_dim)
        self.trane_relation_attention = RelationAttention(hidden_dim, rel_dim=trane_rel_dim)

        self.fuse_extractor_forward = FuseExtractor(hidden_dim)
        self.fuse_extractor_backward = FuseExtractor(hidden_dim)

        self.forward_tail_predict = EntityExtractor(hidden_dim)
        self.backward_head_predict = EntityExtractor(hidden_dim)


        self.threshold_head_start = 0.5
        self.threshold_head_end = 0.5
        self.threshold_tail_start = 0.5
        self.threshold_tail_end = 0.5
        self.max_span_len = 10

    def forward(self, batch, semantic_relation_embeddings, transe_relation_embeddings):


        # B: batch size, L: sequence length, H: hidden dimension
        description_embeddings = batch[0] # shape (B, L, H)
        description_mean_embeddings = batch[1] # shape (B, H)
        description_ids = batch[2] # shape (B,)

        B, L, H = description_embeddings.shape



        # ! I should be careful to not allow logits for [PAD]
        forward_head_start_probs, forward_head_end_probs, f_head_start_logits, f_head_end_logits = self.forward_head_predict(description_embeddings)
        backward_tail_start_probs, backward_tail_end_probs, b_tail_start_logits, b_tail_end_logits = self.backward_tail_predict(description_embeddings)

        forward_c, forward_a = self.semantic_relation_attention(description_embeddings, semantic_relation_embeddings, description_mean_embeddings)
        backward_c, backward_a = self.trane_relation_attention(description_embeddings, transe_relation_embeddings, description_mean_embeddings)

        # ?! During training: You use gold subject spans to build sk directly — no thresholding, no extract_sk. As discussed earlier this is teacher forcing.
        # Extract sk
        # ?! During training, should I train with my silver spans to extract_sk ? the gradients won't flow back through forward_head_start/end to the encoder. The paper trains subject extraction and object extraction jointly with a shared loss (Eq. 19).
        # forward_sk, forward_sk_mask = extract_sk(
        #     description_embeddings=description_embeddings,
        #     start_probs=forward_head_start_probs,
        #     end_probs=forward_head_end_probs,
        #     start_threshold=self.threshold_head_start,
        #     end_threshold=self.threshold_head_end,
        #     max_span_length=self.max_span_len
        #     )
        # backward_sk, backward_sk_mask = extract_sk(
        #     description_embeddings=description_embeddings,
        #     start_probs=backward_tail_start_probs,
        #     end_probs=backward_tail_end_probs,
        #     start_threshold=self.threshold_tail_start,
        #     end_threshold=self.threshold_tail_end,
        #     max_span_length=self.max_span_len
        # )



        forward_hijk = self.fuse_extractor_forward(description_embeddings, forward_c, forward_sk, forward_sk_mask) # (B, R, max_num_subjects, L, H)
        backward_hijk = self.fuse_extractor_backward(description_embeddings, backward_c, backward_sk, backward_sk_mask) # (B, R, max_num_subjects, L, H)

        B, R, S, L, H = forward_hijk.shape
        _, _, forward_tails_start_logits, forward_tail_end_logits = self.forward_tail_predict(forward_hijk) # (B, R, S, L, 1)


        _,_, backward_head_start_logits, backward_head_end_logits = self.backward_head_predict(backward_hijk) # (B, R, S, L, 1)



        return {
            "description_ids": description_ids,
            "froward": {
                "head_start_logits": f_head_start_logits,
                "head_end_logits": f_head_end_logits,
                "tail_start_logits": forward_tails_start_logits, # # (B, R, S, L, 1)
                "tail_end_logits": forward_tail_end_logits, # # (B, R, S, L, 1)
                "forward_s_k_mask": forward_sk_mask
            },
            "backward": {
                "tail_start_logits": b_tail_start_logits, # # (B, R, S, L, 1)
                "tail_end_logits": b_tail_end_logits, # # (B, R, S, L, 1)
                "head_start_logits": backward_head_start_logits,
                "head_end_logits": backward_head_end_logits,
                "backward_s_k_mask": backward_sk_mask 
            }
        }

def entity_extractor_loss(pred_logits, gold, token_mask, pos_weight=10.0):
    """Using BCE loss
    BCE explanation:
    -----------
    BCE answers the question, how wrong your probability prediction for a binary outcome?
        - given p the predicted and y the true label
        - when y =0 (answer should be 0) -> loss would be -log(1-p). then if p is low => loss would be  tiny saying that the answer is correct. if p is high => loss would be huge saying that we are wrong
        - when y=1 (answer should be 1) -> loss would be -log(p). Then if p is low => loss would be high saying that we are wrong. if p is high => loss would tiny saying we are correct.
    What is pos_weight:
    -------------
    
    """
    

def loss_compute():
    # When you compute object predictions over padded subject slots, those slots should be masked out in the loss. Keep track of forward_sk_mask and backward_sk_mask and pass them to the loss function.
    
    pass

def main(use_minimized: bool):
    
    if not check_training_files(use_minimized):
        return

    NUM_EPOCHS = 10
    BATCH_SIZE = 64 if use_cuda else 16
    NUM_WORKERS = 4 if use_cuda else 0
    LEARNING_RATE =   1e-4
    val_split = 0.1


    print("Loading data")
    golden_triples = data_loader.get_golden_triples()
    description_embs_all, description_embs_ids, description_embs_masks = data_loader.get_description_embeddings_all()
    description_embs_mean = data_loader.get_description_embeddings_mean()

    full_dataset = BraskDataset(
        description_embeddings=description_embs_all,
        description_mean_embeddings=        description_embs_mean,
        description_embs_masks=description_embs_masks,
        description_embeddings_ids=description_embs_ids,
        golden_triples=golden_triples
    )
    N = len(description_embs_all)
    L = description_embs_all.shape[1]
    val_size = int(N * val_split)
    train_size = N - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)



    train_loader = DataLoader(train_dataset, 


    sampler = DistributedSampler(train_dataset)
    print("creating data loader")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler = sampler,
        num_workers=NUM_WORKERS,
        pin_memory=use_cuda,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=use_cuda
    )
    


    for ds_batch in dataloader:
        print(ds_batch)
        break


    pass


if __name__ == "__main__":
    answer = input("Train on minimized dataset? [Y/n]: ").strip().lower()
    use_minimized = answer != 'n'
    main(use_minimized)



#DRAFT LOSS:

# triples_per_sentence[b] = [
#     ((h_tok_start, h_tok_end), r_idx, (t_tok_start, t_tok_end)),
#     ...
# ]

# def brask_loss_per_triple(
#     fwd_head_start_logits: torch.Tensor,  # (B, L)
#     fwd_head_end_logits:   torch.Tensor,  # (B, L)
#     fwd_tail_start_logits: torch.Tensor,  # (B, R, S, L)
#     fwd_tail_end_logits:   torch.Tensor,  # (B, R, S, L)
#     triples_per_sentence:  list,          # [b] -> [((hs,he), r, (ts,te)), ...]
#     gold_subject_slots:    list,          # [b] -> [(hs, he), ...] ordered list of subject slots
#     token_mask: torch.Tensor,             # (B, L)
#     L: int,
# ) -> torch.Tensor:

#     L_sub = torch.tensor(0.0, device=fwd_head_start_logits.device)
#     L_obj = torch.tensor(0.0, device=fwd_head_start_logits.device)
#     n_triples = 0

#     for b, triples in enumerate(triples_per_sentence):
#         mask = token_mask[b]  # (L,)

#         def masked_bce(logits, gold):
#             loss = F.binary_cross_entropy_with_logits(
#                 logits, gold, reduction='none'
#             )
#             return (loss * mask).sum() / (mask.sum() + 1e-8)

#         for (h_start, h_end), r_idx, (t_start, t_end) in triples:

#             # Build gold head vector for this specific triple
#             gold_h_start = torch.zeros(L, device=fwd_head_start_logits.device)
#             gold_h_end   = torch.zeros(L, device=fwd_head_start_logits.device)
#             gold_h_start[h_start] = 1.0
#             gold_h_end[h_end]     = 1.0

#             # Build gold tail vector for this specific triple
#             gold_t_start = torch.zeros(L, device=fwd_tail_start_logits.device)
#             gold_t_end   = torch.zeros(L, device=fwd_tail_start_logits.device)
#             gold_t_start[t_start] = 1.0
#             gold_t_end[t_end]     = 1.0

#             # Subject loss — same formula regardless of relation
#             L_sub = L_sub \
#                   + masked_bce(fwd_head_start_logits[b], gold_h_start) \
#                   + masked_bce(fwd_head_end_logits[b],   gold_h_end)

#             # Object loss — need s_idx to index into (B, R, S, L)
#             # Find which slot this head span occupies in the padded sk tensor
#             h_span = (h_start, h_end)
#             if h_span in gold_subject_slots[b]:
#                 s_idx = gold_subject_slots[b].index(h_span)
#                 L_obj = L_obj \
#                       + masked_bce(fwd_tail_start_logits[b, r_idx, s_idx], gold_t_start) \
#                       + masked_bce(fwd_tail_end_logits[b,   r_idx, s_idx], gold_t_end)

#             n_triples += 1

#     return (L_sub + L_obj) / max(n_triples, 1)


# def entity_loss(pred_start: torch.Tensor,
#                 pred_end: torch.Tensor,
#                 gold_start: torch.Tensor,
#                 gold_end: torch.Tensor,
#                 mask: torch.Tensor = None) -> torch.Tensor:
#     """
#     Binary cross-entropy loss for one entity extractor (Eqs. 20-21).

#     Args:
#         pred_start: (B, L) predicted start probabilities (after sigmoid)
#         pred_end:   (B, L) predicted end probabilities (after sigmoid)
#         gold_start: (B, L) binary ground-truth start labels {0, 1}
#         gold_end:   (B, L) binary ground-truth end labels {0, 1}
#         mask:       (B, L) float mask — 1 for real tokens, 0 for padding
#     Returns:
#         scalar loss
#     """
#     eps = 1e-8
#     # BCE manually so we can apply mask
#     def bce(pred, gold):
#         loss = -(gold * torch.log(pred + eps) + (1 - gold) * torch.log(1 - pred + eps))
#         if mask is not None:
#             loss = loss * mask
#             return loss.sum() / (mask.sum() + eps)
#         return loss.mean()

#     return bce(pred_start, gold_start) + bce(pred_end, gold_end)


# def brask_loss(outputs: dict,
#                gold_fwd_head_start: torch.Tensor,  # (B, L)
#                gold_fwd_head_end: torch.Tensor,    # (B, L)
#                gold_fwd_tail_start: torch.Tensor,  # (B, R, S, L)
#                gold_fwd_tail_end: torch.Tensor,    # (B, R, S, L)
#                gold_bwd_tail_start: torch.Tensor,  # (B, L)
#                gold_bwd_tail_end: torch.Tensor,    # (B, L)
#                gold_bwd_head_start: torch.Tensor,  # (B, R, S, L)
#                gold_bwd_head_end: torch.Tensor,    # (B, R, S, L)
#                fwd_sk_mask: torch.Tensor,          # (B, S) — from extract_sk
#                bwd_sk_mask: torch.Tensor,          # (B, S)
#                token_mask: torch.Tensor = None,    # (B, L) padding mask
#                alpha: float = 1.0) -> torch.Tensor:
#     """
#     Total BRASK loss: L_total = sum(L_fwd + L_bwd)  [Eq. 18]

#     L_fwd = L_sub + L_obj                           [Eq. 19]
#     L_bwd = L'_obj + L'_sub                         (symmetric)

#     Args:
#         outputs:  dict returned by BraskModel.forward()
#         alpha:    optional weighting between forward and backward losses
#     """

#     fwd = outputs["froward"]   # note: typo in your code "froward"
#     bwd = outputs["backward"]

#     # --- Forward: subject loss (Eq. 20-21) ---
#     L_sub = entity_loss(
#         fwd["head_start"].squeeze(-1),  # (B, L)
#         fwd["head_end"].squeeze(-1),
#         gold_fwd_head_start,
#         gold_fwd_head_end,
#         mask=token_mask
#     )

#     # --- Forward: object loss ---
#     # fwd tail preds are (B, R, S, L) — mask out padded subject slots
#     # expand sk_mask: (B, 1, S, 1) to broadcast over (B, R, S, L)
#     B, R, S, L = fwd["tail_start"].shape
#     obj_mask = fwd_sk_mask.unsqueeze(1).unsqueeze(-1).expand(B, R, S, L)  # (B, R, S, L)
#     if token_mask is not None:
#         obj_mask = obj_mask * token_mask.unsqueeze(1).unsqueeze(2)  # also mask padding

#     L_obj = entity_loss(
#         fwd["tail_start"],   # (B, R, S, L) — after reshape fix above
#         fwd["tail_end"],
#         gold_fwd_tail_start,
#         gold_fwd_tail_end,
#         mask=obj_mask
#     )

#     L_forward = L_sub + L_obj  # Eq. 19

#     # --- Backward: object loss ---
#     L_prime_obj = entity_loss(
#         bwd["tail_start"].squeeze(-1),
#         bwd["tail_end"].squeeze(-1),
#         gold_bwd_tail_start,
#         gold_bwd_tail_end,
#         mask=token_mask
#     )

#     # --- Backward: subject loss ---
#     obj_mask_bwd = bwd_sk_mask.unsqueeze(1).unsqueeze(-1).expand(B, R, S, L)
#     if token_mask is not None:
#         obj_mask_bwd = obj_mask_bwd * token_mask.unsqueeze(1).unsqueeze(2)

#     L_prime_sub = entity_loss(
#         bwd["head_start"],
#         bwd["head_end"],
#         gold_bwd_head_start,
#         gold_bwd_head_end,
#         mask=obj_mask_bwd
#     )

#     L_backward = L_prime_obj + L_prime_sub  # symmetric to Eq. 19

#     L_total = L_forward + alpha * L_backward  # Eq. 18
#     return L_total, {"L_sub": L_sub, "L_obj": L_obj,
#                      "L_prime_obj": L_prime_obj, "L_prime_sub": L_prime_sub}
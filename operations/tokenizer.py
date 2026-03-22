
from transformers import BertTokenizerFast




def tokenize(texts, max_length):
    tokenizer= BertTokenizerFast.from_pretrained('bert-base-uncased')

    enc = tokenizer(
        texts,
        return_offsets_mapping=False,
        return_attention_mask=False,
        return_token_type_ids = False,
        padding="max_length",
        truncation=True,
        max_length = max_length
    )
    return enc
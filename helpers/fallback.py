import sys
from typing import final
sys.path.insert(0, '..')
import torch
from transformers import BertModel, BertTokenizerFast, AutoModel
from tqdm import tqdm


from utils.files import read_cached_array, read_tensor, init_mmap, cache_array
from utils.settings import settings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main():
    descriptions = read_cached_array(settings.MINIMIZED_FILES.DESCRIPTIONS)
    sentences = list(descriptions.values())
    N = len(descriptions)
    del descriptions
    batch_size= 128
    out_all_embs = settings.MINIMIZED_FILES.DESCRIPTION_EMBEDDINGS_ALL
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = AutoModel.from_pretrained('bert-base-cased')
    max_length = 128
    H = model.config.hidden_size

    final_all_embs = init_mmap(out_all_embs, (N, max_length, H), "float32")
    
    for start in tqdm(range(0, len(sentences), batch_size) , desc="Embedding senteneces"):
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

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            embs = out.last_hidden_state #(B, L, H)

        final_all_embs[start:end] = embs.cpu().numpy()
    final_all_embs.flush() # ensure all data is written to disk
    del final_all_embs # close memmap
    print("finished")
if __name__ == "__main__":
    main()
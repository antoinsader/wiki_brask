## Tasks to do in future:

[] Add semantic comparison inside the golden_triples discovery.
[] Smart truncation while tokenizing to find triples and not just normal 256 truncation.
 
[] in the algorithm , I have to do fusion (token emb, relation emb, entity emb (subject, object)):
 - the way that the paper do is fusion is attention fusion (score =f(x_i, r, global_context)), we might want in the future do other fusion like gating which would be g = sigmoid(W[x_i, r]); h_i = g * x_i + (1-g) * r


[] I have to do relation smart pruning:
    1- pre computed consine similarity between entity embedding and relation embedding
    2- Consider for each description top-k relations + positive relation from ground truth


[] After prediction, I might do filtering of the result spans and keep only spans that matches an alias, I should not do this during training, my objective is to improve precision.


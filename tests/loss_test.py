
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


B, L, H = 3, 5, 2
S, R = 4, 3


X =  torch.tensor([[[-1.1068, -0.1427],
         [ 0.0251,  1.0520],
         [ 0.6303, -0.5684],
         [-0.3796, -0.0531],
         [-0.6234, -0.1572]],

        [[-0.2547,  1.9687],
         [-0.7200, -1.0915],
         [ 2.1149, -1.2067],
         [ 0.7872,  0.2806],
         [ 0.7659,  0.2614]],

        [[ 1.2207, -0.8111],
         [-0.0930, -0.6983],
         [ 0.1360, -0.2004],
         [ 1.0487,  0.4520],
         [-1.0696, -1.7610]]]) # shape: (B, L, H)

gold = torch.tensor([[0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1., 1., 0., 0.]])

class TokenClassifier(nn.Module):
    def __init__(self, input_dim):
        super(TokenClassifier, self).__init__()
        # Projects H -> 1 (the logit for being class '1')
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (B, L, H)
        logits = self.projection(x)  # Result: (B, L, 1)
        return logits.squeeze(-1)
model = TokenClassifier(H)


def bce(pred_logits, gold, pos_weight=2.5):
    pw = torch.tensor([pos_weight], device=pred_logits.device)
    loss = F.binary_cross_entropy_with_logits(pred_logits, gold, pos_weight=pw, reduction="none")
    return loss.sum() / (pred_logits.shape[0] + 1e-8)

criterion = bce


optimizer = optim.Adam(model.parameters(), lr=0.1)

print("start training..")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, gold)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

with torch.no_grad():
    predictions = torch.sigmoid(model(X)) > 0.5
    accuracy = (predictions == gold).float().mean()
    print(f"\nFinal Accuracy: {accuracy.item() * 100:.2f}%")

# forward_tail_start_logits = torch.randn(B, R, S, L)
# tail_start_probs = torch.sigmoid(forward_tail_start_logits.squeeze(-1))

# for b in range(B):
#     predicted_starts = (tail_start_probs[0] >= 0.5).nonzero(as_tuple=False)  # (B, R, S, L) boolean
    
#     print(predicted_starts)
#     break
    



# gold_fwd_tail_start = torch.zeros(B, R, S, L, dtype=torch.float)


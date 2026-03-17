import torch
import torch.nn.functional as F

# Vocabulary mapping: a=0, b=1, c=2, d=3, e=4
V, L = 5, 4
d_model = V + L # 9

# Example sequence: [b, a, d, e] 
# Position 1 is 'a'. (Using 0-indexing, "Position 2" in your rule is index 1)
tokens = torch.tensor([1, 0, 3, 4]) 

def get_input(token_indices):
    content = F.one_hot(token_indices, num_classes=V).float()
    pos = F.one_hot(torch.arange(L), num_classes=L).float()
    return torch.cat([content, pos], dim=-1)

X = get_input(tokens)

# Initialize weights with zeros
W_Q_mat = torch.zeros(d_model, d_model)
W_K_mat = torch.zeros(d_model, d_model)
W_V_mat = torch.zeros(d_model, V)

# --- ATTENTION MECHANISM ---
# Query looks for position 2 (index V+1)
# Key identifies itself via position 2 (index V+1)
W_Q_mat[V+1, V+1] = 10.0 # High weight for strong attention
W_K_mat[V+1, V+1] = 10.0

# --- VALUE MAPPING ---
# If the attended token is 'a' (index 0), output 'c' (index 2)
W_V_mat[0, 2] = 1.0 

# Load into layers
with torch.no_grad():
    q_layer = torch.nn.Linear(d_model, d_model, bias=False)
    k_layer = torch.nn.Linear(d_model, d_model, bias=False)
    v_layer = torch.nn.Linear(d_model, V, bias=False)
    
    q_layer.weight.copy_(W_Q_mat.T)
    k_layer.weight.copy_(W_K_mat.T)
    v_layer.weight.copy_(W_V_mat.T)


Q = q_layer(X)
K = k_layer(X)
V_vecs = v_layer(X)

# S = QK^T, apply causal mask
scores = torch.matmul(Q, K.transpose(-2, -1))
mask = torch.tril(torch.ones(L, L))
scores = scores.masked_fill(mask == 0, float('-inf'))

# P = softmax(S)
P = F.softmax(scores, dim=-1)

# H = PV
H = torch.matmul(P, V_vecs)

print("Attention Matrix (P):")
print(P)
print("\nOutput for last token (should be high on index 2 for 'c'):")
print(torch.round(H[-1], decimals=3))
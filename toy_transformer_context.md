# Continuation Context: Small-Scale Transformer Analysis

## Goal
- Examine attention in highly simplified, small-scale situations (tiny vocabulary and short context) to extract structural properties.
- Explore the probabilistic finite-state machine interpretation of attention.
- Record a consistent notation aligned with standard transformer conventions.

## Key Definitions / Terminology
- **Token embedding**: vector representing a token; combined with positional encoding.
- **Positional encoding**: vector added to token embedding; same dimension as embedding.
- **Query (q), Key (k), Value (v)**: projections of token representations for attention.
- **Attention probability matrix (P)**: row-wise softmax of score matrix `S = QK^T`.
- **Attention output (H)**: weighted sum of value vectors: `H = P V`.
- **Value dimension (d_v)**: dimension of each value vector; in toy model can be set equal to vocabulary size V.
- **Vocabulary size (V)**: number of tokens; also used in toy model to match value dimension.
- **Context length (L)**: number of tokens in input window.
- **Finite-state machine interpretation**: attention as a probabilistic pointer to previous tokens, producing a stochastic process over outputs.

## Notation and Variables
- `V = 5`, `L = 4`, `d = d_model = V + L = 9`
- Standard basis vectors: `e_i in R^{1 x d}` (row vectors)
- Token representation: `x(t,p) = e_t + e_{V+p} in R^{1 x d}`
- Input matrix: `X in R^{L x d}`
- Linear projections: `Q = X W_Q, K = X W_K, V = X W_V`, with `W_Q, W_K, W_V in R^{d x d}`
- Score matrix: `S = Q K^T in R^{L x L}`
- Attention probabilities: `P = softmax(S)`
- Attention output: `H = P V in R^{L x d_v}`
- Final output (optional): `z = h_L W_O, p = softmax(z), W_O in R^{d_v x V}`

## Important Results / Conclusions
- Only the last row of `P` matters when predicting a single output token; earlier rows can be ignored.
- Row-wise softmax symmetry: adding a constant to a row of the score matrix does not change attention probabilities.
- Basis-vector trick: identity embeddings with row-vector convention allow decomposition of key/value matrices into token and position lookup tables.
- With `d_v = V` and `v_j = e_{t_j}`, the attention output `h_L = sum_j P_{Lj} v_j` directly represents a probability-like vector over tokens.
- The toy attention head behaves like a **probabilistic pointer** to previous tokens, yielding a finite-order Markov process for sequence generation.
- Attention in this setup is analogous to a **latent-variable mixture model** but not a strict HMM because the latent variable depends on the full context.

## Open Questions / Next Steps
- Formalize the finite-state-machine interpretation for the `V=5, L=4` toy model.
- Explore minimal parameter settings and symmetries for analytic solutions.
- Investigate how learned k
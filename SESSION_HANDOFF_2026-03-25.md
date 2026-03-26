# Session Handoff (2026-03-25)

## Scope
This session extended the toy attention notebook with position-sensitive and first-token-gated logic experiments.

Primary working notebook:
- rank1_qk_experiment.ipynb

## Core Conventions (kept consistent)
- Vocabulary size: V = 5
- Context length: L = 4
- Embedding dimension: d = V + L = 9
- Token vectors are row vectors.
- Projection convention: Q = X W_Q, K = X W_K, values via V_mat = X W_V.
- Relationship matrix view: R = W_Q W_K^T and scores S = X R X^T.
- Attention scaling uses explicit alpha multipliers.

## What Was Added
### Example 4
Rule framing:
- In fixed-final-token setup, test whether position 2 contains t1.
- If yes -> predict t2, else -> predict t4.

Mechanism:
- Position-specific key at pos2.
- Two value channels: token detector + constant channel.
- Decode from final row.

### Example 5
Requested rule:
- If first token is t3 and pos2 is t1 -> t2, else t4.

Implemented by:
- Gating via first-row query behavior.
- Decode from first row (not final row).

Note:
- This worked, but did not satisfy strict final-row decoding requirement.

### Example 6
Same logical rule as Example 5, but **strict final-row decoding**.

Rule:
- If first token is t3 and pos2 is t1 -> t2
- Otherwise -> t4

Construction summary:
- Single head, single layer.
- Q selects final row (query active at final position only).
- K includes competition between pos0 and pos2 so final-row softmax routing depends on first token and pos2 content.
- V channel 0 is position-aware (token t1 detector with pos0 suppression term), channel 1 is constant.
- Output matrix routes strong channel-0 signal to t2; default bias-like path to t4.

## Example 6 Validation Cases
Observed behavior in the notebook:
1) [3,0,1,0] -> t2 (true branch)
2) [0,0,1,3] -> t4 (first token condition false)
3) [3,0,0,1] -> t4 (pos2 condition false)
4) [1,2,0,4] -> t4 (both false)

Conclusion:
- In this toy setup, strict final-row decoding is achievable with one head and one layer.

## Comparison Note Added
A compact comparison table was appended contrasting Examples 4/5/6 by:
- decode row
- where the gating condition is encoded
- value channel design
- practical takeaway

## Current Notebook State
- Notebook now includes 39 cells total.
- New sections added near the end:
  - Example 6 markdown + code + visualization + interpretation
  - comparison table markdown

Important runtime note:
- Current notebook metadata indicates cells are not executed in the fresh summary snapshot.
- Kernel variable listing still shows prior variables from a past run.
- Next session should rerun relevant cells before relying on outputs.

## Fast Restart Plan For Next Session
1) Open rank1_qk_experiment.ipynb.
2) Run all setup/base cells first (imports, dimensions, helper functions).
3) Run Example 6 cells and verify outputs/plots.
4) Start new experiments from Example 6 as baseline.

## Suggested Next Experiments
- Sensitivity analysis of Example 6 parameters (alpha6, gamma6, key/value coefficients) to map robustness margins.
- Add exhaustive or randomized sequence sweeps to estimate rule accuracy over all length-4 token sequences.
- Try a cleaner multi-head single-layer version that factorizes:
  - head A: first-token gate
  - head B: position-2 content check
  - head C: default/bias channel
- Add a second-layer variant and compare complexity vs reliability against single-layer single-head.

## Files Modified This Session
- rank1_qk_experiment.ipynb
- SESSION_HANDOFF_2026-03-25.md (this file)

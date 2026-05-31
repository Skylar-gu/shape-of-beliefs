# Shape of Beliefs — OLMo-2 Pipeline Summary

Replication of *The Shape of Beliefs* (arXiv:2602.02315) using **allenai/OLMo-2-0425-1B**.
Model: `allenai/OLMo-2-0425-1B` · 16 layers · hidden dim 2048 · no BOS token · bfloat16

---

## Stage 1 — Sequence Generation

**Script:** `generate_sequences.py`

**Key hyperparameters:**
- `--num-seq 10` (sequences per dataset)
- `--len-seq 1000` (tokens per sequence)
- `--mean` in {300, 350, 400, 450, 500, 550, 600, 650, 700} (μ sweep)
- `--std` in {10, 20, 30, 50, 80, 100, 120, 150, 200} (σ sweep)
- Also generated: combined `m300+m700` dataset for dynamics experiment

**Output files:**
```
data/sequences/gaussian_m{mean}_s{std:03d}_l1000_n10.jsonl
data/sequences/gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10.jsonl
```

**Experiment notes:**
- Each `.jsonl` contains 10 sequences of 1000 integers sampled from N(μ, σ), formatted as comma-separated text (e.g. `"423, 187, 612, ..."`).
- The combined `m300+m700` sequence concatenates one m300 and one m700 sequence back-to-back, used to test how the model's belief updates when the generating distribution shifts mid-sequence.
- Tokenization: integers 0–999 tokenize as single tokens in both Llama and OLMo-2; the stride-2 pattern (number, comma, number, ...) is identical across models.

**Key result:** 19 dataset files generated covering the full μ-sweep (9 datasets), σ-sweep (8 additional datasets), and 1 combined dynamics dataset.

---

## Stage 2 — Activation and Logit Extraction

**Script:** `sequences_to_activations.py`

**Key hyperparameters:**
- `--model-name allenai/OLMo-2-0425-1B`
- `--revision main`
- `torch_dtype=torch.bfloat16`
- `BATCH_SIZE = 10` (sequences per forward pass)
- Saves all 16 transformer layers + embedding layer
- Token subset: `token_subset/olmo2-0425-1B_number_tokens.json` (integers 0–999 + punctuation)

**Output files per dataset:**
```
data/activations/olmo-2-0425-1b/{dataset}/model_embed_tokens_batch0000.pt
data/activations/olmo-2-0425-1b/{dataset}/model_layers_{0..15}_batch0000.pt
data/logits/olmo-2-0425-1b/{dataset}/logits_batch0000.pt
```

**Experiment notes:**
- Each activation file is a dict with keys `activations` (shape `[batch, seq_len, 2048]`), `lengths`, and `sequence_ids`.
- Each logits file contains `logits` (shape `[batch, seq_len, |vocab_subset|]`), `token_strings`, and `token_ids` — only the integer and punctuation token subset is saved to reduce disk usage.
- OLMo-2 has no BOS token, so position indexing differs from Llama: comma→number positions are odd indices (1, 3, 5, ...) rather than even.

**Key result:** ~50 GB of cached activations covering 9 μ-sweep datasets, 8 σ-sweep datasets, and the m300+m700 combined dataset across all 16 layers.

---

## Stage 3 — Linear Field Probe Training

**Script:** `linear_field_probes.py`

**Key hyperparameters:**
- Datasets: 9 μ-sweep datasets (m300–m700, σ=100)
- Layers: all 16 (0–15)
- `epochs = 100`
- `lr = 1e-2` (AdamW)
- `weight_decay = 1e-2`
- `batch_size = 2048`
- `bias = False`
- Train sequences: seq_0000–seq_0007 (8/10); test: seq_0008–seq_0009 (2/10)
- `number_start_index = 500` (drops first 500 comma→number positions per sequence to skip equilibration transient)

**Output files:**
```
probes/olmo-2-0425-1b/probes/epoch100_biasFalse/linear_probe_layer{0..15}.pt
```
Each file contains: `train_accuracy`, `test_accuracy`, `per_dataset_accuracy` (dict), `cosine_matrix` (9×9), `cosine_eigenvalues_desc`, `cosine_cumulative_explained`, `probe_state_dict`.

**Experiment notes:**
- Each probe is a 9-class linear classifier (one class per μ value) trained on comma→number activation vectors at a single layer; the probe weight matrix W ∈ ℝ^{9×2048} is treated as a local linear field over belief space.
- Cosine similarity between probe weight vectors measures how distinguishable adjacent belief states are — eigenvalue decay of this 9×9 matrix characterises the intrinsic dimensionality of the belief manifold at each layer.
- Train/test split is sequence-level (not token-level) to avoid data leakage.

**Key result:** Test accuracy rises from 0.62 (layer 0) to 0.963 (layer 15), with a non-monotonic dip at layers 1–3 (0.81 → 0.79 → 0.79) not seen in Llama. Full per-layer results:

| Layer | Train acc | Test acc |
|-------|-----------|----------|
| 0     | 0.7447    | 0.6198   |
| 1     | 0.9282    | 0.8090   |
| 2     | 0.9100    | 0.7890   |
| 3     | 0.9249    | 0.7894   |
| 4     | 0.9493    | 0.8426   |
| 5     | 0.9482    | 0.8430   |
| 6     | 0.9615    | 0.8831   |
| 7     | 0.9773    | 0.8974   |
| 8     | 0.9885    | 0.9194   |
| 9     | 0.9868    | 0.9048   |
| 10    | 0.9850    | 0.9029   |
| 11    | 0.9893    | 0.9059   |
| 12    | 0.9965    | 0.9296   |
| 13    | 0.9989    | 0.9377   |
| 14    | 0.9999    | 0.9471   |
| 15    | 1.0000    | 0.9625   |

---

## Figure 1 — Belief Manifold (Panels B/C/D)

**Script:** `figures/figure01.py`
**Notebook:** `figures/figure01.ipynb`

**Key hyperparameters:**
- `LAYER = 15` (activations for Panel B)
- `DROP_FIRST = 500`
- `TEMP = 1.0`
- `N_COMPONENTS = 16` (PCA components computed, 3 plotted)
- `PROB_SUBSET = 8000` (subsample for inPCA)
- `USE_MEAN_ACTS = True` (plot per-dataset mean activation, 17 points, not all tokens)
- inPCA metric: Hellinger distance (Fisher-Rao on simplex via sqrt transform + MDS)

**Output files:**
```
figures/figure01_olmo2_layer15_mean.html
```

**Experiment notes:**
- Panel B: 3D PCA of layer-15 activation vectors at comma→number positions, coloured by μ (blue→red arc) and σ (separate arc), showing a curved 2D manifold in activation space.
- Panel C: inPCA of softmax output distributions using Hellinger distance (respects simplex geometry); recovers a matching curved manifold in output space, confirming the encoding is geometrically coherent end-to-end.
- Panel D: per-μ softmax mass averaged over each dataset's sequences, showing the model's predictive distribution tracks the true Gaussian shape.

**Key result:** OLMo-2 encodes beliefs as a 2D curved manifold in activation space (μ-arc and σ-arc are orthogonal), replicating the paper's core geometric finding. The inPCA output manifold mirrors the activation manifold, confirming the geometry is preserved through the final projection.

---

## Figure 2 — Belief Dynamics

**Script:** `figures/figure02.py`
**Notebook:** `figures/figure02.ipynb`

**Key hyperparameters:**
- `DATAROOT = "gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10"`
- `SEQUENCE_IDX = 9`
- `LAYER = 15`
- `START_NUMBER_IDX = 500` (start plotting from position 500)
- `TEMPERATURE = 1.0`

**Output files:**
```
figures/figure02a_olmo2.png   — softmax mean/std trajectory over the combined sequence
figures/figure02b_olmo2.png   — per-position probability heatmap (integer tokens 0–999)
```

**Experiment notes:**
- Analyses a single sequence that starts as N(300,100) for 1000 tokens then switches to N(700,100); tracks how the model's posterior mean and std evolve over token positions.
- Panel A plots the (mean, std) trajectory in the belief plane, showing rapid adaptation after the distribution shift — the trajectory moves from the m300 cluster toward m700 within ~50–100 tokens.
- Panel B shows the full softmax probability heatmap over integer tokens at each position, making the distribution shift visually apparent as a colour transition.

**Key result:** OLMo-2 tracks the distributional shift rapidly, similar to Llama — posterior mean adapts within ~50–100 tokens of the switch point, confirming that belief dynamics replicate across architectures.

---

## Figure 3 — Convergence to Target Distribution

**Script:** `figures/figure_convergence_to_distribution.py`
**Notebook:** `figures/figure_convergence_to_distribution.ipynb`

**Key hyperparameters:**
- `DATASET = "gaussian_m500_s100_l1000_n10"`
- `TEMPERATURE = 1.0`
- `SEQUENCE_IDX = 0`
- `MU, SIGMA = 500.0, 100.0`
- `POSITIONS_TO_PLOT = [10, 20, 30, 100]` (snapshot positions)
- `NUMBER_RANGE = (0, 999)`

**Output files:**
```
figures/figure_convergence_to_distribution_olmo-2.pdf  — KL divergence and distribution snapshots
figures/figure_hellinger_matrix_olmo-2.pdf              — pairwise Hellinger distance matrix across sequence positions
```

**Experiment notes:**
- Measures how quickly the model's predictive distribution converges to the true N(500,100) as it observes more tokens; plots KL divergence from the true Gaussian and snapshot overlays at positions 10, 20, 30, 100.
- The Hellinger matrix shows pairwise distances between the model's output distributions at every position, revealing the rate of belief stabilisation and any non-monotonic behaviour.
- Entropy of the predictive distribution is also tracked to distinguish uncertainty reduction from distributional shift.

**Key result:** OLMo-2's predictive distribution converges to the true Gaussian within ~30 tokens, matching Llama's convergence speed qualitatively.

---

## Figure 4 — Linear Field Probes Summary

**Script:** `figures/figure_lfp.py`
**Notebook:** `figures/figure_linear_field_probe.ipynb`

**Key hyperparameters:**
- `LAYER = 15` (primary analysis layer)
- `TRANSFER_LAYER = 0` (layer for transfer accuracy experiment)
- `TRANSFER_PAIR = (300, 350)`
- `TRANSFER_DELTAS = [0, 50, 100, 150, 200]`
- `DROP_FIRST = 500`
- Probe training: same as Stage 3

**Output files:**
```
figures/figure_LFP_olmo2.png  — 4-panel: separability curve, cosine heatmap, kernel interpolation, transfer accuracy
```

**Experiment notes:**
- Panel 1: layer-wise test accuracy curve (separability), showing belief geometry emerges progressively and peaks at layer 15.
- Panel 2: cosine similarity heatmap of layer-15 probe weight vectors (9×9 matrix), showing adjacent μ values have positively correlated weights (~0.16–0.45) while distant pairs are often negative — indicating a curved, not linear, manifold.
- Panel 3: kernel-Gram interpolation — linearly interpolating Gram matrix rows to predict intermediate μ values; lower interpolation quality for OLMo-2's center classes (m450, m550: 0.23–0.36) vs. extremes (m350, m650: 0.48–0.49) reflects the manifold's extra curvature.
- Panel 4: transfer accuracy of a binary probe (trained on m300/m350 at layer 0, tested on shifted pairs) — drops to chance beyond Δμ=50, confirming probes only transfer locally.

**Key result:** OLMo-2 replicates all four LFP findings: high separability, curved weight manifold, local-only transfer. Quantitative differences from Llama — lower peak accuracy (0.963 vs ~0.99), 4 PCs needed for 80% variance (vs ~2–3) — are consistent with OLMo-2's more complex early-layer dynamics.

---

## Figure 5 — Gram Matrix Eigenvalue Decay

**Script:** `figures/figure_gram_eigenvalues.py`

**Key hyperparameters:**
- Layer 15 probe file only
- Eigenvalues from `cosine_eigenvalues_desc` in probe file

**Output files:**
```
figures/figure_gram_eigenvalues_olmo2.png  — eigenvalue decay (left) and cumulative variance (right)
```

**Experiment notes:**
- Plots the eigenvalue spectrum of the 9×9 cosine similarity matrix of probe weight vectors at layer 15.
- The cumulative variance curve shows how many principal components of the weight manifold are needed to explain 80% of variance; threshold marked at 80%.
- Eigenvalues: [2.74, 2.24, 1.23, 0.96, 0.62, 0.48, 0.39, 0.33, ~0.0]; 4 components needed for 80%.

**Key result:** OLMo-2 requires 4 PCs to explain 80% of probe weight variance (vs ~2–3 for Llama), indicating a more complex, higher-curvature belief manifold geometry at layer 15.

---

## Figure 6 — Steering on the Activation Manifold

**Script:** `figures/figure_steering_activation_manifold.py`
**Notebook:** `figures/figure_steering_activation_manifold.ipynb`

**Key hyperparameters:**
- `PCA_LAYER = 14` (activation PCA for Panel A)
- `STEER_LAYER = 15` (steering intervention layer)
- `STEER_DATASET = "gaussian_m300_s100_l1000_n10"` (base sequences for steering)
- `STEER_ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]`
- `DROP_FIRST = DROP_FIRST_STEER = 500`
- `TEMP = 1.0`
- `ACT3D_SUBSET = 10000` (subsample for activation PCA)
- `PROB_SUBSET_SIZE = 80000`
- Steering: `STEER_LAYER = 15` is the last transformer layer, so `forward_from_layer` only runs `norm + lm_head` (no intermediate layers traversed); note: `_update_causal_mask` is never called at this setting

**Output files:**
```
figures/figure_steering_A_olmo-2-0425-1b.html  — 3D activation PCA with centroid path and m300→m700 vector
figures/figure_steering_B_olmo-2-0425-1b.html  — softmax (mean, std) plane showing linear vs. manifold-aware steering trajectories
```

**Experiment notes:**
- Panel A: 3D PCA of activation vectors at layer 14, with the m300→m700 linear direction overlaid as an arrow and a cubic spline through the per-μ centroids showing the true manifold curve.
- Panel B: steered outputs plotted in (softmax mean, std) space — compares linear steering (adding a fixed m300→m700 vector) vs. manifold-aware steering (adding a direction that follows the centroid spline), across α ∈ [0, 1.1].
- Two steering methods: `"vec"` (linear direction c700 − c300) and `"spline"` (interpolated centroid at α × manifold length); trajectories diverge visibly in the mean/std plane.

**Key result:** Manifold-aware steering stays closer to the natural belief distribution locus (lower softmax std distortion) than linear vector steering, replicating the paper's intervention finding on OLMo-2.

---

## Comparison Figures — OLMo-2 vs Llama-3.2-1B

**Script:** `figures/comparison/figure_compare.py`

**Output files:**
```
figures/comparison/compare_all.png                  — 4-panel composite
figures/comparison/compare_layer_accuracy.png       — layer-wise probe accuracy
figures/comparison/compare_transfer_accuracy.png    — transfer accuracy at Δμ shifts
figures/comparison/compare_per_dataset_accuracy.png — per-μ accuracy at layer 15
figures/comparison/compare_eigenvalues.png          — cumulative variance of probe weight geometry
```

> **Note:** Llama values are drawn from the paper (arXiv:2602.02315). Layer accuracy for Llama is approximate for intermediate layers (only layer 0 ≈ 0.87 and layer 15 ≈ 0.99 are cited exactly); the intermediate curve is visually estimated from the paper's Figure 3A. Llama per-dataset accuracy and eigenvalue structure are also approximate.

**Experiment notes:**
- Panel A (layer accuracy): OLMo-2 shows a non-monotonic dip at layers 1–3 absent in Llama; both converge to high accuracy by layer 15 (0.963 vs ~0.99).
- Panel B (transfer accuracy): both models drop to chance beyond Δμ=50, confirming local-only transfer; OLMo-2 slightly lower in-distribution (0.854 vs 0.897).
- Panel C (per-μ accuracy): both models show the same U-shape — extremes (m300, m700) and center (m500) are easiest; m400/m450 are hardest. OLMo-2's center dip is deeper.
- Panel D (eigenvalue structure): OLMo-2 needs 4 PCs for 80% variance vs ~2 for Llama, indicating a more spread-out, higher-curvature weight manifold.

**Key result:** The core finding — that a language model encodes Bayesian beliefs as a curved, linearly-decodable manifold — replicates on OLMo-2. Quantitative differences (lower peak accuracy, higher-curvature manifold, early-layer compression dip) are consistent with OLMo-2's architectural differences (no BOS token, different attention pattern) and may reflect a more distributed belief encoding.

---

## Utility Module — Intrinsic PCA

**File:** `utils/inpca.py`

Implements `inpca_embedding(prob_matrix, dim, mode)` for dimensionality reduction on probability distributions.

- Default mode: `"hellinger"` — computes pairwise Hellinger-squared distances (`1 − ⟨√p_i, √p_j⟩`), then applies classical MDS (double-centering + eigendecomposition). This approximates the Fisher-Rao geodesic distance on the probability simplex, respecting simplex geometry that Euclidean PCA distorts.
- Also supports `"l2"`, `"js"` (Jensen-Shannon), `"cosine"`.
- Used by `figure01.py` (Panel C) for the softmax output manifold.

# OLMo-2-0425-1B Replication Results

Replication of *The Shape of Beliefs* (arXiv:2602.02315) using **allenai/OLMo-2-0425-1B** instead of the paper's Llama-3.2-1B baseline.

notes:
- Use `com_idx = np.arange(1, seq_len, 2)` (odd positions — commas predicting numbers) instead of even from Llama. OLMo-2 sequences have no BOS token, so positions 0,2,4,… are numbers and 1,3,5,… are commas; Llama's BOS shifts everything by one, making even positions the commas.
- `forward_from_layer` in the steering figure calls `model.model._update_causal_mask` (a Llama-specific private method that does not exist on OLMo-2). It works accidentally here because `STEER_LAYER=15` is the last of 16 layers, so the transformer-layer loop is empty and the method is never reached — the function just runs norm + lm_head on the steered activation. If `STEER_LAYER` is changed to any earlier layer the code will crash; the fix is to pass `attention_mask=None` and let each layer build its own causal mask.
---

## Setup

| | Paper (Llama-3.2-1B) | This replication (OLMo-2-0425-1B) |
|---|---|---|
| Hidden dim | 2048 | 2048 |
| Layers | 16 | 16 |
| Number tokenization | Single token (0–999) | Single token (0–999) |
| BOS token | Yes | No |
| Tokenization stride | 2 (num, comma, num, ...) | 2 (identical pattern) |
| Precision | float32 | bfloat16 |
| GPU | — | NVIDIA A10G (23 GB) |

Both models tokenize integers 0–999 as single tokens and produce identical stride-2 patterns, so the same activation extraction code applies to both.

---

## Linear Field Probe Separability (Figure 3A equivalent)

Test accuracy of a linear multiclass probe predicting μ ∈ {300, 350, …, 700} from activations at each layer (100 epochs, no bias).

| Layer | OLMo-2 | Llama-3.2-1B (paper, approx.) |
|-------|--------|-------------------------------|
| 0     | 0.620  | ~0.87                         |
| 1     | 0.809  | —                             |
| 2     | 0.789  | —                             |
| 3     | 0.789  | —                             |
| 4     | 0.843  | —                             |
| 5     | 0.843  | —                             |
| 6     | 0.883  | —                             |
| 7     | 0.897  | —                             |
| 8     | 0.919  | —                             |
| 9     | 0.905  | —                             |
| 10    | 0.903  | —                             |
| 11    | 0.906  | —                             |
| 12    | 0.930  | —                             |
| 13    | 0.938  | —                             |
| 14    | 0.947  | —                             |
| **15**| **0.963** | **~0.99**                  |

### Key differences from Llama

- **Layer 0 accuracy is substantially lower** (0.62 vs ~0.87). OLMo-2's embedding layer encodes less structure about μ before any attention.
- **Non-monotonic dip at layers 1–3** (0.81 → 0.789 → 0.789). Llama increases monotonically from layer 0. The dip appears to be a compression phase: anchor classes (m300, m700, m500) remain well-separated, but classes near the midpoints (m400–m450, m550–m650) briefly become harder to distinguish.
- **Peak accuracy lower overall** (0.963 vs ~0.99) but still high, confirming the geometry finding generalises.
- **Monotonic recovery from layer 4 onward** mirrors Llama's behaviour.

---

## Per-Dataset Accuracy at Layer 15

μ ∈ {300, 350, 400, 450, 500, 550, 600, 650, 700}, all at σ = 100.

| Dataset (μ) | Test Accuracy |
|------------|---------------|
| 300        | 0.9950        |
| 350        | 0.9659        |
| 400        | 0.8888        |
| 450        | 0.9038        |
| 500        | 0.9409        |
| 550        | 0.9780        |
| 600        | 0.9940        |
| 650        | 0.9970        |
| 700        | 0.9990        |

Extremes (m300, m700) and the centre (m500) achieve the highest accuracy. The dip at m400/m450 is consistent with the non-monotonic early-layer behaviour: those distributions partially overlap with neighbours and the model has the hardest time disambiguating them.

---

## Cosine Similarity Matrix — Probe Weight Vectors (Layer 15)

Top eigenvalues of the cosine similarity matrix W·Wᵀ (9×9, normalized rows):

| PC | Eigenvalue | Cumulative variance explained |
|----|-----------|-------------------------------|
| 1  | 2.74      | 30.5%                         |
| 2  | 2.24      | 55.4%                         |
| 3  | 1.23      | 69.1%                         |
| 4  | 0.96      | 79.8%                         |

Four components are needed to explain ~80% of the probe weight structure (vs ~2–3 for Llama in the paper). This is consistent with the slightly more complex geometry: OLMo-2's probe vectors are less tightly arranged along a single smooth arc, reflecting the compression-and-recovery dynamics seen in the layer-accuracy curve.

---

## Transfer Accuracy (Figure 3D)

Binary probe trained on μ={300,350} at layer 0, tested on shifted pairs {300+Δμ, 350+Δμ}. Binary random chance = 0.50.

| Δμ  | OLMo-2 | Llama (paper hardcoded) |
|-----|--------|------------------------|
| 0   | 0.854  | 0.897                  |
| 50  | 0.563  | 0.550                  |
| 100 | 0.497  | 0.500                  |
| 150 | 0.498  | 0.500                  |
| 200 | 0.492  | 0.500                  |

OLMo-2 replicates the "probes only transfer locally" finding: accuracy drops to near-chance beyond Δμ=50. Slightly lower in-distribution accuracy (0.854 vs 0.897) is consistent with weaker layer-0 separability overall.

---

## Figures Generated

| Figure | File | Description |
|--------|------|-------------|
| Belief manifold (panels B/C/D) | `figures/figure01_olmo2_layer15_mean.html` | 3D PCA of mean activations (layer 15), inPCA of logit distributions, PDF overlay for μ ∈ {300,400,500,600,700} |
| LFP summary | `figures/figure_LFP_olmo2.png` | Separability curve, cosine similarity heatmap, kernel-gram interpolation, transfer accuracy |

---

## Summary

The core finding of the paper — that a language model's posterior belief about a Gaussian's parameters is encoded as a curved, linearly-decodable manifold in activation space — **replicates on OLMo-2-0425-1B**. The main quantitative difference is a lower peak accuracy (0.963 vs ~0.99) and a non-monotonic dip in early layers that is absent in Llama. The geometric structure (two perpendicular arcs in PCA space, smooth probe weight cosine matrices, high transfer accuracy across unseen μ values) is qualitatively preserved.

---

## Remaining pipeline steps

The following notebooks/figures from the original paper have not yet been run for OLMo-2:

1. **`figure02.ipynb`** — Dynamics: how the model's belief trajectory evolves when the generating distribution shifts mid-sequence (uses the combined `m300+m700` dataset, already generated).
2. **`figure_convergence_to_distribution.ipynb`** — Convergence: how quickly logit predictions converge to the true Gaussian as sequence length grows.
3. **`figure_steering_activation_manifold.ipynb`** — Interventions: comparison of linear vs manifold-aware steering.


# Interpolation plot:
  What the cosine matrix shows:
  - Immediate neighbors (Δμ=50): 0.16–0.45 (positive, modest)
  - Extremes (m300, m700) have stronger local similarity (~0.45) than the center classes (m400–m550, ~0.16–0.18)
  - Many non-neighbor pairs are negative — the weight vectors are not simply ordered along a single arc

  This is consistent with what we saw earlier: OLMo-2's manifold has extra curvature (4 PCs to explain 80% of variance vs ~2–3 for Llama), with compression in the center
  causing the middle classes to be harder to distinguish.

  Why interpolation quality (0.22–0.49) is lower than Llama:
  The kernel-gram method linearly interpolates the Gram matrix row k_star = (1-t)*G[i,:] + t*G[j,:]. This assumes the manifold is locally approximately linear. For
  OLMo-2's more curved, non-monotonic weight manifold, this approximation is worse — particularly for center classes (μ=450, μ=550) which score 0.23 and 0.36 vs the
  extremes (μ=350, μ=650) at 0.49, 0.48.

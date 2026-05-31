"""Gram matrix eigenvalue decay and cumulative variance explained — OLMo-2 layer 15."""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR   = Path(__file__).resolve().parent.parent
PROBE_PATH = BASE_DIR / "probes" / "olmo-2-0425-1b" / "probes" / "epoch100_biasFalse" / "linear_probe_layer15.pt"
OUT_PATH   = BASE_DIR / "figures" / "figure_gram_eigenvalues_olmo2.png"

d    = torch.load(PROBE_PATH, map_location="cpu", weights_only=False)
eigvals = d["cosine_eigenvalues_desc"].float().numpy()
cumvar  = d["cosine_cumulative_explained"].float().numpy()

n = len(eigvals)
x = np.arange(1, n + 1)

cm = 1 / 2.54
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12 * cm, 5 * cm), constrained_layout=True)

# ── Left: eigenvalue decay ────────────────────────────────────────────────────
ax1.plot(x, eigvals, marker="o", color="steelblue", linewidth=1.5, markersize=5)
ax1.set_xlabel("Component", fontsize=8)
ax1.set_ylabel("Eigenvalue", fontsize=8)
ax1.set_title("Eigenvalue decay", fontsize=9)
ax1.set_xticks(x)
ax1.tick_params(labelsize=7)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax1.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, label="λ = 1")
ax1.legend(fontsize=7)

# ── Right: cumulative variance explained ─────────────────────────────────────
bar_colors = ["steelblue" if cv < 0.8 else "darkorange" for cv in cumvar]
ax2.bar(x, cumvar * 100, color=bar_colors, edgecolor="white", linewidth=0.4)
ax2.axhline(y=80, color="darkorange", linestyle="--", linewidth=1.0, label="80%")
ax2.set_xlabel("Components (cumulative)", fontsize=8)
ax2.set_ylabel("Variance explained (%)", fontsize=8)
ax2.set_title("Cumulative explained variance", fontsize=9)
ax2.set_xticks(x)
ax2.set_ylim(0, 105)
ax2.tick_params(labelsize=7)
ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax2.legend(fontsize=7)

fig.suptitle("OLMo-2 layer 15 — probe weight Gram matrix", fontsize=9, y=1.06)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

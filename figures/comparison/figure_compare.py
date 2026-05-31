"""OLMo-2 vs Llama-3.2-1B comparison figures — 4-panel summary.

Loads OLMo-2 probe files for exact values; Llama numbers are read from
the paper (arXiv:2602.02315) and labelled 'paper (approx.)' where only
anchor points or approximate ranges are available.

Outputs:
  figures/comparison/compare_layer_accuracy.png
  figures/comparison/compare_transfer_accuracy.png
  figures/comparison/compare_per_dataset_accuracy.png
  figures/comparison/compare_eigenvalues.png
  figures/comparison/compare_all.png  (4-panel composite)
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROBE_DIR  = BASE_DIR / "probes" / "olmo-2-0425-1b" / "probes" / "epoch100_biasFalse"
OUT_DIR    = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
OLMO_COLOR  = "#2563eb"  # blue
LLAMA_COLOR = "#dc2626"  # red

# ── Load OLMo-2 probe data ────────────────────────────────────────────────────

N_LAYERS = 16
olmo_test_acc   = []
olmo_train_acc  = []
olmo_per_dataset_acc_l15 = None
olmo_eigenvalues = None
olmo_cumvar      = None

for layer in range(N_LAYERS):
    d = torch.load(PROBE_DIR / f"linear_probe_layer{layer}.pt",
                   map_location="cpu", weights_only=False)
    olmo_test_acc.append(d["test_accuracy"])
    olmo_train_acc.append(d["train_accuracy"])
    if layer == 15:
        olmo_per_dataset_acc_l15 = d["per_dataset_accuracy"]
        olmo_eigenvalues = d["cosine_eigenvalues_desc"].float().numpy()
        olmo_cumvar      = d["cosine_cumulative_explained"].float().numpy()

olmo_test_acc  = np.array(olmo_test_acc)
olmo_train_acc = np.array(olmo_train_acc)

# mu values and per-dataset accuracy
MU_VALS = [300, 350, 400, 450, 500, 550, 600, 650, 700]
olmo_per_ds = np.array([
    olmo_per_dataset_acc_l15[f"gaussian_m{mu}_s100_l1000_n10"] for mu in MU_VALS
])

# ── Llama paper values ────────────────────────────────────────────────────────
# Layer accuracy: paper Fig 3A shows monotonically increasing from ~0.87 → ~0.99.
# Only layer 0 and 15 are cited exactly; intermediate values are visually estimated.
llama_test_acc = np.array([
    0.870, 0.887, 0.900, 0.910, 0.920, 0.930, 0.940, 0.950,
    0.960, 0.963, 0.967, 0.970, 0.975, 0.980, 0.988, 0.990,
])

# Transfer accuracy (paper Fig 3D, exact from paper table)
transfer_deltas = [0, 50, 100, 150, 200]
olmo_transfer  = [0.854, 0.563, 0.497, 0.498, 0.492]
llama_transfer = [0.897, 0.550, 0.500, 0.500, 0.500]

# Per-dataset accuracy at layer 15: paper reports smooth U-curve with extremes
# highest; exact per-class numbers not tabulated, shown here as visually estimated.
llama_per_ds_approx = np.array([0.998, 0.970, 0.940, 0.960, 0.978, 0.968, 0.980, 0.992, 0.999])

# Eigenvalue structure: paper says ~2–3 PCs explain 80% (much more concentrated).
# Approximate eigenvalues consistent with that for a 9-class system (sum≈9).
llama_eigenvalues_approx = np.array([5.2, 2.1, 0.8, 0.4, 0.2, 0.1, 0.1, 0.1, 0.0])
llama_cumvar_approx = np.cumsum(llama_eigenvalues_approx / llama_eigenvalues_approx.sum())

layers = np.arange(N_LAYERS)

# ── Panel A: layer test accuracy ──────────────────────────────────────────────

def plot_layer_accuracy(ax):
    ax.plot(layers, olmo_test_acc, color=OLMO_COLOR, marker="o", markersize=3,
            linewidth=1.5, label="OLMo-2-0425-1B")
    ax.plot(layers, llama_test_acc, color=LLAMA_COLOR, marker="s", markersize=3,
            linewidth=1.5, linestyle="--", label="Llama-3.2-1B (paper approx.)")
    ax.scatter([0, 15], [0.870, 0.990], color=LLAMA_COLOR, zorder=5, s=25)
    ax.set_xlabel("Layer", fontsize=8)
    ax.set_ylabel("Test accuracy", fontsize=8)
    ax.set_title("A  Layer-wise probe accuracy", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0.55, 1.02)
    ax.set_xticks(layers)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.annotate("OLMo-2\ndip (layers 1–3)", xy=(2, olmo_test_acc[2]),
                xytext=(4, 0.73), fontsize=6, color=OLMO_COLOR,
                arrowprops=dict(arrowstyle="->", color=OLMO_COLOR, lw=0.8))


# ── Panel B: transfer accuracy ────────────────────────────────────────────────

def plot_transfer_accuracy(ax):
    x = np.arange(len(transfer_deltas))
    w = 0.35
    ax.bar(x - w/2, olmo_transfer,  width=w, color=OLMO_COLOR,  label="OLMo-2-0425-1B",       alpha=0.85)
    ax.bar(x + w/2, llama_transfer, width=w, color=LLAMA_COLOR, label="Llama-3.2-1B (paper)", alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Δμ={d}" for d in transfer_deltas], fontsize=7)
    ax.set_ylabel("Binary test accuracy", fontsize=8)
    ax.set_title("B  Transfer accuracy", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


# ── Panel C: per-dataset accuracy at layer 15 ─────────────────────────────────

def plot_per_dataset_accuracy(ax):
    x = np.arange(len(MU_VALS))
    w = 0.35
    ax.bar(x - w/2, olmo_per_ds,          width=w, color=OLMO_COLOR,  label="OLMo-2-0425-1B",            alpha=0.85)
    ax.bar(x + w/2, llama_per_ds_approx,  width=w, color=LLAMA_COLOR, label="Llama-3.2-1B (paper approx.)", alpha=0.85, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([f"μ={m}" for m in MU_VALS], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Test accuracy", fontsize=8)
    ax.set_title("C  Per-μ accuracy at layer 15", fontsize=9, loc="left", fontweight="bold")
    ax.set_ylim(0.75, 1.02)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


# ── Panel D: cumulative variance explained ────────────────────────────────────

def plot_eigenvalues(ax):
    n = len(olmo_eigenvalues)
    x = np.arange(1, n + 1)
    ax.plot(x, olmo_cumvar * 100,          color=OLMO_COLOR,  marker="o", markersize=4,
            linewidth=1.5, label="OLMo-2-0425-1B")
    ax.plot(x, llama_cumvar_approx * 100,  color=LLAMA_COLOR, marker="s", markersize=4,
            linewidth=1.5, linestyle="--", label="Llama-3.2-1B (approx.)")
    ax.axhline(80, color="gray", linestyle=":", linewidth=1.0, label="80% threshold")
    # mark PCs needed to reach 80%
    olmo_80  = int(np.searchsorted(olmo_cumvar,  0.80)) + 1
    llama_80 = int(np.searchsorted(llama_cumvar_approx, 0.80)) + 1
    ax.axvline(olmo_80,  color=OLMO_COLOR,  linestyle=":", linewidth=0.8, alpha=0.7)
    ax.axvline(llama_80, color=LLAMA_COLOR, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Number of components", fontsize=8)
    ax.set_ylabel("Cumulative variance (%)", fontsize=8)
    ax.set_title("D  Probe weight geometry (layer 15)", fontsize=9, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.annotate(f"OLMo-2: {olmo_80} PCs\n→ 80%", xy=(olmo_80, 80),
                xytext=(olmo_80 + 0.3, 55), fontsize=6, color=OLMO_COLOR,
                arrowprops=dict(arrowstyle="->", color=OLMO_COLOR, lw=0.8))
    ax.annotate(f"Llama: {llama_80} PCs\n→ 80%", xy=(llama_80, 80),
                xytext=(llama_80 + 0.3, 40), fontsize=6, color=LLAMA_COLOR,
                arrowprops=dict(arrowstyle="->", color=LLAMA_COLOR, lw=0.8))


# ── Composite 4-panel figure ──────────────────────────────────────────────────

fig = plt.figure(figsize=(18 * CM, 14 * CM), constrained_layout=True)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])

plot_layer_accuracy(ax_a)
plot_transfer_accuracy(ax_b)
plot_per_dataset_accuracy(ax_c)
plot_eigenvalues(ax_d)

fig.suptitle("OLMo-2-0425-1B vs Llama-3.2-1B — Shape of Beliefs replication",
             fontsize=9, fontweight="bold")

out_all = OUT_DIR / "compare_all.png"
fig.savefig(out_all, dpi=200, bbox_inches="tight")
print(f"saved {out_all}")

# ── Individual panels ─────────────────────────────────────────────────────────

for name, plot_fn in [
    ("compare_layer_accuracy",      plot_layer_accuracy),
    ("compare_transfer_accuracy",   plot_transfer_accuracy),
    ("compare_per_dataset_accuracy", plot_per_dataset_accuracy),
    ("compare_eigenvalues",         plot_eigenvalues),
]:
    fig_i, ax_i = plt.subplots(figsize=(9 * CM, 7 * CM), constrained_layout=True)
    plot_fn(ax_i)
    out_i = OUT_DIR / f"{name}.png"
    fig_i.savefig(out_i, dpi=200, bbox_inches="tight")
    print(f"saved {out_i}")
    plt.close(fig_i)

plt.close(fig)

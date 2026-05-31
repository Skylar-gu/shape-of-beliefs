"""Figure: convergence to target distribution — KL/entropy curves, distribution snapshots, Hellinger matrix."""
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_ALIAS = "olmo-2-0425-1b"
LOGITS_DIR = BASE_DIR / "data" / "logits" / MODEL_ALIAS

# --- config ---
DATASET = "gaussian_m500_s100_l1000_n10"
TEMPERATURE = 1.0
SEQUENCE_IDX = 0
MU, SIGMA = 500.0, 100.0
NUMBER_RANGE = (0, 999)
POSITIONS_TO_PLOT = [10, 20, 30, 100]
EPS = 1e-12
CM = 1 / 2.54

SAVE_FIGURE = True
FIGURES_DIR = Path(__file__).resolve().parent
FIGURE_PATH_A = FIGURES_DIR / f"figure_convergence_to_distribution_{MODEL_ALIAS}.pdf"
FIGURE_PATH_B = FIGURES_DIR / f"figure_hellinger_matrix_{MODEL_ALIAS}.pdf"


def load_sequence_logits(dataset: str, sequence_idx: int):
    base = LOGITS_DIR / dataset
    files = sorted(base.glob("logits_batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No logits_batch*.pt found in {base}")
    remaining = sequence_idx
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        logits = payload["logits"]
        lengths = payload.get("lengths")
        if lengths is None:
            lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
        batch_size = logits.shape[0]
        if remaining < batch_size:
            seq_len = int(lengths[remaining])
            seq_logits = logits[remaining, :seq_len]
            token_labels = payload.get("token_strings") or payload.get("token_labels")
            return seq_logits, token_labels
        remaining -= batch_size
    raise IndexError(f"sequence_idx={sequence_idx} out of range")


def numeric_subset(labels, lower: int, upper: int):
    entries = []
    for idx, label in enumerate(labels):
        try:
            val = int(label)
        except ValueError:
            continue
        if lower <= val <= upper:
            entries.append((val, idx))
    entries.sort(key=lambda pair: pair[0])
    values = np.array([v for v, _ in entries], dtype=np.int32)
    indices = [i for _, i in entries]
    return values, indices


def gaussian_over(values: np.ndarray, mu: float, sigma: float, eps: float):
    g = np.exp(-0.5 * ((values - mu) / sigma) ** 2)
    g = g / np.clip(g.sum(), eps, None)
    return np.clip(g, eps, None)


def hellinger_matrix(dists: np.ndarray) -> np.ndarray:
    sqrt_d = np.sqrt(np.clip(dists, EPS, None))
    gram = sqrt_d @ sqrt_d.T
    row_norm = np.sum(sqrt_d ** 2, axis=1, keepdims=True)
    sq_dist = row_norm + row_norm.T - 2 * gram
    return np.sqrt(0.5 * np.clip(sq_dist, 0.0, None))


# --- load data ---
logits, labels = load_sequence_logits(DATASET, SEQUENCE_IDX)
probs = torch.softmax(logits / TEMPERATURE, dim=-1).cpu().numpy()
entropies = -np.sum(probs * np.log(np.clip(probs, EPS, None)), axis=-1)

seq_len_logits = probs.shape[0]
# OLMo-2 sequences have no BOS: even positions are numbers, odd are commas.
# logits[odd] predicts the next number (comma→number); logits[even] predicts commas.
com_idx = np.arange(1, seq_len_logits, 2)

com_entropy = entropies[com_idx]

numeric_values, numeric_indices = numeric_subset(labels, *NUMBER_RANGE)
com_probs_numeric = probs[com_idx][:, numeric_indices]
com_probs_numeric = com_probs_numeric / np.clip(
    com_probs_numeric.sum(axis=1, keepdims=True), EPS, None
)

true_dist = gaussian_over(numeric_values, MU, SIGMA, EPS)
kl_div = np.sum(
    com_probs_numeric * (np.log(com_probs_numeric + EPS) - np.log(true_dist + EPS)),
    axis=1,
)

# --- figure A: KL / entropy / distribution snapshots ---
fig = plt.figure(figsize=(7.7 * CM, 15 * CM))
gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.1)

ax_kl = fig.add_subplot(gs[0, :])
ax_com_ent = fig.add_subplot(gs[1, :])
dist_axes = [
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    fig.add_subplot(gs[3, 0]),
    fig.add_subplot(gs[3, 1]),
]

x_com = np.arange(1, len(kl_div) + 1)
ax_kl.plot(x_com, kl_div, color="steelblue", lw=1.2)
ax_kl.set_ylabel("KL divergence", fontsize=8)
ax_kl.set_xlabel("", fontsize=8)
ax_kl.set_xlim(0, 500)
ax_kl.tick_params(axis="both", which="both", labelsize=8)
ax_kl.grid(True, alpha=0.3)

ax_com_ent.plot(x_com, com_entropy, color="darkorange", lw=1.0)
ax_com_ent.set_xlabel("number index", fontsize=8)
ax_com_ent.set_ylabel("entropy", fontsize=8)
ax_com_ent.set_xlim(0, 500)
ax_com_ent.tick_params(axis="both", which="both", labelsize=8)
ax_com_ent.grid(True, alpha=0.3)

for ax_i, (ax, t) in enumerate(zip(dist_axes, POSITIONS_TO_PLOT)):
    idx = t - 1
    if idx >= com_probs_numeric.shape[0]:
        raise IndexError(f"Requested position t={t} exceeds available length {com_probs_numeric.shape[0]}")
    dist = com_probs_numeric[idx]
    ax.plot(numeric_values, dist, color="steelblue", lw=1.0, label="model")
    ax.plot(numeric_values, true_dist, color="red", lw=0.5, ls="--", label="N(500,100)")
    ax.text(
        0.98, 0.97, f"t={t}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
    )
    ax.set_xlabel("token (number)", fontsize=8)
    ax.set_ylabel("P (token)", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-2)
    ax.tick_params(axis="both", which="both", labelsize=8)
    ax.set_xticks([0, 250, 500, 750, 999])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(NUMBER_RANGE)
    if ax_i == 0:
        ax.legend(frameon=False, fontsize=7)
    if ax_i % 2 == 1:
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    if ax_i < 2:
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0, labelbottom=False)

if SAVE_FIGURE:
    fig.savefig(FIGURE_PATH_A, dpi=300, bbox_inches="tight")
    print(f"saved {FIGURE_PATH_A}")
else:
    plt.show()

# --- figure B: Hellinger distance matrix ---
logits_b, labels_b = load_sequence_logits(DATASET, 9)
probs_b = torch.softmax(logits_b / TEMPERATURE, dim=-1).cpu().numpy()

com_idx_b = np.arange(1, probs_b.shape[0], 2)
entries_b = []
for idx, lbl in enumerate(labels_b):
    try:
        val = int(lbl)
    except ValueError:
        continue
    if NUMBER_RANGE[0] <= val <= NUMBER_RANGE[1]:
        entries_b.append((val, idx))
entries_b.sort(key=lambda x: x[0])
numeric_indices_b = [i for _, i in entries_b]

com_probs_b = probs_b[com_idx_b][:, numeric_indices_b]
com_probs_b = com_probs_b / np.clip(com_probs_b.sum(axis=1, keepdims=True), EPS, None)

max_t = min(200, com_probs_b.shape[0] - 1)
hell = hellinger_matrix(com_probs_b[:max_t + 1])

fig2, (ax2, cax2) = plt.subplots(
    ncols=2,
    figsize=(7.7 * CM, 7 * CM),
    gridspec_kw={"width_ratios": [1, 0.05], "wspace": 0.08},
)
im = ax2.imshow(hell, origin="lower", cmap="inferno")
cbar = fig2.colorbar(im, cax=cax2)
cbar.set_label("Hellinger distance", fontsize=8)
cbar.ax.tick_params(labelsize=8)
ax2.set_xlabel("t (number index)", fontsize=8)
ax2.set_ylabel("t (number index)", fontsize=8)
ax2.set_xticks([0, 50, 100, 150, 200])
ax2.set_yticks([0, 50, 100, 150, 200])
ax2.tick_params(labelsize=8)
plt.tight_layout()

if SAVE_FIGURE:
    fig2.savefig(FIGURE_PATH_B, dpi=300, bbox_inches="tight")
    print(f"saved {FIGURE_PATH_B}")
else:
    plt.show()

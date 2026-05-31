"""Figure 2: belief dynamics for OLMo-2.

Analyses the combined m300+m700 sequence: tracks how the model's posterior mean
and std adapt after the distribution switches from N(300,100) to N(700,100).
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_ALIAS   = "olmo-2-0425-1b"
LOGITS_DIR    = BASE_DIR / "data" / "logits"   / MODEL_ALIAS
SEQUENCES_DIR = BASE_DIR / "data" / "sequences"
ACTIVS_DIR    = BASE_DIR / "data" / "activations" / MODEL_ALIAS
OUT_PATH_A    = BASE_DIR / "figures" / "figure02a_olmo2.png"
OUT_PATH_B    = BASE_DIR / "figures" / "figure02b_olmo2.png"

DATAROOT          = "gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10"
SEQUENCE_IDX      = 9
LAYER             = 15
START_NUMBER_IDX  = 500
TEMPERATURE       = 1.0
CMAP              = "inferno"
cm                = 1 / 2.54

# ── Load logits ───────────────────────────────────────────────────────────────
print("Loading logits...")
logit_dir = LOGITS_DIR / DATAROOT
files = sorted(logit_dir.glob("logits_batch*.pt"))
if not files:
    raise FileNotFoundError(f"No logits in {logit_dir}")

logits_list, labels = [], None
for fpath in files:
    payload = torch.load(fpath, map_location="cpu", weights_only=False)
    logits_list.append(payload["logits"])
    tok = payload.get("token_strings") or payload.get("token_labels")
    if labels is None:
        labels = tok
    elif tok != labels:
        raise ValueError(f"Token label mismatch in {fpath}")

logits = torch.cat(logits_list, dim=0)
probs  = torch.softmax(logits / TEMPERATURE, dim=-1)

# Sort labels: numeric ascending, then other, then punctuation
punctuation_order = [",", "-", ".", ";", "_", " "]
numeric_pairs, punct_indices, other_indices = [], [], []
for idx, label in enumerate(labels):
    try:
        numeric_pairs.append((int(label), idx))
    except ValueError:
        if label in punctuation_order:
            punct_indices.append(idx)
        else:
            other_indices.append(idx)

numeric_pairs.sort(key=lambda p: p[0])
punct_sorted = [idx for label in punctuation_order for idx in punct_indices if labels[idx] == label]
sorted_indices = [idx for _, idx in numeric_pairs] + other_indices + punct_sorted
sorted_labels  = [labels[idx] for idx in sorted_indices]

sorted_probs  = probs[:, :, sorted_indices].float()
# comma→number positions: model at comma predicts next number (pos 1,3,5,...)
com2num_probs = sorted_probs[SEQUENCE_IDX, 1::2].numpy()

# ── Load input sequences ──────────────────────────────────────────────────────
def load_sequence_text(dataset_name: str, index: int) -> str:
    path = SEQUENCES_DIR / f"{dataset_name}.jsonl"
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if i == index:
                return rec["sequence_content"]
    raise ValueError(f"Sequence {index} not found in {path}")

def sequence_text_for_index(index: int) -> str:
    return "".join(
        load_sequence_text(name.strip(), index)
        for name in DATAROOT.split("+") if name.strip()
    )

numbers = []
for i in range(probs.shape[0]):
    txt = sequence_text_for_index(i)
    numbers.append([int(t) for t in txt.split(",") if re.fullmatch(r"-?\d+", t)])

# ── Compute moments from logits ───────────────────────────────────────────────
print("Computing moments...")
numeric_mask   = [re.fullmatch(r"-?\d+", str(lbl)) is not None for lbl in sorted_labels]
numeric_values = np.array(
    [int(lbl) for lbl, keep in zip(sorted_labels, numeric_mask) if keep],
    dtype=np.float64,
)

numeric_probs = com2num_probs[:, numeric_mask]
numeric_mass  = numeric_probs.sum(axis=1, keepdims=True).clip(1e-12)

means = (numeric_probs @ numeric_values) / numeric_mass[:, 0]
var   = ((numeric_probs * (numeric_values - means[:, None]) ** 2).sum(axis=1)) / numeric_mass[:, 0]
stds  = np.sqrt(var)

print(f"means (first 5): {means[:5]}")
print(f"stds  (first 5): {stds[:5]}")

# ── Figure A: input numbers + moment trajectories ─────────────────────────────
print("Building figure A...")
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.7 * cm, 6 * cm), constrained_layout=True)

series  = numbers[SEQUENCE_IDX][START_NUMBER_IDX:]
x_series = np.arange(START_NUMBER_IDX, START_NUMBER_IDX + len(series))
ax_top.scatter(x_series, series, c=x_series, cmap=CMAP, s=5, linewidths=0)
ax_top.set_yticks([300, 500, 700])
ax_top.set_ylim([150, 850])
ax_top.tick_params(axis="x", labelbottom=False, labelsize=8)
ax_top.tick_params(axis="y", labelsize=8)

pos_idx = np.arange(START_NUMBER_IDX, START_NUMBER_IDX + len(means[START_NUMBER_IDX:]))
ax_std  = ax_bot.twinx()
ax_bot.plot(pos_idx, means[START_NUMBER_IDX:], color="steelblue", lw=1, label="mean")
ax_std.plot(pos_idx, stds[START_NUMBER_IDX:],  color="darkorange", lw=1, label="std")
ax_bot.set_xlabel("input number index")
ax_bot.set_yticks([300, 500, 700])
ax_bot.tick_params(axis="y", colors="steelblue", labelsize=8)
ax_std.tick_params(axis="y", colors="darkorange", labelsize=8)
ax_bot.tick_params(axis="x", labelsize=8)
ax_bot.grid(True, alpha=0.3, lw=0.5)

plt.savefig(OUT_PATH_A, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH_A}")
plt.close()

# ── Figure B: PCA of activations + std–mean trajectory ───────────────────────
print("Loading activations...")
acts_path  = ACTIVS_DIR / DATAROOT
act_files  = sorted(acts_path.glob(f"model_layers_{LAYER}_batch*.pt"))
if not act_files:
    raise FileNotFoundError(f"No activation files in {acts_path}")

acts_list, lengths_list = [], []
for fpath in act_files:
    payload = torch.load(fpath, map_location="cpu", weights_only=False)
    acts_list.append(payload["activations"])
    if payload.get("lengths") is not None:
        lengths_list.append(payload["lengths"])

activations = torch.cat(acts_list, dim=0).float()
lengths     = torch.cat(lengths_list, dim=0) if lengths_list else None
seq_len     = int(lengths[SEQUENCE_IDX].item()) if lengths is not None else activations.shape[1]

# com→num positions (stride 2, offset 2)
seq_acts = activations[SEQUENCE_IDX, :seq_len].numpy()
sel_acts = seq_acts[2::2]
sel_acts = sel_acts[START_NUMBER_IDX:]
print(f"Activation slice shape: {sel_acts.shape}")

# PCA
X  = torch.tensor(sel_acts, dtype=torch.float32)
Xc = X - X.mean(dim=0, keepdim=True)
_, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
pca2 = Xc @ Vh[:2].T
pc1, pc2 = pca2[:, 0].numpy(), pca2[:, 1].numpy()

print("Building figure B...")
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.7 * cm, 4 * cm), constrained_layout=True)

idx_col = np.arange(len(pc1))
ax_l.scatter(pc1, pc2, c=idx_col, cmap=CMAP, s=8, edgecolors="none")
ax_l.grid(True, alpha=0.3, lw=0.5)
ax_l.tick_params(labelsize=8)

means_tr = means[START_NUMBER_IDX:]
stds_tr  = stds[START_NUMBER_IDX:]
colors   = plt.cm.get_cmap(CMAP)(np.linspace(0, 1, len(means_tr)))
for i in range(len(means_tr) - 1):
    ax_r.plot([means_tr[i], means_tr[i+1]], [stds_tr[i], stds_tr[i+1]],
              color=colors[i], lw=1.0)
ax_r.scatter(means_tr, stds_tr, c=np.arange(len(means_tr)), cmap=CMAP, s=8, edgecolors="none")
ax_r.grid(True, alpha=0.3, lw=0.5)
ax_r.set_xticks([300, 500, 700])
ax_r.set_yticks([100, 150, 200, 250])
ax_r.tick_params(labelsize=8)

plt.savefig(OUT_PATH_B, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH_B}")
plt.close()

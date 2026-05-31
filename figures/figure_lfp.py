"""Figure: Linear Field Probes (4-panel summary) for OLMo-2."""
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

BASE_DIR   = Path(__file__).resolve().parent.parent
PROBE_PATH = BASE_DIR / "probes" / "olmo-2-0425-1b" / "probes" / "epoch100_biasFalse"
ACTIVS_DIR = BASE_DIR / "data" / "activations" / "olmo-2-0425-1b"
LAYER      = 15
OUT_PATH   = BASE_DIR / "figures" / "figure_LFP_olmo2.png"

# Transfer experiment (paper Fig 3D):
#   binary probe trained on mu={300,350} at TRANSFER_LAYER=0,
#   tested on shifted pairs {300+Δμ, 350+Δμ}
TRANSFER_LAYER  = 0
TRANSFER_PAIR   = (300, 350)
TRANSFER_DELTAS = [0, 50, 100, 150, 200]
DROP_FIRST      = 500   # skip first 500 com→num positions (equilibration)


def _parse_seq_idx(seq_id: str) -> int:
    return int(seq_id.split("_")[-1])


def _load_acts_split(mu: int, layer: int):
    """Return (train_acts, test_acts) for gaussian_m{mu}_s100 at given layer."""
    ds   = f"gaussian_m{mu}_s100_l1000_n10"
    site = f"model_layers_{layer}"
    files = sorted((ACTIVS_DIR / ds).glob(f"{site}_batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No activation files for {ds} layer {layer}")
    train_rows, test_rows = [], []
    for fp in files:
        payload  = torch.load(fp, map_location="cpu", weights_only=False)
        acts     = payload["activations"]
        lengths  = payload.get("lengths")
        seq_ids  = payload.get("sequence_ids", [])
        for i, sid in enumerate(seq_ids):
            length  = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            c2n     = acts[i, 2:length:2].float()
            if c2n.shape[0] > DROP_FIRST:
                c2n = c2n[DROP_FIRST:]
            if _parse_seq_idx(sid) < 8:
                train_rows.append(c2n)
            else:
                test_rows.append(c2n)
    return torch.cat(train_rows, dim=0), torch.cat(test_rows, dim=0)


def compute_transfer_accuracy():
    """Binary probe on TRANSFER_PAIR at TRANSFER_LAYER, tested at Δμ shifts."""
    print(f"Computing transfer accuracy (binary probe on mu={TRANSFER_PAIR}, layer={TRANSFER_LAYER})...")
    mu_a, mu_b = TRANSFER_PAIR
    tr_a, _ = _load_acts_split(mu_a, TRANSFER_LAYER)
    tr_b, _ = _load_acts_split(mu_b, TRANSFER_LAYER)

    X_tr = torch.cat([tr_a, tr_b], dim=0)
    y_tr = torch.cat([
        torch.zeros(len(tr_a), dtype=torch.long),
        torch.ones(len(tr_b), dtype=torch.long),
    ])

    probe = torch.nn.Linear(X_tr.shape[1], 2, bias=False)
    opt   = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=2048, shuffle=True,
    )
    for epoch in range(100):
        for bx, by in loader:
            opt.zero_grad()
            F.cross_entropy(probe(bx), by).backward()
            opt.step()
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1}/100")

    delta_mus, accuracies = [], []
    with torch.no_grad():
        for delta in TRANSFER_DELTAS:
            _, te_lo = _load_acts_split(mu_a + delta, TRANSFER_LAYER)
            _, te_hi = _load_acts_split(mu_b + delta, TRANSFER_LAYER)
            X_te = torch.cat([te_lo, te_hi], dim=0)
            y_te = torch.cat([
                torch.zeros(len(te_lo), dtype=torch.long),
                torch.ones(len(te_hi), dtype=torch.long),
            ])
            acc = (probe(X_te).argmax(dim=1) == y_te).float().mean().item()
            print(f"  Δμ={delta:3d}: {acc:.4f}")
            delta_mus.append(float(delta))
            accuracies.append(float(acc))
    return delta_mus, accuracies


def parse_mean(name: str) -> float:
    for pat in (r"mu[_=]?(\d+)", r"m(\d+)"):
        m = re.search(pat, name)
        if m:
            return float(m.group(1))
    digits = re.findall(r"\d+", name)
    if digits:
        return float(digits[0])
    raise ValueError(f"Cannot parse mean from '{name}'")


@torch.no_grad()
def normalize_rows(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return W / W.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def interpolate_kernel_gram(W, mu_vals, mu_star, lam=1e-3, eps=1e-12):
    Wn = normalize_rows(W, eps=eps)
    j = int(torch.searchsorted(mu_vals, torch.tensor(mu_star, device=mu_vals.device)).item())
    if j <= 0 or j >= mu_vals.numel():
        raise ValueError("mu_star must lie strictly between two values in mu_vals")
    i = j - 1
    mu_i, mu_j = float(mu_vals[i].item()), float(mu_vals[j].item())
    t = 0.0 if mu_j == mu_i else float((mu_star - mu_i) / (mu_j - mu_i))
    t = max(0.0, min(1.0, t))
    G = Wn @ Wn.T
    k_star = (1.0 - t) * G[i, :] + t * G[j, :]
    A = G + lam * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    alpha = torch.linalg.solve(A, k_star.to(dtype=A.dtype))
    w_hat = alpha @ Wn
    return w_hat / w_hat.norm().clamp_min(eps)


# ── Load all layer files ───────────────────────────────────────────────────────
layer_files = sorted(
    PROBE_PATH.glob("linear_probe_layer*.pt"),
    key=lambda p: int(p.stem.split("layer")[-1]),
)
if not layer_files:
    raise FileNotFoundError(f"No probe files found in {PROBE_PATH}")

layers, test_accs = [], []
for fp in layer_files:
    data = torch.load(fp, map_location="cpu", weights_only=False)
    layer_idx = int(data.get("layer", fp.stem.split("layer")[-1]))
    layers.append(layer_idx)
    test_accs.append(float(data["test_accuracy"]))

layer_to_file = {
    int(torch.load(fp, map_location="cpu", weights_only=False).get("layer", fp.stem.split("layer")[-1])): fp
    for fp in layer_files
}
if LAYER not in layer_to_file:
    LAYER = max(layer_to_file.keys())
    print(f"Layer {LAYER} not found; using layer={LAYER}")

probe_layer = torch.load(layer_to_file[LAYER], map_location="cpu", weights_only=False)

# ── Panel B: cosine similarity matrix ─────────────────────────────────────────
cosine_matrix = probe_layer["cosine_matrix"].cpu().numpy()
n_mu = cosine_matrix.shape[0]
mu_labels = [rf"$\mu_{i}$" for i in range(1, n_mu + 1)]

# ── Panel C: kernel-gram interpolation ────────────────────────────────────────
w = probe_layer["probe_state_dict"]["weight"].to(torch.float32)
train_datasets = probe_layer["train_datasets"]
mu_vals_full = torch.tensor([parse_mean(n) for n in train_datasets], dtype=w.dtype)
mu_vals_full, sort_idx = torch.sort(mu_vals_full)
W_full = w[sort_idx]
Wn_full = normalize_rows(W_full)

mu_targets = [350.0, 450.0, 550.0, 650.0]
cos_sims = []
for mu_star in mu_targets:
    idx_true = (mu_vals_full == mu_star).nonzero(as_tuple=True)[0]
    if idx_true.numel() == 0:
        raise ValueError(f"mu={mu_star} not found in train_datasets")
    w_true = Wn_full[idx_true[0]]
    mask = mu_vals_full != mu_star
    mu_interp = mu_vals_full[mask]
    W_interp = W_full[mask]
    w_hat = interpolate_kernel_gram(W_interp, mu_interp, mu_star, lam=1e-3)
    cos_sims.append(F.cosine_similarity(w_true, w_hat, dim=0).item())

# ── Panel D: transfer ─────────────────────────────────────────────────────────
delta_mu, accuracy = compute_transfer_accuracy()

# ── Plot ───────────────────────────────────────────────────────────────────────
cm = 1 / 2.54
fig, axes = plt.subplots(
    1, 4,
    figsize=(5.5 * 2.54 * cm, 1.8 * 2.54 * cm),
    gridspec_kw={"wspace": 0.36, "width_ratios": [1, 1.25, 1, 1]},
)
ax_acc, ax_cos, ax_bl, ax_br = axes
fig.subplots_adjust(left=0.08, right=0.965, bottom=0.33, top=0.80)

# A — separability
ax_acc.plot(layers, test_accs, marker=".")
ax_acc.set_xlabel("Layer", fontsize=7)
ax_acc.set_ylabel("test accuracy", fontsize=7)
min_acc = max(0.0, min(test_accs) - 0.05)
ax_acc.set_ylim(round(min_acc, 1), 1.02)
ax_acc.tick_params(labelsize=7)

# B — cosine similarity matrix
ax_cos = sns.heatmap(
    cosine_matrix, ax=ax_cos, cmap="coolwarm",
    vmin=-1, vmax=1, center=0, square=True, cbar=False,
)
ax_cos.set_anchor("N")
ax_cos.set_xticks([i + 0.5 for i in range(len(mu_labels))])
ax_cos.set_yticks([i + 0.5 for i in range(len(mu_labels))])
ax_cos.set_xticklabels(mu_labels, rotation=90, fontsize=7)
ax_cos.set_yticklabels(mu_labels, rotation=0, fontsize=7)
ax_cos.set_yticks([])
ax_cos.tick_params(labelsize=7)
divider = make_axes_locatable(ax_cos)
cax = divider.append_axes("left", size="5%", pad=0.04)
cb = fig.colorbar(ax_cos.collections[0], cax=cax, orientation="vertical", ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=7, pad=1)
cb.outline.set_visible(False)
cax.yaxis.set_ticks_position("left")

# C — interpolation
ax_bl.bar(range(len(mu_targets)), cos_sims, color="tab:blue")
ax_bl.tick_params(labelsize=7)
ax_bl.set_xticks(range(len(mu_targets)))
ax_bl.set_xticklabels([str(int(m)) for m in mu_targets], fontsize=7)
ax_bl.set_xlabel(r"$\mu$", fontsize=7)
ax_bl.set_ylim(0, 1.05)
ax_bl.set_ylabel("cosine similarity", fontsize=7, labelpad=1)

# D — transfer
ax_br.plot(delta_mu, accuracy, marker=".", linestyle="-", color="tab:orange")
ax_br.tick_params(labelsize=7)
ax_br.set_xticks(delta_mu)
ax_br.set_xlabel(r"$\Delta \mu$", fontsize=7)
ax_br.set_ylim(0.45, 1.0)
ax_br.set_ylabel("transfer accuracy", fontsize=7)
ax_br.yaxis.tick_left()
ax_br.yaxis.set_label_position("right")
ax_br.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

panel_labels = ["A - separability:", "B - continuity:", "C - interpolation:", "D - transfer:"]
x_offsets = [-0.4, -0.3, -0.3, 0.0]
for label, ax, x in zip(panel_labels, axes, x_offsets):
    ax.text(x, 1.2, label, transform=ax.transAxes, fontsize=10, va="bottom", ha="left")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
plt.show()

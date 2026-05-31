"""Figure 1 panels B/C/D — belief manifold for OLMo-2."""
from pathlib import Path
import sys
import torch
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from utils.inpca import inpca_embedding

MODEL_ALIAS  = "olmo-2-0425-1b"
ACTIVS_DIR   = BASE_DIR / "data" / "activations" / MODEL_ALIAS
LOGITS_DIR   = BASE_DIR / "data" / "logits"      / MODEL_ALIAS
OUT_PATH     = BASE_DIR / "figures" / "figure01_olmo2_layer15_mean.html"

DATASETS = [
    "gaussian_m300_s100_l1000_n10",
    "gaussian_m350_s100_l1000_n10",
    "gaussian_m400_s100_l1000_n10",
    "gaussian_m450_s100_l1000_n10",
    "gaussian_m500_s010_l1000_n10",
    "gaussian_m500_s020_l1000_n10",
    "gaussian_m500_s030_l1000_n10",
    "gaussian_m500_s050_l1000_n10",
    "gaussian_m500_s080_l1000_n10",
    "gaussian_m500_s120_l1000_n10",
    "gaussian_m500_s150_l1000_n10",
    "gaussian_m500_s200_l1000_n10",
    "gaussian_m500_s100_l1000_n10",
    "gaussian_m550_s100_l1000_n10",
    "gaussian_m600_s100_l1000_n10",
    "gaussian_m650_s100_l1000_n10",
    "gaussian_m700_s100_l1000_n10",
]

LAYER         = 15
DROP_FIRST    = 500
TEMP          = 1.0
N_COMPONENTS  = 16
PROB_SUBSET   = 8000
USE_MEAN_ACTS = True   # plot mean activation per dataset (17 pts) instead of all tokens
FIG_W_CM, FIG_H_CM = 7.7, 10


def cm_to_px(c):
    return c * 37.7952755906


def load_com2num_acts(dataset: str) -> torch.Tensor:
    site = f"model_layers_{LAYER}"
    files = sorted((ACTIVS_DIR / dataset).glob(f"{site}_batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No activation files for {dataset}")
    seqs = []
    for fp in files:
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        acts    = payload["activations"]
        lengths = payload.get("lengths")
        for i in range(acts.shape[0]):
            length  = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            com2num = acts[i, 2:length:2]
            if DROP_FIRST:
                com2num = com2num[DROP_FIRST:]
            seqs.append(com2num)
    return torch.cat(seqs, dim=0)


def load_com2num_probs(dataset: str, expected_labels=None):
    """Load next-number probabilities at comma positions (pos 1,3,5,...).

    At comma positions the model predicts the following number, so this gives
    the Gaussian belief distribution we want to visualise.  Number positions
    (2,4,6,...) predict the next comma and are useless here.
    """
    files = sorted((LOGITS_DIR / dataset).glob("logits_batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No logits files for {dataset}")
    seqs, token_labels = [], None
    for fp in files:
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        logits  = payload["logits"]
        labels  = payload.get("token_strings") or payload.get("token_labels")
        if token_labels is None:
            token_labels = labels
        lengths = payload.get("lengths")

        # Restrict to numeric tokens only before softmax so comma never wins
        numeric_mask = torch.tensor([lbl.isdigit() for lbl in token_labels], dtype=torch.bool)
        logits_num   = logits[..., numeric_mask]   # [batch, seq, n_numbers]
        probs_num    = torch.softmax(logits_num / TEMP, dim=-1)

        for i in range(probs_num.shape[0]):
            length = int(lengths[i].item()) if lengths is not None else probs_num.shape[1]
            # Comma positions: 1, 3, 5, ... — model predicts the *next* number here
            sl = probs_num[i, 1:length:2]
            if DROP_FIRST:
                sl = sl[DROP_FIRST:]
            seqs.append(sl)

    # Build a filtered label list (digits only, sorted numerically)
    num_labels = sorted([lbl for lbl in token_labels if lbl.isdigit()], key=lambda x: int(x))
    if expected_labels is not None and num_labels != expected_labels:
        raise ValueError(f"Token label mismatch in {dataset}")
    return torch.cat(seqs, dim=0), num_labels


def ds_mu(ds):  return int(ds.split("_")[1][1:])
def ds_sigma(ds): return int(ds.split("_")[2][1:])


# ── Load activations ──────────────────────────────────────────────────────────
print("Loading activations...")
all_act_list, act_mu_list, act_sigma_list = [], [], []
for ds in DATASETS:
    acts = load_com2num_acts(ds)
    if USE_MEAN_ACTS:
        acts = acts.mean(dim=0, keepdim=True)   # 1 mean vector per dataset
    n    = acts.shape[0]
    all_act_list.append(acts)
    act_mu_list.extend([ds_mu(ds)] * n)
    act_sigma_list.extend([ds_sigma(ds)] * n)

all_acts      = torch.cat(all_act_list, dim=0).float().numpy()
act_mu        = np.array(act_mu_list)
act_sigma     = np.array(act_sigma_list)

# ── Load logits ───────────────────────────────────────────────────────────────
print("Loading logits...")
all_prob_list, prob_mu_list, prob_sigma_list = [], [], []
label_order = None
for ds in DATASETS:
    probs, label_order = load_com2num_probs(ds, expected_labels=label_order)
    n = probs.shape[0]
    all_prob_list.append(probs)
    prob_mu_list.extend([ds_mu(ds)] * n)
    prob_sigma_list.extend([ds_sigma(ds)] * n)

all_probs   = torch.cat(all_prob_list, dim=0).float().numpy()
prob_mu     = np.array(prob_mu_list)
prob_sigma  = np.array(prob_sigma_list)

# ── PCA on activations ────────────────────────────────────────────────────────
print("Running PCA...")
pca        = PCA(n_components=max(3, N_COMPONENTS))
act_coords = pca.fit_transform(all_acts)[:, :3]

mask_mu    = (act_sigma == 100) & (act_mu != 500)
mask_sigma = ~mask_mu

# ── inPCA on softmax ──────────────────────────────────────────────────────────
print("Running inPCA...")
m          = all_probs.shape[0]
rng        = np.random.default_rng(0)
idx        = rng.choice(m, size=min(PROB_SUBSET, m), replace=False)
probs_sub  = all_probs[idx]
mu_sub     = prob_mu[idx]
sigma_sub  = prob_sigma[idx]
mask_mu_sub    = (sigma_sub == 100) & (mu_sub != 500)
mask_sigma_sub = ~mask_mu_sub
inpca_coords, _ = inpca_embedding(probs_sub, dim=3)

# ── PDF panel ─────────────────────────────────────────────────────────────────
# label_order is now a sorted list of digit strings: ['0','1',...,'999']
ord_lbl = label_order   # already sorted numerically by load_com2num_probs

target_mus  = {300, 400, 500, 600, 700}
line_traces, annotations, max_y = [], [], 0.0
for ds, probs in zip(DATASETS, all_prob_list):
    mu, s = ds_mu(ds), ds_sigma(ds)
    if s == 100 and mu in target_mus:
        t     = (mu - 100) / 800
        color = px.colors.sample_colorscale("Phase", t)[0]
        vec   = probs.mean(dim=0).numpy()   # already in sorted numeric order
        max_y = max(max_y, vec.max())
        line_traces.append(go.Scatter(
            x=list(range(len(ord_lbl))), y=vec,
            mode="lines", line=dict(color=color, width=1.2), showlegend=False,
        ))
        xpos = int(mu)   # label '300' is at index 300 in [0,1,...,999]
        annotations.append(dict(
            x=xpos, y=None, xref="x", yref="y",
            text=str(mu), showarrow=False, font=dict(color=color, size=10),
        ))
if max_y == 0:
    max_y = 1.0
for a in annotations:
    a["y"] = max_y * 1.05

# ── Assemble figure ───────────────────────────────────────────────────────────
print("Building figure...")
zoom   = 0.75
camera = dict(eye=dict(x=-1.0*zoom, y=1.6*zoom, z=0.8*zoom), up=dict(x=0, y=0, z=1))

fig = make_subplots(
    rows=3, cols=1,
    specs=[[{"type": "scene"}], [{"type": "scene"}], [{"type": "xy"}]],
    row_heights=[0.4, 0.4, 0.2],
    vertical_spacing=0.04,
)

# PCA traces
fig.add_trace(go.Scatter3d(
    x=act_coords[mask_mu, 0], y=-act_coords[mask_mu, 1], z=act_coords[mask_mu, 2],
    mode="markers",
    marker=dict(size=8 if USE_MEAN_ACTS else 0.5, color=act_mu[mask_mu], colorscale="Phase", cmin=100, cmax=900, showscale=False),
    showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=act_coords[mask_sigma, 0], y=act_coords[mask_sigma, 1], z=act_coords[mask_sigma, 2],
    mode="markers",
    marker=dict(size=8 if USE_MEAN_ACTS else 0.5, color=act_sigma[mask_sigma], colorscale="dense", cmin=-50, cmax=250, showscale=False),
    showlegend=False,
), row=1, col=1)

# inPCA traces
fig.add_trace(go.Scatter3d(
    x=inpca_coords[mask_mu_sub, 0], y=inpca_coords[mask_mu_sub, 1], z=-inpca_coords[mask_mu_sub, 2],
    mode="markers",
    marker=dict(size=1, color=mu_sub[mask_mu_sub], colorscale="Phase", cmin=100, cmax=900, showscale=False),
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Scatter3d(
    x=inpca_coords[mask_sigma_sub, 0], y=inpca_coords[mask_sigma_sub, 1], z=-inpca_coords[mask_sigma_sub, 2],
    mode="markers",
    marker=dict(size=1, color=sigma_sub[mask_sigma_sub], colorscale="dense", cmin=-50, cmax=250, showscale=False),
    showlegend=False,
), row=2, col=1)

for tr in line_traces:
    fig.add_trace(tr, row=3, col=1)

tick_labels = ["0", "300", "400", "500", "600", "700", "999"]
tick_vals   = [int(lbl) for lbl in tick_labels]   # index == int value for sorted [0..999]

axis_style = dict(
    showbackground=True, backgroundcolor="white", gridcolor="#dcdcdc",
    zerolinecolor="#dcdcdc", title="", showticklabels=False, showgrid=True, zeroline=False, ticks="",
)
fig.update_layout(
    width=cm_to_px(FIG_W_CM), height=cm_to_px(FIG_H_CM) + 100,
    showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style, camera=camera),
    scene2=dict(xaxis=axis_style, yaxis=axis_style, zaxis=axis_style, camera=camera),
    plot_bgcolor="white",
)
fig.update_xaxes(
    row=3, col=1,
    tickmode="array", tickvals=tick_vals, ticktext=tick_labels,
    range=[-0.5, len(ord_lbl) - 0.5],
    showgrid=True, gridcolor="black", zeroline=False,
    showline=True, linecolor="black", ticks="outside", tickfont=dict(size=10),
)
fig.update_yaxes(
    row=3, col=1,
    showticklabels=False, showgrid=False, zeroline=False,
    showline=True, linecolor="black", ticks="outside",
)
for text, yref_frac in [("B", 1.0), ("C", 0.5), ("D", 0.1)]:
    fig.add_annotation(
        x=-0.1, y=yref_frac, xref="paper", yref="paper",
        text=text, showarrow=False, font=dict(size=14, color="black"), align="left",
    )
for ann in annotations:
    fig.add_annotation(**ann)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUT_PATH))
print(f"Saved: {OUT_PATH}")
fig.show()

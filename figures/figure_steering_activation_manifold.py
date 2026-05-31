"""Figure: steering on the activation manifold — Panel A (PCA 3D) and Panel B (mean/std plane)."""
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM
import matplotlib.cm as cm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_ALIAS = "olmo-2-0425-1b"
LOGITS_DIR = BASE_DIR / "data" / "logits" / MODEL_ALIAS
ACTIVS_DIR = BASE_DIR / "data" / "activations" / MODEL_ALIAS

# --- config ---
DATASETS = [
    "gaussian_m300_s100_l1000_n10",
    "gaussian_m350_s100_l1000_n10",
    "gaussian_m400_s100_l1000_n10",
    "gaussian_m450_s100_l1000_n10",
    "gaussian_m500_s100_l1000_n10",
    "gaussian_m550_s100_l1000_n10",
    "gaussian_m600_s100_l1000_n10",
    "gaussian_m650_s100_l1000_n10",
    "gaussian_m700_s100_l1000_n10",
]
ORDER = ["m300", "m350", "m400", "m450", "m500", "m550", "m600", "m650", "m700"]

TEMP = 1.0
DROP_FIRST = 500
PROB_SUBSET_SIZE = 80000

PCA_LAYER = 14         # layer index for activation PCA (panel A)
STEER_DATASET = "gaussian_m300_s100_l1000_n10"
STEER_LAYER = 15       # layer index for steering (panel B)
STEER_ALPHAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
DROP_FIRST_STEER = 500
MODEL_NAME = "allenai/OLMo-2-0425-1B"

ACT3D_SUBSET = 10000
EPS = 1e-12

FIGURES_DIR = Path(__file__).resolve().parent
OUT_PATH_A = FIGURES_DIR / f"figure_steering_A_{MODEL_ALIAS}.html"
OUT_PATH_B = FIGURES_DIR / f"figure_steering_B_{MODEL_ALIAS}.html"


# --- data loading helpers ---

def load_com2num_probs(dataset: str, expected_labels=None):
    logit_dir = LOGITS_DIR / dataset
    files = sorted(logit_dir.glob("logits_batch*.pt"))
    if not files:
        raise FileNotFoundError(f"No logits files found in {logit_dir}")
    logits_list, token_labels = [], None
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        logits_list.append(payload["logits"])
        labels = payload.get("token_strings") or payload.get("token_labels")
        if token_labels is None:
            token_labels = labels
        elif labels != token_labels:
            raise ValueError(f"Token label order mismatch in {fpath}")
    logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits / TEMP, dim=-1)
    if expected_labels is not None and token_labels != expected_labels:
        raise ValueError(f"Token label order mismatch in {dataset}")
    com_to_num_idx = range(1, probs.shape[1], 2)  # odd positions: commas predicting numbers (no BOS)
    seq_vectors = []
    for seq_probs in probs:
        sl = seq_probs[com_to_num_idx]
        if DROP_FIRST:
            sl = sl[DROP_FIRST:]
        seq_vectors.append(sl)
    if not seq_vectors:
        raise ValueError(f"No probabilities found for {dataset}")
    return torch.cat(seq_vectors, dim=0), token_labels


def load_layer_acts_index(dataset_name: str, layer_idx: int):
    ds_dir = ACTIVS_DIR / dataset_name
    pattern = f"model_layers_{layer_idx}_batch*.pt"
    files = sorted(ds_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activation files for {dataset_name}: {ds_dir / pattern}")
    out = {}
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        acts = payload["activations"]
        lengths = payload.get("lengths")
        seq_ids = payload.get("sequence_ids")
        if seq_ids is None:
            raise ValueError(f"Missing sequence_ids in {fpath}")
        for i, seq_id in enumerate(seq_ids):
            length = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            out[seq_id] = acts[i, :length].clone()
    return out


def load_com2num_acts_for_pca(dataset_name: str, layer_idx: int):
    ds_dir = ACTIVS_DIR / dataset_name
    pattern = f"model_layers_{layer_idx}_batch*.pt"
    files = sorted(ds_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No activation files for {dataset_name}: {ds_dir / pattern}")
    seq_tensors = []
    for fpath in files:
        payload = torch.load(fpath, map_location="cpu")
        acts = payload["activations"]
        lengths = payload.get("lengths")
        for i in range(acts.shape[0]):
            length = int(lengths[i].item()) if lengths is not None else acts.shape[1]
            com2num = acts[i, 1:length:2]
            if DROP_FIRST and com2num.size(0) > DROP_FIRST:
                com2num = com2num[DROP_FIRST:]
            if com2num.size(0):
                seq_tensors.append(com2num)
    if not seq_tensors:
        raise ValueError(f"No activations for {dataset_name} after slicing.")
    return torch.cat(seq_tensors, dim=0)


def compute_centroids(layer_idx: int):
    cents = {}
    for ds in DATASETS:
        acts_index = load_layer_acts_index(ds, layer_idx)
        rows = []
        for acts in acts_index.values():
            com2num = acts[1::2]
            if DROP_FIRST and com2num.size(0) > DROP_FIRST:
                com2num = com2num[DROP_FIRST:]
            rows.append(com2num)
        if rows:
            cents[ds] = torch.cat(rows, dim=0).mean(dim=0)
    centroid_matrix = torch.stack([cents[d] for d in DATASETS], dim=0)
    return cents, centroid_matrix


def forward_from_layer(model, hidden: torch.Tensor, layer_idx: int):
    attention_mask = torch.ones(hidden.size(0), hidden.size(1), device=hidden.device)
    position_ids = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
    h = hidden
    for i in range(layer_idx + 1, model.model.config.num_hidden_layers):
        layer = model.model.layers[i]
        h = layer(
            hidden_states=h,
            attention_mask=model.model._update_causal_mask(attention_mask, h, i),
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
        )[0]
    h = model.model.norm(h)
    return model.lm_head(h)


def run_steering_probs(dataset_name: str, layer_idx: int, alphas: list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    model.to(device)

    centroids, centroid_matrix = compute_centroids(layer_idx)
    c300 = centroids["gaussian_m300_s100_l1000_n10"].to(device)
    c700 = centroids["gaussian_m700_s100_l1000_n10"].to(device)
    vec_300_700 = c700 - c300

    def spline_target(alpha: float):
        u = alpha * (len(centroid_matrix) - 1)
        i = int(np.clip(np.floor(u), 0, len(centroid_matrix) - 2))
        f = u - i
        return (1 - f) * centroid_matrix[i].to(device) + f * centroid_matrix[i + 1].to(device)

    ref_dir = LOGITS_DIR / dataset_name
    ref_file = sorted(ref_dir.glob("logits_batch*.pt"))[0]
    ref_payload = torch.load(ref_file, map_location="cpu")
    token_labels = ref_payload.get("token_strings") or ref_payload.get("token_labels")
    token_ids = ref_payload["token_ids"]
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

    acts_index = load_layer_acts_index(dataset_name, layer_idx)
    seq_ids = sorted(acts_index.keys())

    probs_rows, tags, alpha_list = [], [], []

    for alpha in alphas:
        for seq_id in seq_ids:
            base = acts_index[seq_id].to(device).unsqueeze(0)
            steered = base + vec_300_700.view(1, 1, -1) * alpha
            with torch.no_grad():
                logits = forward_from_layer(model, steered, layer_idx)
            subset_logits = logits.index_select(dim=-1, index=token_ids_tensor)
            p = torch.softmax(subset_logits / TEMP, dim=-1)
            com2num = p[0, 1::2]
            if DROP_FIRST_STEER:
                com2num = com2num[DROP_FIRST_STEER:]
            if com2num.size(0) == 0:
                continue
            probs_rows.append(com2num.cpu())
            tags.extend(["vec"] * com2num.size(0))
            alpha_list.extend([alpha] * com2num.size(0))

    for alpha in alphas:
        target = spline_target(alpha)
        dir_spline = target - c300
        for seq_id in seq_ids:
            base = acts_index[seq_id].to(device).unsqueeze(0)
            steered = base + dir_spline.view(1, 1, -1) * alpha
            with torch.no_grad():
                logits = forward_from_layer(model, steered, layer_idx)
            subset_logits = logits.index_select(dim=-1, index=token_ids_tensor)
            p = torch.softmax(subset_logits / TEMP, dim=-1)
            com2num = p[0, 1::2]
            if DROP_FIRST_STEER:
                com2num = com2num[DROP_FIRST_STEER:]
            if com2num.size(0) == 0:
                continue
            probs_rows.append(com2num.cpu())
            tags.extend(["spline"] * com2num.size(0))
            alpha_list.extend([alpha] * com2num.size(0))

    if not probs_rows:
        raise ValueError("No steered probabilities collected.")
    return torch.cat(probs_rows, dim=0), tags, alpha_list, token_labels


# --- load reference logits and steered probs ---
dataset_labels, all_prob_list, label_order = [], [], None
for ds in DATASETS:
    probs, label_order = load_com2num_probs(ds, expected_labels=label_order)
    all_prob_list.append(probs)
    dataset_labels.extend([ds.split("_")[1]] * probs.shape[0])

probabilities_reference = torch.cat(all_prob_list, dim=0).cpu().numpy()

probabilities_steering, steer_tags, steer_alphas, steer_token_labels = run_steering_probs(
    STEER_DATASET, STEER_LAYER, STEER_ALPHAS
)

probabilities_all = np.vstack([probabilities_reference, probabilities_steering.detach().numpy()])
labels_all = np.array(list(dataset_labels) + list(steer_tags))
alphas_all = np.array([np.nan] * len(dataset_labels) + list(steer_alphas), dtype=float)

# subsample
m = probabilities_all.shape[0]
rng = np.random.default_rng(0)
subset_idx = rng.choice(m, size=min(PROB_SUBSET_SIZE, m), replace=False)
probabilities_all_subset = probabilities_all[subset_idx]
labels_raw_subset = labels_all[subset_idx]
alphas_all_subset = alphas_all[subset_idx]

label_map = {"spline": "steering along manifold", "vec": "steering along vector"}
labels_with_steer = np.array([label_map.get(lab, lab) for lab in labels_raw_subset])


# =========================================================
# Panel A: 3D PCA of activations
# =========================================================
acts_list, labels_list = [], []
for ds in DATASETS:
    acts = load_com2num_acts_for_pca(ds, PCA_LAYER).cpu().numpy()
    acts_list.append(acts)
    labels_list.extend([ds.split("_")[1]] * acts.shape[0])

acts_all = np.vstack(acts_list)
labels_act_all = np.array(labels_list)

sel = rng.choice(len(acts_all), size=min(ACT3D_SUBSET, len(acts_all)), replace=False)
acts_sample = acts_all[sel]
labels_sample = labels_act_all[sel]

pca3 = PCA(n_components=6, random_state=0)
coords3d = pca3.fit_transform(acts_sample)

palette = cm.get_cmap("tab10")
cmap = {
    lbl: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)"
    for lbl, (r, g, b, _) in zip(ORDER, [palette(i % 10) for i in range(len(ORDER))])
}

traces_a = []
for lbl in ORDER:
    mask = labels_sample == lbl
    if not mask.any():
        continue
    mu_val = float(lbl[1:])
    traces_a.append(go.Scatter3d(
        x=coords3d[mask, 0], y=coords3d[mask, 1], z=coords3d[mask, 3],
        mode="markers",
        marker=dict(size=1, color=[mu_val] * int(mask.sum()), colorscale="Phase",
                    cmin=100, cmax=900, opacity=0.8, showscale=False),
        name=lbl, showlegend=False,
        hovertemplate=f"{lbl}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}<extra></extra>",
    ))

fig_a = go.Figure(data=traces_a)

# Centroid line m300→m700
centroids_2d = {}
for mu in ["m300", "m700"]:
    mask = labels_sample == mu
    if mask.any():
        centroids_2d[mu] = coords3d[mask].mean(axis=0)

if all(k in centroids_2d for k in ("m300", "m700")):
    start, end = centroids_2d["m300"], centroids_2d["m700"]
    mid = 0.5 * (start + end)
    vec = end - start
    fig_a.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines", line=dict(color="#fa5102", width=3),
        showlegend=False, hoverinfo="skip",
    ))
    fig_a.add_trace(go.Cone(
        x=[mid[0]], y=[mid[1]], z=[mid[2]],
        u=[vec[0]], v=[vec[1]], w=[vec[2]],
        colorscale=[[0, "orange"], [1, "orange"]], showscale=False,
        sizemode="absolute", sizeref=1.0, anchor="tip",
        name="", hoverinfo="skip",
    ))

# Spline through all centroids
centroid_seq = [(lbl, coords3d[labels_sample == lbl].mean(axis=0))
                for lbl in ORDER if (labels_sample == lbl).any()]
if len(centroid_seq) >= 2:
    t = np.arange(len(centroid_seq))
    cent = np.stack([c[1] for c in centroid_seq])
    tt = np.linspace(t[0], t[-1], 200)
    cs_x = CubicSpline(t, cent[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, cent[:, 1], bc_type="natural")
    cs_z = CubicSpline(t, cent[:, 2], bc_type="natural")
    fig_a.add_trace(go.Scatter3d(
        x=cs_x(tt), y=cs_y(tt), z=cs_z(tt),
        mode="lines", line=dict(color="#ffc004", width=5),
        showlegend=False, hoverinfo="skip",
    ))

fig_a.update_layout(
    width=290, height=150,
    margin=dict(l=10, r=10, t=0, b=0),
    showlegend=False,
    scene=dict(
        xaxis=dict(title="", showticklabels=False, showgrid=True, zeroline=False, ticks="",
                   showbackground=True, backgroundcolor="white", gridcolor="#dcdcdc", zerolinecolor="#dcdcdc"),
        yaxis=dict(title="", showticklabels=False, showgrid=True, zeroline=False, ticks="",
                   showbackground=True, backgroundcolor="white", gridcolor="#dcdcdc", zerolinecolor="#dcdcdc"),
        zaxis=dict(title="", showticklabels=False, showgrid=True, zeroline=False, ticks="",
                   showbackground=True, backgroundcolor="white", gridcolor="#dcdcdc", zerolinecolor="#dcdcdc"),
        aspectmode="manual",
        aspectratio=dict(x=2, y=1.0, z=0.8),
        domain=dict(x=[0, 1], y=[0, 1]),
    ),
    scene_camera=dict(eye=dict(x=-0.2, y=-1.6, z=0.6)),
)
fig_a.add_annotation(
    x=0.0, y=1.0, xref="paper", yref="paper",
    text="A", showarrow=False, font=dict(size=12, color="black"), align="left",
)

fig_a.write_html(str(OUT_PATH_A), include_plotlyjs="cdn", full_html=True)
print(f"saved {OUT_PATH_A}")


# =========================================================
# Panel B: mean/std plane
# =========================================================
base_dir = LOGITS_DIR / DATASETS[0]
base_file = sorted(base_dir.glob("logits_batch*.pt"))[0]
base_payload = torch.load(base_file, map_location="cpu")
token_labels_base = base_payload.get("token_strings") or base_payload.get("token_labels")

labels_list_b = list(token_labels_base)
numeric_pairs = [(i, int(lbl)) for i, lbl in enumerate(labels_list_b) if lbl.isdigit()]
if not numeric_pairs:
    raise ValueError("No numeric tokens found in token_labels_base.")
numeric_pairs.sort(key=lambda x: x[1])
num_indices = np.array([i for i, _ in numeric_pairs], dtype=int)
num_values = np.array([val for _, val in numeric_pairs], dtype=float)


def mean_std_over_numeric(probs_row: np.ndarray):
    p = probs_row[num_indices]
    s = p.sum()
    if s == 0:
        return np.nan, np.nan
    p = p / s
    mean = (p * num_values).sum()
    diff = num_values - mean
    var = (p * diff * diff).sum()
    return mean, np.sqrt(var)


means, stds = [], []
for row in probabilities_all_subset:
    mn, sd = mean_std_over_numeric(row)
    means.append(mn)
    stds.append(sd)
means, stds = np.array(means), np.array(stds)

base_mask = (labels_with_steer != label_map.get("vec", "vec")) & (
    labels_with_steer != label_map.get("spline", "spline")
)

fig_b = px.scatter(
    x=means[base_mask],
    y=stds[base_mask],
    color=means[base_mask],
    color_continuous_scale="Phase",
    range_color=(100, 900),
    opacity=0.4,
    labels={"x": "mean", "y": "std", "color": "mean"},
    hover_data={"label": labels_with_steer[base_mask]},
    render_mode="svg",
)
fig_b.update_traces(marker=dict(size=2))

steer_mask = ~np.isnan(alphas_all_subset)
traj_points = []
for tag in ["vec", "spline"]:
    tag_mask = steer_mask & (labels_raw_subset == tag)
    if not tag_mask.any():
        continue
    for alpha in np.unique(alphas_all_subset[tag_mask]):
        alpha_mask = tag_mask & np.isclose(alphas_all_subset, alpha)
        traj_points.append({
            "tag": tag, "alpha": alpha,
            "mean": float(np.mean(means[alpha_mask])),
            "std": float(np.mean(stds[alpha_mask])),
        })

traj_colors = {"vec": "#fa5102", "spline": "#ffc004"}

for tag in ["vec", "spline"]:
    pts = sorted([p for p in traj_points if p["tag"] == tag], key=lambda x: x["alpha"])
    if not pts:
        continue
    name = f"{label_map.get(tag, tag)} trajectory"
    fig_b.add_trace(go.Scatter(
        x=[p["mean"] for p in pts],
        y=[p["std"] for p in pts],
        mode="lines+markers",
        name=name,
        line=dict(shape="spline", width=3, color=traj_colors.get(tag)),
        marker=dict(size=6, line=dict(width=1, color="white")),
        customdata=np.array([[p["alpha"]] for p in pts]),
        hovertemplate="alpha=%{customdata[0]:.1f}<br>mean=%{x:.3f}<br>std=%{y:.3f}<extra></extra>",
    ))

highlight_vec_name = f"{label_map.get('vec', 'vec')} alpha=0.5"
highlight_spline_name = f"{label_map.get('spline', 'spline')} alpha=0.6"

highlight_vec_mask = (labels_raw_subset == "vec") & np.isclose(alphas_all_subset, 0.5)
if highlight_vec_mask.any():
    fig_b.add_trace(go.Scatter(
        x=means[highlight_vec_mask], y=stds[highlight_vec_mask],
        mode="markers", name=highlight_vec_name,
        marker=dict(size=2, color="#fa5102"), opacity=1.0,
    ))

highlight_spline_mask = (labels_raw_subset == "spline") & np.isclose(alphas_all_subset, 0.6)
if highlight_spline_mask.any():
    fig_b.add_trace(go.Scatter(
        x=means[highlight_spline_mask], y=stds[highlight_spline_mask],
        mode="markers", name=highlight_spline_name,
        marker=dict(size=2, color="#ffc004"), opacity=1.0,
    ))

traj_names = {
    f"{label_map.get('vec', 'vec')} trajectory",
    f"{label_map.get('spline', 'spline')} trajectory",
}
traces_all = list(fig_b.data)
traj_traces = [t for t in traces_all if t.name in traj_names]
highlight_traces = [t for t in traces_all if t.name in {highlight_vec_name, highlight_spline_name}]
base_traces = [t for t in traces_all if t not in traj_traces and t not in highlight_traces]
fig_b.data = tuple(base_traces + traj_traces + highlight_traces)

fig_b.update_xaxes(range=[250, 750], tickvals=[300, 500, 700], ticktext=["300", "500", "700"],
                   tickfont=dict(size=10), title_text="softmax mean", title_font=dict(size=10))
fig_b.update_yaxes(range=[80, 205], tickvals=[100, 150, 200], ticktext=["100", "150", "200"],
                   tickfont=dict(size=10), title_text="softmax std", title_font=dict(size=10))

DPI = 96
PX_PER_CM = DPI / 2.54

fig_b.update_coloraxes(showscale=False)
fig_b.update_layout(
    showlegend=False,
    width=int(7.7 * PX_PER_CM),
    height=int(7.0 * PX_PER_CM),
    margin=dict(
        l=int(0.2 * PX_PER_CM), r=int(0.2 * PX_PER_CM),
        t=int(0.2 * PX_PER_CM), b=int(0.2 * PX_PER_CM),
    ),
)
fig_b.add_annotation(
    x=-0.2, y=1.0, xref="paper", yref="paper",
    text="B", showarrow=False, font=dict(size=12, color="black"), align="left",
)

fig_b.write_html(str(OUT_PATH_B), include_plotlyjs="cdn", full_html=True)
print(f"saved {OUT_PATH_B}")

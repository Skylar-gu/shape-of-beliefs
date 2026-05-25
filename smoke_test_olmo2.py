"""
Smoke tests for OLMo-2 before running the full pipeline.
Run from repo root on Colab after setup:
    python smoke_test_olmo2.py

Checks:
  1. CUDA availability
  2. Model + tokenizer loading
  3. Layer hook paths exist
  4. Tiny forward pass (2 seqs x 100 numbers)
  5. Output files land in model-namespaced path
  6. Equilibration timing (to calibrate number_start_index)
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID     = "allenai/OLMo-2-0425-1B"
TOKEN_SUBSET = "token_subset/olmo-2-0425-1B_number_tokens.json"
BASE_DIR     = Path(__file__).resolve().parent

PASS = "✓"
FAIL = "✗"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ---------------------------------------------------------------------------
# 1. CUDA
# ---------------------------------------------------------------------------
section("1. CUDA")
device = "cuda" if torch.cuda.is_available() else "cpu"
icon = PASS if device == "cuda" else FAIL
print(f"  {icon} device = {device}")
if device == "cpu":
    print("  WARNING: running on CPU — forward passes will be slow.")

# ---------------------------------------------------------------------------
# 2. Model + tokenizer loading
# ---------------------------------------------------------------------------
section("2. Model + tokenizer loading")
print(f"  Loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.to(device)
model.eval()
print(f"  {PASS} Loaded. Hidden dim: {model.config.hidden_size}  Layers: {model.config.num_hidden_layers}")

# ---------------------------------------------------------------------------
# 3. Layer hook paths
# ---------------------------------------------------------------------------
section("3. Layer hook paths")
sites = ["model.embed_tokens", *[f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]]
all_ok = True
for site in sites:
    try:
        model.get_submodule(site)
    except AttributeError:
        print(f"  {FAIL} Missing: {site}")
        all_ok = False
if all_ok:
    print(f"  {PASS} All {len(sites)} hook sites found (embed_tokens + {model.config.num_hidden_layers} layers)")

# ---------------------------------------------------------------------------
# 4. Tiny forward pass — 2 sequences of 100 numbers from N(500, 100)
# ---------------------------------------------------------------------------
section("4. Tiny forward pass")
rng = np.random.default_rng(42)

def make_sequence(mean, std, length, rng):
    nums = rng.normal(mean, std, length)
    nums = np.clip(np.round(nums), 0, 999).astype(int)
    return ",".join(str(n) for n in nums)

seqs = [make_sequence(500, 100, 100, rng) for _ in range(2)]
enc = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=3000)
input_ids     = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)
print(f"  Input shape: {input_ids.shape}  (batch=2, seq_len={input_ids.shape[1]})")

captures = {}
hooks = []

def make_hook(name):
    def hook(_m, _i, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        captures[name] = output.detach().cpu()
    return hook

for site in sites:
    hooks.append(model.get_submodule(site).register_forward_hook(make_hook(site)))

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

for h in hooks:
    h.remove()

print(f"  {PASS} Forward pass complete")
for site in sites[:3]:
    print(f"    {site}: {captures[site].shape}")
print(f"    ...")

# ---------------------------------------------------------------------------
# 5. Output path uses model alias
# ---------------------------------------------------------------------------
section("5. Output path naming")
model_alias = MODEL_ID.split("/")[-1].lower()
out_dir = BASE_DIR / "data" / "activations" / model_alias / "smoke_test"
out_dir.mkdir(parents=True, exist_ok=True)
test_file = out_dir / "test.pt"
torch.save({"ok": True}, test_file)
print(f"  {PASS} Saved to: {test_file.relative_to(BASE_DIR)}")
test_file.unlink()
out_dir.rmdir()

# ---------------------------------------------------------------------------
# 6. Equilibration timing
# ---------------------------------------------------------------------------
section("6. Equilibration timing (calibrates number_start_index)")

token_map = json.loads((BASE_DIR / TOKEN_SUBSET).read_text())
token_ids = torch.tensor(
    [v for k, v in sorted(token_map.items(), key=lambda kv: kv[1]) if k.isdigit()],
    dtype=torch.long,
)

# Use a longer sequence for this check
long_seq  = make_sequence(500, 100, 500, rng)
enc_long  = tokenizer([long_seq], return_tensors="pt", truncation=True, max_length=13000)
input_ids_long = enc_long["input_ids"].to(device)
attn_long      = enc_long["attention_mask"].to(device)

with torch.no_grad():
    out_long = model(input_ids=input_ids_long, attention_mask=attn_long)

logits = out_long.logits[0].cpu()                          # [seq_len, vocab]
num_logits = logits.index_select(1, token_ids)             # [seq_len, n_numbers]
probs = torch.softmax(num_logits, dim=-1)                  # [seq_len, n_numbers]

# Find comma token id to locate com2num positions
comma_id = token_map.get(",")
input_ids_cpu = input_ids_long[0].cpu().tolist()
com2num_positions = [i + 1 for i, t in enumerate(input_ids_cpu[:-1]) if t == comma_id]

if len(com2num_positions) < 10:
    print("  WARNING: fewer than 10 com2num positions found — comma token ID may differ.")
else:
    # KL divergence from the final distribution (proxy for convergence)
    final_dist = probs[com2num_positions[-1]]
    kl_divs = []
    for pos in com2num_positions:
        p = probs[pos] + 1e-10
        q = final_dist + 1e-10
        kl = (p * (p / q).log()).sum().item()
        kl_divs.append(kl)

    plt.figure(figsize=(9, 4))
    plt.plot(kl_divs, linewidth=1.5)
    plt.axvline(500, color="red", linestyle="--", label="current number_start_index=500")
    plt.xlabel("com2num position index")
    plt.ylabel("KL divergence from final distribution")
    plt.title("OLMo-2 equilibration — lower = converged")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "smoke_test_equilibration.png", dpi=120)
    plt.show()

    # Suggest a cutoff: first position where KL stays below 5% of initial
    threshold = 0.05 * kl_divs[0]
    suggested = next((i for i, k in enumerate(kl_divs) if k < threshold), None)
    print(f"  {PASS} com2num positions found: {len(com2num_positions)}")
    print(f"  Suggested number_start_index: {suggested}  (current default: 500)")
    print(f"  Plot saved to smoke_test_equilibration.png")

# ---------------------------------------------------------------------------
section("Summary")
print("  If all checks passed, OLMo-2 is ready for the full pipeline.")
print("  Run with:")
print(f"    MODEL='{MODEL_ID}' TOKEN_SUBSET='olmo-2-0425-1B_number_tokens.json' ./scripts/generate_all.sh")

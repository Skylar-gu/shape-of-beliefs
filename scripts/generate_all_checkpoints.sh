#!/usr/bin/env bash
set -euo pipefail

# Run the full activation pipeline across OLMo-2-0425-1B training checkpoints.
# Run from repo root:
#   chmod +x scripts/generate_all_checkpoints.sh
#   ./scripts/generate_all_checkpoints.sh
#
# Outputs land in:
#   data/activations/olmo-2-0425-1b/<revision>/<dataset>/
#
# Stage 1 checkpoints live on the main repo (allenai/OLMo-2-0425-1B) as branches.
# Stage 1 fine-grained early training (1K-step resolution) lives on a separate repo:
#   allenai/OLMo-2-0425-1B-early-training  (verify names before using)
#
# To re-verify all available revisions:
#   python -c "
#   from huggingface_hub import list_repo_refs
#   refs = list_repo_refs('allenai/OLMo-2-0425-1B')
#   for b in sorted(refs.branches, key=lambda x: x.name): print(b.name)
#   "

MODEL="allenai/OLMo-2-0425-1B"
TOKEN_SUBSET="olmo-2-0425-1B_number_tokens.json"

# ---------------------------------------------------------------------------
# Stage 1 — sampled checkpoints from the main repo
# Full list goes step0 → step1907359 (4001B tokens), every 10K steps.
# We sample: every 10K for the first 100K (early dynamics), then every 100K.
# ---------------------------------------------------------------------------
STAGE1_REVISIONS=(
  "stage1-step0-tokens0B"
  "stage1-step300-tokens1B"
  "stage1-step10000-tokens21B"
  "stage1-step20000-tokens42B"
  "stage1-step30000-tokens63B"
  "stage1-step40000-tokens84B"
  "stage1-step50000-tokens105B"
  "stage1-step60000-tokens126B"
  "stage1-step70000-tokens147B"
  "stage1-step80000-tokens168B"
  "stage1-step90000-tokens189B"
  "stage1-step100000-tokens210B"
  "stage1-step200000-tokens420B"
  "stage1-step300000-tokens630B"
  "stage1-step400000-tokens839B"
  "stage1-step500000-tokens1049B"
  "stage1-step600000-tokens1259B"
  "stage1-step700000-tokens1469B"
  "stage1-step800000-tokens1678B"
  "stage1-step900000-tokens1888B"
  "stage1-step1000000-tokens2098B"
  "stage1-step1907359-tokens4001B"
)

# ---------------------------------------------------------------------------
# Stage 2 — ingredient 3 only (the one merged into the final model)
# Ingredients 1 and 2 are exploratory runs; add them here if needed.
# ---------------------------------------------------------------------------
STAGE2_REVISIONS=(
  "stage2-ingredient3-step1000-tokens3B"
  "stage2-ingredient3-step2000-tokens5B"
  "stage2-ingredient3-step3000-tokens7B"
  "stage2-ingredient3-step4000-tokens9B"
  "stage2-ingredient3-step5000-tokens11B"
  "stage2-ingredient3-step6000-tokens13B"
  "stage2-ingredient3-step7000-tokens15B"
  "stage2-ingredient3-step8000-tokens17B"
  "stage2-ingredient3-step9000-tokens19B"
  "stage2-ingredient3-step10000-tokens21B"
  "stage2-ingredient3-step11000-tokens24B"
  "stage2-ingredient3-step12000-tokens26B"
  "stage2-ingredient3-step13000-tokens28B"
  "stage2-ingredient3-step14000-tokens30B"
  "stage2-ingredient3-step15000-tokens32B"
  "stage2-ingredient3-step16000-tokens34B"
  "stage2-ingredient3-step17000-tokens36B"
  "stage2-ingredient3-step18000-tokens38B"
  "stage2-ingredient3-step19000-tokens40B"
  "stage2-ingredient3-step20000-tokens42B"
  "stage2-ingredient3-step21000-tokens45B"
  "stage2-ingredient3-step22000-tokens47B"
  "stage2-ingredient3-step23000-tokens49B"
  "stage2-ingredient3-step23852-tokens51B"
)

# Two datasets: m300 and m700 (extremes). Combined is used for dynamics analysis.
# To run a single dataset only, set DATASETS=("gaussian_m500_s100_l1000_n10") and remove COMBINED_DATASET.
DATASETS=(
  "gaussian_m300_s100_l1000_n10"
  "gaussian_m700_s100_l1000_n10"
)
COMBINED_DATASET="gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10"

# Sequences only need to be generated once (independent of model)
echo "=== Generating sequences (once) ==="
for ds in "${DATASETS[@]}"; do
  if [[ "$ds" =~ ^gaussian_m([0-9]+)_s([0-9]+)_l([0-9]+)_n([0-9]+)$ ]]; then
    uv run python generate_sequences.py \
      --num-seq "${BASH_REMATCH[4]}" \
      --len-seq "${BASH_REMATCH[3]}" \
      --mean   "${BASH_REMATCH[1]}" \
      --std    "${BASH_REMATCH[2]}"
  fi
done

# ---------------------------------------------------------------------------
run_checkpoint() {
  local model="$1"
  local rev="$2"
  echo ""
  echo "--- $model @ $rev ---"
  for ds in "${DATASETS[@]}"; do
    uv run python sequences_to_activations.py \
      --dataset-name "$ds" \
      --model        "$model" \
      --token-subset "$TOKEN_SUBSET" \
      --revision     "$rev"
  done
  uv run python sequences_to_activations.py \
    --dataset-name "$COMBINED_DATASET" \
    --model        "$model" \
    --token-subset "$TOKEN_SUBSET" \
    --revision     "$rev"
}

echo ""
echo "=== Stage 1 (${#STAGE1_REVISIONS[@]} checkpoints) ==="
for rev in "${STAGE1_REVISIONS[@]}"; do
  run_checkpoint "$MODEL" "$rev"
done

echo ""
echo "=== Stage 2 ingredient 3 (${#STAGE2_REVISIONS[@]} checkpoints) ==="
for rev in "${STAGE2_REVISIONS[@]}"; do
  run_checkpoint "$MODEL" "$rev"
done

echo ""
echo "=== Done. ==="

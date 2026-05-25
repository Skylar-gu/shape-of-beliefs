#!/usr/bin/env bash
set -euo pipefail

# Run the full activation pipeline across all OLMo-2-0425-1B training checkpoints.
# Run from repo root:
#   chmod +x scripts/generate_all_checkpoints.sh
#   ./scripts/generate_all_checkpoints.sh
#
# Outputs land in:
#   data/activations/olmo-2-0425-1b-early-training/<revision>/<dataset>/
#   data/activations/olmo-2-0425-1b/<revision>/<dataset>/       (stage2)
#
# To verify exact revision names before running:
#   python -c "
#   from huggingface_hub import list_repo_refs
#   refs = list_repo_refs('allenai/OLMo-2-0425-1B-early-training')
#   for b in sorted(refs.branches, key=lambda x: x.name): print(b.name)
#   "

TOKEN_SUBSET="olmo-2-0425-1B_number_tokens.json"

# ---------------------------------------------------------------------------
# Stage 1 — early training checkpoints (every 1000 steps, 0–37000)
# Hosted as branches on allenai/OLMo-2-0425-1B-early-training
# ---------------------------------------------------------------------------
EARLY_MODEL="allenai/OLMo-2-0425-1B-early-training"

EARLY_REVISIONS=(
  "stage1-step0-tokens0B"
  "stage1-step1000-tokens2.1B"
  "stage1-step2000-tokens4.2B"
  "stage1-step3000-tokens6.3B"
  "stage1-step4000-tokens8.4B"
  "stage1-step5000-tokens10.5B"
  "stage1-step6000-tokens12.6B"
  "stage1-step7000-tokens14.7B"
  "stage1-step8000-tokens16.8B"
  "stage1-step9000-tokens18.9B"
  "stage1-step10000-tokens21B"
  "stage1-step11000-tokens23.1B"
  "stage1-step12000-tokens25.2B"
  "stage1-step13000-tokens27.3B"
  "stage1-step14000-tokens29.4B"
  "stage1-step15000-tokens31.5B"
  "stage1-step16000-tokens33.6B"
  "stage1-step17000-tokens35.7B"
  "stage1-step18000-tokens37.8B"
  "stage1-step19000-tokens39.9B"
  "stage1-step20000-tokens42B"
  "stage1-step21000-tokens44.1B"
  "stage1-step22000-tokens46.2B"
  "stage1-step23000-tokens48.3B"
  "stage1-step24000-tokens50.4B"
  "stage1-step25000-tokens52.5B"
  "stage1-step26000-tokens54.6B"
  "stage1-step27000-tokens56.7B"
  "stage1-step28000-tokens58.8B"
  "stage1-step29000-tokens60.9B"
  "stage1-step30000-tokens63B"
  "stage1-step31000-tokens65.1B"
  "stage1-step32000-tokens67.2B"
  "stage1-step33000-tokens69.3B"
  "stage1-step34000-tokens71.4B"
  "stage1-step35000-tokens73.5B"
  "stage1-step36000-tokens75.6B"
  "stage1-step37000-tokens77.7B"
)

# ---------------------------------------------------------------------------
# Stage 2 — ingredient 3 checkpoints (every 1000 steps, 1000–23852)
# Hosted as branches on allenai/OLMo-2-0425-1B (main repo)
# ---------------------------------------------------------------------------
STAGE2_MODEL="allenai/OLMo-2-0425-1B"

STAGE2_REVISIONS=(
  "stage2-ingredient3-step1000"
  "stage2-ingredient3-step2000"
  "stage2-ingredient3-step3000"
  "stage2-ingredient3-step4000"
  "stage2-ingredient3-step5000"
  "stage2-ingredient3-step6000"
  "stage2-ingredient3-step7000"
  "stage2-ingredient3-step8000"
  "stage2-ingredient3-step9000"
  "stage2-ingredient3-step10000"
  "stage2-ingredient3-step11000"
  "stage2-ingredient3-step12000"
  "stage2-ingredient3-step13000"
  "stage2-ingredient3-step14000"
  "stage2-ingredient3-step15000"
  "stage2-ingredient3-step16000"
  "stage2-ingredient3-step17000"
  "stage2-ingredient3-step18000"
  "stage2-ingredient3-step19000"
  "stage2-ingredient3-step20000"
  "stage2-ingredient3-step21000"
  "stage2-ingredient3-step22000"
  "stage2-ingredient3-step23000"
  "stage2-ingredient3-step23852"
)

DATASETS=(
  "gaussian_m300_s100_l1000_n10"
  "gaussian_m350_s100_l1000_n10"
  "gaussian_m400_s100_l1000_n10"
  "gaussian_m450_s100_l1000_n10"
  "gaussian_m500_s010_l1000_n10"
  "gaussian_m500_s020_l1000_n10"
  "gaussian_m500_s030_l1000_n10"
  "gaussian_m500_s050_l1000_n10"
  "gaussian_m500_s080_l1000_n10"
  "gaussian_m500_s100_l1000_n10"
  "gaussian_m500_s120_l1000_n10"
  "gaussian_m500_s150_l1000_n10"
  "gaussian_m500_s200_l1000_n10"
  "gaussian_m550_s100_l1000_n10"
  "gaussian_m600_s100_l1000_n10"
  "gaussian_m650_s100_l1000_n10"
  "gaussian_m700_s100_l1000_n10"
)
COMBINED_DATASET="gaussian_m300_s100_l1000_n10+gaussian_m700_s100_l1000_n10"

# Sequences only need to be generated once (they don't depend on the model)
echo "=== Generating sequences (once) ==="
for ds in "${DATASETS[@]}"; do
  if [[ "$ds" =~ ^gaussian_m([0-9]+)_s([0-9]+)_l([0-9]+)_n([0-9]+)$ ]]; then
    mean="${BASH_REMATCH[1]}"
    std="${BASH_REMATCH[2]}"
    len="${BASH_REMATCH[3]}"
    num="${BASH_REMATCH[4]}"
    uv run python generate_sequences.py --num-seq "$num" --len-seq "$len" --mean "$mean" --std "$std"
  fi
done

# ---------------------------------------------------------------------------
# Stage 1 early training
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 1: early training (${#EARLY_REVISIONS[@]} checkpoints) ==="

for rev in "${EARLY_REVISIONS[@]}"; do
  echo ""
  echo "--- $EARLY_MODEL @ $rev ---"
  for ds in "${DATASETS[@]}"; do
    uv run python sequences_to_activations.py \
      --dataset-name "$ds" \
      --model "$EARLY_MODEL" \
      --token-subset "$TOKEN_SUBSET" \
      --revision "$rev"
  done
  uv run python sequences_to_activations.py \
    --dataset-name "$COMBINED_DATASET" \
    --model "$EARLY_MODEL" \
    --token-subset "$TOKEN_SUBSET" \
    --revision "$rev"
done

# ---------------------------------------------------------------------------
# Stage 2 ingredient 3
# ---------------------------------------------------------------------------
echo ""
echo "=== Stage 2 ingredient 3 (${#STAGE2_REVISIONS[@]} checkpoints) ==="

for rev in "${STAGE2_REVISIONS[@]}"; do
  echo ""
  echo "--- $STAGE2_MODEL @ $rev ---"
  for ds in "${DATASETS[@]}"; do
    uv run python sequences_to_activations.py \
      --dataset-name "$ds" \
      --model "$STAGE2_MODEL" \
      --token-subset "$TOKEN_SUBSET" \
      --revision "$rev"
  done
  uv run python sequences_to_activations.py \
    --dataset-name "$COMBINED_DATASET" \
    --model "$STAGE2_MODEL" \
    --token-subset "$TOKEN_SUBSET" \
    --revision "$rev"
done

echo ""
echo "=== Done. All checkpoints processed. ==="

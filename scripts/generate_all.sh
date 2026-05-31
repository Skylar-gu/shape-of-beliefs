#!/usr/bin/env bash
set -euo pipefail

# Generate sequences, activations, and logits.
# Run from repo root:
#   chmod +x scripts/generate_all.sh
#   ./scripts/generate_all.sh
#
# To run with a different model:
#   MODEL=allenai/OLMo-2-0425-1B TOKEN_SUBSET=olmo-2-0425-1B_number_tokens.json ./scripts/generate_all.sh

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
TOKEN_SUBSET="${TOKEN_SUBSET:-llama3-2-1B_number_tokens.json}"

echo "Model:        $MODEL"
echo "Token subset: $TOKEN_SUBSET"
echo ""

# 1) Explicit dataset list
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

# 2) Generate sequences for each dataset
for ds in "${DATASETS[@]}"; do
  if [[ "$ds" =~ ^gaussian_m([0-9]+)_s([0-9]+)_l([0-9]+)_n([0-9]+)$ ]]; then
    mean="${BASH_REMATCH[1]}"
    std="${BASH_REMATCH[2]}"
    len="${BASH_REMATCH[3]}"
    num="${BASH_REMATCH[4]}"
    echo "Generating sequences: $ds"
    uv run python generate_sequences.py --num-seq "$num" --len-seq "$len" --mean "$mean" --std "$std"
  else
    echo "Skipping unrecognized dataset name: $ds"
  fi
done

# 3) Produce activations + logits for each dataset
for ds in "${DATASETS[@]}"; do
  echo "Generating activations/logits: $ds"
  uv run python sequences_to_activations.py --dataset-name "$ds" --model "$MODEL" --token-subset "$TOKEN_SUBSET"
done

echo "Generating activations/logits: $COMBINED_DATASET"
uv run python sequences_to_activations.py --dataset-name "$COMBINED_DATASET" --model "$MODEL" --token-subset "$TOKEN_SUBSET"

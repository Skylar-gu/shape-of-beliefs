"""
Generate number_tokens JSON files for OLMo-1B and OLMo-2-1B,
matching the format of llama3-2-1B_number_tokens.json.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

MODELS = {
    "olmo-1B":        "allenai/OLMo-1B-hf",
    "olmo-2-0425-1B": "allenai/OLMo-2-0425-1B",
}

SPECIAL_TOKENS = [",", ";", ".", "_", " ", "-"]

OUT_DIR = Path(__file__).resolve().parent

for label, model_id in MODELS.items():
    print(f"\n{model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    token_map = {}

    for i in range(1000):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        if len(ids) == 1:
            token_map[str(i)] = ids[0]

    for s in SPECIAL_TOKENS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            token_map[s] = ids[0]
        else:
            print(f"  Warning: '{s}' is not a single token -> {ids}")

    out_path = OUT_DIR / f"{label}_number_tokens.json"
    with open(out_path, "w") as f:
        json.dump(token_map, f, indent=2)

    print(f"  {len([k for k in token_map if k.isdigit()])} integers + {len([k for k in token_map if not k.isdigit()])} delimiters -> {out_path.name}")

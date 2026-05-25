"""
Test how many integers 0-999 tokenize as a single token under OLMo and OLMo 2.
Prints a summary and the full list of multi-token integers for each model.

Usage (on Colab after installing deps):
    python token_subset/test_olmo_tokenization.py
"""

from transformers import AutoTokenizer

MODELS = {
    "OLMo-1B":   "allenai/OLMo-1B-hf",
    "OLMo-2-1B": "allenai/OLMo-2-1124-1B",
}

for label, model_id in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  {label}  ({model_id})")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  FAILED to load tokenizer: {e}")
        continue

    single, multi = [], []

    for i in range(1000):
        ids = tokenizer.encode(str(i), add_special_tokens=False)
        if len(ids) == 1:
            single.append(i)
        else:
            multi.append((i, ids))

    print(f"  Single-token integers : {len(single)} / 1000")
    print(f"  Multi-token integers  : {len(multi)} / 1000")

    if multi:
        print(f"\n  Multi-token examples (integer -> token ids -> decoded pieces):")
        for n, ids in multi[:30]:
            pieces = [tokenizer.decode([t]) for t in ids]
            print(f"    {n:>4}  ->  {ids}  ->  {pieces}")
        if len(multi) > 30:
            print(f"    ... and {len(multi) - 30} more")

    # Show coverage by range
    print(f"\n  Coverage by range:")
    for lo, hi in [(0, 9), (10, 99), (100, 999)]:
        subset_single = [n for n in single if lo <= n <= hi]
        total = hi - lo + 1
        print(f"    {lo:>3}-{hi:<3}  :  {len(subset_single)}/{total}")

print()

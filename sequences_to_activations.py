from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import chain
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# DATASET_NAME = "gaussian_m500_s100_l1000_n10" # if several, combine with a +
# can't do more than 10 sequences of l1000 on one GPU... make updates for that
DEFAULT_DATASET_NAME = "gaussian_m500_s100_l1000_n10"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEFAULT_TOKEN_SUBSET = "llama3-2-1B_number_tokens.json"

BASE_DIR = Path(__file__).resolve().parent

JSONL_DIR = Path(BASE_DIR) / "data" / "sequences"
BATCH_SIZE = 10

ALLOW_OVERWRITE = False

TOKENIZER_MAX_LENGTH = 13000


@dataclass
class Sequence:
    sequence_id: str
    sequence_content: str


def load_sequences_from_jsonl(jsonl_path: Path) -> list[Sequence]:
    sequences: list[Sequence] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_no}") from e
            if "sequence_id" not in record or "sequence_content" not in record:
                raise ValueError(f"Missing keys in {jsonl_path}:{line_no}")
            sequences.append(
                Sequence(
                    sequence_id=record["sequence_id"],
                    sequence_content=record["sequence_content"],
                )
            )
    return sequences


def combine_datasets(datasets: list[list[Sequence]]) -> list[Sequence]:
    if len(datasets) == 1:
        return datasets[0]

    base = datasets[0]
    base_ids = [seq.sequence_id for seq in base]
    base_id_set = set(base_ids)

    other_maps = []
    for ds in datasets[1:]:
        ids = [seq.sequence_id for seq in ds]
        if set(ids) != base_id_set:
            raise ValueError("Datasets have different sequence IDs.")
        other_maps.append({seq.sequence_id: seq.sequence_content for seq in ds})

    fused = []
    for seq in base:
        content = seq.sequence_content
        for m in other_maps:
            content = f"{content}{m[seq.sequence_id]}"
        fused.append(Sequence(sequence_id=seq.sequence_id, sequence_content=content))
    return fused


def iter_batches(sequences: list[Sequence], batch_size: int):
    for i in range(0, len(sequences), batch_size):
        yield sequences[i : i + batch_size]


def initialize_batch_iterator_and_seq_len(batch_iter, tokenizer, max_length=TOKENIZER_MAX_LENGTH):
    """Prime the iterator, compute sequence length from the first batch, and rewind."""
    batch_iter = iter(batch_iter)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        raise ValueError("Dataset iterator is empty.")

    prompts = [seq.sequence_content for seq in first_batch]
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    first_attention_mask = enc["attention_mask"]
    first_lengths = first_attention_mask.sum(dim=1).to(dtype=torch.int64, device="cpu")
    seq_len = int(first_lengths[0].item())
    if not torch.all(first_lengths == seq_len):
        raise ValueError(f"Unequal sequence lengths in first batch: {first_lengths.tolist()}")

    return first_batch, seq_len, chain([first_batch], batch_iter)


def ensure_empty_or_overwrite(path: Path):
    if path.exists() and not ALLOW_OVERWRITE:
        raise FileExistsError(f"Output path exists (set ALLOW_OVERWRITE=True to overwrite): {path}")


def batch_already_computed(batch_idx: int, sites: list[str], activs_output_dir: Path, logits_output_dir: Path) -> bool:
    logits_path = logits_output_dir / f"logits_batch{batch_idx:04d}.pt"
    if not logits_path.exists():
        return False
    for site in sites:
        acts_path = activs_output_dir / f"{site.replace('.', '_')}_batch{batch_idx:04d}.pt"
        if not acts_path.exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate activations and logits from sequences.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Dataset name or '+'-joined list of dataset names.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model ID (default: meta-llama/Llama-3.2-1B).",
    )
    parser.add_argument(
        "--token-subset",
        type=str,
        default=DEFAULT_TOKEN_SUBSET,
        help="Filename of the token subset JSON in token_subset/ (default: llama3-2-1B_number_tokens.json).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace revision (branch/tag) to load, e.g. stage1-step1000-tokens2.1B.",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model
    TOKEN_SUBSET_PATH = Path(BASE_DIR) / "token_subset" / args.token_subset

    dataset_names = [name.strip() for name in args.dataset_name.split("+") if name.strip()]
    combined_dataset_name = "+".join(dataset_names)
    model_alias = MODEL_NAME.split("/")[-1].lower()
    revision_part = args.revision if args.revision else ""
    activs_output_dir = Path(BASE_DIR) / "data" / "activations" / model_alias / revision_part / combined_dataset_name
    logits_output_dir = Path(BASE_DIR) / "data" / "logits" / model_alias / revision_part / combined_dataset_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=args.revision,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()

    sites = ["model.embed_tokens", *[f"model.layers.{i}" for i in range(model.config.num_hidden_layers)]]
    # sites = [*[f"model.layers.{i}" for i in range(0,16)]]

    jsonl_paths = [JSONL_DIR / f"{name}.jsonl" for name in dataset_names]
    datasets = [load_sequences_from_jsonl(p) for p in jsonl_paths]
    sequences = combine_datasets(datasets)
    batch_iter = iter_batches(sequences, BATCH_SIZE)
    first_batch, seq_len, batch_iter = initialize_batch_iterator_and_seq_len(batch_iter, tokenizer)

    activs_output_dir.mkdir(parents=True, exist_ok=True)
    logits_output_dir.mkdir(parents=True, exist_ok=True)

    if not TOKEN_SUBSET_PATH.exists():
        raise FileNotFoundError(f"Token subset file not found: {TOKEN_SUBSET_PATH}")
    token_map = json.loads(TOKEN_SUBSET_PATH.read_text(encoding="utf-8"))
    subset_items = sorted(token_map.items(), key=lambda kv: kv[1])
    token_strings = [k for k, _ in subset_items]
    token_ids = [v for _, v in subset_items]
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Duplicate token IDs in token subset file.")
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

    def resolve_module(name: str):
        try:
            return model.get_submodule(name)
        except AttributeError as e:
            raise ValueError(f"Module not found: {name}") from e

    with torch.no_grad():
        for batch_idx, batch in enumerate(batch_iter):
            if batch_already_computed(batch_idx, sites, activs_output_dir, logits_output_dir):
                print(f"Skipping batch {batch_idx} (already computed).")
                continue

            seq_ids = [seq.sequence_id for seq in batch]
            prompts = [seq.sequence_content for seq in batch]

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TOKENIZER_MAX_LENGTH,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            lengths = attention_mask.sum(dim=1).to(dtype=torch.int64, device="cpu")
            batch_seq_len = int(lengths[0].item())
            if not torch.all(lengths == batch_seq_len):
                raise ValueError(f"Unequal sequence lengths: {lengths.tolist()}")
            if batch_seq_len != seq_len:
                raise ValueError(f"Inconsistent sequence length: expected {seq_len}, got {batch_seq_len}")

            captures: dict[str, torch.Tensor] = {}
            hooks = []

            def make_hook(site_name: str):
                def hook(_module, _inputs, output):
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    captures[site_name] = output.detach().cpu()
                return hook

            for site in sites:
                module = resolve_module(site)
                hooks.append(module.register_forward_hook(make_hook(site)))

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            for h in hooks:
                h.remove()

            logits = outputs.logits.detach().cpu()[:, :batch_seq_len, :]
            input_ids_cpu = input_ids.detach().cpu()[:, :batch_seq_len]
            subset_logits = logits.index_select(dim=2, index=token_ids_tensor)

            logits_path = logits_output_dir / f"logits_batch{batch_idx:04d}.pt"
            torch.save(
                {
                    "sequence_ids": seq_ids,
                    "logits": subset_logits,
                    "input_ids": input_ids_cpu,
                    "lengths": lengths,
                    "token_ids": token_ids,
                    "token_strings": token_strings,
                },
                logits_path,
            )

            for site, acts in captures.items():
                acts_path = activs_output_dir / f"{site.replace('.', '_')}_batch{batch_idx:04d}.pt"
                torch.save(
                    {
                        "sequence_ids": seq_ids,
                        "activations": acts[:, :batch_seq_len],
                        "lengths": lengths,
                    },
                    acts_path,
                )

            del captures, logits, subset_logits, input_ids, attention_mask, input_ids_cpu
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

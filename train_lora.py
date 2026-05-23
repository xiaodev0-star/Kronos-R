import argparse
import glob
import json
import os

from config import DataConfig, LoRAConfig
from reproducibility import set_global_seed
from sft_service import train_lora_adapter


def _expand_inputs(inputs):
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "*.csv")))
            continue
        matches = glob.glob(item)
        if matches:
            paths.extend(matches)
        elif os.path.isfile(item):
            paths.append(item)
    deduped = []
    seen = set()
    for path in paths:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on user-selected CSV files.")
    parser.add_argument("inputs", nargs="+", help="CSV files, wildcard patterns, or folders.")
    parser.add_argument("--adapter-name", default="user_adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-sequences-per-file", type=int, default=0)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=getattr(LoRAConfig, "random_seed", DataConfig.random_seed))
    args = parser.parse_args()

    set_global_seed(int(args.seed), deterministic=True)
    print(f"Seed: {int(args.seed)}")

    csv_paths = _expand_inputs(args.inputs)
    if not csv_paths:
        raise FileNotFoundError("No valid CSV files found from inputs.")

    result = train_lora_adapter(
        csv_paths=csv_paths,
        adapter_name=args.adapter_name,
        base_checkpoint_path=args.base_checkpoint,
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=max(float(args.lr), 1e-7),
        seq_len=args.seq_len,
        max_sequences_per_file=max(0, int(args.max_sequences_per_file)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

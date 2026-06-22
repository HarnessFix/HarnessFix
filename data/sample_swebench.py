#!/usr/bin/env python3
"""Sample reproducible SWE-Bench Verified train/val/test subsets.

The paper experiments use the official SWE-Bench Verified ``test`` split as the
source pool, then create three non-overlapping subsets:

- train: used for trace collection and repair iteration
- val: used as a regression gate during repair
- test: held out for final evaluation

Defaults preserve the original two-step sampling scripts: train/test are sampled
with seed 42, then val is sampled from the remaining instances with seed 123.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def save_subset(split_name: str, dataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    DatasetDict({"test": dataset}).save_to_disk(str(output_dir))
    instance_ids = sorted(dataset["instance_id"])
    ids_file = output_dir / "instance_ids.txt"
    ids_file.write_text("\n".join(instance_ids) + "\n")
    print(f"[{split_name}] {len(instance_ids)} instances -> {output_dir.resolve()}")
    print(f"  instance IDs: {ids_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample non-overlapping train/val/test subsets from SWE-Bench Verified."
    )
    parser.add_argument("--train-n", type=int, default=100, help="Number of train instances.")
    parser.add_argument("--test-n", type=int, default=100, help="Number of held-out test instances.")
    parser.add_argument("--val-n", type=int, default=50, help="Number of validation instances.")
    parser.add_argument("--train-test-seed", type=int, default=42, help="Shuffle seed for train/test.")
    parser.add_argument("--val-seed", type=int, default=123, help="Shuffle seed for val after train/test exclusion.")
    parser.add_argument("--train-output", type=Path, default=Path("data/verified_train_100"))
    parser.add_argument("--test-output", type=Path, default=Path("data/verified_test_100"))
    parser.add_argument("--val-output", type=Path, default=Path("data/verified_val_50"))
    args = parser.parse_args()

    print("Loading SWE-Bench Verified (official test split)...")
    dataset = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    total = len(dataset)
    required = args.train_n + args.test_n + args.val_n
    if required > total:
        raise ValueError(f"Need {required} instances but SWE-Bench Verified only has {total}.")

    shuffled = dataset.shuffle(seed=args.train_test_seed)
    train_end = args.train_n
    test_end = train_end + args.test_n
    train = shuffled.select(range(0, train_end))
    test = shuffled.select(range(train_end, test_end))

    train_ids = set(train["instance_id"])
    test_ids = set(test["instance_id"])
    if train_ids & test_ids:
        raise RuntimeError("Sampled train/test splits overlap.")

    excluded = train_ids | test_ids
    remaining = dataset.filter(lambda row: row["instance_id"] not in excluded)
    if args.val_n > len(remaining):
        raise ValueError(f"Need {args.val_n} val instances but only {len(remaining)} remain.")
    val = remaining.shuffle(seed=args.val_seed).select(range(args.val_n))

    val_ids = set(val["instance_id"])
    if val_ids & excluded:
        raise RuntimeError("Sampled val split overlaps with train/test.")

    save_subset("train", train, args.train_output)
    save_subset("test", test, args.test_output)
    save_subset("val", val, args.val_output)
    print("Done.")


if __name__ == "__main__":
    main()

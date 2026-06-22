#!/usr/bin/env python3
"""
从 SWE-Bench Verified 测试集中随机采样两组互不重合的数据集：
  - train set：100 条，用于训练/实验
  - test set：100 条，用于评估

固定随机种子保证可复现。先 shuffle，取前 100 条作 train，再取后 100 条作 test。

用法:
    python sample_train_test.py
    python sample_train_test.py --n 100 --seed 42 --train-output ./verified_train_100 --test-output ./verified_test_100
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample two non-overlapping splits from SWE-Bench Verified")
    parser.add_argument("--n", type=int, default=100, help="Number of instances per split (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train-output", type=str, default="./verified_train_100", help="Output directory for train set")
    parser.add_argument("--test-output", type=str, default="./verified_test_100", help="Output directory for test set")
    args = parser.parse_args()

    print("Loading SWE-Bench Verified (test split)...")
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    total = len(ds)
    print(f"Total instances: {total}")

    required = args.n * 2
    if required > total:
        raise ValueError(f"Need {required} instances but only {total} available. Reduce --n.")

    shuffled = ds.shuffle(seed=args.seed)
    train = shuffled.select(range(args.n))
    test = shuffled.select(range(args.n, args.n * 2))

    # 验证无重叠
    train_ids = set(train["instance_id"])
    test_ids = set(test["instance_id"])
    assert train_ids.isdisjoint(test_ids), "BUG: train/test overlap!"

    for split_name, split_ds, output_str in [("train", train, args.train_output), ("test", test, args.test_output)]:
        output_path = Path(output_str)
        DatasetDict({"test": split_ds}).save_to_disk(str(output_path))
        ids = sorted(split_ds["instance_id"])
        ids_file = output_path / "instance_ids.txt"
        ids_file.write_text("\n".join(ids) + "\n")
        print(f"\n[{split_name}] {len(ids)} instances → {output_path.resolve()}")
        for iid in ids:
            print(f"  {iid}")
        print(f"  Instance IDs saved to: {ids_file}")

    print(f"\nDone. train={args.train_output}  test={args.test_output}")
    print("Run with:")
    print(f"  ./run_swebench.sh --subset $(pwd)/{args.train_output} --split test --model <model>")
    print(f"  ./run_swebench.sh --subset $(pwd)/{args.test_output}  --split test --model <model>")


if __name__ == "__main__":
    main()

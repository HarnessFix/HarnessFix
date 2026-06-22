#!/usr/bin/env python3
"""
从 SWE-Bench Verified 测试集中随机采样 100 条，保存为独立的 HuggingFace Dataset，
供 mini-swe-agent 的 --subset 参数使用。

采样固定随机种子 42，保证可复现。

用法:
    python sample_verified_100.py
    python sample_verified_100.py --n 100 --seed 42 --output ./verified_sample_100
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample SWE-Bench Verified instances")
    parser.add_argument("--n", type=int, default=100, help="Number of instances to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default="./verified_sample_100", help="Output directory for the sampled dataset")
    args = parser.parse_args()

    print(f"Loading SWE-Bench Verified (test split)...")
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    total = len(ds)
    print(f"Total instances: {total}")

    print(f"Sampling {args.n} instances with seed={args.seed}...")
    sample = ds.shuffle(seed=args.seed).select(range(args.n))

    # 打印采样到的 instance ID，方便核查
    ids = sample["instance_id"]
    print(f"Sampled {len(ids)} instances:")
    for iid in sorted(ids):
        print(f"  {iid}")

    # 保存为 DatasetDict，使得 load_dataset(path, split="test") 可以直接读取
    output_path = Path(args.output)
    DatasetDict({"test": sample}).save_to_disk(str(output_path))
    print(f"\nSaved to: {output_path.resolve()}")

    # 同时写一份 instance_ids.txt 方便人工查看
    ids_file = output_path / "instance_ids.txt"
    ids_file.write_text("\n".join(sorted(ids)) + "\n")
    print(f"Instance IDs saved to: {ids_file}")


if __name__ == "__main__":
    main()

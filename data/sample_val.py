#!/usr/bin/env python3
"""
从 SWE-Bench Verified 剩余实例中采样验证集（val set）。

排除已有的 train set 和 test set，从剩余实例中随机采样 N 条作为验证集。
验证集用于每轮 agent 改动后的回归检测，不用于指导改进方向。

用法:
    python sample_val.py
    python sample_val.py --n 50 --seed 42 \
        --train-ids ./verified_train_100/instance_ids.txt \
        --test-ids  ./verified_test_100/instance_ids.txt \
        --output    ./verified_val_50
"""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset


def main():
    parser = argparse.ArgumentParser(description="Sample val set from remaining SWE-Bench Verified instances")
    parser.add_argument("--n", type=int, default=50, help="Number of val instances (default: 50)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123, different from train/test seed=42)")
    parser.add_argument("--train-ids", type=str, default="./verified_train_100/instance_ids.txt",
                        help="Path to train set instance_ids.txt")
    parser.add_argument("--test-ids", type=str, default="./verified_test_100/instance_ids.txt",
                        help="Path to test set instance_ids.txt")
    parser.add_argument("--output", type=str, default="./verified_val_50",
                        help="Output directory for val set")
    args = parser.parse_args()

    # 加载已有 train/test 的 instance_id
    train_ids = set(Path(args.train_ids).read_text().split())
    test_ids  = set(Path(args.test_ids).read_text().split())
    excluded  = train_ids | test_ids
    print(f"Excluding {len(train_ids)} train + {len(test_ids)} test = {len(excluded)} instances")

    # 加载完整数据集
    print("Loading SWE-Bench Verified (test split)...")
    ds = load_dataset("princeton-nlp/SWE-Bench_Verified", split="test")
    print(f"Total instances in Verified: {len(ds)}")

    # 过滤掉已使用的
    remaining = ds.filter(lambda x: x["instance_id"] not in excluded)
    print(f"Remaining after exclusion: {len(remaining)}")

    if args.n > len(remaining):
        raise ValueError(f"Need {args.n} instances but only {len(remaining)} remaining. Reduce --n.")

    # 采样
    val = remaining.shuffle(seed=args.seed).select(range(args.n))

    # 验证无重叠
    val_ids = set(val["instance_id"])
    assert val_ids.isdisjoint(train_ids), "BUG: val/train overlap!"
    assert val_ids.isdisjoint(test_ids),  "BUG: val/test overlap!"
    print(f"Overlap check passed: val has no overlap with train or test")

    # 保存
    output_path = Path(args.output)
    DatasetDict({"test": val}).save_to_disk(str(output_path))
    ids = sorted(val["instance_id"])
    ids_file = output_path / "instance_ids.txt"
    ids_file.write_text("\n".join(ids) + "\n")

    print(f"\n[val] {len(ids)} instances → {output_path.resolve()}")
    for iid in ids:
        print(f"  {iid}")
    print(f"\nInstance IDs saved to: {ids_file}")
    print(f"\nNext steps:")
    print(f"  # 预拉取 Docker 镜像")
    print(f"  ./pull_swebench_images.sh $(pwd)/{args.output} test 4")
    print(f"")
    print(f"  # 跑原始版本 baseline（只需跑一次）")
    print(f"  ./run_swebench.sh --subset $(pwd)/{args.output} --split test \\")
    print(f"      --model openai/Qwen3-Coder --workers 4 \\")
    print(f"      --output ../traces/swe_val_qwen3_coder_original --cost-limit 1")
    print(f"")
    print(f"  # 每次改动后跑 enhanced 版本并对比")
    print(f"  PYTHONPATH=enhanced_swe_v1/src ./run_swebench.sh \\")
    print(f"      --subset $(pwd)/{args.output} --split test \\")
    print(f"      --model openai/Qwen3-Coder --workers 4 \\")
    print(f"      --output ../traces/val_qwen3_coder_swe_enhanced_vN --cost-limit 1")


if __name__ == "__main__":
    main()

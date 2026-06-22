#!/usr/bin/env python3
"""Sample GAIA benchmark data for local train/val/test splits.

Creates local 2:1:2 splits with ground-truth answers:
  data/gaia_train_60/ — 60 instances sampled from official validation split
  data/gaia_val_30/   — 30 instances sampled from official validation split
  data/gaia_test_60/  — 60 instances sampled from official validation split

Requirements:
  HF_TOKEN env var (GAIA dataset is gated on HuggingFace)
  pip install datasets huggingface_hub python-dotenv

Usage:
  python3 data/sample_gaia.py
  # or with explicit token:
  HF_TOKEN=hf_... python3 data/sample_gaia.py
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── Sampling configuration ─────────────────────────────────────────────────────

SEED = 42

TRAIN_COUNTS = {"1": 19, "2": 31, "3": 10}  # total = 60
VAL_COUNTS = {"1": 10, "2": 16, "3": 4}     # total = 30
TEST_COUNTS = {"1": 19, "2": 31, "3": 10}   # total = 60


# ── Data loading ───────────────────────────────────────────────────────────────

def load_gaia_split(split: str):
    """Load a GAIA split from HuggingFace (requires HF_TOKEN)."""
    from datasets import load_dataset

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set.\n"
            "GAIA dataset is gated — visit https://huggingface.co/datasets/gaia-benchmark/GAIA "
            "to request access, then set HF_TOKEN."
        )

    print(f"Loading gaia-benchmark/GAIA ({split} split) from HuggingFace...")
    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split=split,
        token=hf_token,
    )
    return dataset


def normalize_task(item: dict) -> dict:
    """Normalize a raw GAIA dataset row to our standard format."""
    # Handle both raw dataset field names and pre-processed ones
    task_id = item.get("task_id") or item.get("id") or ""
    question = item.get("Question") or item.get("question") or ""
    true_answer = (
        item.get("Final answer")
        or item.get("true_answer")
        or item.get("answer")
        or ""
    )
    level = str(item.get("Level") or item.get("level") or item.get("task") or "1")
    # Level may be an int
    level = str(int(level)) if level.isdigit() else "1"
    file_name = item.get("file_name") or item.get("file_path") or ""

    return {
        "task_id": task_id,
        "question": question,
        "true_answer": true_answer,
        "level": int(level),
        "file_name": file_name,
    }


# ── Sampling ───────────────────────────────────────────────────────────────────

def stratified_sample_no_overlap(
    items_by_level: dict[str, list[dict]],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Sample train, val, and test sets from each level without overlap."""
    rng = random.Random(seed)
    train_tasks: list[dict] = []
    val_tasks: list[dict] = []
    test_tasks: list[dict] = []

    for level in sorted(train_counts.keys()):
        pool = list(items_by_level.get(level, []))
        rng.shuffle(pool)

        n_train = train_counts[level]
        n_val = val_counts.get(level, 0)
        n_test = test_counts.get(level, 0)
        needed = n_train + n_val + n_test

        if len(pool) < needed:
            raise ValueError(
                f"Not enough level-{level} instances: need {needed} "
                f"(train={n_train} + val={n_val} + test={n_test}), have {len(pool)}"
            )

        train_tasks.extend(pool[:n_train])
        val_tasks.extend(pool[n_train:n_train + n_val])
        test_tasks.extend(pool[n_train + n_val:n_train + n_val + n_test])

    return train_tasks, val_tasks, test_tasks


# ── Output ─────────────────────────────────────────────────────────────────────

def write_subset(name: str, tasks: list[dict]) -> Path:
    """Write a subset to disk: data.jsonl + instance_ids.txt."""
    out_dir = _SCRIPT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "data.jsonl").open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    ids = [t["task_id"] for t in tasks]
    (out_dir / "instance_ids.txt").write_text("\n".join(ids) + "\n")

    # Summary by level
    level_counts = {}
    for t in tasks:
        lvl = str(t["level"])
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    level_str = ", ".join(f"L{k}={v}" for k, v in sorted(level_counts.items()))
    print(f"  → {out_dir}/ ({len(tasks)} tasks: {level_str})")
    return out_dir


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    validation_dataset = load_gaia_split("validation")

    # Organize validation by level
    validation_by_level: dict[str, list[dict]] = {"1": [], "2": [], "3": []}
    for item in validation_dataset:
        task = normalize_task(dict(item))
        level = str(task["level"])
        if level not in validation_by_level:
            level = "1"
            task["level"] = 1
        validation_by_level[level].append(task)

    validation_dist = ", ".join(f"L{k}={len(v)}" for k, v in sorted(validation_by_level.items()))
    print(f"Validation split level distribution: {validation_dist}")
    print(f"Validation total: {sum(len(v) for v in validation_by_level.values())} instances\n")

    train_tasks, val_tasks, test_tasks = stratified_sample_no_overlap(
        validation_by_level, TRAIN_COUNTS, VAL_COUNTS, TEST_COUNTS, seed=SEED
    )

    print(f"Sampled: train={len(train_tasks)}, val={len(val_tasks)}, test={len(test_tasks)}")
    print("Writing splits...")
    write_subset("gaia_train_60", train_tasks)
    write_subset("gaia_val_30", val_tasks)
    write_subset("gaia_test_60", test_tasks)
    print("\nDone!")


if __name__ == "__main__":
    main()

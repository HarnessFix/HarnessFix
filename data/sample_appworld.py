#!/usr/bin/env python3
"""Create sampled AppWorld train/val/test subsets from the local task cache."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CACHE = Path(os.environ.get("APPWORLD_TASK_CACHE", "data/appworld_task_cache.json"))


def _load_cache() -> dict:
    cache_path = DEFAULT_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"AppWorld task cache not found at {cache_path}. "
            "Set up the local AppWorld dataset first."
        )
    return json.loads(cache_path.read_text())


def _write_subset(name: str, tasks: list[dict]) -> None:
    out_dir = DATA_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.jsonl").write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n"
    )
    (out_dir / "instance_ids.txt").write_text("\n".join(task["task_id"] for task in tasks) + "\n")


def _sample(tasks: list[dict], size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if size >= len(tasks):
        return list(tasks)
    return rng.sample(tasks, size)


def _sample_mixed_test(cache: dict, size: int, seed: int) -> list[dict]:
    normal = list(cache.get("test_normal", []))
    challenge = list(cache.get("test_challenge", []))
    total = len(normal) + len(challenge)
    if total == 0:
        return []

    normal_count = round(size * len(normal) / total)
    normal_count = min(normal_count, len(normal), size)
    challenge_count = min(size - normal_count, len(challenge))
    if normal_count + challenge_count < size:
        normal_count = min(len(normal), normal_count + size - normal_count - challenge_count)

    sampled_normal = _sample(normal, normal_count, seed=seed)
    sampled_challenge = _sample(challenge, challenge_count, seed=seed + 1)
    mixed = [
        {**task, "source_split": "test_normal"}
        for task in sampled_normal
    ] + [
        {**task, "source_split": "test_challenge"}
        for task in sampled_challenge
    ]
    rng = random.Random(seed + 2)
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    cache = _load_cache()
    train_90 = _sample(cache.get("train", []), 90, seed=42)
    val_45 = _sample(cache.get("dev", []), 45, seed=123)
    test_90 = _sample_mixed_test(cache, 90, seed=31415)

    _write_subset("appworld_train_90", train_90)
    _write_subset("appworld_val_45", val_45)
    _write_subset("appworld_test_90", test_90)

    print("Wrote:")
    print(f"  {DATA_DIR / 'appworld_train_90'}")
    print(f"  {DATA_DIR / 'appworld_val_45'}")
    print(f"  {DATA_DIR / 'appworld_test_90'}")


if __name__ == "__main__":
    main()

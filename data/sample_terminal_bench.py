#!/usr/bin/env python3
"""Create local Terminal-Bench 2.0 train/val/test split directories.

The script expects a local clone at data/terminal_bench_2_verified by default.
It creates split folders containing symlinks to task directories plus
HarnessFix metadata files: instance_ids.txt and data.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "data" / "terminal_bench_2_verified"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "terminal_bench_splits.json"


def _find_task_dirs(source: Path) -> list[Path]:
    candidates = []
    for task_toml in source.rglob("task.toml"):
        task_dir = task_toml.parent
        if any(part.startswith(".") for part in task_dir.relative_to(source).parts):
            continue
        candidates.append(task_dir)
    return sorted(candidates, key=lambda p: p.name)


def _read_instruction(task_dir: Path) -> str:
    path = task_dir / "instruction.md"
    if path.exists():
        return path.read_text(errors="replace")
    return ""


def _write_split(name: str, tasks: list[Path], output_root: Path, source: Path, copy: bool) -> None:
    split_dir = output_root / name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ids = []
    for task_dir in tasks:
        task_id = task_dir.name
        ids.append(task_id)
        dest = split_dir / task_id
        if copy:
            shutil.copytree(task_dir, dest)
        else:
            dest.symlink_to(task_dir.resolve(), target_is_directory=True)
        rows.append({
            "task_id": task_id,
            "source_dir": str(task_dir.relative_to(source)),
            "instruction": _read_instruction(task_dir),
        })

    (split_dir / "instance_ids.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    with (split_dir / "data.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{name}: {len(tasks)} tasks -> {split_dir}")


def _load_manifest_splits(path: Path, task_by_id: dict[str, Path]) -> dict[str, list[Path]]:
    manifest = json.loads(path.read_text())
    split_ids = manifest.get("splits")
    if not isinstance(split_ids, dict):
        raise SystemExit(f"Terminal-Bench split manifest has no splits object: {path}")

    splits = {}
    seen = set()
    for split_name in ("terminal_bench_train", "terminal_bench_val", "terminal_bench_test"):
        ids = split_ids.get(split_name)
        if not isinstance(ids, list):
            raise SystemExit(f"Terminal-Bench split manifest missing list for {split_name}: {path}")
        duplicates = sorted(task_id for task_id in ids if task_id in seen)
        if duplicates:
            raise SystemExit("Duplicate Terminal-Bench task ids in manifest: " + ", ".join(duplicates))
        missing = sorted(task_id for task_id in ids if task_id not in task_by_id)
        if missing:
            raise SystemExit(
                f"Terminal-Bench source is missing manifest task ids for {split_name}: "
                + ", ".join(missing)
            )
        seen.update(ids)
        splits[split_name] = [task_by_id[task_id] for task_id in ids]
    return splits


def _sample_random_splits(
    tasks: list[Path],
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = tasks[:]
    rng.shuffle(shuffled)
    requested = train_count + val_count + test_count
    if requested > len(shuffled):
        raise SystemExit(f"Requested {requested} tasks but only found {len(shuffled)}")

    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:train_count + val_count + test_count]
    return {
        "terminal_bench_train": train,
        "terminal_bench_val": val,
        "terminal_bench_test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Terminal-Bench 2.0 local splits")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train", type=int, default=34)
    parser.add_argument("--val", type=int, default=17)
    parser.add_argument("--test", type=int, default=34)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--copy", action="store_true", help="Copy task dirs instead of symlinking them")
    parser.add_argument("--random-split", action="store_true", help="Ignore the manifest and sample with --seed")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Terminal-Bench source not found: {args.source}")
    tasks = _find_task_dirs(args.source)
    if not tasks:
        raise SystemExit(f"No task.toml files found under {args.source}")

    if args.random_split:
        splits = _sample_random_splits(tasks, args.train, args.val, args.test, args.seed)
    else:
        task_by_id = {task_dir.name: task_dir for task_dir in tasks}
        splits = _load_manifest_splits(args.manifest, task_by_id)

    for name, split_tasks in splits.items():
        _write_split(name, split_tasks, args.output_root, args.source, args.copy)


if __name__ == "__main__":
    main()

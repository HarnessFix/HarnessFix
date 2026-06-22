#!/usr/bin/env python3
"""AppWorld benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_subset_data(subset_dir: Path) -> list[dict]:
    data_file = subset_dir / "data.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"data.jsonl not found in {subset_dir}")
    tasks = []
    for line in data_file.read_text().splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def load_traj_info(traces_dir: Path, task_id: str) -> dict:
    traj_path = traces_dir / task_id / f"{task_id}.traj.json"
    if not traj_path.exists():
        return {}
    try:
        return json.loads(traj_path.read_text()).get("info", {})
    except Exception:
        return {}


def load_eval_result(traces_dir: Path, task_id: str) -> dict:
    result_path = traces_dir / task_id / "result.json"
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text())
    except Exception:
        return {}


def evaluate(tasks: list[dict], traces_dir: Path) -> dict:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    empty_patch_ids: list[str] = []
    error_ids: list[str] = []
    by_difficulty_total: dict[str, int] = {}
    by_difficulty_resolved: dict[str, int] = {}

    for task in tasks:
        task_id = task["task_id"]
        difficulty = str(task.get("difficulty", ""))
        by_difficulty_total[difficulty] = by_difficulty_total.get(difficulty, 0) + 1

        info = load_traj_info(traces_dir, task_id)
        eval_result = load_eval_result(traces_dir, task_id)

        if eval_result.get("success"):
            resolved_ids.append(task_id)
            by_difficulty_resolved[difficulty] = by_difficulty_resolved.get(difficulty, 0) + 1
            continue

        exit_status = info.get("exit_status", "")
        prediction = info.get("prediction", "")
        if exit_status == "error" or eval_result.get("error"):
            error_ids.append(task_id)
        elif not prediction and exit_status != "unsolved":
            empty_patch_ids.append(task_id)
        else:
            unresolved_ids.append(task_id)

    total = len(tasks)
    resolved = len(resolved_ids)
    accuracy = resolved / total if total else 0.0
    by_difficulty = {}
    for difficulty, count in sorted(by_difficulty_total.items()):
        resolved_count = by_difficulty_resolved.get(difficulty, 0)
        by_difficulty[difficulty] = {
            "total": count,
            "resolved": resolved_count,
            "accuracy": round(resolved_count / count, 4) if count else 0.0,
        }

    return {
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "error_ids": sorted(error_ids),
        "total": total,
        "resolved": resolved,
        "accuracy": round(accuracy, 4),
        "by_difficulty": by_difficulty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AppWorld benchmark evaluator")
    parser.add_argument("--subset", required=True, type=Path,
                        help="Subset directory containing data.jsonl")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output path for results.json")
    parser.add_argument("--traces-dir", required=True, type=Path,
                        help="Directory containing per-task result.json and traj files")
    args = parser.parse_args()

    if not args.subset.exists():
        print(f"ERROR: subset directory not found: {args.subset}", file=sys.stderr)
        raise SystemExit(1)
    if not args.traces_dir.exists():
        print(f"ERROR: traces directory not found: {args.traces_dir}", file=sys.stderr)
        raise SystemExit(1)

    tasks = load_subset_data(args.subset)
    results = evaluate(tasks, args.traces_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"Evaluated {len(tasks)} AppWorld tasks")
    print(f"Resolved: {results['resolved']} / {results['total']} ({results['accuracy']*100:.1f}%)")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()

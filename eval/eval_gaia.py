#!/usr/bin/env python3
"""GAIA benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "task_agent" / "open_deep_research" / "src"))
from open_deep_research.scoring import question_scorer  # noqa: E402


def load_preds(preds_path: Path) -> dict[str, str]:
    return json.loads(preds_path.read_text())


def load_subset_data(subset_dir: Path) -> dict[str, dict]:
    data_file = subset_dir / "data.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"data.jsonl not found in {subset_dir}")
    tasks: dict[str, dict] = {}
    for line in data_file.read_text().splitlines():
        line = line.strip()
        if line:
            task = json.loads(line)
            tasks[task["task_id"]] = task
    return tasks


def load_traj_exit_status(traces_dir: Path, task_id: str) -> str | None:
    traj_path = traces_dir / task_id / f"{task_id}.traj.json"
    if not traj_path.exists():
        return None
    try:
        data = json.loads(traj_path.read_text())
        return data.get("info", {}).get("exit_status")
    except Exception:
        return None


def evaluate(preds: dict[str, str], tasks: dict[str, dict], traces_dir: Path | None) -> dict:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    empty_patch_ids: list[str] = []
    error_ids: list[str] = []
    by_level_total: dict[str, int] = {}
    by_level_resolved: dict[str, int] = {}

    for task_id, task in tasks.items():
        true_answer = task.get("true_answer", "")
        level = str(task.get("level", "1"))
        by_level_total[level] = by_level_total.get(level, 0) + 1

        prediction = preds.get(task_id, None)
        traj_status = load_traj_exit_status(traces_dir, task_id) if traces_dir else None

        if traj_status == "error" or prediction is None:
            error_ids.append(task_id)
            continue
        if prediction == "" or prediction is None:
            empty_patch_ids.append(task_id)
            continue

        if question_scorer(prediction, true_answer):
            resolved_ids.append(task_id)
            by_level_resolved[level] = by_level_resolved.get(level, 0) + 1
        else:
            unresolved_ids.append(task_id)

    total = len(tasks)
    resolved = len(resolved_ids)
    accuracy = resolved / total if total > 0 else 0.0
    by_level = {}
    for level in sorted(by_level_total):
        lvl_total = by_level_total[level]
        lvl_resolved = by_level_resolved.get(level, 0)
        by_level[level] = {
            "total": lvl_total,
            "resolved": lvl_resolved,
            "accuracy": round(lvl_resolved / lvl_total, 4) if lvl_total > 0 else 0.0,
        }

    return {
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "error_ids": sorted(error_ids),
        "total": total,
        "resolved": resolved,
        "accuracy": round(accuracy, 4),
        "by_level": by_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA benchmark evaluator")
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--traces-dir", type=Path, default=None)
    args = parser.parse_args()

    if not args.preds.exists():
        print(f"ERROR: preds.json not found: {args.preds}", file=sys.stderr)
        raise SystemExit(1)
    if not args.subset.exists():
        print(f"ERROR: subset directory not found: {args.subset}", file=sys.stderr)
        raise SystemExit(1)

    preds = load_preds(args.preds)
    tasks = load_subset_data(args.subset)
    traces_dir = args.traces_dir or args.preds.parent
    results = evaluate(preds, tasks, traces_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"Evaluated {len(tasks)} GAIA tasks")
    print(f"Resolved: {results['resolved']} / {results['total']} ({results['accuracy']*100:.1f}%)")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()

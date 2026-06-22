#!/usr/bin/env python3
"""Evaluate HarnessFix-formatted Terminal-Bench/Harbor outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _task_ids_from_subset(subset: Path) -> list[str]:
    ids_file = subset / "instance_ids.txt"
    if ids_file.exists():
        return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    data_file = subset / "data.jsonl"
    if data_file.exists():
        ids = []
        for line in data_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id") or row.get("instance_id") or row.get("name")
            if task_id:
                ids.append(str(task_id))
        return ids
    if subset.exists():
        return sorted(path.name for path in subset.iterdir() if path.is_dir())
    return []


def _reward_score(result: dict[str, Any]) -> float | int | None:
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    if not isinstance(rewards, dict) or not rewards:
        return None
    if "reward" in rewards:
        return rewards["reward"]
    numeric = [v for v in rewards.values() if isinstance(v, (int, float))]
    if not numeric:
        return None
    return min(numeric) if len(numeric) > 1 else numeric[0]


def _load_traj_info(traces_dir: Path, task_id: str) -> dict[str, Any]:
    traj_path = traces_dir / task_id / f"{task_id}.traj.json"
    if not traj_path.exists():
        return {}
    try:
        return _load_json(traj_path).get("info", {})
    except Exception:
        return {}


def _load_result(traces_dir: Path, task_id: str) -> dict[str, Any]:
    path = traces_dir / task_id / "result.json"
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def evaluate(task_ids: list[str], traces_dir: Path) -> dict[str, Any]:
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    empty_patch_ids: list[str] = []
    error_ids: list[str] = []
    rewards: dict[str, float | int | None] = {}

    for task_id in task_ids:
        info = _load_traj_info(traces_dir, task_id)
        result = _load_result(traces_dir, task_id)
        reward = info.get("reward")
        if reward is None:
            reward = _reward_score(result)
        rewards[task_id] = reward

        exception = result.get("exception_info") or info.get("exception")
        if exception or info.get("exit_status") == "error":
            error_ids.append(task_id)
        elif reward is None:
            empty_patch_ids.append(task_id)
        elif float(reward) >= 1.0:
            resolved_ids.append(task_id)
        else:
            unresolved_ids.append(task_id)

    total = len(task_ids)
    resolved = len(resolved_ids)
    accuracy = resolved / total if total else 0.0
    return {
        "resolved_ids": sorted(resolved_ids),
        "unresolved_ids": sorted(unresolved_ids),
        "empty_patch_ids": sorted(empty_patch_ids),
        "error_ids": sorted(error_ids),
        "total": total,
        "resolved": resolved,
        "accuracy": round(accuracy, 4),
        "rewards": rewards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal-Bench evaluator")
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--traces-dir", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()

    if not args.traces_dir.exists():
        print(f"ERROR: traces directory not found: {args.traces_dir}", file=sys.stderr)
        raise SystemExit(1)
    task_ids = _task_ids_from_subset(args.subset)
    if not task_ids:
        run_info = args.traces_dir / "run_info.json"
        if run_info.exists():
            task_ids = [str(item["task_id"]) for item in _load_json(run_info).get("tasks", [])]
    if not task_ids:
        print(f"ERROR: no Terminal-Bench task ids found from {args.subset} or run_info.json", file=sys.stderr)
        raise SystemExit(1)

    results = evaluate(task_ids, args.traces_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")

    print(f"Evaluated {len(task_ids)} Terminal-Bench tasks")
    print(f"Resolved: {results['resolved']} / {results['total']} ({results['accuracy']*100:.1f}%)")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()

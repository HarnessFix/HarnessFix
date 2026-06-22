#!/usr/bin/env python3
"""AppWorld benchmark runner entry point.

Output layout:
  <output>/<task_id>/<task_id>.traj.json
  <output>/<task_id>/result.json
  <output>/<task_id>/eval_report.md
  <output>/preds.json
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)
sys.path.insert(0, str(_PROJECT_ROOT / "agent_framework" / "src"))

from appworld_agent.core import load_preds, load_subset, save_preds
from appworld_agent.official_react_adapter import run_official_react_task


_preds_lock = threading.Lock()


def main() -> None:
    parser = argparse.ArgumentParser(description="AppWorld benchmark runner")
    parser.add_argument("--subset", required=True, type=Path,
                        help="Path to AppWorld subset directory containing data.jsonl")
    parser.add_argument("--model", default="openai/gpt-5-mini",
                        help="LiteLLM-compatible model ID")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel task workers")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Alias for interface compatibility")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory")
    parser.add_argument("--filter", default=None,
                        help="Only run this specific task_id")
    args = parser.parse_args()

    effective_workers = max(args.workers, args.concurrency // 4 if args.concurrency > 4 else 1)
    args.output.mkdir(parents=True, exist_ok=True)

    tasks = load_subset(args.subset)
    print(f"Loaded {len(tasks)} tasks from {args.subset}")

    if args.filter:
        tasks = [task for task in tasks if task.task_id == args.filter]
        if not tasks:
            print(f"ERROR: No task found with task_id={args.filter!r}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Filter applied: running 1 task ({args.filter})")

    existing_preds = load_preds(args.output)
    if existing_preds:
        before = len(tasks)
        tasks = [task for task in tasks if task.task_id not in existing_preds]
        print(f"Resume: skipped {before - len(tasks)} already-completed tasks")

    if not tasks:
        print(f"All tasks already completed ({len(existing_preds)} in preds.json). Nothing to do.")
        return

    print(
        f"Running {len(tasks)} AppWorld tasks | model={args.model} "
        f"| backend=official_simplified_react_code | workers={effective_workers}"
    )
    print(f"Output: {args.output}")
    print("")

    preds = dict(existing_preds)

    def _run_one(task):
        try:
            result = run_official_react_task(task, args.model, args.output)
            with _preds_lock:
                preds[result["task_id"]] = result["prediction"]
                save_preds(args.output, preds)
            return result
        except Exception as exc:
            with _preds_lock:
                preds[task.task_id] = ""
                save_preds(args.output, preds)
            return {
                "task_id": task.task_id,
                "prediction": "",
                "exit_status": "error",
                "score": 0.0,
                "success": False,
                "error": str(exc),
            }

    if effective_workers <= 1:
        for task in tasks:
            result = _run_one(task)
            print(f"  [{result['exit_status']:8s}] {result['task_id']}: score={result['score']:.2f}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                print(f"  [{result['exit_status']:8s}] {result['task_id']}: score={result['score']:.2f}")

    print(f"\nDone. {len(preds)} total predictions in {args.output}/preds.json")


if __name__ == "__main__":
    main()

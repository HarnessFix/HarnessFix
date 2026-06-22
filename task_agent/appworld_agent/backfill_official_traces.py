#!/usr/bin/env python3
"""Backfill normalized AppWorld official ReAct traces from official LM logs.

The runner now stores each chat-completion call as ``model_calls`` in
``<task_id>.traj.json``. This utility upgrades older official ReAct runs that
already have ``official_logs/lm_calls.jsonl`` without rerunning AppWorld tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2] if HERE.parent.name == "appworld_agent" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "task_agent" / "appworld_agent" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "agent_framework" / "src"))

from appworld_agent.official_trace import (  # noqa: E402
    SENSITIVE_LM_ARGUMENT_KEYS,
    messages_from_model_calls,
    read_official_messages,
    read_official_model_calls,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _task_dir_for_lm_calls(path: Path) -> Path:
    # .../<task_id>/official_logs/lm_calls.jsonl
    return path.parent.parent


def iter_official_trace_dirs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for lm_calls_path in sorted(root.glob("**/official_logs/lm_calls.jsonl")):
        task_dir = _task_dir_for_lm_calls(lm_calls_path)
        traj_paths = sorted(task_dir.glob("*.traj.json"))
        if not traj_paths:
            continue
        pairs.append((task_dir, traj_paths[0]))
    return pairs


def backfill_trace(task_dir: Path, traj_path: Path, *, dry_run: bool, force: bool) -> dict[str, Any]:
    traj = _load_json(traj_path)
    existing_model_calls = traj.get("model_calls") or []
    if existing_model_calls and not force:
        return {
            "status": "skipped",
            "reason": "model_calls already present",
            "task_dir": str(task_dir),
            "model_calls": len(existing_model_calls),
        }

    model_calls = read_official_model_calls(task_dir)
    logger_messages = read_official_messages(task_dir)
    messages = messages_from_model_calls(model_calls) or logger_messages
    if not model_calls and not logger_messages:
        return {
            "status": "skipped",
            "reason": "no parseable official logs",
            "task_dir": str(task_dir),
            "model_calls": 0,
        }

    info = traj.get("info")
    if not isinstance(info, dict):
        info = {}
        traj["info"] = info
    info["chat_trace"] = {
        "source": "official_logs/lm_calls.jsonl",
        "message_count": len(messages),
        "model_call_count": len(model_calls),
        "request_response_granularity": "message_level_chat_completion",
        "raw_provider_payload_available": bool(model_calls),
        "sensitive_fields_redacted": sorted(SENSITIVE_LM_ARGUMENT_KEYS),
    }
    traj["messages"] = messages
    traj["model_calls"] = model_calls
    traj["official_logger_messages"] = logger_messages
    traj.setdefault("trajectory_format", "appworld-official-simplified-react-adapter")

    if not dry_run:
        traj_path.write_text(json.dumps(traj, indent=2, ensure_ascii=False))

    return {
        "status": "updated",
        "task_dir": str(task_dir),
        "messages": len(messages),
        "model_calls": len(model_calls),
        "logger_messages": len(logger_messages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("traces"), help="Trace root to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report updates without writing files")
    parser.add_argument("--force", action="store_true", help="Rewrite traces that already contain model_calls")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N matching trace dirs")
    args = parser.parse_args()

    pairs = iter_official_trace_dirs(args.root)
    if args.limit:
        pairs = pairs[: args.limit]

    counts: dict[str, int] = {"updated": 0, "skipped": 0}
    total_model_calls = 0
    total_messages = 0
    examples: list[dict[str, Any]] = []
    for task_dir, traj_path in pairs:
        result = backfill_trace(task_dir, traj_path, dry_run=args.dry_run, force=args.force)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        total_model_calls += int(result.get("model_calls", 0) or 0)
        total_messages += int(result.get("messages", 0) or 0)
        if len(examples) < 5:
            examples.append(result)

    print(
        json.dumps(
            {
                "root": str(args.root),
                "dry_run": args.dry_run,
                "matched": len(pairs),
                "counts": counts,
                "total_model_calls": total_model_calls,
                "total_messages": total_messages,
                "examples": examples,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

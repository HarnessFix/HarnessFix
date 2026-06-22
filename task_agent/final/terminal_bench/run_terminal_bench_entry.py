#!/usr/bin/env python3
"""Run Terminal-Bench 2.0 tasks through local Harbor Terminus-2."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parents[1]
HARBOR_DIR = ADAPTER_DIR / "harbor"
HARBOR_SRC = HARBOR_DIR / "src"
DEFAULT_DATASET = REPO_ROOT / "data" / "terminal_bench_2_verified"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _model_info_for(model: str) -> dict[str, Any]:
    registry_path = REPO_ROOT / "task_agent" / "model_registry.json"
    if not registry_path.exists():
        return {}
    registry = _load_json(registry_path)
    item = registry.get(model, {})
    info: dict[str, Any] = {}
    for src, dst in (
        ("max_input_tokens", "max_input_tokens"),
        ("max_output_tokens", "max_output_tokens"),
        ("input_cost_per_token", "input_cost_per_token"),
        ("output_cost_per_token", "output_cost_per_token"),
    ):
        if item.get(src) is not None:
            info[dst] = item[src]
    return info


def _task_names_from_subset(subset: Path) -> list[str]:
    ids_file = subset / "instance_ids.txt"
    if ids_file.exists():
        return [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    data_file = subset / "data.jsonl"
    if data_file.exists():
        names: list[str] = []
        for line in data_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id") or row.get("instance_id") or row.get("name")
            if task_id:
                names.append(str(task_id))
        return names
    return []


def _dataset_args(subset: Path, limit: int | None) -> list[str]:
    if subset.exists():
        args = ["--path", str(subset)]
    else:
        args = ["--dataset", "terminal-bench@2.0"]
    for task_name in _task_names_from_subset(subset):
        args.extend(["--include-task-name", task_name])
    if limit is not None:
        args.extend(["--n-tasks", str(limit)])
    return args


def _harbor_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(HARBOR_SRC)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run_harbor(args: argparse.Namespace) -> Path:
    if not HARBOR_SRC.exists():
        raise SystemExit(f"Harbor source not found at {HARBOR_SRC}. Clone it locally first.")

    harbor_jobs_dir = args.output / "harbor_jobs"
    job_name = args.job_name or args.output.name
    cmd = [
        sys.executable,
        "-m", "harbor.cli.main", "run",
        "--agent", "terminus-2",
        "--model", args.model,
        "--jobs-dir", str(harbor_jobs_dir),
        "--job-name", job_name,
        "--n-concurrent", str(args.workers),
        "--yes", "--quiet",
        "--ak", f"store_all_messages={str(args.store_all_messages).lower()}",
        "--ak", "record_terminal_session=false",
    ]
    if args.max_turns is not None:
        cmd.extend(["--ak", f"max_turns={args.max_turns}"])
        cmd.extend(["--ak", "suppress_max_turns_warning=true"])

    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")
    if api_base:
        cmd.extend(["--ak", f"api_base={api_base}"])

    model_info = _model_info_for(args.model)
    if model_info:
        cmd.extend(["--ak", "model_info=" + json.dumps(model_info, separators=(",", ":"))])

    cmd.extend(_dataset_args(args.subset, args.limit))

    print("[terminal_bench] Running Harbor Terminus-2", flush=True)
    print("  $ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=_harbor_env(), check=True)
    return harbor_jobs_dir / job_name


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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


def _terminal_messages(result: dict[str, Any]) -> list[dict[str, Any]]:
    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    messages = metadata.get("all_messages") or []
    return messages if isinstance(messages, list) else []


def _model_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls.append({
            "call_index": len(calls) + 1,
            "request_messages": messages[:idx],
            "response_message": message,
        })
    return calls


def _normalize_task_id(value: Any, trial_name: str) -> str:
    if isinstance(value, str) and value:
        task_id = value.rsplit("/", 1)[-1]
    elif isinstance(value, dict):
        task_path = value.get("path") or value.get("name") or value.get("ref")
        task_id = Path(str(task_path)).name if task_path else trial_name
    else:
        task_id = trial_name
    if "__" in task_id:
        task_id = task_id.split("__", 1)[0]
    return task_id


def _trial_records(job_dir: Path, job_result: dict[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    records: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    for trial in job_result.get("trial_results", []) or []:
        trial_name = str(trial.get("trial_name") or trial.get("task_name") or "")
        trial_dir = job_dir / trial_name if trial_name else job_dir
        records.append((trial, trial_dir))
        if trial_name:
            seen.add(trial_name)

    for candidate in sorted(job_dir.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir() or candidate.name in seen:
            continue
        result_path = candidate / "result.json"
        config_path = candidate / "config.json"
        if result_path.exists():
            records.append((_load_json(result_path), candidate))
        elif config_path.exists():
            config = _load_json(config_path)
            records.append({"trial_name": candidate.name, "config": config}, candidate)
    return records


def _convert_outputs(job_dir: Path, output: Path, model: str, store_all_messages: bool | None = None) -> None:
    job_result_path = job_dir / "result.json"
    if not job_result_path.exists():
        raise SystemExit(f"Harbor result.json not found: {job_result_path}")

    job_result = _load_json(job_result_path)
    preds: dict[str, str] = {}
    index: list[dict[str, Any]] = []

    for trial, trial_dir in _trial_records(job_dir, job_result):
        result_path = trial_dir / "result.json"
        result = _load_json(result_path) if result_path.exists() else trial
        trial_name = str(result.get("trial_name") or trial.get("trial_name") or trial_dir.name)
        task_id = _normalize_task_id(result.get("task_name") or result.get("task_id"), trial_name)
        task_out = output / task_id
        task_out.mkdir(parents=True, exist_ok=True)

        reward = _reward_score(result)
        exception = result.get("exception_info")
        solved = bool(reward is not None and float(reward) >= 1.0 and not exception)
        exit_status = "solved" if solved else ("error" if exception else "unsolved")
        preds[task_id] = str(reward) if reward is not None else ""

        _write_json(task_out / "result.json", result)
        _copy_if_exists(trial_dir / "agent" / "trajectory.json", task_out / f"{task_id}.atif.json")
        _copy_if_exists(trial_dir / "agent" / "terminus_2.pane", task_out / "terminus_2.pane")
        _copy_if_exists(trial_dir / "verifier" / "test-stdout.txt", task_out / "test-stdout.txt")
        _copy_if_exists(trial_dir / "verifier" / "test-stderr.txt", task_out / "test-stderr.txt")
        _copy_if_exists(trial_dir / "verifier" / "reward.json", task_out / "reward.json")
        _copy_if_exists(trial_dir / "trial.log", task_out / "trial.log")
        _copy_if_exists(trial_dir / "exception.txt", task_out / "exception.txt")

        agent_result = result.get("agent_result") or {}
        messages = _terminal_messages(result)
        model_calls = _model_calls_from_messages(messages)
        chat_trace = {
            "source": "agent_result.metadata.all_messages",
            "message_count": len(messages),
            "model_call_count": len(model_calls),
            "request_response_granularity": "message_level_chat_history",
            "raw_provider_payload_available": False,
        }
        traj = {
            "trajectory_format": "harbor-terminus-2-atif-adapter",
            "instance_id": task_id,
            "messages": messages,
            "model_calls": model_calls,
            "info": {
                "exit_status": exit_status,
                "prediction": preds[task_id],
                "reward": reward,
                "model": model,
                "task_name": task_id,
                "trial_name": trial_name,
                "harbor_job_dir": str(job_dir),
                "agent": "terminus-2",
                "trajectory_atif_path": str(task_out / f"{task_id}.atif.json"),
                "exception": exception,
                "chat_trace": chat_trace,
                "model_stats": {
                    "api_calls": (agent_result.get("metadata") or {}).get("n_episodes"),
                    "instance_cost": agent_result.get("cost_usd"),
                    "input_tokens": agent_result.get("n_input_tokens"),
                    "output_tokens": agent_result.get("n_output_tokens"),
                },
            },
            "harbor_trial_result": result,
        }
        _write_json(task_out / f"{task_id}.traj.json", traj)
        index.append({
            "task_id": task_id,
            "trial_name": trial_name,
            "exit_status": exit_status,
            "reward": reward,
            "exception_type": (exception or {}).get("exception_type") if isinstance(exception, dict) else None,
        })

    _write_json(output / "preds.json", preds)
    _write_json(output / "run_info.json", {
        "benchmark": "terminal_bench_2",
        "agent": "terminus-2",
        "model": model,
        "trajectory_format": "harbor-terminus-2-atif-adapter",
        "harbor_job_dir": str(job_dir),
        "store_all_messages": store_all_messages,
        "n_predictions": len(preds),
        "tasks": index,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Terminal-Bench 2.0 with local Harbor Terminus-2")
    parser.add_argument("--subset", type=Path, default=DEFAULT_DATASET, help="Local Terminal-Bench dataset/subset directory")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--workers", type=int, default=2, help="Harbor --n-concurrent value")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--max-turns", type=int, default=None, help="Optional Terminus-2 max_turns cap")
    parser.add_argument("--no-store-all-messages", dest="store_all_messages", action="store_false", help="Do not store full Terminus-2 chat messages in Harbor TrialResult metadata")
    parser.set_defaults(store_all_messages=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    job_dir = _run_harbor(args)
    _convert_outputs(job_dir, args.output, args.model, args.store_all_messages)

    print(f"Done. Traces saved to: {args.output}")
    print(f"  preds.json   : {args.output / 'preds.json'}")
    print(f"  traj files   : {args.output}/<task_id>/<task_id>.traj.json")
    print(f"  Harbor job   : {job_dir}")


if __name__ == "__main__":
    main()

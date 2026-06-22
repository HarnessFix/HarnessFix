#!/usr/bin/env python3
"""GAIA benchmark runner entry point.

This script is the CLI entry point called by run_gaia.sh.
It closely mirrors the logic from smolagents/examples/open_deep_research/run_gaia.py,
adapted to work with our local dataset format and pipeline trace conventions.

Output layout:
  <output>/<task_id>/<task_id>.traj.json   — per-task trace
  <output>/preds.json                       — {task_id: prediction_str, ...}

Traj JSON format:
  {
    "instance_id": "<task_id>",
    "info": {
      "exit_status": "solved" | "unsolved" | "error",
      "prediction": "<answer string>",
      "elapsed_seconds": 12.3,
      "model_id": "openai/gpt-4o",
      "parsing_error": false,
      "iteration_limit_exceeded": false
    },
    "trajectory": [<agent memory steps as strings>]
  }

Resume support: tasks already in preds.json are skipped automatically.

DO NOT MODIFY THIS FILE — it is the pipeline's stable interface.
To change agent behavior, modify files in src/open_deep_research/ instead.
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load env from project root
_HERE = Path(__file__).parent
_PROJECT_ROOT = _HERE.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# HF login (required for gated models; GAIA dataset uses local files so optional here)
try:
    from huggingface_hub import login as hf_login
    _hf_token = os.environ.get("HF_TOKEN")
    if _hf_token:
        hf_login(_hf_token)
except Exception:
    pass

# Import agent — PYTHONPATH override (set by pipeline for enhanced versions) takes precedence
from open_deep_research.agent import create_agent_team
from open_deep_research.scoring import question_scorer
from open_deep_research.prompts import AUGMENTED_QUESTION_PREFIX, ANSWER_FORMAT_REMINDER
from open_deep_research.scripts.reformulator import prepare_response
from open_deep_research.scripts.run_agents import get_single_file_description, get_zip_description
from open_deep_research.scripts.text_inspector_tool import TextInspectorTool
from open_deep_research.scripts.visual_qa import visualizer


DEFAULT_TASK_TIMEOUT_SECONDS = int(os.environ.get("GAIA_TASK_TIMEOUT_SECONDS", "1800"))


# ── Step serialization ─────────────────────────────────────────────────────────

def _step_to_dict(step) -> dict:
    """Convert a smolagents memory step to a JSON-serializable dict."""
    import dataclasses
    import enum

    def _safe(obj):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, enum.Enum):
            return obj.value
        if isinstance(obj, dict):
            return {str(_safe(k)): _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: _safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if hasattr(obj, "dict") and callable(obj.dict):
            try:
                return _safe(obj.dict())
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return _safe(vars(obj))
        return str(obj)

    raw = _safe(step)
    raw["_type"] = type(step).__name__
    return raw


# ── Data loading ───────────────────────────────────────────────────────────────

def load_subset(subset_dir: Path) -> list[dict]:
    """Load tasks from a GAIA subset directory (data.jsonl)."""
    data_file = subset_dir / "data.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"data.jsonl not found in {subset_dir}")
    tasks = []
    for line in data_file.read_text().splitlines():
        line = line.strip()
        if line:
            task = json.loads(line)
            task["_subset_dir"] = str(subset_dir)
            tasks.append(task)
    return tasks


def load_preds(output_dir: Path) -> dict[str, str]:
    """Load existing predictions (for resume support)."""
    preds_file = output_dir / "preds.json"
    if preds_file.exists():
        try:
            return json.loads(preds_file.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


_preds_lock = threading.Lock()


def save_preds(output_dir: Path, preds: dict[str, str]) -> None:
    """Atomically save predictions dict to preds.json."""
    preds_file = output_dir / "preds.json"
    tmp_file = preds_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    tmp_file.rename(preds_file)


def _find_local_attachment(task: dict, subset_dir: Path | None, task_dir: Path) -> Path | None:
    """Return a usable local path for a GAIA attachment, downloading it if needed."""
    file_name = task.get("file_name") or ""
    if not file_name:
        return None

    candidates: list[Path] = []
    file_path = task.get("file_path") or ""
    for raw_path in (file_name, file_path):
        if not raw_path:
            continue
        path = Path(raw_path)
        candidates.append(path)
        if subset_dir is not None and not path.is_absolute():
            candidates.append(subset_dir / path)
            candidates.append(subset_dir / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None

    remote_names = []
    if file_path:
        remote_names.append(file_path)
    remote_names.extend([f"2023/validation/{Path(file_name).name}", file_name])

    seen: set[str] = set()
    for remote_name in remote_names:
        if not remote_name or remote_name in seen:
            continue
        seen.add(remote_name)
        try:
            from huggingface_hub import hf_hub_download

            downloaded = Path(
                hf_hub_download(
                    repo_id="gaia-benchmark/GAIA",
                    repo_type="dataset",
                    filename=remote_name,
                    token=hf_token,
                )
            )
        except Exception:
            continue
        if downloaded.exists():
            local_path = task_dir / Path(file_name).name
            if not local_path.exists():
                shutil.copy2(downloaded, local_path)
            return local_path.resolve()

    return None


# ── Single-task runner ─────────────────────────────────────────────────────────

def _build_augmented_question(task: dict, model, document_inspection_tool, attachment_path: Path | None = None) -> str:
    """Construct the augmented question with file context (mirrors run_gaia.py)."""
    augmented = AUGMENTED_QUESTION_PREFIX + task["question"]

    file_name = str(attachment_path) if attachment_path else task.get("file_name", "")
    if file_name and os.path.exists(file_name):
        if ".zip" in file_name:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                file_name, task["question"], visualizer, document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                file_name, task["question"], visualizer, document_inspection_tool
            )
        augmented += prompt_use_files

    if ANSWER_FORMAT_REMINDER:
        augmented += "\n\n" + ANSWER_FORMAT_REMINDER

    return augmented


def run_single_task(task: dict, model_id: str, output_dir: Path) -> dict:
    """Run the agent on a single GAIA task.

    Writes the traj file and returns a result dict.
    """
    from open_deep_research.tracing import TracingLiteLLMModel, model_trace_context
    from open_deep_research.config import CUSTOM_ROLE_CONVERSIONS, TEXT_LIMIT, DEFAULT_MAX_TOKENS

    task_id = task["task_id"]
    true_answer = task.get("true_answer", "")
    subset_dir = Path(task["_subset_dir"]) if task.get("_subset_dir") else None

    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    traj_path = task_dir / f"{task_id}.traj.json"

    start_time = time.time()
    prediction = ""
    exit_status = "error"
    parsing_error = False
    iteration_limit_exceeded = False
    trajectory_steps: list = []
    model_calls: list = []
    trace_recorder = None
    agent = None

    try:
        # Build model for document inspection
        model_kwargs: dict = {
            "model_id": model_id,
            "custom_role_conversions": CUSTOM_ROLE_CONVERSIONS,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
        if api_base:
            model_kwargs["api_base"] = api_base
        if api_key:
            model_kwargs["api_key"] = api_key
        with model_trace_context(task_id) as trace_recorder:
            file_context_model = TracingLiteLLMModel(**model_kwargs, trace_component="file_context", trace_recorder=trace_recorder)
            document_inspection_tool = TextInspectorTool(file_context_model, TEXT_LIMIT)

            # Build augmented question
            attachment_path = _find_local_attachment(task, subset_dir, task_dir)
            augmented_question = _build_augmented_question(
                task, file_context_model, document_inspection_tool, attachment_path=attachment_path
            )

            # Create agent and run
            agent = create_agent_team(model_id)
            raw_result = agent.run(augmented_question)

            # Use reformulator to clean up the answer (mirrors run_gaia.py)
            try:
                agent_memory = agent.write_memory_to_messages()
                reformulation_model = TracingLiteLLMModel(**model_kwargs, trace_component="reformulator", trace_recorder=trace_recorder)
                final_result = prepare_response(augmented_question, agent_memory, reformulation_model=reformulation_model)
                prediction = str(final_result)
            except Exception:
                prediction = str(raw_result) if raw_result is not None else ""

            model_calls = list(trace_recorder.model_calls)

        # Detect error flags
        iteration_limit_exceeded = "Agent stopped due to iteration limit" in prediction
        try:
            agent_memory_str = str(agent.write_memory_to_messages())
            parsing_error = "AgentParsingError" in agent_memory_str
        except Exception:
            parsing_error = False

        exit_status = "solved" if question_scorer(prediction, true_answer) else "unsolved"

        # Collect trajectory as JSON-serializable dicts
        try:
            for step in agent.memory.steps:
                try:
                    trajectory_steps.append(_step_to_dict(step))
                except Exception:
                    trajectory_steps.append({"_raw": str(step)})
        except Exception:
            pass

    except Exception:
        prediction = ""
        exit_status = "error"
        if trace_recorder is not None:
            model_calls = list(trace_recorder.model_calls)
        trajectory_steps.append({"_error": traceback.format_exc()})

    elapsed = time.time() - start_time

    traj_data = {
        "instance_id": task_id,
        "info": {
            "exit_status": exit_status,
            "prediction": prediction,
            "elapsed_seconds": round(elapsed, 2),
            "model_id": model_id,
            "parsing_error": parsing_error,
            "iteration_limit_exceeded": iteration_limit_exceeded,
            "chat_trace": {
                "source": "open_deep_research.tracing.TracingLiteLLMModel",
                "model_call_count": len(model_calls),
                "request_response_granularity": "message_level_model_call",
                "raw_provider_payload_available": True,
                "sensitive_fields_redacted": sorted(["Authorization", "api_key", "authorization"]),
            },
        },
        "trajectory": trajectory_steps,
        "model_calls": model_calls,
        "trajectory_format": "open-deep-research-smolagents-trace",
    }
    traj_path.write_text(json.dumps(traj_data, ensure_ascii=False, indent=2))

    return {
        "task_id": task_id,
        "prediction": prediction,
        "exit_status": exit_status,
    }


def _write_timeout_trace(task: dict, model_id: str, output_dir: Path, elapsed: float) -> dict:
    task_id = task["task_id"]
    prediction = "Unable to determine"
    exit_status = "unsolved"
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    traj_path = task_dir / f"{task_id}.traj.json"
    traj_data = {
        "instance_id": task_id,
        "info": {
            "exit_status": exit_status,
            "prediction": prediction,
            "elapsed_seconds": round(elapsed, 2),
            "model_id": model_id,
            "parsing_error": False,
            "iteration_limit_exceeded": False,
            "timeout_exceeded": True,
            "chat_trace": {
                "source": "open_deep_research.tracing.TracingLiteLLMModel",
                "model_call_count": 0,
                "request_response_granularity": "message_level_model_call",
                "raw_provider_payload_available": False,
                "sensitive_fields_redacted": sorted(["Authorization", "api_key", "authorization"]),
            },
        },
        "trajectory": [{"_error": f"Task exceeded timeout after {round(elapsed, 2)} seconds."}],
        "model_calls": [],
        "trajectory_format": "open-deep-research-smolagents-trace",
    }
    traj_path.write_text(json.dumps(traj_data, ensure_ascii=False, indent=2))
    return {"task_id": task_id, "prediction": prediction, "exit_status": exit_status}


def _write_child_error_trace(task: dict, model_id: str, output_dir: Path, elapsed: float, message: str) -> dict:
    task_id = task["task_id"]
    prediction = ""
    task_dir = output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    traj_path = task_dir / f"{task_id}.traj.json"
    traj_data = {
        "instance_id": task_id,
        "info": {
            "exit_status": "error",
            "prediction": prediction,
            "elapsed_seconds": round(elapsed, 2),
            "model_id": model_id,
            "parsing_error": False,
            "iteration_limit_exceeded": False,
            "timeout_exceeded": False,
            "chat_trace": {
                "source": "open_deep_research.tracing.TracingLiteLLMModel",
                "model_call_count": 0,
                "request_response_granularity": "message_level_model_call",
                "raw_provider_payload_available": False,
                "sensitive_fields_redacted": sorted(["Authorization", "api_key", "authorization"]),
            },
        },
        "trajectory": [{"_error": message}],
        "model_calls": [],
        "trajectory_format": "open-deep-research-smolagents-trace",
    }
    traj_path.write_text(json.dumps(traj_data, ensure_ascii=False, indent=2))
    return {"task_id": task_id, "prediction": prediction, "exit_status": "error"}


def _run_single_task_child(task: dict, model_id: str, output_dir: str, result_queue) -> None:
    try:
        result_queue.put(run_single_task(task, model_id, Path(output_dir)))
    except Exception:
        result_queue.put(
            {
                "task_id": task.get("task_id", "unknown"),
                "prediction": "",
                "exit_status": "error",
                "error": traceback.format_exc(),
            }
        )


def run_single_task_with_timeout(
    task: dict,
    model_id: str,
    output_dir: Path,
    timeout_seconds: int,
) -> dict:
    if timeout_seconds <= 0:
        return run_single_task(task, model_id, output_dir)

    start_time = time.time()
    ctx = mp.get_context(os.environ.get("GAIA_TASK_TIMEOUT_START_METHOD", "spawn"))
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_run_single_task_child,
        args=(task, model_id, str(output_dir), result_queue),
        daemon=False,
    )
    proc.start()
    proc.join(timeout_seconds)

    elapsed = time.time() - start_time
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return _write_timeout_trace(task, model_id, output_dir, elapsed)

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        return _write_child_error_trace(
            task,
            model_id,
            output_dir,
            elapsed,
            f"Task worker exited with code {proc.exitcode} without returning a result.",
        )

    if result.get("error"):
        return _write_child_error_trace(task, model_id, output_dir, elapsed, result["error"])
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GAIA benchmark runner (open_deep_research)")
    parser.add_argument("--subset", required=True, type=Path,
                        help="Path to GAIA subset directory containing data.jsonl")
    parser.add_argument("--model", default="openai/gpt-5-mini",
                        help="LiteLLM-compatible model ID (default: openai/gpt-5-mini)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel task workers (default: 1)")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Alias for --workers, kept for script compatibility")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for traces and preds.json")
    parser.add_argument("--filter", default=None,
                        help="Only run this specific task_id (for smoke testing)")
    parser.add_argument("--task-timeout-seconds", type=int, default=DEFAULT_TASK_TIMEOUT_SECONDS,
                        help="Per-task wall-clock timeout. Set 0 to disable.")
    args = parser.parse_args()

    effective_workers = max(args.workers, args.concurrency // 4 if args.concurrency > 4 else 1)

    # Setup output dir
    args.output.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_subset(args.subset)
    print(f"Loaded {len(tasks)} tasks from {args.subset}")

    # Apply filter
    if args.filter:
        tasks = [t for t in tasks if t["task_id"] == args.filter]
        if not tasks:
            print(f"ERROR: No task found with task_id={args.filter!r}", file=sys.stderr)
            sys.exit(1)
        print(f"Filter applied: running 1 task ({args.filter})")

    # Resume: skip already-completed tasks
    existing_preds = load_preds(args.output)
    if existing_preds:
        before = len(tasks)
        tasks = [t for t in tasks if t["task_id"] not in existing_preds]
        print(f"Resume: skipped {before - len(tasks)} already-completed tasks")

    if not tasks:
        print(f"All tasks already completed ({len(existing_preds)} in preds.json). Nothing to do.")
        return

    print(
        f"Running {len(tasks)} tasks | model={args.model} | "
        f"workers={effective_workers} | task_timeout={args.task_timeout_seconds}s"
    )
    print(f"Output: {args.output}")
    print("")

    preds: dict[str, str] = dict(existing_preds)

    def _run_one(task: dict) -> tuple[str, str, str]:
        try:
            result = run_single_task_with_timeout(
                task,
                args.model,
                args.output,
                args.task_timeout_seconds,
            )
            task_id = result["task_id"]
            pred = result["prediction"]
            status = result["exit_status"]
            with _preds_lock:
                preds[task_id] = pred
                save_preds(args.output, preds)
            return task_id, pred, status
        except Exception as exc:
            task_id = task.get("task_id", "unknown")
            print(f"  [error] {task_id}: {exc}", file=sys.stderr)
            with _preds_lock:
                preds[task_id] = ""
                save_preds(args.output, preds)
            return task_id, "", "error"

    if effective_workers <= 1:
        for task in tasks:
            task_id, pred, status = _run_one(task)
            print(f"  [{status:8s}] {task_id}: {pred[:80]!r}")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_run_one, t): t for t in tasks}
            for future in as_completed(futures):
                task_id, pred, status = future.result()
                print(f"  [{status:8s}] {task_id}: {pred[:80]!r}")

    print(f"\nDone. {len(preds)} total predictions in {args.output}/preds.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {
    "empty_patch_rate",
    "error_rate",
    "loop_rate",
    "invalid_submission_rate",
    "repeated_search_rate",
    "repeated_command_rate",
    "syntax_error_rate",
    "avg_instance_cost",
    "avg_steps",
    "avg_api_calls",
    "collateral_damage_rate",
    "assertion_failure_rate",
    "repeated_api_call_rate",
    "repeated_bad_repair_rate",
    "iterations_to_pass_gate",
    "missing_evidence_rate",
    "collateral_regression_rate",
    "delegation_failure_rate",
    "completion_without_effect_rate",
}

HIGHER_IS_BETTER = {
    "resolved_rate",
    "accuracy",
    "exact_match",
    "task_success_rate",
    "trace_coverage",
    "attribution_agreement",
    "terminal_task_success_rate",
}


def load_eval_json(path_or_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path_or_dir)
    if path.is_dir():
        json_files = [item for item in path.glob("*.json") if not item.name.startswith("run_")]
        if not json_files:
            raise FileNotFoundError(f"No JSON report found in {path}")
        path = json_files[0]
    return path, json.loads(path.read_text())


def _normalized_commands_from_swe(messages: list[dict[str, Any]]) -> list[str]:
    commands = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for action in message.get("extra", {}).get("actions", []):
            command = action.get("command", "")
            if not command:
                continue
            commands.append(re.sub(r"\s+", " ", command.strip()))
    return commands


def _max_run(commands: list[str]) -> int:
    if not commands:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(commands)):
        if commands[idx] == commands[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _gave_up_prediction(prediction: str) -> bool:
    text = (prediction or "").lower()
    return not text.strip() or any(
        token in text
        for token in (
            "unable to determine",
            "cannot find",
            "could not find",
            "cannot answer",
            "not available",
        )
    )


def _swe_eval_metrics(eval_data: dict[str, Any], total: int) -> dict[str, float]:
    resolved = len(eval_data.get("resolved_ids", []))
    unresolved = len(eval_data.get("unresolved_ids", []))
    errors = len(eval_data.get("error_ids", []))
    empty = len(eval_data.get("empty_patch_ids", []))
    return {
        "resolved_rate": resolved / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "empty_patch_rate": empty / total if total else 0.0,
        "invalid_submission_rate": errors / total if total else 0.0,
        "unresolved_rate": unresolved / total if total else 0.0,
    }


def compute_swe_metrics(traces_dir: str | Path, eval_path: str | Path) -> dict[str, Any]:
    traces_dir = Path(traces_dir)
    _, eval_data = load_eval_json(eval_path)
    traj_files = sorted(traces_dir.glob("*/*.traj.json"))
    total = len(traj_files)
    api_calls = []
    costs = []
    loop_instances = 0
    validation_instances = 0
    syntax_error_instances = 0

    for traj_file in traj_files:
        data = json.loads(traj_file.read_text())
        info = data.get("info", {})
        api_calls.append(info.get("model_stats", {}).get("api_calls", 0) or 0)
        costs.append(info.get("model_stats", {}).get("instance_cost", 0.0) or 0.0)
        commands = _normalized_commands_from_swe(data.get("messages", []))
        if _max_run(commands) >= 5:
            loop_instances += 1
        if any(token in command for command in commands for token in ("pytest", "py_compile", "unittest", "runtests.py")):
            validation_instances += 1
        if any(
            any(error_token in str(message.get("content", "")) for error_token in ("SyntaxError", "IndentationError"))
            for message in data.get("messages", [])
            if message.get("role") == "user"
        ):
            syntax_error_instances += 1

    metrics = _swe_eval_metrics(eval_data, total)
    metrics.update(
        {
            "avg_api_calls": sum(api_calls) / total if total else 0.0,
            "avg_instance_cost": sum(costs) / total if total else 0.0,
            "loop_rate": loop_instances / total if total else 0.0,
            "verification_effort_ratio": validation_instances / total if total else 0.0,
            "syntax_error_rate": syntax_error_instances / total if total else 0.0,
        }
    )
    return {"mode": "swe", "total_instances": total, "metrics": metrics}


def compute_gaia_metrics(traces_dir: str | Path, eval_path: str | Path) -> dict[str, Any]:
    traces_dir = Path(traces_dir)
    _, eval_data = load_eval_json(eval_path)
    traj_files = sorted(traces_dir.glob("*/*.traj.json"))
    total = len(traj_files)
    steps = []
    elapsed = []
    empty_predictions = 0
    repeated_search = 0

    for traj_file in traj_files:
        data = json.loads(traj_file.read_text())
        info = data.get("info", {})
        prediction = info.get("prediction", "") or ""
        if _gave_up_prediction(prediction):
            empty_predictions += 1
        trajectory = data.get("trajectory", [])
        steps.append(len(trajectory))
        elapsed.append(float(info.get("elapsed_seconds", 0.0) or 0.0))
        search_mentions = [
            json.dumps(step, ensure_ascii=False).lower().count("google")
            + json.dumps(step, ensure_ascii=False).lower().count("search")
            for step in trajectory
        ]
        if sum(search_mentions) >= 4:
            repeated_search += 1

    if "accuracy" in eval_data:
        accuracy = float(eval_data.get("accuracy", 0.0) or 0.0)
    else:
        total_correct = len(eval_data.get("correct_ids", eval_data.get("resolved_ids", [])))
        accuracy = total_correct / total if total else 0.0
    metrics = {
        "accuracy": accuracy,
        "exact_match": accuracy,
        "error_rate": len(eval_data.get("error_ids", [])) / total if total else 0.0,
        "empty_prediction_rate": empty_predictions / total if total else 0.0,
        "repeated_search_rate": repeated_search / total if total else 0.0,
        "avg_steps": sum(steps) / total if total else 0.0,
        "avg_elapsed_seconds": sum(elapsed) / total if total else 0.0,
    }
    return {"mode": "gaia", "total_instances": total, "metrics": metrics}


def _appworld_completion_attempted(data: dict[str, Any], commands: list[str]) -> bool:
    info = data.get("info", {})
    if info.get("prediction") == "__TASK_COMPLETED__" or info.get("agent_exit_status") in {"Submitted", "OfficialReActCompleted"}:
        return True
    joined = "\n".join(commands).lower()
    return "complete_task" in joined or "__task_completed__" in joined


def _appworld_missing_effect_failure(failure_text: str) -> bool:
    text = failure_text.lower()
    return any(
        token in text
        for token in (
            "model changes match",
            "changed records",
            "changed_model_names",
            "number of new",
            "left operand is empty",
            "set() ==",
            "no new",
            "not created",
            "missing",
            "paymentrequest",
            "notification",
        )
    )


def compute_appworld_metrics(traces_dir: str | Path, eval_path: str | Path) -> dict[str, Any]:
    traces_dir = Path(traces_dir)
    _, eval_data = load_eval_json(eval_path)
    traj_files = sorted(traces_dir.glob("*/*.traj.json"))
    total = len(traj_files)
    api_calls = []
    costs = []
    step_counts = []
    repeated_code = 0
    unsafe_failures = 0
    assertion_failures = 0
    completion_without_effect = 0

    for traj_file in traj_files:
        data = json.loads(traj_file.read_text())
        info = data.get("info", {})
        api_calls.append(info.get("model_stats", {}).get("api_calls", 0) or 0)
        costs.append(info.get("model_stats", {}).get("instance_cost", 0.0) or 0.0)
        messages = data.get("messages", [])
        commands = _normalized_commands_from_swe(messages)
        step_counts.append(len([m for m in messages if m.get("role") == "assistant"]))
        if _max_run(commands) >= 4:
            repeated_code += 1

        result_path = traj_file.parent / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                result = {}
            failures = result.get("failures", [])
            failure_text = " ".join(json.dumps(item, ensure_ascii=False) for item in failures).lower()
            if any(token in failure_text for token in ("collateral", "unexpected", "should not", "extra change")):
                unsafe_failures += 1
            if failures:
                assertion_failures += 1
            if _appworld_completion_attempted(data, commands) and _appworld_missing_effect_failure(failure_text):
                completion_without_effect += 1

    task_success_rate = len(eval_data.get("resolved_ids", [])) / total if total else 0.0
    metrics = {
        "resolved_rate": task_success_rate,
        "task_success_rate": task_success_rate,
        "error_rate": len(eval_data.get("error_ids", [])) / total if total else 0.0,
        "empty_patch_rate": len(eval_data.get("empty_patch_ids", [])) / total if total else 0.0,
        "invalid_submission_rate": len(eval_data.get("error_ids", [])) / total if total else 0.0,
        "collateral_damage_rate": unsafe_failures / total if total else 0.0,
        "assertion_failure_rate": assertion_failures / total if total else 0.0,
        "repeated_api_call_rate": repeated_code / total if total else 0.0,
        "completion_without_effect_rate": completion_without_effect / total if total else 0.0,
        "avg_api_calls": sum(api_calls) / total if total else 0.0,
        "avg_instance_cost": sum(costs) / total if total else 0.0,
        "avg_steps": sum(step_counts) / total if total else 0.0,
    }
    return {"mode": "appworld", "total_instances": total, "metrics": metrics}



def compute_terminal_bench_metrics(traces_dir: str | Path, eval_path: str | Path) -> dict[str, Any]:
    traces_dir = Path(traces_dir)
    _, eval_data = load_eval_json(eval_path)
    traj_files = sorted(traces_dir.glob("*/*.traj.json"))
    total = len(traj_files)
    api_calls = []
    costs = []
    steps = []
    repeated_commands = 0
    missing_atif = 0
    verifier_failures = 0

    for traj_file in traj_files:
        data = json.loads(traj_file.read_text())
        info = data.get("info", {})
        stats = info.get("model_stats", {}) or {}
        api_calls.append(stats.get("api_calls", 0) or 0)
        costs.append(stats.get("instance_cost", 0.0) or 0.0)
        atif_path = traj_file.parent / f"{traj_file.parent.name}.atif.json"
        if not atif_path.exists():
            missing_atif += 1
            steps.append(0)
            continue
        try:
            atif = json.loads(atif_path.read_text())
        except json.JSONDecodeError:
            missing_atif += 1
            steps.append(0)
            continue
        atif_steps = atif.get("steps", [])
        steps.append(len(atif_steps))
        commands = []
        for step in atif_steps:
            for call in step.get("tool_calls", []) or []:
                commands.append(re.sub(r"\s+", " ", json.dumps(call, ensure_ascii=False).strip()))
        if _max_run(commands) >= 4:
            repeated_commands += 1
        result_path = traj_file.parent / "result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                result = {}
            if result.get("verifier_result") and info.get("exit_status") != "solved":
                verifier_failures += 1

    success_rate = len(eval_data.get("resolved_ids", [])) / total if total else 0.0
    metrics = {
        "resolved_rate": success_rate,
        "terminal_task_success_rate": success_rate,
        "error_rate": len(eval_data.get("error_ids", [])) / total if total else 0.0,
        "empty_patch_rate": len(eval_data.get("empty_patch_ids", [])) / total if total else 0.0,
        "repeated_command_rate": repeated_commands / total if total else 0.0,
        "missing_evidence_rate": missing_atif / total if total else 0.0,
        "assertion_failure_rate": verifier_failures / total if total else 0.0,
        "avg_api_calls": sum(api_calls) / total if total else 0.0,
        "avg_instance_cost": sum(costs) / total if total else 0.0,
        "avg_steps": sum(steps) / total if total else 0.0,
    }
    return {"mode": "terminal_bench", "total_instances": total, "metrics": metrics}

def compute_run_metrics(mode: str, traces_dir: str | Path, eval_path: str | Path) -> dict[str, Any]:
    if mode == "gaia":
        return compute_gaia_metrics(traces_dir, eval_path)
    if mode == "appworld":
        return compute_appworld_metrics(traces_dir, eval_path)
    if mode == "terminal_bench":
        return compute_terminal_bench_metrics(traces_dir, eval_path)
    return compute_swe_metrics(traces_dir, eval_path)


def _metric_direction(metric_name: str) -> str:
    if metric_name in LOWER_IS_BETTER:
        return "lower"
    if metric_name in HIGHER_IS_BETTER:
        return "higher"
    if metric_name.endswith("_rate") or metric_name.endswith("_cost"):
        return "lower"
    return "higher"


def evaluate_target_metrics(plan_spec: dict[str, Any], baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    requested_metrics: list[str] = []
    for fix in plan_spec.get("fixes", []):
        for metric in fix.get("target_metrics", []):
            if metric not in requested_metrics:
                requested_metrics.append(metric)
    if not requested_metrics:
        requested_metrics = ["accuracy"] if baseline.get("mode") == "gaia" else ["resolved_rate"]

    results = []
    improvement_signals = []
    for metric in requested_metrics:
        baseline_value = baseline["metrics"].get(metric, 0.0)
        current_value = current["metrics"].get(metric, 0.0)
        direction = _metric_direction(metric)
        delta = current_value - baseline_value if direction == "higher" else baseline_value - current_value
        improved = delta > 1e-9
        results.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "current": current_value,
                "direction": direction,
                "delta": delta,
                "improved": improved,
            }
        )
        improvement_signals.append(delta)

    return {
        "metrics": results,
        "aggregate_delta": sum(improvement_signals) / len(improvement_signals) if improvement_signals else 0.0,
        "improved_metric_count": sum(1 for item in results if item["improved"]),
    }


def compute_cost_ratio(baseline: dict[str, Any], current: dict[str, Any]) -> float:
    baseline_cost = baseline["metrics"].get("avg_instance_cost")
    current_cost = current["metrics"].get("avg_instance_cost")
    if baseline_cost and current_cost:
        return current_cost / baseline_cost if baseline_cost else 1.0
    baseline_steps = baseline["metrics"].get("avg_steps") or baseline["metrics"].get("avg_api_calls") or 1.0
    current_steps = current["metrics"].get("avg_steps") or current["metrics"].get("avg_api_calls") or baseline_steps
    return current_steps / baseline_steps if baseline_steps else 1.0

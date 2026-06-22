#!/usr/bin/env python3
"""Closed-loop HarnessFix pipeline for Terminal-Bench 2.0 via local Harbor Terminus-2.

The default task agent is only Harbor `terminus-2`. Other Harbor agents are not
wired into the experiment path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

REPO_ROOT = Path(__file__).parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
TASK_AGENT_DIR = REPO_ROOT / "task_agent"
TERMINAL_AGENT_DIR = TASK_AGENT_DIR / "terminal_bench_agent"
DATA_DIR = REPO_ROOT / "data"
FAILURE_ANALYSIS_DIR = REPO_ROOT / "failure_analysis"
IMPROVEMENT_PLANS_DIR = REPO_ROOT / "improvement_plans"
TRACES_DIR = REPO_ROOT / "traces"
EVAL_DIR = REPO_ROOT / "eval"

TRAIN_DIR = DATA_DIR / "terminal_bench_train"
VAL_DIR = DATA_DIR / "terminal_bench_val"
TEST_DIR = DATA_DIR / "terminal_bench_test"
FULL_DATASET_DIR = DATA_DIR / "terminal_bench_2_verified"


def log(msg: str) -> None:
    print(f"\n[pipeline_terminal_bench] {msg}", flush=True)


def run(cmd: list[str], env: dict | None = None, check: bool = True,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    merged_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, env=merged_env, check=check, timeout=timeout)


def find_eval_json(directory: Path) -> Path | None:
    jsons = [item for item in directory.glob("*.json") if not item.name.startswith("run_")]
    return jsons[0] if jsons else None


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace("-", "_").lower()


def plan_spec_path(plan_path: Path) -> Path:
    return plan_path.with_suffix(".json")


def base_agent_dir(version: int) -> Path:
    if version <= 0:
        return TERMINAL_AGENT_DIR
    return TASK_AGENT_DIR / f"enhanced_terminal_bench_v{version}"


def base_agent_label(version: int) -> str:
    return "original" if version <= 0 else f"terminus2_enhanced_v{version}"


def train_run_id(model: str, version: int) -> str:
    return f"terminal_bench_train_{model_slug(model)}_{base_agent_label(version)}"


def val_baseline_run_id(model: str) -> str:
    return f"terminal_bench_val_{model_slug(model)}_original"


def val_enhanced_run_id(model: str, version: int) -> str:
    return f"terminal_bench_val_{model_slug(model)}_enhanced_v{version}"


def train_traces_dir(model: str, version: int) -> Path:
    return TRACES_DIR / train_run_id(model, version)


def train_eval_dir(model: str, version: int) -> Path:
    return EVAL_DIR / f"terminal_bench_results_{train_run_id(model, version)}"


def val_enhanced_traces_dir(model: str, version: int) -> Path:
    return TRACES_DIR / val_enhanced_run_id(model, version)


def val_enhanced_eval_dir(model: str, version: int) -> Path:
    return EVAL_DIR / f"terminal_bench_results_{val_enhanced_run_id(model, version)}"


def val_traces_dir_for_base(model: str, base_version: int) -> Path:
    return TRACES_DIR / val_baseline_run_id(model) if base_version <= 0 else val_enhanced_traces_dir(model, base_version)


def val_eval_dir_for_base(model: str, base_version: int) -> Path:
    if base_version <= 0:
        return EVAL_DIR / f"terminal_bench_results_{val_baseline_run_id(model)}"
    return val_enhanced_eval_dir(model, base_version)


def resolved_count_from_eval(eval_dir: Path) -> int:
    eval_json = find_eval_json(eval_dir)
    if not eval_json:
        return 0
    data = json.loads(eval_json.read_text())
    subset = set((VAL_DIR / "instance_ids.txt").read_text().split())
    return len(set(data.get("resolved_ids", [])) & subset)


def best_version_by_val(model: str, versions: list[int]) -> int:
    best_version = 0
    best_score = resolved_count_from_eval(val_eval_dir_for_base(model, 0))
    for version in versions:
        score = resolved_count_from_eval(val_enhanced_eval_dir(model, version))
        if score > best_score:
            best_version = version
            best_score = score
    return best_version


def step_val_baseline_inference(model: str, workers: int, force: bool, dry_run: bool) -> Path:
    output_dir = TRACES_DIR / val_baseline_run_id(model)
    log(f"Bootstrap: val baseline inference -> {output_dir.name}/")
    if (output_dir / "preds.json").exists() and not force:
        log("  val baseline preds already exists, skipping")
        return output_dir
    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_terminal_bench.sh"),
        "--subset", str(VAL_DIR),
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
    ]
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return output_dir
    run(cmd, timeout=48 * 3600)
    return output_dir


def step_val_baseline_evaluate(model: str, baseline_traces: Path, force: bool, dry_run: bool) -> Path:
    report_dir = EVAL_DIR / f"terminal_bench_results_{val_baseline_run_id(model)}"
    log(f"Bootstrap: val baseline evaluate -> {report_dir.name}/")
    if find_eval_json(report_dir) and not force:
        log("  val baseline eval results already exists, skipping")
        return report_dir
    output_file = report_dir / "results.json"
    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_terminal_bench.py",
        "--subset", str(VAL_DIR),
        "--traces-dir", str(baseline_traces),
        "--output", str(output_file),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir
    run(cmd)
    return report_dir


def step_train_inference(model: str, base_version: int, workers: int, force: bool, dry_run: bool) -> Path:
    output_dir = train_traces_dir(model, base_version)
    source_dir = base_agent_dir(base_version)
    log(f"Step 0: train inference on {source_dir.name} -> {output_dir.name}/")
    if (output_dir / "preds.json").exists() and not force:
        log("  train preds.json already exists, skipping")
        return output_dir
    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_terminal_bench.sh"),
        "--subset", str(TRAIN_DIR),
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
    ]
    env = {"AGENT_SRC_DIR": str(source_dir)} if base_version > 0 else None
    if dry_run:
        prefix = f"AGENT_SRC_DIR={source_dir} " if env else ""
        print(f"  [DRY RUN] {prefix}would run: {' '.join(str(c) for c in cmd)}")
        return output_dir
    run(cmd, env=env, timeout=48 * 3600)
    return output_dir


def step_train_evaluate(model: str, base_version: int, train_traces: Path, force: bool, dry_run: bool) -> Path:
    report_dir = train_eval_dir(model, base_version)
    log(f"Step 1: train evaluate -> {report_dir.name}/")
    if find_eval_json(report_dir) and not force:
        log("  train eval results already exists, skipping")
        return report_dir
    output_file = report_dir / "results.json"
    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_terminal_bench.py",
        "--subset", str(TRAIN_DIR),
        "--traces-dir", str(train_traces),
        "--output", str(output_file),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir
    run(cmd)
    return report_dir


def step_train_failure_analysis(model: str, base_version: int, train_traces: Path, train_eval: Path,
                                analysis_model: str, workers: int, force: bool, dry_run: bool) -> Path:
    output_file = FAILURE_ANALYSIS_DIR / "results" / f"{train_run_id(model, base_version)}_analysis.jsonl"
    log(f"Step 2: train failure analysis -> {output_file.name}")
    if output_file.exists() and not force:
        log("  train failure analyses already exists, skipping")
        return output_file
    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
        "--mode", "terminal_bench",
        "--model", analysis_model,
        "--traces-dir", str(train_traces),
        "--eval-results", str(find_eval_json(train_eval) or train_eval / "results.json"),
        "--terminal-bench-subset", str(TRAIN_DIR),
        "--output-file", str(output_file),
        "--workers", str(workers),
        "--agent-source-dir", str(base_agent_dir(base_version)),
        "--resume",
    ]
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return output_file
    run(cmd)
    return output_file


def step_aggregate(version: int, model: str, analysis_results: Path, analysis_model: str,
                   val_analyses_path: Path | None, prev_plan_path: Path | None,
                   force: bool, dry_run: bool) -> Path:
    plan_path = IMPROVEMENT_PLANS_DIR / f"terminal_bench_{model_slug(model)}_v{version}.md"
    spec_path = plan_spec_path(plan_path)
    log(f"Step 3: aggregate -> {plan_path.name}")
    if plan_path.exists() and spec_path.exists() and not force:
        log("  improvement plan already exists, skipping")
        return plan_path
    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "aggregate_results.py",
        "--mode", "terminal_bench",
        "--model", analysis_model,
        "--results-file", str(analysis_results),
        "--output", str(plan_path),
        "--spec-output", str(spec_path),
        "--force",
    ]
    if val_analyses_path and val_analyses_path.exists():
        cmd += ["--val-analyses", str(val_analyses_path)]
    if prev_plan_path and prev_plan_path.exists():
        cmd += ["--prev-plan", str(prev_plan_path)]
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return plan_path
    run(cmd)
    return plan_path


def _run_modify_versioned_terminal_bench(version: int, base_version: int, plan_path: Path,
                                         spec_path: Path, target_dir: Path, modify_model: str,
                                         train_compare_path: Path | None = None) -> None:
    import yaml

    source_dir = base_agent_dir(base_version)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    log(f"  Copying {source_dir.name} -> {target_dir.name} ...")
    shutil.copytree(
        source_dir,
        target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.egg-info"),
    )

    config_path = REPO_ROOT / "enhancement_implementation" / "config_terminal_bench.yaml"
    config = yaml.safe_load(config_path.read_text())

    sys.path.insert(0, str(REPO_ROOT / "agent_framework" / "src"))
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    agent_config = dict(config.get("agent", {}))
    model_config = config.get("model", {})
    env_config = config.get("environment", {})
    traj_path = REPO_ROOT / "enhancement_implementation" / "results" / f"modify_terminal_bench_v{version}.traj.json"
    traj_path.parent.mkdir(parents=True, exist_ok=True)

    model = LitellmTextbasedModel(
        model_name=modify_model,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}),
        cost_tracking="ignore_errors",
    )
    extra_env = {
        "TARGET_DIR": str(target_dir),
        "ORIGINAL_DIR": str(source_dir),
        "PLAN_PATH": str(plan_path),
        "PLAN_JSON_PATH": str(spec_path),
    }
    if train_compare_path and train_compare_path.exists():
        extra_env["TRAIN_COMPARE_PATH"] = str(train_compare_path)
    env = LocalEnvironment(env={**env_config.get("env", {}), **extra_env})
    agent = DefaultAgent(
        model,
        env,
        output_path=traj_path,
        **{k: v for k, v in agent_config.items() if k not in ("system_template", "instance_template")},
        system_template=agent_config["system_template"],
        instance_template=agent_config["instance_template"],
    )
    agent.run(
        target_dir=str(target_dir),
        original_dir=str(source_dir),
        plan_path=str(plan_path),
        plan_json_path=str(spec_path),
    )


def step_modify(version: int, base_version: int, plan_path: Path, modify_model: str,
                force: bool, dry_run: bool, train_compare_path: Path | None = None) -> Path:
    target_dir = base_agent_dir(version)
    log(f"Step 4: modify -> {target_dir.name}/")
    if target_dir.exists() and not force:
        log("  enhanced agent already exists, skipping")
        return target_dir
    if dry_run:
        print(f"  [DRY RUN] would create {target_dir}")
        return target_dir
    _run_modify_versioned_terminal_bench(
        version=version,
        base_version=base_version,
        plan_path=plan_path,
        spec_path=plan_spec_path(plan_path),
        target_dir=target_dir,
        modify_model=modify_model,
        train_compare_path=train_compare_path,
    )
    return target_dir


def step_plan_diff_audit(version: int, base_version: int, plan_path: Path, target_dir: Path, dry_run: bool) -> dict:
    output_path = FAILURE_ANALYSIS_DIR / "results" / f"audit_terminal_bench_v{version}.json"
    log(f"Step 4.5: audit candidate diff -> {target_dir.name}")
    if dry_run:
        return {"passed": True, "dry_run": True, "changed_files": []}
    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "plan_diff_audit.py",
        "--mode", "terminal_bench",
        "--original-dir", str(base_agent_dir(base_version)),
        "--candidate-dir", str(target_dir),
        "--spec", str(plan_spec_path(plan_path)),
        "--output", str(output_path),
    ]
    subprocess.run(cmd, capture_output=False, text=True, check=False)
    return json.loads(output_path.read_text()) if output_path.exists() else {"passed": False, "violations": ["audit_output_missing"]}


def step_val_inference(version: int, enhanced_dir: Path, model: str, workers: int, dry_run: bool) -> Path:
    output_dir = val_enhanced_traces_dir(model, version)
    log(f"Step 5: val inference -> {output_dir.name}/")
    if (output_dir / "preds.json").exists():
        log("  preds.json already exists, skipping")
        return output_dir
    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_terminal_bench.sh"),
        "--subset", str(VAL_DIR),
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
    ]
    if dry_run:
        print(f"  [DRY RUN] AGENT_SRC_DIR={enhanced_dir} would run: {' '.join(str(c) for c in cmd)}")
        return output_dir
    run(cmd, env={"AGENT_SRC_DIR": str(enhanced_dir)}, timeout=48 * 3600)
    return output_dir


def step_val_evaluate(version: int, val_traces: Path, model: str, dry_run: bool) -> Path:
    report_dir = val_enhanced_eval_dir(model, version)
    log(f"Step 6: val evaluate -> {report_dir.name}/")
    if find_eval_json(report_dir):
        log("  eval results already exists, skipping")
        return report_dir
    output_file = report_dir / "results.json"
    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_terminal_bench.py",
        "--subset", str(VAL_DIR),
        "--traces-dir", str(val_traces),
        "--output", str(output_file),
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir
    run(cmd)
    return report_dir


def step_compare_train(version: int, model: str, baseline_train_eval: Path,
                       enhanced_train_eval: Path, baseline_train_traces: Path,
                       enhanced_train_traces: Path, plan_path: Path, dry_run: bool) -> dict:
    output = FAILURE_ANALYSIS_DIR / "results" / f"terminal_bench_train_compare_v{version}.json"
    log(f"Step 6.5: compare train baseline vs enhanced v{version}")
    if dry_run:
        return {
            "net_change": 0,
            "dry_run": True,
            "regressed_ids": [],
            "improved_ids": [],
            "baseline_count": 0,
            "current_count": 0,
            "regression_count": 0,
            "improvement_count": 0,
        }
    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "check_val_gate.py",
        "--mode", "terminal_bench",
        "--baseline-eval", str(baseline_train_eval),
        "--current-eval", str(enhanced_train_eval),
        "--baseline-traces", str(baseline_train_traces),
        "--current-traces", str(enhanced_train_traces),
        "--plan-spec", str(plan_spec_path(plan_path)),
        "--val-ids", str(TRAIN_DIR / "instance_ids.txt"),
        "--output", str(output),
        "--max-regression", "9999",
        "--min-improvement", "0",
    ]
    subprocess.run(cmd, capture_output=False, text=True, check=False)
    if output.exists():
        return json.loads(output.read_text())
    return {
        "net_change": 0,
        "regressed_ids": [],
        "improved_ids": [],
        "baseline_count": 0,
        "current_count": 0,
        "regression_count": 0,
        "improvement_count": 0,
    }


def _cleanup_for_redo(version: int, model: str) -> None:
    dirs_to_remove = [
        base_agent_dir(version),
        train_traces_dir(model, version),
        train_eval_dir(model, version),
        val_enhanced_traces_dir(model, version),
        val_enhanced_eval_dir(model, version),
    ]
    for path in dirs_to_remove:
        if path.exists():
            shutil.rmtree(path)
            log(f"  Removed {path.name} for redo")


def step_gate_check(version: int, baseline_eval: Path, current_eval: Path, baseline_traces: Path,
                    current_traces: Path, plan_path: Path, max_regression: int, min_improvement: int,
                    max_cost_ratio: float, dry_run: bool) -> dict:
    output_path = FAILURE_ANALYSIS_DIR / "results" / f"gate_terminal_bench_v{version}.json"
    log("Step 7: gate check")
    if dry_run:
        return {"passed": False, "regressed_ids": [], "dry_run": True}
    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "check_val_gate.py",
        "--mode", "terminal_bench",
        "--baseline-eval", str(baseline_eval),
        "--current-eval", str(current_eval),
        "--baseline-traces", str(baseline_traces),
        "--current-traces", str(current_traces),
        "--plan-spec", str(plan_spec_path(plan_path)),
        "--val-ids", str(VAL_DIR / "instance_ids.txt"),
        "--output", str(output_path),
        "--max-regression", str(max_regression),
        "--min-improvement", str(min_improvement),
        "--max-cost-ratio", str(max_cost_ratio),
    ]
    subprocess.run(cmd, capture_output=False, text=True, check=False)
    return json.loads(output_path.read_text()) if output_path.exists() else {"passed": False, "regressed_ids": []}


def step_analyze_regressions(version: int, regressed_ids: list[str], val_traces: Path,
                              analysis_model: str, workers: int, dry_run: bool) -> Path:
    output_file = FAILURE_ANALYSIS_DIR / "results" / f"terminal_bench_val_regression_analyses_v{version}.jsonl"
    if not regressed_ids:
        return output_file
    if dry_run:
        return output_file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(regressed_ids) + "\n")
        ids_file = Path(handle.name)
    try:
        cmd = [
            VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
            "--mode", "terminal_bench",
            "--model", analysis_model,
            "--traces-dir", str(val_traces),
            "--terminal-bench-subset", str(VAL_DIR),
            "--output-file", str(output_file),
            "--instance-ids-file", str(ids_file),
            "--workers", str(workers),
            "--agent-source-dir", str(base_agent_dir(version)),
            "--no-resume",
        ]
        run(cmd)
    finally:
        ids_file.unlink(missing_ok=True)
    return output_file


def step_record_memory(version: int, plan_path: Path, audit: dict | None, gate: dict | None, dry_run: bool) -> None:
    if dry_run:
        return
    from failure_analysis.harness_memory import build_memory_entry, default_memory_root, store_memory_entry

    spec = json.loads(plan_spec_path(plan_path).read_text())
    summary = spec.get("plan_metadata", {}).get("summary", plan_path.stem)
    outcome = "accepted" if gate and gate.get("passed") and (audit is None or audit.get("passed")) else "rejected"
    entry = build_memory_entry(
        mode="terminal_bench",
        version=version,
        outcome=outcome,
        plan_path=plan_path,
        spec=spec,
        summary=summary,
        changed_files=[] if audit is None else audit.get("changed_files", []),
        audit=audit,
        gate=gate,
    )
    store_memory_entry(default_memory_root(REPO_ROOT), entry)


def print_test_eval_command(enhanced_dir: Path | None, model: str) -> None:
    if enhanced_dir is None:
        label = "terminus2"
        env_line = ""
    else:
        label = enhanced_dir.name
        env_line = f"  AGENT_SRC_DIR={enhanced_dir} \\\n"
    print(f"""
Next: run final Terminal-Bench held-out test evaluation with {label}

{env_line}  ./task_agent/run_terminal_bench.sh \\
      --subset data/terminal_bench_test \\
      --model {model} \\
      --workers 2 \\
      --output traces/terminal_bench_test_{model_slug(model)}_{label}

  .venv/bin/python3 eval/eval_terminal_bench.py \\
      --subset data/terminal_bench_test \\
      --traces-dir traces/terminal_bench_test_{model_slug(model)}_{label} \\
      --output eval/terminal_bench_results_test_{model_slug(model)}_{label}/results.json
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="HarnessFix pipeline for Terminal-Bench 2.0")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--analysis-model", default="openai/claude-opus-4-5-20251101-thinking")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-modify-retries", type=int, default=2)
    parser.add_argument("--max-regression", type=int, default=2)
    parser.add_argument("--min-improvement", type=int, default=1)
    parser.add_argument("--max-cost-ratio", type=float, default=1.25)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (TERMINAL_AGENT_DIR / "harbor" / "src").exists():
        print("ERROR: task_agent/terminal_bench_agent/harbor/src not found. Clone Harbor locally first.")
        raise SystemExit(1)
    if not FULL_DATASET_DIR.exists() and not args.dry_run:
        print("ERROR: data/terminal_bench_2_verified/ not found. Clone Terminal-Bench 2.0 verified locally first.")
        raise SystemExit(1)
    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not split_dir.exists() and not args.dry_run:
            print(f"ERROR: {split_dir} not found. Create local Terminal-Bench split directories first.")
            raise SystemExit(1)

    print(f"""
{'='*60}
  HarnessFix Pipeline (Terminal-Bench 2.0 / Harbor Terminus-2)
  model:            {args.model}
  analysis_model:   {args.analysis_model}
  start_version:    v{args.start_version}
  max_iterations:   {args.max_iterations}
  val gate:         regression <= {args.max_regression}, fallback net >= {args.min_improvement}, cost <= {args.max_cost_ratio}
  workers:          {args.workers}
{'='*60}
""")

    baseline_traces = step_val_baseline_inference(args.model, args.workers, args.force, args.dry_run)
    baseline_eval = step_val_baseline_evaluate(args.model, baseline_traces, args.force, args.dry_run)
    val_analyses_path: Path | None = None
    prev_plan_path: Path | None = None
    current_base_version = args.start_version - 1
    promoted_versions: list[int] = []

    for i in range(args.max_iterations):
        version = args.start_version + i
        print(f"\n{'#'*60}")
        print(f"  TERMINAL-BENCH ITERATION {i+1}/{args.max_iterations}  (candidate v{version}, base v{current_base_version})")
        print(f"{'#'*60}")

        train_traces = step_train_inference(args.model, current_base_version, args.workers, args.force, args.dry_run)
        train_eval = step_train_evaluate(args.model, current_base_version, train_traces, args.force, args.dry_run)
        train_analysis = step_train_failure_analysis(args.model, current_base_version, train_traces, train_eval, args.analysis_model, args.workers, args.force, args.dry_run)
        plan_path = step_aggregate(version, args.model, train_analysis, args.analysis_model, val_analyses_path, prev_plan_path, args.force, args.dry_run)

        train_compare_path: Path | None = None
        target_dir = base_agent_dir(version)
        audit: dict | None = None
        for retry in range(args.max_modify_retries + 1):
            if retry > 0:
                log(f"  Modify retry {retry}/{args.max_modify_retries} with train feedback")
            target_dir = step_modify(version, current_base_version, plan_path, args.analysis_model, (retry > 0) or args.force, args.dry_run, train_compare_path)
            audit = step_plan_diff_audit(version, current_base_version, plan_path, target_dir, args.dry_run)
            if not audit.get("passed", True):
                log(f"  Audit failed: {audit.get('violations', [])}")
                if retry < args.max_modify_retries:
                    if not args.dry_run:
                        _cleanup_for_redo(version, args.model)
                    continue
                break

            enhanced_train_traces = step_train_inference(args.model, version, args.workers, (retry > 0) or args.force, args.dry_run)
            enhanced_train_eval = step_train_evaluate(args.model, version, enhanced_train_traces, (retry > 0) or args.force, args.dry_run)
            train_compare = step_compare_train(
                version,
                args.model,
                train_eval,
                enhanced_train_eval,
                train_traces,
                enhanced_train_traces,
                plan_path,
                args.dry_run,
            )
            train_net = train_compare.get("net_change", 0)
            train_improved = train_net > 0
            log(
                f"  Train compare (v{version} retry={retry}): "
                f"net={train_net:+d} improvements={train_compare.get('improvement_count', 0)} "
                f"regressions={train_compare.get('regression_count', 0)}"
            )
            if train_improved or args.dry_run:
                break
            if retry < args.max_modify_retries:
                train_compare_path = FAILURE_ANALYSIS_DIR / "results" / f"terminal_bench_train_compare_v{version}.json"
                if not args.dry_run:
                    _cleanup_for_redo(version, args.model)
            else:
                log(f"  Train did not improve after {retry + 1} attempt(s); proceeding to val gate")

        if audit is not None and not audit.get("passed", True):
            gate = {"passed": False, "failure_reasons": ["audit_failed"], "regressed_ids": [], "improved_ids": []}
            step_record_memory(version, plan_path, audit, gate, args.dry_run)
            prev_plan_path = plan_path
            val_analyses_path = None
            continue

        val_traces = step_val_inference(version, target_dir, args.model, args.workers, args.dry_run)
        val_eval = step_val_evaluate(version, val_traces, args.model, args.dry_run)
        base_val_traces = val_traces_dir_for_base(args.model, current_base_version)
        base_val_eval = val_eval_dir_for_base(args.model, current_base_version)
        gate = step_gate_check(
            version,
            base_val_eval,
            val_eval,
            base_val_traces,
            val_traces,
            plan_path,
            args.max_regression,
            args.min_improvement,
            args.max_cost_ratio,
            args.dry_run,
        )
        step_record_memory(version, plan_path, audit, gate, args.dry_run)
        if gate.get("passed"):
            log(f"PROMOTED v{version}: val {gate.get('baseline_count')} -> {gate.get('current_count')} resolved")
            current_base_version = version
            promoted_versions.append(version)
        else:
            log(
                f"NOT PROMOTED v{version}: {gate.get('regression_count', '?')} regressions, "
                f"{gate.get('improvement_count', '?')} improvements"
            )

        regressed_ids = gate.get("regressed_ids", [])
        val_analyses_path = step_analyze_regressions(version, regressed_ids, val_traces, args.analysis_model, args.workers, args.dry_run)
        prev_plan_path = plan_path

    best_version = best_version_by_val(args.model, promoted_versions)
    best_dir = base_agent_dir(best_version)
    best_eval = val_eval_dir_for_base(args.model, best_version)
    log(f"Finished iterations. Best-so-far version: v{best_version} ({best_dir.name})")
    if (VAL_DIR / "instance_ids.txt").exists():
        log(f"Best validation resolved: {resolved_count_from_eval(best_eval)}/{len((VAL_DIR / 'instance_ids.txt').read_text().split())}")
    print_test_eval_command(best_dir if best_version > 0 else None, args.model)


if __name__ == "__main__":
    main()

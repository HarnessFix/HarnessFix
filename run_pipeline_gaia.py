#!/usr/bin/env python3
"""Closed-loop HarnessFix pipeline for GAIA benchmark.

Mirrors run_pipeline_swe.py but uses:
  - open_deep_research as the task agent (instead of mini-swe-agent)
  - run_gaia.sh for inference (instead of run_swebench.sh)
  - eval/eval_gaia.py for evaluation (instead of swebench harness)
  - failure_analysis/run_analysis.py --mode gaia for failure analysis
  - enhancement_implementation/config_gaia.yaml for the modify step

Bootstrap phase:
  0. val_base_infer   — run original open_deep_research on gaia_val_30 if missing
  1. val_base_eval    — evaluate val baseline predictions if missing

Iteration phase:
  0. train_infer      — run current base agent on gaia_train_60
  1. train_eval       — evaluate train predictions
  2. train_analysis   — analyze failed train tasks for current base agent
  3. aggregate        — generate improvement_plan_gaia_vN.md + .json
  4. modify           — create enhanced_odr_vN from base agent + plan
  5. val_infer        — run enhanced agent on gaia_val_30
  6. val_eval         — evaluate val predictions
  7. gate             — compare vs baseline; pass → done, fail → analyze regressions

Usage:
  .venv/bin/python3 run_pipeline_gaia.py --model openai/gpt-5-mini
  .venv/bin/python3 run_pipeline_gaia.py --dry-run --model openai/gpt-5-mini --max-iterations 1
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

REPO_ROOT = Path(__file__).parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python3"
TASK_AGENT_DIR = REPO_ROOT / "task_agent"
DATA_DIR = REPO_ROOT / "data"
FAILURE_ANALYSIS_DIR = REPO_ROOT / "failure_analysis"
IMPROVEMENT_PLANS_DIR = REPO_ROOT / "improvement_plans"
TRACES_DIR = REPO_ROOT / "traces"
EVAL_DIR = REPO_ROOT / "eval"

# GAIA-specific paths
GAIA_AGENT_DIR = TASK_AGENT_DIR / "open_deep_research"
TRAIN_DIR = DATA_DIR / "gaia_train_60"
VAL_DIR = DATA_DIR / "gaia_val_30"
TEST_DIR = DATA_DIR / "gaia_test_60"


# ── Utilities ──────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"\n[pipeline_gaia] {msg}", flush=True)


def run(cmd: list, env: dict | None = None, check: bool = True,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output."""
    merged_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        return subprocess.run(cmd, env=merged_env, check=check, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"\n[pipeline_gaia] ERROR: subprocess timed out after {timeout}s: {cmd[0]}", flush=True)
        raise


def find_eval_json(directory: Path) -> Path | None:
    jsons = [j for j in directory.glob("*.json") if not j.name.startswith("run_")]
    return jsons[0] if jsons else None


def load_gate_result(path: Path) -> dict:
    return json.loads(path.read_text())


def model_short(model: str) -> str:
    return model.replace("/", "-").replace("_", "-")


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace("-", "_").lower()


def plan_spec_path(plan_path: Path) -> Path:
    return plan_path.with_suffix(".json")


def base_agent_dir(model: str, version: int) -> Path:
    """Return the source directory for the given base version."""
    if version <= 0:
        return GAIA_AGENT_DIR
    return TASK_AGENT_DIR / f"enhanced_odr_{model_slug(model)}_v{version}"


def base_agent_label(version: int) -> str:
    if version <= 0:
        return "original"
    return f"odr_enhanced_v{version}"


def train_run_id(model: str, base_version: int) -> str:
    return f"gaia_train60_{model_slug(model)}_{base_agent_label(base_version)}"


def val_baseline_run_id(model: str) -> str:
    return f"gaia_val30_{model_slug(model)}_original"


def train_traces_dir(model: str, base_version: int) -> Path:
    return TRACES_DIR / train_run_id(model, base_version)


def train_eval_dir(model: str, base_version: int) -> Path:
    return EVAL_DIR / f"gaia_results_{train_run_id(model, base_version)}"


def train_analysis_path(model: str, base_version: int) -> Path:
    return FAILURE_ANALYSIS_DIR / "results" / f"{train_run_id(model, base_version)}_analysis.jsonl"


def val_baseline_traces_dir(model: str) -> Path:
    return TRACES_DIR / val_baseline_run_id(model)


def val_baseline_eval_dir(model: str) -> Path:
    return EVAL_DIR / f"gaia_results_{val_baseline_run_id(model)}"


def enhanced_train_run_id(model: str, version: int) -> str:
    return f"gaia_train60_{model_slug(model)}_odr_enhanced_v{version}"


def enhanced_train_traces_dir(model: str, version: int) -> Path:
    return TRACES_DIR / enhanced_train_run_id(model, version)


def enhanced_train_eval_dir(model: str, version: int) -> Path:
    return EVAL_DIR / f"gaia_results_{enhanced_train_run_id(model, version)}"


def val_enhanced_run_id(model: str, version: int) -> str:
    ms = model_short(model)
    return f"gaia_val30_{ms}_odr_enhanced_v{version}"


def val_enhanced_traces_dir(model: str, version: int) -> Path:
    return TRACES_DIR / val_enhanced_run_id(model, version)


def val_enhanced_eval_dir(model: str, version: int) -> Path:
    return EVAL_DIR / f"gaia_results_{val_enhanced_run_id(model, version)}"


def val_traces_dir_for_base(model: str, base_version: int) -> Path:
    return val_baseline_traces_dir(model) if base_version <= 0 else val_enhanced_traces_dir(model, base_version)


def val_eval_dir_for_base(model: str, base_version: int) -> Path:
    return val_baseline_eval_dir(model) if base_version <= 0 else val_enhanced_eval_dir(model, base_version)


def resolved_count_from_eval(eval_dir: Path) -> int:
    eval_json = find_eval_json(eval_dir)
    if not eval_json:
        return 0
    data = json.loads(eval_json.read_text())
    subset = set((VAL_DIR / "instance_ids.txt").read_text().split())
    return len(set(data.get("resolved_ids", [])) & subset)


def best_version_by_val(model: str, versions: list[int]) -> int:
    best_version = 0
    best_score = resolved_count_from_eval(val_baseline_eval_dir(model))
    for version in versions:
        score = resolved_count_from_eval(val_enhanced_eval_dir(model, version))
        if score > best_score:
            best_version = version
            best_score = score
    return best_version


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_train_inference(model: str, base_version: int, workers: int, concurrency: int,
                         force: bool, dry_run: bool) -> Path:
    output_dir = train_traces_dir(model, base_version)
    source_dir = base_agent_dir(model, base_version)
    log(f"Step 0: train inference on {source_dir.name} → {output_dir.name}/")

    if (output_dir / "preds.json").exists() and not force:
        log("  train preds.json already exists, skipping")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_gaia.sh"),
        "--subset", str(TRAIN_DIR),
        "--model", model,
        "--workers", str(workers),
        "--concurrency", str(concurrency),
        "--output", str(output_dir),
    ]

    if dry_run:
        agent_src = source_dir / "src"
        prefix = f"AGENT_SRC_DIR={agent_src} " if base_version > 0 else ""
        print(f"  [DRY RUN] would run: {prefix}{' '.join(str(c) for c in cmd)}")
        return output_dir

    env = {"AGENT_SRC_DIR": str(source_dir / "src")} if base_version > 0 else None
    run(cmd, env=env, timeout=24 * 3600)
    return output_dir


def step_train_evaluate(model: str, base_version: int, train_traces: Path,
                        force: bool, dry_run: bool) -> Path:
    report_dir = train_eval_dir(model, base_version)
    log(f"Step 1: train evaluate → {report_dir.name}/")

    if find_eval_json(report_dir) and not force:
        log("  train eval results already exist, skipping")
        return report_dir

    preds_path = train_traces / "preds.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_file = report_dir / "results.json"

    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_gaia.py",
        "--preds", str(preds_path),
        "--subset", str(TRAIN_DIR),
        "--output", str(output_file),
        "--traces-dir", str(train_traces),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir

    run(cmd)
    return report_dir


def step_train_failure_analysis(model: str, base_version: int,
                                train_traces: Path, train_eval_dir_path: Path,
                                analysis_model: str, workers: int,
                                force: bool, dry_run: bool) -> Path:
    output_file = train_analysis_path(model, base_version)
    log(f"Step 2: train failure analysis → {output_file.name}")

    if output_file.exists() and not force:
        log("  train failure analyses already exist, skipping")
        return output_file

    eval_json = find_eval_json(train_eval_dir_path)

    cmd_base = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
        "--mode", "gaia",
        "--model", analysis_model,
        "--traces-dir", str(train_traces),
        "--eval-results", str(eval_json) if eval_json else "<eval-json>",
        "--gaia-subset", str(TRAIN_DIR),
        "--output-file", str(output_file),
        "--workers", str(workers),
        "--agent-source-dir", str(base_agent_dir(model, base_version) / "src" / "open_deep_research"),
        "--resume",
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd_base)}")
        return output_file

    run(cmd_base)
    return output_file


def step_val_baseline_inference(model: str, workers: int, concurrency: int,
                                force: bool, dry_run: bool) -> Path:
    output_dir = val_baseline_traces_dir(model)
    log(f"Bootstrap: val baseline inference → {output_dir.name}/")

    if (output_dir / "preds.json").exists() and not force:
        log("  val baseline preds.json already exists, skipping")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_gaia.sh"),
        "--subset", str(VAL_DIR),
        "--model", model,
        "--workers", str(workers),
        "--concurrency", str(concurrency),
        "--output", str(output_dir),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return output_dir

    run(cmd, timeout=12 * 3600)
    return output_dir


def step_val_baseline_evaluate(model: str, baseline_traces: Path,
                               force: bool, dry_run: bool) -> Path:
    report_dir = val_baseline_eval_dir(model)
    log(f"Bootstrap: val baseline evaluate → {report_dir.name}/")

    if find_eval_json(report_dir) and not force:
        log("  val baseline eval results already exist, skipping")
        return report_dir

    preds_path = baseline_traces / "preds.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_file = report_dir / "results.json"

    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_gaia.py",
        "--preds", str(preds_path),
        "--subset", str(VAL_DIR),
        "--output", str(output_file),
        "--traces-dir", str(baseline_traces),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir

    run(cmd)
    return report_dir


def step_aggregate(version: int, model: str, analysis_results_path: Path, analysis_model: str,
                   val_analyses_path: Path | None,
                   prev_plan_path: Path | None,
                   force: bool, dry_run: bool) -> Path:
    """Generate improvement_plans/gaia_{model_slug}_vN.md and .json."""
    IMPROVEMENT_PLANS_DIR.mkdir(exist_ok=True)
    plan_path = IMPROVEMENT_PLANS_DIR / f"gaia_{model_slug(model)}_v{version}.md"
    spec_path = plan_spec_path(plan_path)
    log(f"Step 3: aggregate → {plan_path.name}")

    if plan_path.exists() and spec_path.exists() and not force:
        log(f"  {plan_path.name} already exists, skipping (use --force to regenerate)")
        return plan_path

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "aggregate_results.py",
        "--mode", "gaia",
        "--model", analysis_model,
        "--results-file", str(analysis_results_path),
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


def step_modify(version: int, base_version: int, plan_path: Path, model: str,
                modify_model: str, force: bool, dry_run: bool,
                train_compare_path: Path | None = None) -> Path:
    """Create enhanced_odr_{model_slug}_vN from improvement plan.

    train_compare_path: optional path to a previous train comparison JSON (redo context).
    """
    target_dir = TASK_AGENT_DIR / f"enhanced_odr_{model_slug(model)}_v{version}"
    log(f"Step 4: modify → {target_dir.name}/"
        + (f" [with train feedback: {train_compare_path.name}]" if train_compare_path else ""))

    if target_dir.exists() and not force:
        log(f"  {target_dir.name}/ already exists, skipping (use --force to regenerate)")
        return target_dir

    if dry_run:
        print(f"  [DRY RUN] would create {target_dir.name}/ from improvement plan"
              + (f" with train feedback" if train_compare_path else ""))
        return target_dir

    _run_modify_versioned_gaia(
        version=version,
        base_version=base_version,
        model=model,
        plan_path=plan_path,
        spec_path=plan_spec_path(plan_path),
        target_dir=target_dir,
        modify_model=modify_model,
        train_compare_path=train_compare_path,
    )
    return target_dir


def _run_modify_versioned_gaia(version: int, base_version: int, model: str,
                               plan_path: Path, spec_path: Path,
                               target_dir: Path, modify_model: str,
                               train_compare_path: Path | None = None) -> None:
    """Copy base agent and run modify_agent with GAIA config."""
    import shutil
    import yaml

    source_dir = base_agent_dir(model, base_version)

    # Copy current base agent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    log(f"  Copying {source_dir.name} → {target_dir.name} ...")
    shutil.copytree(
        source_dir, target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.egg-info"),
    )

    # Load GAIA modify_agent config
    config_path = REPO_ROOT / "enhancement_implementation" / "config_gaia.yaml"
    config = yaml.safe_load(config_path.read_text())

    # Load agent + model
    sys.path.insert(0, str(REPO_ROOT / "agent_framework" / "src"))
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    agent_config = dict(config.get("agent", {}))
    model_config = config.get("model", {})
    env_config = config.get("environment", {})

    results_dir = REPO_ROOT / "enhancement_implementation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    traj_path = results_dir / f"modify_agent_gaia_v{version}.traj.json"

    model = LitellmTextbasedModel(
        model_name=modify_model,
        format_error_template=model_config.get("format_error_template", "Format error: {{error}}"),
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
        model, env,
        output_path=traj_path,
        **{k: v for k, v in agent_config.items()
           if k not in ("system_template", "instance_template")},
        system_template=agent_config["system_template"],
        instance_template=agent_config["instance_template"],
    )
    template_vars = {
        "target_dir": str(target_dir),
        "original_dir": str(source_dir),
        "plan_path": str(plan_path),
        "plan_json_path": str(spec_path),
    }
    result = agent.run(**template_vars)
    log(f"  modify_agent finished: exit_status={result.get('exit_status')}, traj={traj_path}")


def step_train_inference_enhanced(version: int, enhanced_src: Path, model: str,
                                  workers: int, concurrency: int,
                                  force: bool, dry_run: bool) -> Path:
    """Run enhanced agent on gaia_train_60 to check if the modification improved training set."""
    output_dir = enhanced_train_traces_dir(model, version)
    log(f"Step 4b: enhanced train inference → {output_dir.name}/")

    if (output_dir / "preds.json").exists() and not force:
        log("  enhanced train preds.json already exists, skipping")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_gaia.sh"),
        "--subset", str(TRAIN_DIR),
        "--model", model,
        "--workers", str(workers),
        "--concurrency", str(concurrency),
        "--output", str(output_dir),
    ]

    if dry_run:
        print(f"  [DRY RUN] AGENT_SRC_DIR={enhanced_src} {' '.join(str(c) for c in cmd)}")
        return output_dir

    run(cmd, env={"AGENT_SRC_DIR": str(enhanced_src)}, timeout=24 * 3600)
    return output_dir


def step_train_evaluate_enhanced(version: int, model: str,
                                 enhanced_train_traces: Path,
                                 force: bool, dry_run: bool) -> Path:
    """Evaluate enhanced agent predictions on gaia_train_60."""
    report_dir = enhanced_train_eval_dir(model, version)
    log(f"Step 4c: enhanced train evaluate → {report_dir.name}/")

    if find_eval_json(report_dir) and not force:
        log("  enhanced train eval results already exist, skipping")
        return report_dir

    preds_path = enhanced_train_traces / "preds.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_file = report_dir / "results.json"

    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_gaia.py",
        "--preds", str(preds_path),
        "--subset", str(TRAIN_DIR),
        "--output", str(output_file),
        "--traces-dir", str(enhanced_train_traces),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir

    run(cmd)
    return report_dir


def step_compare_train(version: int, model: str,
                       baseline_train_eval: Path, enhanced_train_eval: Path,
                       dry_run: bool) -> dict:
    """Compare enhanced vs baseline on train set. Returns comparison dict (no hard gate)."""
    output = FAILURE_ANALYSIS_DIR / "results" / f"gaia_train_compare_v{version}.json"
    log(f"Step 4d: compare train baseline vs enhanced v{version}")

    if dry_run:
        print(f"  [DRY RUN] would compare {enhanced_train_eval.name} vs {baseline_train_eval.name}")
        return {"net_change": 0, "dry_run": True, "regressed_ids": [], "improved_ids": [],
                "baseline_count": 0, "current_count": 0,
                "regression_count": 0, "improvement_count": 0}

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "check_val_gate.py",
        "--mode", "gaia",
        "--baseline-eval", str(baseline_train_eval),
        "--current-eval", str(enhanced_train_eval),
        "--val-ids", str(TRAIN_DIR / "instance_ids.txt"),
        "--output", str(output),
        "--max-regression", "9999",  # No hard gate on train — just record
        "--min-improvement", "0",
    ]
    subprocess.run(cmd, capture_output=False, text=True, check=False)

    if output.exists():
        return load_gate_result(output)
    return {"net_change": 0, "regressed_ids": [], "improved_ids": [],
            "baseline_count": 0, "current_count": 0,
            "regression_count": 0, "improvement_count": 0}


def _cleanup_for_redo(version: int, model: str) -> None:
    """Remove enhanced agent dir and its train/val inference results to allow a redo."""
    import shutil
    dirs_to_remove = [
        TASK_AGENT_DIR / f"enhanced_odr_{model_slug(model)}_v{version}",
        enhanced_train_traces_dir(model, version),
        enhanced_train_eval_dir(model, version),
        val_enhanced_traces_dir(model, version),
        val_enhanced_eval_dir(model, version),
    ]
    for d in dirs_to_remove:
        if d.exists():
            shutil.rmtree(d)
            log(f"  Removed {d.name} for redo")


def step_val_inference(version: int, enhanced_src: Path, model: str,
                       workers: int, concurrency: int, dry_run: bool) -> Path:
    """Run enhanced agent on gaia_val_30."""
    output_dir = val_enhanced_traces_dir(model, version)
    log(f"Step 5: val inference → {output_dir.name}/")

    if (output_dir / "preds.json").exists():
        log(f"  preds.json already exists, skipping val inference")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_gaia.sh"),
        "--subset", str(VAL_DIR),
        "--model", model,
        "--workers", str(workers),
        "--concurrency", str(concurrency),
        "--output", str(output_dir),
    ]

    if dry_run:
        print(f"  [DRY RUN] AGENT_SRC_DIR={enhanced_src} {' '.join(str(c) for c in cmd)}")
        return output_dir

    run(cmd, env={"AGENT_SRC_DIR": str(enhanced_src)}, timeout=12 * 3600)
    return output_dir


def step_val_evaluate(version: int, val_traces: Path, model: str, dry_run: bool) -> Path:
    """Evaluate enhanced agent on gaia_val_30."""
    report_dir = val_enhanced_eval_dir(model, version)
    log(f"Step 6: val evaluate → {report_dir.name}/")

    if find_eval_json(report_dir):
        log(f"  eval results already exist, skipping")
        return report_dir

    preds_path = val_traces / "preds.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_file = report_dir / "results.json"

    cmd = [
        VENV_PYTHON, EVAL_DIR / "eval_gaia.py",
        "--preds", str(preds_path),
        "--subset", str(VAL_DIR),
        "--output", str(output_file),
        "--traces-dir", str(val_traces),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return report_dir

    run(cmd)
    return report_dir


def step_plan_diff_audit(version: int, model: str, base_version: int, plan_path: Path,
                         target_dir: Path, dry_run: bool) -> dict:
    log(f"Step 4.5: audit candidate diff → {target_dir.name}")
    output_path = FAILURE_ANALYSIS_DIR / "results" / f"gaia_audit_v{version}.json"

    if dry_run:
        print(f"  [DRY RUN] would audit {target_dir.name} against {base_agent_dir(model, base_version).name}")
        return {"passed": True, "dry_run": True, "changed_files": []}

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "plan_diff_audit.py",
        "--mode", "gaia",
        "--original-dir", str(base_agent_dir(model, base_version)),
        "--candidate-dir", str(target_dir),
        "--spec", str(plan_spec_path(plan_path)),
        "--output", str(output_path),
    ]
    subprocess.run(cmd, capture_output=False, text=True, check=False)
    if output_path.exists():
        return json.loads(output_path.read_text())
    return {"passed": False, "violations": ["audit_output_missing"], "changed_files": []}


def step_gate_check(version: int, baseline_eval_dir: Path, val_eval_dir: Path,
                    baseline_traces_dir: Path, val_traces_dir: Path, plan_path: Path,
                    max_regression: int, min_improvement: int, max_cost_ratio: float,
                    dry_run: bool) -> dict:
    """Compare val results vs baseline with target/regression/cost checks."""
    log(
        "Step 7: gate check "
        f"(max_regression={max_regression}, min_improvement={min_improvement}, "
        f"max_cost_ratio={max_cost_ratio})"
    )

    gate_output = FAILURE_ANALYSIS_DIR / "results" / f"gaia_gate_v{version}.json"

    if dry_run:
        print(f"  [DRY RUN] would compare {val_eval_dir.name} vs {baseline_eval_dir.name}")
        return {"passed": False, "regressed_ids": [], "dry_run": True}

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "check_val_gate.py",
        "--mode", "gaia",
        "--baseline-eval", str(baseline_eval_dir),
        "--current-eval", str(val_eval_dir),
        "--baseline-traces", str(baseline_traces_dir),
        "--current-traces", str(val_traces_dir),
        "--plan-spec", str(plan_spec_path(plan_path)),
        "--val-ids", str(VAL_DIR / "instance_ids.txt"),
        "--output", str(gate_output),
        "--max-regression", str(max_regression),
        "--min-improvement", str(min_improvement),
        "--max-cost-ratio", str(max_cost_ratio),
    ]
    result = subprocess.run(cmd, capture_output=False, text=True, check=False)
    if gate_output.exists():
        return load_gate_result(gate_output)
    return {"passed": result.returncode == 0, "regressed_ids": []}


def step_record_memory(mode: str, version: int, plan_path: Path, audit: dict | None,
                       gate: dict | None, dry_run: bool) -> None:
    if dry_run:
        return
    from failure_analysis.harness_memory import build_memory_entry, default_memory_root, store_memory_entry

    spec = json.loads(plan_spec_path(plan_path).read_text())
    summary = spec.get("plan_metadata", {}).get("summary", plan_path.stem)
    outcome = "accepted" if gate and gate.get("passed") and (audit is None or audit.get("passed")) else "rejected"
    entry = build_memory_entry(
        mode=mode,
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


def step_attribution_analysis(version: int, improved_ids: list[str],
                              analysis_jsonl: Path, label: str) -> dict:
    """Cross-reference newly-passing instances with failure analysis records.

    Computes the 'true fix rate': what fraction of newly-passing instances
    were previously diagnosed in the failure analysis (i.e., genuine fixes
    vs. untracked improvements / noise).

    Saves result to failure_analysis/results/gaia_attribution_{label}_v{version}.json.
    Returns the attribution dict.
    """
    log(f"Attribution analysis ({label}, v{version}): {len(improved_ids)} newly-passing instances")

    # Load failure analysis records
    diagnosed_by_id: dict[str, dict] = {}
    if analysis_jsonl.exists():
        for line in analysis_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                iid = rec.get("instance_id", "")
                if iid:
                    diagnosed_by_id[iid] = rec
            except json.JSONDecodeError:
                pass
    else:
        log(f"  Warning: failure analysis file not found: {analysis_jsonl}")

    diagnosed_ids = set(diagnosed_by_id)
    improved_set = set(improved_ids)
    genuine_fix_ids = sorted(improved_set & diagnosed_ids)
    untracked_ids = sorted(improved_set - diagnosed_ids)
    true_fix_rate = len(genuine_fix_ids) / len(improved_ids) if improved_ids else 0.0

    # Breakdown by affected_component for genuine fixes
    component_counts: dict[str, int] = {}
    for iid in genuine_fix_ids:
        comp = diagnosed_by_id[iid].get("affected_component", "unknown")
        component_counts[comp] = component_counts.get(comp, 0) + 1

    # Print report
    print(f"\n  {'─'*58}")
    print(f"  Attribution Analysis — {label} (v{version})")
    print(f"  {'─'*58}")
    print(f"  Newly passing total:       {len(improved_ids)}")
    print(f"  Had failure diagnosis:     {len(genuine_fix_ids)}"
          f"  ({true_fix_rate:.0%} of newly passing)")
    print(f"  No diagnosis (untracked):  {len(untracked_ids)}")
    if component_counts:
        print(f"  Genuine fixes by component:")
        for comp, cnt in sorted(component_counts.items(), key=lambda x: -x[1]):
            print(f"    {comp}: {cnt}")
    if genuine_fix_ids:
        print(f"\n  Genuine fixes (had diagnosis):")
        for iid in genuine_fix_ids:
            rec = diagnosed_by_id[iid]
            comp = rec.get("affected_component", "?")
            cat = rec.get("failure_category", "?")
            gave_up = rec.get("gave_up_prematurely", False)
            fix_scope = rec.get("fix_scope", "?")
            print(f"    + {iid}  [{comp}] cat={cat}"
                  f"  gave_up={gave_up}  fix_scope={fix_scope}")
    if untracked_ids:
        print(f"\n  Untracked improvements (no prior diagnosis):")
        for iid in untracked_ids:
            print(f"    + {iid}")
    print(f"  {'─'*58}\n")

    result = {
        "version": version,
        "label": label,
        "analysis_file": str(analysis_jsonl),
        "total_newly_passing": len(improved_ids),
        "genuine_fix_count": len(genuine_fix_ids),
        "untracked_count": len(untracked_ids),
        "true_fix_rate": round(true_fix_rate, 4),
        "genuine_fix_ids": genuine_fix_ids,
        "untracked_ids": untracked_ids,
        "genuine_fixes_by_component": component_counts,
    }

    out_path = FAILURE_ANALYSIS_DIR / "results" / f"gaia_attribution_{label}_v{version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log(f"  Attribution result saved → {out_path.name}")

    return result


def step_analyze_regressions(version: int, model: str, regressed_ids: list[str],
                              val_traces: Path, gaia_subset: Path,
                              analysis_model: str, workers: int, dry_run: bool) -> Path:
    """Run failure analysis on val regressions."""
    output_file = FAILURE_ANALYSIS_DIR / "results" / f"gaia_val_regression_analyses_v{version}.jsonl"
    log(f"Step 8: analyze {len(regressed_ids)} val regressions → {output_file.name}")

    if not regressed_ids:
        log("  No regressions to analyze")
        return output_file

    if dry_run:
        print(f"  [DRY RUN] would analyze: {regressed_ids}")
        return output_file

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(regressed_ids) + "\n")
        ids_file = Path(f.name)

    try:
        cmd = [
            VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
            "--mode", "gaia",
            "--model", analysis_model,
            "--traces-dir", str(val_traces),
            "--gaia-subset", str(gaia_subset),
            "--output-file", str(output_file),
            "--instance-ids-file", str(ids_file),
            "--workers", str(workers),
            "--agent-source-dir", str(base_agent_dir(model, version) / "src" / "open_deep_research"),
            "--no-resume",
        ]
        run(cmd)
    finally:
        ids_file.unlink(missing_ok=True)

    return output_file


# ── Main loop ──────────────────────────────────────────────────────────────────

def print_test_eval_command(enhanced_dir: Path, model: str) -> None:
    version = enhanced_dir.name
    print(f"""
Next: run final GAIA held-out test evaluation with {version}

  AGENT_SRC_DIR={enhanced_dir}/src \\
  ./task_agent/run_gaia.sh \\
      --subset "$(pwd)/data/gaia_test_60" \\
      --model {model} \\
      --workers 2 --concurrency 4 \\
      --output traces/gaia_test_{model_slug(model)}_{version}

  mkdir -p eval/gaia_results_test_{version}
  .venv/bin/python3 eval/eval_gaia.py \\
      --preds traces/gaia_test_{model_slug(model)}_{version}/preds.json \\
      --subset data/gaia_test_60 \\
      --output eval/gaia_results_test_{version}/results.json \\
      --traces-dir traces/gaia_test_{model_slug(model)}_{version}
""")

def main() -> None:
    parser = argparse.ArgumentParser(description="HarnessFix pipeline for GAIA")
    parser.add_argument("--model", default="openai/gpt-5-mini",
                        help="Model for GAIA inference (default: openai/gpt-5-mini)")
    parser.add_argument("--analysis-model", default="openai/gpt-5-mini",
                        help="Model for failure analysis + aggregation")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel task workers for inference (default: 2)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrency within the GAIA runner (default: 4)")
    parser.add_argument("--start-version", type=int, default=1,
                        help="Starting enhanced version number (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max outer loop iterations (default: 3)")
    parser.add_argument("--max-modify-retries", type=int, default=2,
                        help="Max redo-modify retries when train set does not improve (default: 2)")
    parser.add_argument("--max-regression", type=int, default=2,
                        help="Max regressions allowed to pass val gate (default: 2)")
    parser.add_argument("--min-improvement", type=int, default=1,
                        help="Min net improvement required to pass val gate (default: 1)")
    parser.add_argument("--max-cost-ratio", type=float, default=1.25,
                        help="Max allowed validation cost ratio for the regression-aware gate (default: 1.25)")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate all steps even if outputs exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without running anything")
    args = parser.parse_args()

    # Validate datasets
    if not TRAIN_DIR.exists():
        print("ERROR: data/gaia_train_60/ not found.")
        print("Run: python3 data/sample_gaia.py")
        sys.exit(1)
    if not VAL_DIR.exists():
        print("ERROR: data/gaia_val_30/ not found.")
        print("Run: python3 data/sample_gaia.py")
        sys.exit(1)
    if not TEST_DIR.exists():
        print("ERROR: data/gaia_test_60/ not found.")
        print("Run: python3 data/sample_gaia.py")
        sys.exit(1)

    print(f"""
{'='*60}
  HarnessFix Pipeline (GAIA)
  model:             {args.model}
  analysis_model:    {args.analysis_model}
  start_version:     v{args.start_version}
  max_iterations:    {args.max_iterations}
  max_modify_retries:{args.max_modify_retries}
  val gate:          regression <= {args.max_regression}, fallback net >= {args.min_improvement}, cost <= {args.max_cost_ratio}
  workers:           {args.workers}  concurrency: {args.concurrency}
{'='*60}
""")

    baseline_traces = step_val_baseline_inference(
        model=args.model,
        workers=args.workers,
        concurrency=args.concurrency,
        force=args.force,
        dry_run=args.dry_run,
    )
    baseline_eval = step_val_baseline_evaluate(
        model=args.model,
        baseline_traces=baseline_traces,
        force=args.force,
        dry_run=args.dry_run,
    )

    val_analyses_path: Path | None = None
    prev_plan_path: Path | None = None
    current_base_version = args.start_version - 1
    promoted_versions: list[int] = []

    for i in range(args.max_iterations):
        version = args.start_version + i
        print(f"\n{'#'*60}")
        print(f"  GAIA ITERATION {i+1}/{args.max_iterations}  (candidate v{version}, base v{current_base_version})")
        print(f"{'#'*60}")
        # ── Baseline train inference + eval + analysis (on base agent) ─────────
        train_traces = step_train_inference(
            model=args.model,
            base_version=current_base_version,
            workers=args.workers,
            concurrency=args.concurrency,
            force=args.force,
            dry_run=args.dry_run,
        )
        train_eval = step_train_evaluate(
            model=args.model,
            base_version=current_base_version,
            train_traces=train_traces,
            force=args.force,
            dry_run=args.dry_run,
        )
        train_analysis = step_train_failure_analysis(
            model=args.model,
            base_version=current_base_version,
            train_traces=train_traces,
            train_eval_dir_path=train_eval,
            analysis_model=args.analysis_model,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
        )

        # ── Aggregate failure analyses → improvement plan ──────────────────────
        plan_path = step_aggregate(
            version=version,
            model=args.model,
            analysis_results_path=train_analysis,
            analysis_model=args.analysis_model,
            val_analyses_path=val_analyses_path,
            prev_plan_path=prev_plan_path,
            force=args.force,
            dry_run=args.dry_run,
        )

        # ── Inner retry loop: modify → check train → redo if not improved ──────
        train_compare_path: Path | None = None  # feedback from previous retry
        target_dir: Path = TASK_AGENT_DIR / f"enhanced_odr_{model_slug(args.model)}_v{version}"
        audit: dict | None = None

        for retry in range(args.max_modify_retries + 1):
            if retry > 0:
                print(f"\n  {'─'*50}")
                log(f"  Modify retry {retry}/{args.max_modify_retries} "
                    f"(train did not improve — redoing with feedback)")
                print(f"  {'─'*50}")

            target_dir = step_modify(
                version=version,
                base_version=current_base_version,
                plan_path=plan_path,
                model=args.model,
                modify_model=args.analysis_model,
                force=(retry > 0) or args.force,
                dry_run=args.dry_run,
                train_compare_path=train_compare_path,
            )

            audit = step_plan_diff_audit(
                version=version,
                model=args.model,
                base_version=current_base_version,
                plan_path=plan_path,
                target_dir=target_dir,
                dry_run=args.dry_run,
            )
            if not audit.get("passed", True):
                log(f"  Audit failed: {audit.get('violations', [])}")
                if retry < args.max_modify_retries:
                    if not args.dry_run:
                        _cleanup_for_redo(version, args.model)
                    continue
                break

            # Run enhanced agent on train set and compare vs baseline
            enh_train_traces = step_train_inference_enhanced(
                version=version,
                enhanced_src=target_dir / "src",
                model=args.model,
                workers=args.workers,
                concurrency=args.concurrency,
                force=(retry > 0) or args.force,
                dry_run=args.dry_run,
            )
            enh_train_eval = step_train_evaluate_enhanced(
                version=version,
                model=args.model,
                enhanced_train_traces=enh_train_traces,
                force=(retry > 0) or args.force,
                dry_run=args.dry_run,
            )
            train_compare = step_compare_train(
                version=version,
                model=args.model,
                baseline_train_eval=train_eval,
                enhanced_train_eval=enh_train_eval,
                dry_run=args.dry_run,
            )

            train_net = train_compare.get("net_change", 0)
            train_improved = train_net > 0
            log(f"  Train compare (v{version} retry={retry}): "
                f"net={train_net:+d}  "
                f"improvements={train_compare.get('improvement_count', 0)}  "
                f"regressions={train_compare.get('regression_count', 0)}  "
                f"({'IMPROVED' if train_improved else 'NOT IMPROVED'})")

            if not args.dry_run and train_compare.get("improved_ids"):
                step_attribution_analysis(
                    version=version,
                    improved_ids=train_compare["improved_ids"],
                    analysis_jsonl=train_analysis,
                    label=f"train_retry{retry}",
                )

            if train_improved:
                log(f"  Train improved — proceeding to val gate")
                break

            if retry < args.max_modify_retries:
                log(f"  Train did not improve — cleaning up and retrying modify with feedback ...")
                train_compare_path = (
                    FAILURE_ANALYSIS_DIR / "results" / f"gaia_train_compare_v{version}.json"
                )
                if not args.dry_run:
                    _cleanup_for_redo(version, args.model)
            else:
                log(f"  Train did not improve after {retry + 1} attempt(s) — "
                    f"proceeding to val gate anyway (best effort)")

        if audit is not None and not audit.get("passed", True):
            gate = {
                "passed": False,
                "failure_reasons": ["audit_failed"],
                "regressed_ids": [],
                "improved_ids": [],
            }
            step_record_memory("gaia", version, plan_path, audit, gate, args.dry_run)
            prev_plan_path = plan_path
            val_analyses_path = None
            continue

        # ── Val inference + eval ───────────────────────────────────────────────
        val_traces_dir = step_val_inference(
            version=version,
            enhanced_src=target_dir / "src",
            model=args.model,
            workers=args.workers,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
        val_eval_dir = step_val_evaluate(
            version=version,
            val_traces=val_traces_dir,
            model=args.model,
            dry_run=args.dry_run,
        )

        # ── Val gate check ─────────────────────────────────────────────────────
        base_val_traces = val_traces_dir_for_base(args.model, current_base_version)
        base_val_eval = val_eval_dir_for_base(args.model, current_base_version)
        gate = step_gate_check(
            version=version,
            baseline_eval_dir=base_val_eval,
            val_eval_dir=val_eval_dir,
            baseline_traces_dir=base_val_traces,
            val_traces_dir=val_traces_dir,
            plan_path=plan_path,
            max_regression=args.max_regression,
            min_improvement=args.min_improvement,
            max_cost_ratio=args.max_cost_ratio,
            dry_run=args.dry_run,
        )
        step_record_memory("gaia", version, plan_path, audit, gate, args.dry_run)

        if not args.dry_run and gate.get("improved_ids"):
            step_attribution_analysis(
                version=version,
                improved_ids=gate["improved_ids"],
                analysis_jsonl=train_analysis,
                label="val",
            )

        if gate.get("passed"):
            log(f"✓ PROMOTED v{version}: "
                f"val {gate.get('baseline_count')} → {gate.get('current_count')} resolved")
            current_base_version = version
            promoted_versions.append(version)
        else:
            log(f"✗ NOT PROMOTED v{version}: "
                f"{gate.get('regression_count', '?')} regressions, "
                f"{gate.get('improvement_count', '?')} improvements")

        # ── Analyze val regressions → feed into next outer iteration ───────────
        regressed_ids = gate.get("regressed_ids", [])
        val_analyses_path = step_analyze_regressions(
            version=version,
            model=args.model,
            regressed_ids=regressed_ids,
            val_traces=val_traces_dir,
            gaia_subset=VAL_DIR,
            analysis_model=args.analysis_model,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        prev_plan_path = plan_path

    best_version = best_version_by_val(args.model, promoted_versions)
    best_dir = base_agent_dir(args.model, best_version)
    best_eval = val_eval_dir_for_base(args.model, best_version)
    log(f"Finished iterations. Best-so-far version: v{best_version} ({best_dir.name})")
    log(f"Best validation resolved: {resolved_count_from_eval(best_eval)}/{len((VAL_DIR / 'instance_ids.txt').read_text().split())}")
    if best_version > 0:
        print_test_eval_command(best_dir, args.model)
    else:
        log("No enhanced candidate beat the original baseline under promotion rules.")


if __name__ == "__main__":
    main()

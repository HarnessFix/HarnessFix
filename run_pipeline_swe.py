#!/usr/bin/env python3
"""Closed-loop HarnessFix pipeline.

Bootstrap phase:
  0. val_base_infer   — run original agent on val_50 if missing
  1. val_base_eval    — evaluate val baseline predictions if missing

Iteration phase:
  0. train_infer      — run the current base agent on train_100
  1. train_eval       — evaluate train predictions
  2. train_analysis   — analyze failed train instances for the current base agent
  3. aggregate        — generate improvement_plan_vN.md + improvement_plan_vN.json
                        (uses val regression analyses from previous iteration if any)
  4. modify           — create enhanced_swe_vN from the current base agent + plan
  5. train_enh_infer  — run enhanced agent on train_100 for train-side comparison
  6. train_enh_eval   — evaluate enhanced train predictions
  7. train_compare    — compare enhanced train vs current base train (reported only)
  8. val_infer        — run enhanced agent on val_50
  9. val_eval         — evaluate val predictions with swebench harness
 10. val_compare      — compare enhanced val vs current base val
 11. promote          — keep candidate as next base if it improves validation behavior
 12. iterate          — continue until max iterations or consecutive promotion failures

Usage:
  # Full run from scratch (version 1)
  .venv/bin/python3 run_pipeline_swe.py

  # Resume from a specific version
  .venv/bin/python3 run_pipeline_swe.py --start-version 2

  # Dry run
  .venv/bin/python3 run_pipeline_swe.py --dry-run
"""

import argparse
import difflib
import json
import os
import re
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
LOGS_DIR = REPO_ROOT / "logs" / "run_evaluation"
TRAIN_DATASET_DIR = DATA_DIR / "verified_train_100"
VAL_DATASET_DIR = DATA_DIR / "verified_val_50"
TRAIN_IDS_FILE = TRAIN_DATASET_DIR / "instance_ids.txt"
VAL_IDS_FILE = VAL_DATASET_DIR / "instance_ids.txt"
ORIGINAL_AGENT_DIR = TASK_AGENT_DIR / "mini-swe-agent"
STATE_FILE = REPO_ROOT / "pipeline_state.json"
RUN_LABEL = ""
TRAIN_INFERENCE_TIMEOUT_SECONDS = int(os.environ.get("HARNESSFIX_SWE_TRAIN_TIMEOUT_SECONDS", 36 * 3600))
VAL_INFERENCE_TIMEOUT_SECONDS = int(os.environ.get("HARNESSFIX_SWE_VAL_TIMEOUT_SECONDS", 18 * 3600))


# ── Utilities ─────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"\n[pipeline] {msg}", flush=True)


def run(cmd: list, env: dict | None = None, check: bool = True,
        timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess command, streaming output."""
    merged_env = {**os.environ, **(env or {})}
    merged_env["PATH"] = f"{REPO_ROOT / '.venv' / 'bin'}:{merged_env.get('PATH', '')}"
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    try:
        return subprocess.run(cmd, env=merged_env, check=check, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"\n[pipeline] ERROR: subprocess timed out after {timeout}s: {cmd[0]}", flush=True)
        raise


def find_eval_json(directory: Path) -> Path | None:
    jsons = [j for j in directory.glob("*.json") if not j.name.startswith("run_")]
    return jsons[0] if jsons else None


def model_short(model: str) -> str:
    return model.replace("/", "-").replace("_", "-")


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace("-", "_").lower()


def eval_model_slug(model: str) -> str:
    return model.replace("/", "__")


def plan_spec_path(plan_path: Path) -> Path:
    return plan_path.with_suffix(".json")


def run_label_suffix() -> str:
    return f"_{RUN_LABEL}" if RUN_LABEL else ""


def sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("_")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def ids_filter_regex(ids_file: Path) -> str:
    ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    if not ids:
        return r"$^"
    return "^(?:" + "|".join(re.escape(instance_id) for instance_id in ids) + ")$"


def enhanced_agent_dir(version: int) -> Path:
    suffix = run_label_suffix()
    return TASK_AGENT_DIR / f"enhanced_swe_v{version}{suffix}"


def base_agent_dir(version: int) -> Path:
    if version <= 0:
        return ORIGINAL_AGENT_DIR
    return enhanced_agent_dir(version)


def base_agent_label(version: int) -> str:
    if version <= 0:
        return "original"
    return f"enhanced_v{version}"


def train_run_id(model: str, base_version: int) -> str:
    return f"swe_train_{model_slug(model)}_{base_agent_label(base_version)}{run_label_suffix()}"


def val_baseline_run_id(model: str) -> str:
    return f"swe_val_{model_slug(model)}_original{run_label_suffix()}"


def train_traces_dir(model: str, base_version: int) -> Path:
    return TRACES_DIR / train_run_id(model, base_version)


def train_eval_dir(model: str, base_version: int) -> Path:
    return EVAL_DIR / f"results_{train_run_id(model, base_version)}"


def train_logs_dir(model: str, base_version: int) -> Path:
    return LOGS_DIR / train_run_id(model, base_version) / eval_model_slug(model)


def train_analysis_path(model: str, base_version: int) -> Path:
    return FAILURE_ANALYSIS_DIR / "results" / f"{train_run_id(model, base_version)}_analysis.jsonl"


def val_baseline_traces_dir(model: str) -> Path:
    return TRACES_DIR / val_baseline_run_id(model)


def val_baseline_eval_dir(model: str) -> Path:
    return EVAL_DIR / f"results_{val_baseline_run_id(model)}"


def val_enhanced_run_id(model: str, version: int) -> str:
    ms = model_short(model)
    return f"val_{ms}_swe_enhanced_v{version}{run_label_suffix()}"


def val_enhanced_traces_dir(model: str, version: int) -> Path:
    return TRACES_DIR / val_enhanced_run_id(model, version)


def val_enhanced_eval_dir(model: str, version: int) -> Path:
    return EVAL_DIR / f"results_{val_enhanced_run_id(model, version)}"


def val_traces_dir_for_base(model: str, base_version: int) -> Path:
    return val_baseline_traces_dir(model) if base_version <= 0 else val_enhanced_traces_dir(model, base_version)


def val_eval_dir_for_base(model: str, base_version: int) -> Path:
    return val_baseline_eval_dir(model) if base_version <= 0 else val_enhanced_eval_dir(model, base_version)


def iteration_report_path(version: int) -> Path:
    return FAILURE_ANALYSIS_DIR / "results" / f"iteration_report_swe_v{version}{run_label_suffix()}.json"


def resolved_count_from_eval(eval_dir: Path, ids_file: Path) -> int:
    eval_json = find_eval_json(eval_dir)
    if not eval_json:
        return 0
    subset = set(ids_file.read_text().split())
    data = json.loads(eval_json.read_text())
    return len(set(data.get("resolved_ids", [])) & subset)


def expected_failed_ids_from_eval(eval_json: Path) -> set[str]:
    data = json.loads(eval_json.read_text())
    failed: set[str] = set()
    for key in ("empty_patch_ids", "unresolved_ids", "error_ids"):
        failed.update(str(instance_id) for instance_id in data.get(key, []) if instance_id)
    return failed


def completed_analysis_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        instance_id = record.get("instance_id")
        if instance_id:
            completed.add(str(instance_id))
    return completed


def analysis_complete(path: Path, expected_ids: set[str]) -> bool:
    if not expected_ids:
        return path.exists()
    return expected_ids <= completed_analysis_ids(path)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_train_inference(model: str, base_version: int, workers: int, cost_limit: float,
                         force: bool, dry_run: bool) -> Path:
    output_dir = train_traces_dir(model, base_version)
    source_dir = base_agent_dir(base_version)
    log(f"Step 0: train inference on {source_dir.name} → {output_dir.name}/")

    if (output_dir / "preds.json").exists() and not force:
        log("  train preds.json already exists, skipping")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_swebench.sh"),
        "--subset", str(TRAIN_DATASET_DIR),
        "--split", "test",
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
        "--cost-limit", str(cost_limit),
        "--filter", ids_filter_regex(TRAIN_IDS_FILE),
    ]

    if dry_run:
        prefix = f"PYTHONPATH={source_dir / 'src'} " if base_version > 0 else ""
        print(f"  [DRY RUN] would run: {prefix}{' '.join(str(c) for c in cmd)}")
        return output_dir

    env = {"PYTHONPATH": str(source_dir / "src")} if base_version > 0 else None
    run(cmd, env=env, timeout=TRAIN_INFERENCE_TIMEOUT_SECONDS)
    return output_dir


def step_train_evaluate(model: str, base_version: int, train_traces: Path, workers: int,
                        force: bool, dry_run: bool) -> Path:
    report_dir = train_eval_dir(model, base_version)
    run_id = train_run_id(model, base_version)
    log(f"Step 1: train evaluate → {report_dir.name}/")

    if find_eval_json(report_dir) and not force:
        log("  train eval results already exist, skipping")
        return report_dir

    preds_path = train_traces / "preds.json"
    train_ids = TRAIN_IDS_FILE.read_text().split()
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        VENV_PYTHON, "-m", "swebench.harness.run_evaluation",
        "-d", "princeton-nlp/SWE-Bench_Verified",
        "-s", "test",
        "-p", str(preds_path),
        "--max_workers", str(workers),
        "-id", run_id,
        "--report_dir", str(report_dir),
        "-i", *train_ids,
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd[:8])} ... -i <100 ids>")
        return report_dir

    run(cmd)
    return report_dir


def step_train_failure_analysis(model: str, base_version: int,
                                train_traces: Path, train_eval_dir_path: Path,
                                analysis_model: str, workers: int,
                                force: bool, dry_run: bool) -> Path:
    output_file = train_analysis_path(model, base_version)
    logs_dir = train_logs_dir(model, base_version)
    log(f"Step 2: train failure analysis → {output_file.name}")

    eval_json = find_eval_json(train_eval_dir_path)
    if eval_json is None:
        raise FileNotFoundError(f"No train eval JSON found under {train_eval_dir_path}")
    expected_ids = expected_failed_ids_from_eval(eval_json)

    if output_file.exists() and not force:
        completed_ids = completed_analysis_ids(output_file)
        if expected_ids <= completed_ids:
            log(f"  train failure analyses complete ({len(expected_ids)}/{len(expected_ids)}), skipping")
            return output_file
        missing_ids = sorted(expected_ids - completed_ids)
        log(
            "  train failure analyses incomplete "
            f"({len(completed_ids & expected_ids)}/{len(expected_ids)}); resuming "
            f"{len(missing_ids)} missing instance(s)"
        )

    if dry_run:
        cmd = [
            VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
            "--model", analysis_model,
            "--traces-dir", train_traces,
            "--eval-results", eval_json,
            "--logs-dir", logs_dir,
            "--output-file", output_file,
            "--workers", str(workers),
            "--agent-source-dir", str(base_agent_dir(base_version) / "src" / "minisweagent"),
            "--resume",
        ]
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return output_file

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
        "--model", analysis_model,
        "--traces-dir", train_traces,
        "--eval-results", eval_json,
        "--logs-dir", logs_dir,
        "--output-file", output_file,
        "--workers", str(workers),
        "--agent-source-dir", str(base_agent_dir(base_version) / "src" / "minisweagent"),
        "--resume",
    ]

    run(cmd)
    completed_ids = completed_analysis_ids(output_file)
    if not analysis_complete(output_file, expected_ids):
        missing_ids = sorted(expected_ids - completed_ids)
        raise RuntimeError(
            f"Train failure analysis incomplete for base v{base_version}: "
            f"{len(completed_ids & expected_ids)}/{len(expected_ids)} complete, missing {missing_ids}"
        )
    return output_file


def step_val_baseline_inference(model: str, workers: int, cost_limit: float,
                                force: bool, dry_run: bool) -> Path:
    output_dir = val_baseline_traces_dir(model)
    log(f"Bootstrap 3: val baseline inference → {output_dir.name}/")

    if (output_dir / "preds.json").exists() and not force:
        log("  val baseline preds.json already exists, skipping")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_swebench.sh"),
        "--subset", str(VAL_DATASET_DIR),
        "--split", "test",
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
        "--cost-limit", str(cost_limit),
        "--filter", ids_filter_regex(VAL_IDS_FILE),
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return output_dir

    run(cmd, timeout=VAL_INFERENCE_TIMEOUT_SECONDS)
    return output_dir


def step_val_baseline_evaluate(model: str, baseline_traces: Path, workers: int,
                               force: bool, dry_run: bool) -> Path:
    report_dir = val_baseline_eval_dir(model)
    run_id = val_baseline_run_id(model)
    log(f"Bootstrap 4: val baseline evaluate → {report_dir.name}/")

    if find_eval_json(report_dir) and not force:
        log("  val baseline eval results already exist, skipping")
        return report_dir

    preds_path = baseline_traces / "preds.json"
    val_ids = VAL_IDS_FILE.read_text().split()
    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        VENV_PYTHON, "-m", "swebench.harness.run_evaluation",
        "-d", "princeton-nlp/SWE-Bench_Verified",
        "-s", "test",
        "-p", str(preds_path),
        "--max_workers", str(workers),
        "-id", run_id,
        "--report_dir", str(report_dir),
        "-i", *val_ids,
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd[:8])} ... -i <50 ids>")
        return report_dir

    run(cmd)
    return report_dir

def step_aggregate(version: int, model: str, analysis_results_path: Path, analysis_model: str,
                   val_analyses_path: Path | None,
                   prev_plan_path: Path | None,
                   prev_iteration_report_path: Path | None,
                   force: bool, dry_run: bool) -> Path:
    """Generate improvement_plans/swe_{model_slug}_vN.md and .json."""
    IMPROVEMENT_PLANS_DIR.mkdir(exist_ok=True)
    plan_path = IMPROVEMENT_PLANS_DIR / f"swe_{model_slug(model)}_v{version}{run_label_suffix()}.md"
    spec_path = plan_spec_path(plan_path)
    log(f"Step 3: aggregate → {plan_path.name}")

    if plan_path.exists() and spec_path.exists() and not force:
        log(f"  {plan_path.name} already exists, skipping (use --force to regenerate)")
        return plan_path

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "aggregate_results.py",
        "--model", analysis_model,
        "--results-file", analysis_results_path,
        "--output", plan_path,
        "--spec-output", spec_path,
        "--force",
    ]
    if val_analyses_path and val_analyses_path.exists():
        cmd += ["--val-analyses", val_analyses_path]
    if prev_plan_path and prev_plan_path.exists():
        cmd += ["--prev-plan", prev_plan_path]
    if prev_iteration_report_path and prev_iteration_report_path.exists():
        cmd += ["--prev-iteration-report", prev_iteration_report_path]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd)}")
        return plan_path

    run(cmd)
    return plan_path


def step_modify(version: int, base_version: int, plan_path: Path, modify_model: str,
                force: bool, dry_run: bool) -> Path:
    """Create enhanced_swe_vN from improvement plan."""
    target_dir = enhanced_agent_dir(version)
    log(f"Step 4: modify → {target_dir.name}/")

    if target_dir.exists() and not force:
        log(f"  {target_dir.name}/ already exists, skipping (use --force to regenerate)")
        return target_dir

    if dry_run:
        print(f"  [DRY RUN] would call _run_modify_versioned_swe(version={version}, model={modify_model})")
        return target_dir

    _run_modify_versioned_swe(
        version,
        base_version,
        plan_path,
        plan_spec_path(plan_path),
        target_dir,
        modify_model,
    )
    return target_dir


def step_plan_diff_audit(version: int, base_version: int, plan_path: Path,
                         target_dir: Path, dry_run: bool) -> dict:
    """Audit whether the candidate diff matches the approved plan/spec scope."""
    log(f"Step 4.5: audit candidate diff → {target_dir.name}")
    output_path = FAILURE_ANALYSIS_DIR / "results" / f"audit_swe_v{version}{run_label_suffix()}.json"

    if dry_run:
        print(f"  [DRY RUN] would audit {target_dir.name} against {base_agent_dir(base_version).name}")
        return {"passed": True, "dry_run": True, "changed_files": []}

    cmd = [
        VENV_PYTHON, FAILURE_ANALYSIS_DIR / "plan_diff_audit.py",
        "--mode", "swe",
        "--original-dir", base_agent_dir(base_version),
        "--candidate-dir", target_dir,
        "--spec", plan_spec_path(plan_path),
        "--output", output_path,
    ]
    subprocess.run(cmd, check=False)
    if output_path.exists():
        return json.loads(output_path.read_text())
    return {"passed": False, "violations": ["audit_output_missing"], "changed_files": []}


def _run_modify_versioned_swe(version: int, base_version: int,
                              plan_path: Path, spec_path: Path, target_dir: Path,
                              modify_model: str) -> None:
    """Call modify_agent with versioned paths via direct Python import."""
    import shutil
    import yaml
    source_dir = base_agent_dir(base_version)

    # Copy current base agent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    log(f"  Copying {source_dir.name} → {target_dir.name} ...")
    shutil.copytree(
        source_dir, target_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.egg-info"),
    )

    # Load modify_agent config
    config_path = REPO_ROOT / "enhancement_implementation" / "config_swe.yaml"
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
    traj_path = results_dir / f"modify_agent_v{version}{run_label_suffix()}.traj.json"

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


def step_val_inference(version: int, enhanced_src: Path, model: str,
                       workers: int, cost_limit: float, dry_run: bool) -> Path:
    """Run enhanced agent on val_50."""
    output_dir = val_enhanced_traces_dir(model, version)
    log(f"Step 5: val inference → {output_dir.name}/")

    if (output_dir / "preds.json").exists():
        log(f"  preds.json already exists, skipping val inference")
        return output_dir

    cmd = [
        "bash", str(TASK_AGENT_DIR / "run_swebench.sh"),
        "--subset", str(VAL_DATASET_DIR),
        "--split", "test",
        "--model", model,
        "--workers", str(workers),
        "--output", str(output_dir),
        "--cost-limit", str(cost_limit),
        "--filter", ids_filter_regex(VAL_IDS_FILE),
    ]

    if dry_run:
        print(f"  [DRY RUN] PYTHONPATH={enhanced_src} {' '.join(str(c) for c in cmd)}")
        return output_dir

    run(cmd, env={"PYTHONPATH": str(enhanced_src)}, timeout=VAL_INFERENCE_TIMEOUT_SECONDS)
    return output_dir


def step_val_evaluate(version: int, val_traces_dir: Path,
                      model: str, workers: int, dry_run: bool) -> Path:
    """Run swebench harness on val predictions."""
    run_id = val_enhanced_run_id(model, version)
    report_dir = val_enhanced_eval_dir(model, version)
    log(f"Step 6: val evaluate → {report_dir.name}/")

    if find_eval_json(report_dir):
        log(f"  eval results already exist, skipping")
        return report_dir

    preds_path = val_traces_dir / "preds.json"
    val_ids = VAL_IDS_FILE.read_text().split()

    report_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        VENV_PYTHON, "-m", "swebench.harness.run_evaluation",
        "-d", "princeton-nlp/SWE-Bench_Verified",
        "-s", "test",
        "-p", str(preds_path),
        "--max_workers", str(workers),
        "-id", run_id,
        "--report_dir", str(report_dir),
        "-i", *val_ids,
    ]

    if dry_run:
        print(f"  [DRY RUN] would run: {' '.join(str(c) for c in cmd[:8])} ... -i <50 ids>")
        return report_dir

    run(cmd)
    return report_dir


def step_train_compare(version: int, baseline_eval_dir: Path, current_eval_dir: Path,
                       baseline_traces_dir: Path, current_traces_dir: Path,
                       plan_path: Path, max_cost_ratio: float,
                       dry_run: bool) -> dict:
    """Compare enhanced train results against the current base train run.

    This is reported-only data for iterative analysis, not an acceptance gate.
    """
    output = FAILURE_ANALYSIS_DIR / "results" / f"train_compare_swe_v{version}{run_label_suffix()}.json"
    log(f"Step 6.5: train compare → {output.name} (reported only)")

    if dry_run:
        print(f"  [DRY RUN] would compare {current_eval_dir.name} vs {baseline_eval_dir.name}")
        return {"reported_only": True, "dry_run": True}

    result = build_run_comparison(
        split="train",
        comparison_type="train_current_base_vs_enhanced",
        ids_file=TRAIN_IDS_FILE,
        baseline_eval_dir=baseline_eval_dir,
        current_eval_dir=current_eval_dir,
        baseline_traces_dir=baseline_traces_dir,
        current_traces_dir=current_traces_dir,
        plan_path=plan_path,
        max_cost_ratio=max_cost_ratio,
    )
    result["reported_only"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def step_val_compare(version: int, baseline_eval_dir: Path, current_eval_dir: Path,
                     baseline_traces_dir: Path, current_traces_dir: Path,
                     plan_path: Path, max_cost_ratio: float,
                     dry_run: bool) -> dict:
    """Compare enhanced val results against the current promoted base."""
    output = FAILURE_ANALYSIS_DIR / "results" / f"val_compare_swe_v{version}{run_label_suffix()}.json"
    log(f"Step 7: val compare → {output.name} (promotion evidence)")

    if dry_run:
        print(f"  [DRY RUN] would compare {current_eval_dir.name} vs {baseline_eval_dir.name}")
        return {"dry_run": True, "regressed_ids": [], "improved_ids": []}

    result = build_run_comparison(
        split="val",
        comparison_type="val_current_base_vs_enhanced",
        ids_file=VAL_IDS_FILE,
        baseline_eval_dir=baseline_eval_dir,
        current_eval_dir=current_eval_dir,
        baseline_traces_dir=baseline_traces_dir,
        current_traces_dir=current_traces_dir,
        plan_path=plan_path,
        max_cost_ratio=max_cost_ratio,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def build_run_comparison(split: str, comparison_type: str, ids_file: Path,
                         baseline_eval_dir: Path, current_eval_dir: Path,
                         baseline_traces_dir: Path, current_traces_dir: Path,
                         plan_path: Path, max_cost_ratio: float) -> dict:
    """Build a resolved/metric/cost comparison for one dataset split."""
    from failure_analysis.validation_metrics import (
        compute_cost_ratio,
        compute_run_metrics,
        evaluate_target_metrics,
        load_eval_json,
    )

    subset = set(ids_file.read_text().split())
    baseline_path, _ = load_eval_json(baseline_eval_dir)
    current_path, _ = load_eval_json(current_eval_dir)

    baseline_data = json.loads(baseline_path.read_text())
    current_data = json.loads(current_path.read_text())
    baseline_resolved = set(baseline_data.get("resolved_ids", [])) & subset
    current_resolved = set(current_data.get("resolved_ids", [])) & subset
    regressed_ids = sorted(baseline_resolved - current_resolved)
    improved_ids = sorted(current_resolved - baseline_resolved)

    baseline_metrics = compute_run_metrics("swe", baseline_traces_dir, baseline_path)
    current_metrics = compute_run_metrics("swe", current_traces_dir, current_path)
    plan_spec = json.loads(plan_spec_path(plan_path).read_text()) if plan_spec_path(plan_path).exists() else {"fixes": []}
    target_results = evaluate_target_metrics(plan_spec, baseline_metrics, current_metrics)
    cost_ratio = compute_cost_ratio(baseline_metrics, current_metrics)

    return {
        "split": split,
        "comparison_type": comparison_type,
        "baseline_count": len(baseline_resolved),
        "current_count": len(current_resolved),
        "net_change": len(current_resolved) - len(baseline_resolved),
        "regression_count": len(regressed_ids),
        "improvement_count": len(improved_ids),
        "regressed_ids": regressed_ids,
        "improved_ids": improved_ids,
        "cost_ratio": cost_ratio,
        "target_metric_results": target_results,
        "baseline_metrics": baseline_metrics,
        "current_metrics": current_metrics,
        "comparison_config": {
            "mode": "swe",
            "ids_file": str(ids_file),
            "cost_report_threshold": max_cost_ratio,
        },
    }


def step_promotion_decision(version: int, audit: dict, train_compare: dict,
                            val_compare: dict, min_improvement: int,
                            min_target_metrics: int,
                            max_error_rate_delta: float,
                            max_invalid_rate_delta: float,
                            dry_run: bool) -> dict:
    """Decide whether a candidate becomes the base for the next iteration."""
    output = FAILURE_ANALYSIS_DIR / "results" / f"promotion_swe_v{version}{run_label_suffix()}.json"
    log(f"Step 8: promotion decision → {output.name}")

    if dry_run:
        print("  [DRY RUN] would decide whether to promote candidate")
        return {"passed": False, "promoted": False, "dry_run": True, "regressed_ids": []}

    baseline_metrics = val_compare.get("baseline_metrics", {}).get("metrics", {})
    current_metrics = val_compare.get("current_metrics", {}).get("metrics", {})
    error_delta = current_metrics.get("error_rate", 0.0) - baseline_metrics.get("error_rate", 0.0)
    invalid_delta = (
        current_metrics.get("invalid_submission_rate", 0.0)
        - baseline_metrics.get("invalid_submission_rate", 0.0)
    )
    target_improved = val_compare.get("target_metric_results", {}).get("improved_metric_count", 0)

    checks = {
        "audit_passed": bool(audit.get("passed", True)),
        "net_improvement": val_compare.get("net_change", 0) >= min_improvement,
        "target_metric_improved": target_improved >= min_target_metrics,
        "error_delta_within_limit": error_delta <= max_error_rate_delta,
        "invalid_delta_within_limit": invalid_delta <= max_invalid_rate_delta,
    }
    promoted = all(checks.values())
    failure_reasons = [name for name, passed in checks.items() if not passed]

    result = {
        "passed": promoted,
        "promoted": promoted,
        "decision": "promote" if promoted else "do_not_promote",
        "failure_reasons": failure_reasons,
        "checks": checks,
        "val_net_change": val_compare.get("net_change"),
        "val_regression_count": val_compare.get("regression_count"),
        "val_improvement_count": val_compare.get("improvement_count"),
        "train_net_change": train_compare.get("net_change"),
        "train_regression_count": train_compare.get("regression_count"),
        "train_improvement_count": train_compare.get("improvement_count"),
        "target_improved_metric_count": target_improved,
        "error_rate_delta": error_delta,
        "invalid_submission_rate_delta": invalid_delta,
        "cost_ratio": val_compare.get("cost_ratio"),
        "cost_gate_enabled": False,
        "regressed_ids": val_compare.get("regressed_ids", []),
        "improved_ids": val_compare.get("improved_ids", []),
        "promotion_config": {
            "min_improvement": min_improvement,
            "min_target_metrics": min_target_metrics,
            "max_error_rate_delta": max_error_rate_delta,
            "max_invalid_rate_delta": max_invalid_rate_delta,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def step_iteration_report(version: int, base_version: int, target_dir: Path, plan_path: Path,
                          train_analysis: Path, audit: dict, train_compare: dict,
                          val_compare: dict, promotion: dict, dry_run: bool) -> Path:
    """Persist detailed iteration context for the next aggregate step."""
    output = iteration_report_path(version)
    log(f"Step 8.5: iteration report → {output.name}")

    if dry_run:
        print(f"  [DRY RUN] would write detailed report for v{version}")
        return output

    report = {
        "version": version,
        "base_version": base_version,
        "candidate_dir": str(target_dir),
        "base_dir": str(base_agent_dir(base_version)),
        "plan_path": str(plan_path),
        "plan_spec_path": str(plan_spec_path(plan_path)),
        "train_analysis_path": str(train_analysis),
        "plan_summary": load_plan_summary(plan_path),
        "audit": audit,
        "changed_files": audit.get("changed_files", []),
        "diff_summary": collect_diff_summary(base_agent_dir(base_version), target_dir, audit.get("changed_files", [])),
        "train_compare": train_compare,
        "val_compare": val_compare,
        "promotion": promotion,
        "next_iteration_guidance": build_next_iteration_guidance(train_compare, val_compare, promotion),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return output


def load_plan_summary(plan_path: Path) -> dict:
    spec = {}
    if plan_spec_path(plan_path).exists():
        spec = json.loads(plan_spec_path(plan_path).read_text())
    plan_text = plan_path.read_text() if plan_path.exists() else ""
    return {
        "metadata": spec.get("plan_metadata", {}),
        "fix_count": len(spec.get("fixes", [])),
        "fixes": [
            {
                "fix_id": fix.get("fix_id"),
                "operator_family": fix.get("operator_family"),
                "target_defect_class": fix.get("target_defect_class"),
                "target_files": fix.get("target_files", []),
                "target_metrics": fix.get("target_metrics", []),
                "risk_level": fix.get("risk_level"),
                "regression_risks": fix.get("regression_risks", []),
            }
            for fix in spec.get("fixes", [])
        ],
        "markdown": plan_text,
    }


def collect_diff_summary(base_dir: Path, target_dir: Path, changed_files: list[str]) -> dict:
    summary = {}
    for rel in changed_files:
        base_file = base_dir / rel
        target_file = target_dir / rel
        try:
            before = base_file.read_text().splitlines()
        except Exception:
            before = []
        try:
            after = target_file.read_text().splitlines()
        except Exception:
            after = []
        diff = list(difflib.unified_diff(
            before,
            after,
            fromfile=f"{base_dir.name}/{rel}",
            tofile=f"{target_dir.name}/{rel}",
            lineterm="",
        ))
        summary[rel] = {
            "diff_line_count": len(diff),
            "diff": "\n".join(diff),
            "truncated": False,
        }
    return summary


def build_next_iteration_guidance(train_compare: dict, val_compare: dict, promotion: dict) -> list[str]:
    guidance = []
    if promotion.get("promoted"):
        guidance.append("Candidate was promoted. Build the next version on top of it and preserve its validated gains.")
    else:
        guidance.append("Candidate was not promoted. Treat it as a failed attempt; avoid repeating its harmful edits.")
    if val_compare.get("regressed_ids"):
        guidance.append(
            "Analyze and repair validation regressions: "
            + ", ".join(val_compare.get("regressed_ids", []))
        )
    if train_compare.get("regressed_ids"):
        guidance.append(
            "Use train regressions as additional side-effect evidence: "
            + ", ".join(train_compare.get("regressed_ids", []))
        )
    for metric in val_compare.get("target_metric_results", {}).get("metrics", []):
        if metric.get("improved"):
            guidance.append(
                f"Preserve val metric improvement: {metric['metric']} "
                f"{metric['baseline']:.4f} -> {metric['current']:.4f}."
            )
    failure_reasons = promotion.get("failure_reasons", [])
    if failure_reasons:
        guidance.append("Promotion failed because: " + ", ".join(failure_reasons))
    return guidance


def step_record_memory(mode: str, version: int, plan_path: Path, audit: dict | None,
                       outcome_record: dict | None, dry_run: bool) -> None:
    if dry_run:
        return
    from failure_analysis.harness_memory import build_memory_entry, default_memory_root, store_memory_entry

    spec = json.loads(plan_spec_path(plan_path).read_text())
    summary = spec.get("plan_metadata", {}).get("summary", plan_path.stem)
    outcome = (
        "accepted"
        if outcome_record and outcome_record.get("promoted") and (audit is None or audit.get("passed"))
        else "rejected"
    )
    entry = build_memory_entry(
        mode=mode,
        version=version,
        outcome=outcome,
        plan_path=plan_path,
        spec=spec,
        summary=summary,
        changed_files=[] if audit is None else audit.get("changed_files", []),
        audit=audit,
        gate=outcome_record,
    )
    store_memory_entry(default_memory_root(REPO_ROOT), entry)


def best_version_by_val(model: str, versions: list[int]) -> int:
    best_version = 0
    best_score = resolved_count_from_eval(val_baseline_eval_dir(model), VAL_IDS_FILE)
    for version in versions:
        score = resolved_count_from_eval(val_enhanced_eval_dir(model, version), VAL_IDS_FILE)
        if score > best_score:
            best_version = version
            best_score = score
    return best_version


def step_analyze_regressions(version: int, regressed_ids: list[str],
                              val_traces_dir: Path, analysis_model: str,
                              workers: int, dry_run: bool) -> Path:
    """Run failure analysis on val regressions."""
    output_file = FAILURE_ANALYSIS_DIR / "results" / f"val_regression_analyses_v{version}{run_label_suffix()}.jsonl"
    log(f"Step 8: analyze {len(regressed_ids)} val regressions → {output_file.name}")

    if not regressed_ids:
        log("  No regressions to analyze")
        return output_file

    if dry_run:
        print(f"  [DRY RUN] would analyze: {regressed_ids}")
        return output_file

    # Write regressed instance IDs to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(regressed_ids) + "\n")
        ids_file = Path(f.name)

    output_file.unlink(missing_ok=True)

    try:
        cmd = [
            VENV_PYTHON, FAILURE_ANALYSIS_DIR / "run_analysis.py",
            "--model", analysis_model,
            "--traces-dir", val_traces_dir,
            "--output-file", output_file,
            "--instance-ids-file", ids_file,
            "--workers", str(workers),
            "--agent-source-dir", str(base_agent_dir(version) / "src" / "minisweagent"),
            "--no-resume",   # fresh analysis each time
        ]
        run(cmd)
    finally:
        ids_file.unlink(missing_ok=True)

    expected_ids = set(regressed_ids)
    completed_ids = completed_analysis_ids(output_file)
    if not analysis_complete(output_file, expected_ids):
        missing_ids = sorted(expected_ids - completed_ids)
        raise RuntimeError(
            f"Val regression analysis incomplete for v{version}: "
            f"{len(completed_ids & expected_ids)}/{len(expected_ids)} complete, missing {missing_ids}"
        )

    return output_file


# ── Main loop ─────────────────────────────────────────────────────────────────

def print_test_eval_command(enhanced_dir: Path, model: str) -> None:
    ms = model_short(model)
    version = enhanced_dir.name  # e.g. enhanced_swe_v2
    print(f"""
Next: run final test set evaluation with {version}

  PYTHONPATH={enhanced_dir}/src \\
  ./task_agent/run_swebench.sh \\
      --subset "$(pwd)/data/verified_test_100" \\
      --split test --model {model} \\
      --workers 4 --output traces/test_{ms}_{version} --cost-limit 1

  mkdir -p eval/results_test_{version}
  python -m swebench.harness.run_evaluation \\
      -d princeton-nlp/SWE-Bench_Verified -s test \\
      -p traces/test_{ms}_{version}/preds.json \\
      --max_workers 4 -id test_{version} \\
      --report_dir eval/results_test_{version} \\
      -i $(cat data/verified_test_100/instance_ids.txt | tr '\\n' ' ')
""")


def main():
    global RUN_LABEL, TRAIN_DATASET_DIR, VAL_DATASET_DIR, TRAIN_IDS_FILE, VAL_IDS_FILE

    parser = argparse.ArgumentParser(description="Closed-loop HarnessFix pipeline")
    parser.add_argument("--model", default="openai/gpt-5-mini",
                        help="Model for agent inference (default: openai/gpt-5-mini)")
    parser.add_argument("--analysis-model", default="openai/claude-opus-4-5-20251101-thinking",
                        help="Model for failure analysis + aggregation (default: openai/claude-opus-4-5-20251101-thinking)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for inference + harness (default: 4)")
    parser.add_argument("--cost-limit", type=float, default=1.0,
                        help="Per-instance cost limit for inference (default: 1.0)")
    parser.add_argument("--start-version", type=int, default=1,
                        help="Starting version number (default: 1)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max loop iterations (default: 3)")
    parser.add_argument("--min-improvement", type=int, default=1,
                        help="Min validation net improvement required to promote a candidate (default: 1)")
    parser.add_argument("--min-target-metrics", type=int, default=1,
                        help="Min target metrics that must improve for promotion (default: 1)")
    parser.add_argument("--max-error-rate-delta", type=float, default=0.15,
                        help="Max allowed validation error-rate increase for promotion (default: 0.15)")
    parser.add_argument("--max-invalid-rate-delta", type=float, default=0.15,
                        help="Max allowed validation invalid-submission-rate increase for promotion (default: 0.15)")
    parser.add_argument("--max-promotion-failures", type=int, default=2,
                        help="Stop after this many consecutive non-promoted candidates (default: 2)")
    parser.add_argument("--max-cost-ratio", type=float, default=1.25,
                        help="Reported validation cost ratio threshold; retained for compatibility, not used for pass/fail")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DATASET_DIR,
                        help="Dataset directory for train inference/eval (default: data/verified_train_100)")
    parser.add_argument("--val-dir", type=Path, default=VAL_DATASET_DIR,
                        help="Dataset directory for val inference/eval (default: data/verified_val_50)")
    parser.add_argument("--train-ids-file", type=Path, default=None,
                        help="Instance IDs for train inference/eval (default: <train-dir>/instance_ids.txt)")
    parser.add_argument("--val-ids-file", type=Path, default=None,
                        help="Instance IDs for val inference/eval (default: <val-dir>/instance_ids.txt)")
    parser.add_argument("--run-label", default="",
                        help="Optional suffix for traces/eval/plans, useful for smoke runs")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate all steps even if outputs exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without running anything")
    args = parser.parse_args()

    TRAIN_DATASET_DIR = resolve_repo_path(args.train_dir)
    VAL_DATASET_DIR = resolve_repo_path(args.val_dir)
    TRAIN_IDS_FILE = resolve_repo_path(args.train_ids_file) if args.train_ids_file else TRAIN_DATASET_DIR / "instance_ids.txt"
    VAL_IDS_FILE = resolve_repo_path(args.val_ids_file) if args.val_ids_file else VAL_DATASET_DIR / "instance_ids.txt"
    RUN_LABEL = sanitize_label(args.run_label)

    # Validate datasets
    if not TRAIN_DATASET_DIR.exists():
        print(f"ERROR: train dataset not found: {TRAIN_DATASET_DIR}")
        sys.exit(1)
    if not VAL_DATASET_DIR.exists():
        print(f"ERROR: val dataset not found: {VAL_DATASET_DIR}")
        sys.exit(1)
    if not TRAIN_IDS_FILE.exists():
        print(f"ERROR: train ids file not found: {TRAIN_IDS_FILE}")
        sys.exit(1)
    if not VAL_IDS_FILE.exists():
        print(f"ERROR: val ids file not found: {VAL_IDS_FILE}")
        sys.exit(1)

    print(f"""
{'='*60}
  HarnessFix Pipeline
  model:           {args.model}
  analysis_model:  {args.analysis_model}
  start_version:   v{args.start_version}
  max_iterations:  {args.max_iterations}
  promotion:       net >= {args.min_improvement}, target_metrics >= {args.min_target_metrics}, cost reported only
  stop:            consecutive promotion failures >= {args.max_promotion_failures}
  workers:         {args.workers}
  cost_limit:      ${args.cost_limit}
  train_dataset:   {TRAIN_DATASET_DIR}
  train_ids:       {TRAIN_IDS_FILE}
  val_dataset:     {VAL_DATASET_DIR}
  val_ids:         {VAL_IDS_FILE}
  run_label:       {RUN_LABEL or '(none)'}
{'='*60}
""")

    baseline_traces = step_val_baseline_inference(
        model=args.model,
        workers=args.workers,
        cost_limit=args.cost_limit,
        force=args.force,
        dry_run=args.dry_run,
    )
    baseline_eval = step_val_baseline_evaluate(
        model=args.model,
        baseline_traces=baseline_traces,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
    )

    val_analyses_path: Path | None = None
    prev_plan_path: Path | None = None
    prev_iteration_report: Path | None = None
    current_base_version = args.start_version - 1
    promoted_versions: list[int] = []
    consecutive_promotion_failures = 0

    for i in range(args.max_iterations):
        version = args.start_version + i
        print(f"\n{'#'*60}")
        print(f"  ITERATION {i+1}/{args.max_iterations}  (candidate v{version}, base v{current_base_version})")
        print(f"{'#'*60}")

        train_traces = step_train_inference(
            model=args.model,
            base_version=current_base_version,
            workers=args.workers,
            cost_limit=args.cost_limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        train_eval = step_train_evaluate(
            model=args.model,
            base_version=current_base_version,
            train_traces=train_traces,
            workers=args.workers,
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

        # ── 1. Aggregate ──────────────────────────────────────────
        plan_path = step_aggregate(
            version=version,
            model=args.model,
            analysis_results_path=train_analysis,
            analysis_model=args.analysis_model,
            val_analyses_path=val_analyses_path,
            prev_plan_path=prev_plan_path,
            prev_iteration_report_path=prev_iteration_report,
            force=args.force,
            dry_run=args.dry_run,
        )

        # ── 2. Modify ─────────────────────────────────────────────
        target_dir = step_modify(
            version=version,
            base_version=current_base_version,
            plan_path=plan_path,
            modify_model=args.analysis_model,
            force=args.force,
            dry_run=args.dry_run,
        )

        audit = step_plan_diff_audit(
            version=version,
            base_version=current_base_version,
            plan_path=plan_path,
            target_dir=target_dir,
            dry_run=args.dry_run,
        )
        if not audit.get("passed", True):
            log(f"✗ Audit FAILED at v{version}: {audit.get('violations', [])}")
            promotion = {
                "passed": False,
                "promoted": False,
                "failure_reasons": ["audit_failed"],
                "regressed_ids": [],
                "improved_ids": [],
            }
            step_record_memory("swe", version, plan_path, audit, promotion, args.dry_run)
            prev_iteration_report = step_iteration_report(
                version=version,
                base_version=current_base_version,
                target_dir=target_dir,
                plan_path=plan_path,
                train_analysis=train_analysis,
                audit=audit,
                train_compare={},
                val_compare={"regressed_ids": [], "improved_ids": []},
                promotion=promotion,
                dry_run=args.dry_run,
            )
            prev_plan_path = plan_path
            val_analyses_path = None
            consecutive_promotion_failures += 1
            if consecutive_promotion_failures >= args.max_promotion_failures:
                log(f"Stopping: {consecutive_promotion_failures} consecutive candidates were not promoted.")
                break
            continue

        # ── 3. Enhanced train inference/evaluation ────────────────
        enhanced_train_traces = step_train_inference(
            model=args.model,
            base_version=version,
            workers=args.workers,
            cost_limit=args.cost_limit,
            force=args.force,
            dry_run=args.dry_run,
        )
        enhanced_train_eval = step_train_evaluate(
            model=args.model,
            base_version=version,
            train_traces=enhanced_train_traces,
            workers=args.workers,
            force=args.force,
            dry_run=args.dry_run,
        )
        train_compare = step_train_compare(
            version=version,
            baseline_eval_dir=train_eval,
            current_eval_dir=enhanced_train_eval,
            baseline_traces_dir=train_traces,
            current_traces_dir=enhanced_train_traces,
            plan_path=plan_path,
            max_cost_ratio=args.max_cost_ratio,
            dry_run=args.dry_run,
        )
        log(
            "  train compare: "
            f"{train_compare.get('baseline_count', '?')} → {train_compare.get('current_count', '?')} resolved, "
            f"{train_compare.get('regression_count', '?')} regressions, "
            f"{train_compare.get('improvement_count', '?')} improvements"
        )

        # ── 4. Val inference ──────────────────────────────────────
        val_traces_dir = step_val_inference(
            version=version,
            enhanced_src=target_dir / "src",
            model=args.model,
            workers=args.workers,
            cost_limit=args.cost_limit,
            dry_run=args.dry_run,
        )

        # ── 5. Val evaluate ───────────────────────────────────────
        val_eval_dir = step_val_evaluate(
            version=version,
            val_traces_dir=val_traces_dir,
            model=args.model,
            workers=args.workers,
            dry_run=args.dry_run,
        )

        # ── 6. Val comparison ─────────────────────────────────────
        base_val_traces = val_traces_dir_for_base(args.model, current_base_version)
        base_val_eval = val_eval_dir_for_base(args.model, current_base_version)
        val_compare = step_val_compare(
            version=version,
            baseline_eval_dir=base_val_eval,
            current_eval_dir=val_eval_dir,
            baseline_traces_dir=base_val_traces,
            current_traces_dir=val_traces_dir,
            plan_path=plan_path,
            max_cost_ratio=args.max_cost_ratio,
            dry_run=args.dry_run,
        )

        # ── 7. Promotion decision ─────────────────────────────────
        promotion = step_promotion_decision(
            version=version,
            audit=audit,
            train_compare=train_compare,
            val_compare=val_compare,
            min_improvement=args.min_improvement,
            min_target_metrics=args.min_target_metrics,
            max_error_rate_delta=args.max_error_rate_delta,
            max_invalid_rate_delta=args.max_invalid_rate_delta,
            dry_run=args.dry_run,
        )
        step_record_memory("swe", version, plan_path, audit, promotion, args.dry_run)

        iteration_report = step_iteration_report(
            version=version,
            base_version=current_base_version,
            target_dir=target_dir,
            plan_path=plan_path,
            train_analysis=train_analysis,
            audit=audit,
            train_compare=train_compare,
            val_compare=val_compare,
            promotion=promotion,
            dry_run=args.dry_run,
        )

        if promotion.get("promoted"):
            log(f"✓ PROMOTED v{version}: "
                f"val {val_compare.get('baseline_count')} → {val_compare.get('current_count')} resolved")
            current_base_version = version
            promoted_versions.append(version)
            consecutive_promotion_failures = 0
        else:
            log(f"✗ NOT PROMOTED v{version}: "
                f"{', '.join(promotion.get('failure_reasons', [])) or 'promotion checks failed'}")
            consecutive_promotion_failures += 1

        # ── 8. Analyze regressions (feed into next iteration) ─────
        regressed_ids = val_compare.get("regressed_ids", [])
        val_analyses_path = step_analyze_regressions(
            version=version,
            regressed_ids=regressed_ids,
            val_traces_dir=val_traces_dir,
            analysis_model=args.analysis_model,
            workers=args.workers,
            dry_run=args.dry_run,
        )
        prev_plan_path = plan_path
        prev_iteration_report = iteration_report

        if consecutive_promotion_failures >= args.max_promotion_failures:
            log(f"Stopping: {consecutive_promotion_failures} consecutive candidates were not promoted.")
            break

    best_version = best_version_by_val(args.model, promoted_versions)
    best_dir = base_agent_dir(best_version)
    best_eval = val_eval_dir_for_base(args.model, best_version)
    log(f"Finished iterations. Best-so-far version: v{best_version} ({best_dir.name})")
    log(f"Best validation resolved: {resolved_count_from_eval(best_eval, VAL_IDS_FILE)}/{len(VAL_IDS_FILE.read_text().split())}")
    if best_version > 0:
        print_test_eval_command(best_dir, args.model)
    else:
        log("No enhanced candidate beat the original baseline under promotion rules.")


if __name__ == "__main__":
    main()

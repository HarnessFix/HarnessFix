#!/usr/bin/env python3
"""Batch failure analysis runner for mini-swe-agent and GAIA traces.

Usage examples:
  # SWE-bench (default mode)
  python3 failure_analysis/run_analysis.py --dry-run --limit 5
  python3 failure_analysis/run_analysis.py -i astropy__astropy-12907 -m anthropic/claude-sonnet-4-5
  python3 failure_analysis/run_analysis.py --category empty_patch --resume

  # GAIA mode
  python3 failure_analysis/run_analysis.py --mode gaia \\
      --gaia-subset data/gaia_train_60 \\
      --traces-dir traces/gaia_train_round1 \\
      --eval-results eval/gaia_results_round1/results.json \\
      --output-file failure_analysis/results/gaia_train_round1_analysis.jsonl \\
      --model openai/gpt-5-mini \\
      --workers 2 --resume
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ── project-root relative imports ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent_framework" / "src"))

from failure_analysis.htir import (
    compile_appworld_htir,
    compile_gaia_htir,
    compile_swe_htir,
    compile_terminal_bench_htir,
    write_bundle,
)
from failure_analysis.artifact_sanitizer import sanitized_trace_output_path, write_sanitized_artifact
from failure_analysis.operator_registry import (
    infer_defect_class,
    infer_severity,
    normalize_defect_class,
    normalize_operator_family,
    recommend_operator_family,
)
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
EVAL_RESULTS_PATH = (
    REPO_ROOT
    / "eval"
    / "results_swe_train_qwen3_coder_original"
    / "openai__Qwen3-Coder.swe_train_qwen3_coder_original.json"
)
TRACES_DIR = REPO_ROOT / "traces" / "swe_train_qwen3_coder_original"
LOGS_DIR = REPO_ROOT / "logs" / "run_evaluation" / "swe_train_qwen3_coder_original" / "openai__Qwen3-Coder"
ANALYSIS_DIR = Path(__file__).parent
RESULTS_DIR = ANALYSIS_DIR / "results"
ALL_RESULTS_PATH = RESULTS_DIR / "all_results.jsonl"
CONFIG_PATH_SWE = ANALYSIS_DIR / "analysis_config_swe.yaml"
CONFIG_PATH_GAIA = ANALYSIS_DIR / "analysis_config_gaia.yaml"
CONFIG_PATH_APPWORLD = ANALYSIS_DIR / "analysis_config_appworld.yaml"
CONFIG_PATH_TERMINAL_BENCH = ANALYSIS_DIR / "analysis_config_terminal_bench.yaml"
IMPL_DOC_PATH = ANALYSIS_DIR / "task_agent_impl_doc.md"

DEFAULT_MODEL = "openai/gpt-5-mini"

# Module-level path overrides (set by main() when CLI args are provided)
_TRACES_DIR_OVERRIDE: Path | None = None
_ALL_RESULTS_PATH_OVERRIDE: Path | None = None
_EVAL_RESULTS_PATH_OVERRIDE: Path | None = None
_LOGS_DIR_OVERRIDE: Path | None = None
_GAIA_SUBSET_DIR: Path | None = None
_APPWORLD_SUBSET_DIR: Path | None = None
_TERMINAL_BENCH_SUBSET_DIR: Path | None = None
_AGENT_SOURCE_DIR_OVERRIDE: Path | None = None


def _traces_dir() -> Path:
    return _TRACES_DIR_OVERRIDE if _TRACES_DIR_OVERRIDE else TRACES_DIR


def _all_results_path() -> Path:
    return _ALL_RESULTS_PATH_OVERRIDE if _ALL_RESULTS_PATH_OVERRIDE else ALL_RESULTS_PATH


def _eval_results_path() -> Path:
    return _EVAL_RESULTS_PATH_OVERRIDE if _EVAL_RESULTS_PATH_OVERRIDE else EVAL_RESULTS_PATH


def _logs_dir() -> Path:
    return _LOGS_DIR_OVERRIDE if _LOGS_DIR_OVERRIDE else LOGS_DIR


def _gaia_subset_dir() -> Path:
    return _GAIA_SUBSET_DIR or (REPO_ROOT / "data" / "gaia_train_100")


def _appworld_subset_dir() -> Path:
    return _APPWORLD_SUBSET_DIR or (REPO_ROOT / "data" / "appworld_train_100")


def _terminal_bench_subset_dir() -> Path:
    return _TERMINAL_BENCH_SUBSET_DIR or (REPO_ROOT / "data" / "terminal_bench_train")


def _default_agent_source_dir(mode: str) -> Path:
    if mode == "gaia":
        return REPO_ROOT / "task_agent" / "open_deep_research" / "src" / "open_deep_research"
    if mode == "appworld":
        return REPO_ROOT / "task_agent" / "appworld_agent" / "src" / "appworld_agent"
    if mode == "terminal_bench":
        return REPO_ROOT / "task_agent" / "terminal_bench_agent"
    return REPO_ROOT / "task_agent" / "mini-swe-agent" / "src" / "minisweagent"


def _agent_source_dir(mode: str) -> Path:
    return _AGENT_SOURCE_DIR_OVERRIDE or _default_agent_source_dir(mode)


def _agent_source_context(mode: str) -> dict[str, str]:
    source_dir = _agent_source_dir(mode).resolve()
    return {
        "agent_source_dir": str(source_dir),
        "agent_source_root": str(source_dir.parent),
    }


def _analysis_run_metadata(mode: str) -> dict[str, str]:
    source_context = _agent_source_context(mode)
    return {
        "analysis_mode": mode,
        "agent_source_dir": source_context["agent_source_dir"],
        "agent_source_root": source_context["agent_source_root"],
    }


def _attach_analysis_run_metadata(record: dict, mode: str) -> dict:
    metadata = _analysis_run_metadata(mode)
    for key, value in metadata.items():
        record.setdefault(key, value)
    analysis_config = record.get("analysis_config")
    if not isinstance(analysis_config, dict):
        analysis_config = {}
    for key, value in metadata.items():
        analysis_config.setdefault(key, value)
    record["analysis_config"] = analysis_config
    return record

def _agent_config_from_analysis_config(config: dict) -> dict:
    agent_config = dict(config.get("agent", {}))
    for key in ("step_limit", "cost_limit"):
        if key in config and key not in agent_config:
            agent_config[key] = config[key]
    return agent_config


DEFAULT_OBSERVATION_OUTPUT_MAX_CHARS = 16_000


def _configured_observation_max_chars(env_config: dict) -> int:
    configured = env_config.get("observation_max_chars")
    if configured is None:
        configured = os.environ.get("HARNESSFIX_ANALYSIS_OBSERVATION_MAX_CHARS")
    try:
        value = int(configured) if configured is not None else DEFAULT_OBSERVATION_OUTPUT_MAX_CHARS
    except (TypeError, ValueError):
        value = DEFAULT_OBSERVATION_OUTPUT_MAX_CHARS
    return max(value, 0)


def _truncate_observation_output(output: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(output) <= max_chars:
        return output, False
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    omitted = len(output) - max_chars
    marker = f"\n[... HarnessFix truncated {omitted} observation chars; showing head/tail only ...]\n"
    return output[:head_chars] + marker + output[-tail_chars:], True


class TruncatingLocalEnvironment(LocalEnvironment):
    def __init__(self, *args, output_max_chars: int = DEFAULT_OBSERVATION_OUTPUT_MAX_CHARS, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_max_chars = output_max_chars

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        output = super().execute(action, cwd=cwd, timeout=timeout)
        text = output.get("output")
        if isinstance(text, str):
            truncated, did_truncate = _truncate_observation_output(text, self.output_max_chars)
            if did_truncate:
                output["output"] = truncated
                extra = output.setdefault("extra", {})
                extra["harnessfix_observation_truncated"] = True
                extra["harnessfix_original_output_chars"] = len(text)
                extra["harnessfix_observation_max_chars"] = self.output_max_chars
        return output


def _make_analysis_environment(env_config: dict) -> LocalEnvironment:
    kwargs = {"env": env_config.get("env", {})}
    for key in ("cwd", "timeout"):
        if key in env_config:
            kwargs[key] = env_config[key]
    return TruncatingLocalEnvironment(
        **kwargs,
        output_max_chars=_configured_observation_max_chars(env_config),
    )


def _htir_output_path(instance_id: str) -> Path:
    run_label = _all_results_path().stem.replace("_analysis", "")
    return RESULTS_DIR / "htir" / run_label / f"{instance_id}.htir.json"


def _sanitized_traj_path(instance_id: str) -> Path:
    run_label = _all_results_path().stem.replace("_analysis", "")
    return sanitized_trace_output_path(RESULTS_DIR, run_label, instance_id)


def _default_evidence_spans(bundle: dict) -> list[dict]:
    spans = []
    facets = bundle.get("node_facets", {}) or {}
    for step_id, facet in list(facets.items())[-8:]:
        refs = facet.get("evidence_refs", []) or []
        layers = facet.get("harness_layer", {}).get("implicated_layers", [])
        artifact = facet.get("artifact_state", {}).get("external_effect", {})
        for ref in refs[:2]:
            spans.append(
                {
                    "step_id": step_id,
                    "node_id": ref.get("node_id"),
                    "event_id": ref.get("node_id"),
                    "source_ref": ref.get("source_ref"),
                    "harness_layers": layers,
                    "reason": ref.get("summary"),
                    "artifact_state": artifact,
                }
            )
        if len(spans) >= 8:
            break
    if spans:
        return spans

    causal_nodes = bundle.get("views", {}).get("causal", [])
    if not causal_nodes:
        causal_nodes = bundle.get("views", {}).get("failure", [])
    for node in causal_nodes:
        node_id = node.get("node_id") or node.get("event_id")
        spans.append(
            {
                "node_id": node_id,
                "event_id": node_id,
                "source_ref": node.get("source_ref"),
                "reason": node.get("summary"),
            }
        )
    return spans


def _default_responsible_steps(bundle: dict) -> list[str]:
    steps = bundle.get("agent_trace_steps", []) or []
    if not steps:
        return []
    symptom_steps = [
        step.get("step_id")
        for step in steps
        if step.get("execution_status") in {"exception", "submitted", "verified"}
    ]
    if symptom_steps:
        return [step for step in symptom_steps[-4:] if step]
    return [step.get("step_id") for step in steps[-4:] if step.get("step_id")]


def _layers_for_steps(bundle: dict, step_ids: list[str]) -> list[str]:
    facets = bundle.get("node_facets", {}) or {}
    layers: set[str] = set()
    for step_id in step_ids:
        layers.update(facets.get(step_id, {}).get("harness_layer", {}).get("implicated_layers", []) or [])
    return sorted(layers)


def _legacy_nodes_for_steps(bundle: dict, step_ids: list[str]) -> list[str]:
    step_set = set(step_ids)
    nodes = []
    for step in bundle.get("agent_trace_steps", []) or []:
        if step.get("step_id") in step_set:
            nodes.extend(step.get("legacy_event_ids", []) or [])
    return nodes


def _enrich_analysis_output(parsed: dict, evidence_anchor: dict, bundle: dict, htir_path: Path) -> dict:
    default_responsible_steps = _default_responsible_steps(bundle)
    parsed_steps = parsed.get("responsible_steps") or parsed.get("responsible_step_ids") or []
    responsible_steps = parsed_steps or default_responsible_steps
    candidate_steps = parsed.get("candidate_responsible_steps") or responsible_steps

    default_responsible_nodes = _legacy_nodes_for_steps(bundle, responsible_steps) or [
        node.get("node_id") or node.get("event_id")
        for node in bundle.get("views", {}).get("causal", [])
    ]
    default_downstream_nodes = [
        node.get("node_id") or node.get("event_id")
        for node in bundle.get("views", {}).get("failure", [])
    ]
    responsible_nodes = parsed.get("responsible_nodes") or parsed.get("responsible_events") or default_responsible_nodes
    downstream_nodes = parsed.get("downstream_nodes") or parsed.get("downstream_events") or default_downstream_nodes
    implicated_layers = (
        parsed.get("implicated_harness_layers")
        or parsed.get("harness_layers")
        or _layers_for_steps(bundle, responsible_steps)
    )
    parsed.setdefault("evidence_anchor", evidence_anchor)
    parsed.setdefault("htir_path", str(htir_path))
    parsed.setdefault("htir_summary", f"(see {htir_path})")
    parsed.setdefault("candidate_responsible_steps", candidate_steps)
    parsed.setdefault("responsible_steps", responsible_steps)
    parsed.setdefault("responsible_nodes", responsible_nodes)
    parsed.setdefault("downstream_nodes", downstream_nodes)
    parsed.setdefault("implicated_harness_layers", implicated_layers)
    # Compatibility aliases for historical JSONL consumers.
    parsed.setdefault("responsible_events", responsible_nodes)
    parsed.setdefault("downstream_events", downstream_nodes)
    parsed.setdefault("evidence_spans", _default_evidence_spans(bundle))
    parsed["defect_class"] = normalize_defect_class(parsed.get("defect_class")) or infer_defect_class(parsed)
    parsed.setdefault("defect_labels", [parsed["defect_class"]])
    parsed["recommended_operator_family"] = (
        normalize_operator_family(parsed.get("recommended_operator_family")) or recommend_operator_family(parsed)
    )
    parsed.setdefault("severity", infer_severity(parsed))
    parsed.setdefault("confidence", "medium")
    return parsed


def _fallback_analysis_output(
    instance_id: str,
    failure_category: str,
    exit_status: str,
    api_calls: int,
    evidence_anchor: dict,
    htir_bundle: dict,
    htir_path: Path,
    reason: str,
) -> dict:
    responsible_steps = _default_responsible_steps(htir_bundle)
    layers = _layers_for_steps(htir_bundle, responsible_steps) or ["Observability"]
    parsed = {
        "instance_id": instance_id,
        "failure_category": failure_category,
        "exit_status": exit_status or "analysis_failed",
        "api_calls": api_calls,
        "failure_manifestation": "Analysis agent could not complete a structured diagnosis for this failed SWE-bench instance.",
        "failure_reason": (
            "The trace/evaluation artifacts were available, but the analysis agent did not return valid JSON "
            f"before failing: {reason}"
        ),
        "agent_design_issue": (
            "HarnessFix should keep per-instance failure analysis bounded and robust so one unusually long trace "
            "cannot block aggregate repair planning. This fallback preserves HTIR anchors and compact evidence for "
            "aggregation, but marks the diagnosis as low confidence."
        ),
        "affected_component": "error_handling",
        "defect_class": "context",
        "recommended_operator_family": "instrumentation",
        "candidate_responsible_steps": responsible_steps,
        "responsible_steps": responsible_steps,
        "implicated_harness_layers": layers,
        "defect_labels": ["context", "analysis_fallback", failure_category],
        "severity": "medium",
        "confidence": "low",
        "_analysis_fallback": True,
        "_analysis_failure_reason": reason,
    }
    return _enrich_analysis_output(parsed, evidence_anchor, htir_bundle, htir_path)


# ── Shared utilities ────────────────────────────────────────────────────────────

def load_failed_instances() -> dict[str, str]:
    """Load failed instance IDs from eval results JSON.

    Returns: {instance_id: failure_category}
    """
    logger = logging.getLogger("run_analysis")
    p = _eval_results_path()
    if not p.exists():
        logger.warning(f"Eval results file not found: {p}. Returning empty failed list.")
        return {}
    data = json.loads(p.read_text())
    failed: dict[str, str] = {}
    for iid in data.get("empty_patch_ids", []):
        failed[iid] = "empty_patch"
    for iid in data.get("unresolved_ids", []):
        failed[iid] = "unresolved"
    for iid in data.get("error_ids", []):
        failed[iid] = "error"
    return failed


def load_completed_ids() -> set[str]:
    """Load instance IDs already written to the output JSONL."""
    p = _all_results_path()
    if not p.exists():
        return set()
    completed = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            iid = record.get("instance_id")
            if iid:
                completed.add(iid)
        except json.JSONDecodeError:
            pass
    return completed


def parse_submission(submission: str) -> dict | None:
    """Extract JSON from agent submission string."""
    if not submission:
        return None
    try:
        return json.loads(submission.strip())
    except json.JSONDecodeError:
        pass
    for match in reversed(re.findall(r"\{[\s\S]*?\}", submission)):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    return None


def _clean_excerpt(text: str, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


_write_lock = threading.Lock()


def append_result(record: dict) -> None:
    """Append a result record to the output JSONL (crash-safe, thread-safe)."""
    p = _all_results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with p.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_config(mode: str = "swe") -> dict:
    """Load and parse the appropriate analysis config YAML."""
    if mode == "gaia":
        config_path = CONFIG_PATH_GAIA
    elif mode == "appworld":
        config_path = CONFIG_PATH_APPWORLD
    elif mode == "terminal_bench":
        config_path = CONFIG_PATH_TERMINAL_BENCH
    else:
        config_path = CONFIG_PATH_SWE
    return yaml.safe_load(config_path.read_text())


def load_impl_doc(logger: logging.Logger) -> str:
    """Load task_agent_impl_doc.md. Warn if missing."""
    if IMPL_DOC_PATH.exists():
        doc = IMPL_DOC_PATH.read_text()
        logger.info(f"Loaded impl doc: {IMPL_DOC_PATH} ({len(doc)} chars)")
        return doc
    logger.warning(
        f"Implementation doc not found at {IMPL_DOC_PATH}. "
        "Run explore_task_agent.py first for better analysis quality. "
        "Proceeding with empty impl_doc."
    )
    return "(Implementation doc not available. Run explore_task_agent.py to generate it.)"


# ── SWE-bench mode ──────────────────────────────────────────────────────────────

def get_paths(instance_id: str) -> dict[str, str]:
    """Return file paths needed for SWE-bench analysis of an instance."""
    traj_path = _traces_dir() / instance_id / f"{instance_id}.traj.json"
    report_path = _logs_dir() / instance_id / "report.json"
    test_output_path = _logs_dir() / instance_id / "test_output.txt"
    output_path = RESULTS_DIR / f"{instance_id}.traj.json"
    return {
        "traj_path": str(traj_path),
        "report_path": str(report_path),
        "test_output_path": str(test_output_path),
        "output_path": output_path,
    }


def extract_task_description(traj_path: str) -> str:
    """Extract the original SWE-bench task description from the traj.json first user message."""
    path = Path(traj_path)
    if not path.exists():
        return "(task description unavailable: traj file not found)"
    data = json.loads(path.read_text())
    messages = data.get("messages", [])
    if len(messages) < 2:
        return "(task description unavailable: not enough messages)"
    content = messages[1].get("content", "")
    if not isinstance(content, str):
        return "(task description unavailable: unexpected content type)"
    m = re.search(r"<pr_description>(.*?)</pr_description>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return content.strip()


def _extract_last_assistant_signal(traj_path: str) -> str:
    path = Path(traj_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    messages = data.get("messages", [])
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_messages:
        return ""
    content = assistant_messages[-1].get("content", "")
    if not isinstance(content, str):
        return ""
    cmd_match = re.findall(r"<mswea_bash_command>(.*?)</mswea_bash_command>", content, re.DOTALL)
    if cmd_match:
        return _clean_excerpt(cmd_match[-1])
    return _clean_excerpt(content)


def _extract_last_observation_signal(traj_path: str) -> str:
    path = Path(traj_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    messages = data.get("messages", [])
    user_messages = [m for m in messages[2:] if m.get("role") == "user"]
    if not user_messages:
        return ""
    content = user_messages[-1].get("content", "")
    if not isinstance(content, str):
        return ""
    output_match = re.search(r"<output>(.*?)</output>", content, re.DOTALL)
    if output_match:
        return _clean_excerpt(output_match.group(1))
    return _clean_excerpt(content)


def _extract_report_signal(report_path: str) -> str:
    path = Path(report_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return _clean_excerpt(path.read_text())
    interesting_keys = [
        "resolved", "tests_status", "fail_to_pass", "pass_to_pass",
        "FAIL_TO_PASS", "PASS_TO_PASS", "error",
    ]
    parts = []
    for key in interesting_keys:
        if key in data and data[key]:
            parts.append(f"{key}={data[key]}")
    if not parts:
        parts.append(json.dumps(data, ensure_ascii=False))
    return _clean_excerpt(" | ".join(parts))


def _extract_test_output_signal(test_output_path: str) -> str:
    path = Path(test_output_path)
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    patterns = [
        r"(?m)^FAILED\s+.*$",
        r"(?m)^E\s+.*$",
        r"(?m)^AssertionError.*$",
        r"(?m)^Traceback \(most recent call last\):.*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _clean_excerpt(match.group(0))
    return _clean_excerpt(text)


def build_evidence_anchor(paths: dict[str, str]) -> dict[str, str]:
    """Build compact evidence anchors from SWE-bench artifacts."""
    anchor = {
        "last_action": _extract_last_assistant_signal(paths["traj_path"]),
        "terminal_observation": _extract_last_observation_signal(paths["traj_path"]),
        "report_signal": _extract_report_signal(paths["report_path"]),
        "test_signal": _extract_test_output_signal(paths["test_output_path"]),
    }
    return {k: v for k, v in anchor.items() if v}


def run_analysis(
    instance_id: str,
    failure_category: str,
    model_name: str,
    config: dict,
    impl_doc: str,
    logger: logging.Logger,
) -> dict | None:
    """Run analysis agent on a single SWE-bench instance."""
    paths = get_paths(instance_id)
    evidence_anchor = build_evidence_anchor(paths)
    task_description = extract_task_description(paths["traj_path"])
    sanitized_traj_path = write_sanitized_artifact(paths["traj_path"], _sanitized_traj_path(instance_id))
    htir_bundle = compile_swe_htir(
        instance_id=instance_id,
        failure_category=failure_category,
        paths=paths,
        task_description=task_description,
    )
    htir_path = write_bundle(htir_bundle, _htir_output_path(instance_id))

    agent_config = _agent_config_from_analysis_config(config)
    agent_config["output_path"] = paths["output_path"]
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    model = LitellmTextbasedModel(
        model_name=model_name,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}),
        cost_tracking="ignore_errors",
    )
    env = _make_analysis_environment(env_config)
    agent = DefaultAgent(model, env, **agent_config)

    logger.info(f"Running analysis for {instance_id} ({failure_category})")

    try:
        result = agent.run(
            instance_id=instance_id,
            failure_category=failure_category,
            traj_path=str(sanitized_traj_path),
            raw_traj_path=paths["traj_path"],
            report_path=paths["report_path"],
            test_output_path=paths["test_output_path"],
            impl_doc=impl_doc,
            task_description=task_description,
            **_agent_source_context("swe"),
            htir_path=str(htir_path),
        )
    except Exception as e:
        logger.error(f"Agent raised exception for {instance_id}: {e}")
        return _fallback_analysis_output(
            instance_id=instance_id,
            failure_category=failure_category,
            exit_status="analysis_agent_exception",
            api_calls=getattr(agent, "n_calls", 0),
            evidence_anchor=evidence_anchor,
            htir_bundle=htir_bundle,
            htir_path=htir_path,
            reason=str(e),
        )

    submission = result.get("submission", "")
    logger.info(f"Submission for {instance_id}: {submission!r}")

    parsed = parse_submission(submission)
    if parsed is None:
        logger.warning(f"Could not parse JSON from submission for {instance_id}")
        fallback = _fallback_analysis_output(
            instance_id=instance_id,
            failure_category=failure_category,
            exit_status=result.get("exit_status", "analysis_parse_error"),
            api_calls=agent.n_calls,
            evidence_anchor=evidence_anchor,
            htir_bundle=htir_bundle,
            htir_path=htir_path,
            reason=f"invalid JSON submission: {submission[:1000]}",
        )
        fallback["_parse_error"] = True
        return fallback

    parsed.setdefault("instance_id", instance_id)
    parsed.setdefault("failure_category", failure_category)
    return _enrich_analysis_output(parsed, evidence_anchor, htir_bundle, htir_path)


# ── GAIA mode ───────────────────────────────────────────────────────────────────

_gaia_data_cache: dict[str, dict] | None = None
_gaia_data_lock = threading.Lock()


def _load_gaia_data() -> dict[str, dict]:
    """Load data.jsonl from the GAIA subset, keyed by task_id."""
    global _gaia_data_cache
    with _gaia_data_lock:
        if _gaia_data_cache is not None:
            return _gaia_data_cache
        data_file = _gaia_subset_dir() / "data.jsonl"
        result: dict[str, dict] = {}
        if data_file.exists():
            for line in data_file.read_text().splitlines():
                line = line.strip()
                if line:
                    task = json.loads(line)
                    result[task["task_id"]] = task
        _gaia_data_cache = result
        return result


def _gaia_get_paths(task_id: str) -> dict:
    """Return artifact paths for a GAIA task (traj only — no Docker harness logs)."""
    traj_path = _traces_dir() / task_id / f"{task_id}.traj.json"
    output_path = RESULTS_DIR / f"{task_id}.traj.json"
    return {
        "traj_path": str(traj_path),
        "report_path": "",
        "test_output_path": "",
        "output_path": output_path,
    }


def _gaia_extract_task_description(traj_path: str) -> str:
    """Extract task description from data.jsonl (not from traj)."""
    task_id = Path(traj_path).parent.name
    task = _load_gaia_data().get(task_id)
    if task:
        question = task.get("question", "")
        true_answer = task.get("true_answer", "")
        level = task.get("level", "?")
        return (
            f"Question (Level {level}): {question}\n\n"
            f"Expected answer: {true_answer}"
        )
    return f"(task description unavailable for task_id={task_id!r})"


def _gaia_extract_traj_last_steps(traj_path: str, n_steps: int | None = None) -> str:
    path = Path(traj_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    trajectory = data.get("trajectory", [])
    if not trajectory:
        return ""
    last_steps = trajectory if n_steps is None else trajectory[-n_steps:]
    parts = []
    for i, step in enumerate(last_steps):
        content = step.get("content", str(step)) if isinstance(step, dict) else str(step)
        step_label = i + 1 if n_steps is None else f"-{n_steps - i}"
        parts.append(f"[step {step_label}] {_clean_excerpt(content)}")
    return "\n".join(parts)


def _gaia_extract_traj_prediction(traj_path: str) -> str:
    path = Path(traj_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return ""
    return data.get("info", {}).get("prediction", "")


def _gaia_extract_traj_exit_status(traj_path: str) -> str:
    path = Path(traj_path)
    if not path.exists():
        return "unknown"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "unknown"
    return data.get("info", {}).get("exit_status", "unknown")


def _gaia_build_evidence_anchor(paths: dict) -> dict:
    """Build compact evidence anchors from GAIA traj artifacts."""
    traj_path = paths["traj_path"]
    task_id = Path(traj_path).parent.name
    task = _load_gaia_data().get(task_id, {})

    predicted = _gaia_extract_traj_prediction(traj_path)
    anchor = {
        "predicted_answer": _clean_excerpt(predicted) if predicted else "(empty)",
        "true_answer": _clean_excerpt(task.get("true_answer", "")),
        "exit_status": _gaia_extract_traj_exit_status(traj_path),
    }
    last_steps = _gaia_extract_traj_last_steps(traj_path, n_steps=None)
    if last_steps:
        anchor["last_trajectory_steps"] = last_steps
    return {k: v for k, v in anchor.items() if v}


def _gaia_run_analysis(
    task_id: str,
    failure_category: str,
    model_name: str,
    config: dict,
    impl_doc: str,
    logger: logging.Logger,
) -> dict | None:
    """Run the analysis agent on a single GAIA task."""
    paths = _gaia_get_paths(task_id)
    evidence_anchor = _gaia_build_evidence_anchor(paths)
    task_description = _gaia_extract_task_description(paths["traj_path"])
    gaia_data = _load_gaia_data()
    task_data = gaia_data.get(task_id, {})
    predicted_answer = _gaia_extract_traj_prediction(paths["traj_path"])
    sanitized_traj_path = write_sanitized_artifact(paths["traj_path"], _sanitized_traj_path(task_id))
    htir_bundle = compile_gaia_htir(
        task_id=task_id,
        failure_category=failure_category,
        traj_path=paths["traj_path"],
        question=task_data.get("question", ""),
        true_answer=task_data.get("true_answer", ""),
        predicted_answer=predicted_answer,
    )
    htir_path = write_bundle(htir_bundle, _htir_output_path(task_id))

    agent_config = _agent_config_from_analysis_config(config)
    agent_config["output_path"] = paths["output_path"]
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    model = LitellmTextbasedModel(
        model_name=model_name,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}),
        cost_tracking="ignore_errors",
    )
    env = _make_analysis_environment(env_config)
    agent = DefaultAgent(model, env, **agent_config)

    logger.info(f"Running GAIA analysis for {task_id} ({failure_category})")
    try:
        result = agent.run(
            instance_id=task_id,
            failure_category=failure_category,
            traj_path=str(sanitized_traj_path),
            raw_traj_path=paths["traj_path"],
            report_path=paths["report_path"],
            test_output_path=paths["test_output_path"],
            impl_doc=impl_doc,
            task_description=task_description,
            **_agent_source_context("gaia"),
            question=task_data.get("question", ""),
            true_answer=task_data.get("true_answer", ""),
            predicted_answer=predicted_answer,
            task_level=str(task_data.get("level", "?")),
            htir_path=str(htir_path),
        )
    except Exception as e:
        logger.error(f"Agent raised exception for {task_id}: {e}")
        return None

    submission = result.get("submission", "")
    logger.info(f"Submission for {task_id}: {submission!r}")

    parsed = parse_submission(submission)
    if parsed is None:
        logger.warning(f"Could not parse JSON from submission for {task_id}")
        return {
            "instance_id": task_id,
            "failure_category": failure_category,
            "exit_status": result.get("exit_status", "unknown"),
            "api_calls": agent.n_calls,
            "failure_manifestation": "Analysis agent failed to produce valid JSON",
            "failure_reason": f"Submission was: {submission}",
            "agent_design_issue": "unknown",
            "affected_component": "unknown",
            "evidence_anchor": evidence_anchor,
            "htir_path": str(htir_path),
            "_parse_error": True,
        }

    parsed.setdefault("instance_id", task_id)
    parsed.setdefault("failure_category", failure_category)
    return _enrich_analysis_output(parsed, evidence_anchor, htir_bundle, htir_path)


# ── AppWorld mode ──────────────────────────────────────────────────────────────

_appworld_data_cache: dict[str, dict] | None = None
_appworld_data_lock = threading.Lock()


def _load_appworld_data() -> dict[str, dict]:
    global _appworld_data_cache
    with _appworld_data_lock:
        if _appworld_data_cache is not None:
            return _appworld_data_cache
        data_file = _appworld_subset_dir() / "data.jsonl"
        result: dict[str, dict] = {}
        if data_file.exists():
            for line in data_file.read_text().splitlines():
                line = line.strip()
                if line:
                    task = json.loads(line)
                    result[task["task_id"]] = task
        _appworld_data_cache = result
        return result


def _appworld_get_paths(task_id: str) -> dict[str, str]:
    task_dir = _traces_dir() / task_id
    return {
        "traj_path": str(task_dir / f"{task_id}.traj.json"),
        "report_path": "",
        "test_output_path": "",
        "result_path": str(task_dir / "result.json"),
        "eval_report_path": str(task_dir / "eval_report.md"),
        "output_path": RESULTS_DIR / f"{task_id}.traj.json",
    }


def _appworld_extract_problem_statement(task_id: str) -> str:
    task = _load_appworld_data().get(task_id, {})
    supervisor = task.get("supervisor", {})
    if task and supervisor:
        return (
            f"My name is: {supervisor.get('first_name', '')} {supervisor.get('last_name', '')}. "
            f"My personal email is {supervisor.get('email', '')} and phone number is {supervisor.get('phone_number', '')}.\n\n"
            f"Task: {task.get('instruction', '')}"
        )
    return f"(task description unavailable for task_id={task_id!r})"


def _appworld_extract_traj_info(traj_path: str) -> dict:
    path = Path(traj_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("info", {})
    except json.JSONDecodeError:
        return {}


def _appworld_build_evidence_anchor(paths: dict) -> dict:
    info = _appworld_extract_traj_info(paths["traj_path"])
    result = {}
    result_path = Path(paths["result_path"])
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            result = {}
    failures = result.get("failures", [])
    failure_summary = "; ".join(str(item.get("requirement", item)) for item in failures)
    anchor = {
        "prediction": _clean_excerpt(str(info.get("prediction", ""))) if info.get("prediction") else "(empty)",
        "exit_status": str(info.get("exit_status", "unknown")),
        "required_apps": ", ".join(info.get("required_apps", [])) if info.get("required_apps") else "",
        "evaluation_failure": _clean_excerpt(failure_summary) if failure_summary else "",
        "pass_percentage": str(info.get("pass_percentage", "")),
    }
    return {k: v for k, v in anchor.items() if v}


def _appworld_run_analysis(
    task_id: str,
    failure_category: str,
    model_name: str,
    config: dict,
    impl_doc: str,
    logger: logging.Logger,
) -> dict | None:
    paths = _appworld_get_paths(task_id)
    evidence_anchor = _appworld_build_evidence_anchor(paths)
    problem_statement = _appworld_extract_problem_statement(task_id)
    task_data = _load_appworld_data().get(task_id, {})
    sanitized_traj_path = write_sanitized_artifact(paths["traj_path"], _sanitized_traj_path(task_id))
    htir_bundle = compile_appworld_htir(
        task_id=task_id,
        failure_category=failure_category,
        paths=paths,
        problem_statement=problem_statement,
    )
    htir_path = write_bundle(htir_bundle, _htir_output_path(task_id))

    agent_config = _agent_config_from_analysis_config(config)
    agent_config["output_path"] = paths["output_path"]
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    model = LitellmTextbasedModel(
        model_name=model_name,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}),
        cost_tracking="ignore_errors",
    )
    env = _make_analysis_environment(env_config)
    agent = DefaultAgent(model, env, **agent_config)

    logger.info(f"Running AppWorld analysis for {task_id} ({failure_category})")
    try:
        result = agent.run(
            instance_id=task_id,
            failure_category=failure_category,
            traj_path=str(sanitized_traj_path),
            raw_traj_path=paths["traj_path"],
            result_path=paths["result_path"],
            eval_report_path=paths["eval_report_path"],
            impl_doc=impl_doc,
            task_description=problem_statement,
            **_agent_source_context("appworld"),
            instruction=task_data.get("instruction", ""),
            required_apps=", ".join(task_data.get("required_apps", [])),
            difficulty=str(task_data.get("difficulty", "")),
            htir_path=str(htir_path),
        )
    except Exception as e:
        logger.error(f"Agent raised exception for {task_id}: {e}")
        return None

    submission = result.get("submission", "")
    parsed = parse_submission(submission)
    if parsed is None:
        logger.warning(f"Could not parse JSON from submission for {task_id}")
        return {
            "instance_id": task_id,
            "failure_category": failure_category,
            "exit_status": result.get("exit_status", "unknown"),
            "api_calls": agent.n_calls,
            "failure_manifestation": "Analysis agent failed to produce valid JSON",
            "failure_reason": f"Submission was: {submission}",
            "agent_design_issue": "unknown",
            "affected_component": "unknown",
            "evidence_anchor": evidence_anchor,
            "htir_path": str(htir_path),
            "_parse_error": True,
        }

    parsed.setdefault("instance_id", task_id)
    parsed.setdefault("failure_category", failure_category)
    return _enrich_analysis_output(parsed, evidence_anchor, htir_bundle, htir_path)


# ── Terminal-Bench mode ────────────────────────────────────────────────────────

_terminal_bench_data_cache: dict[str, dict] | None = None
_terminal_bench_data_lock = threading.Lock()


def _load_terminal_bench_data() -> dict[str, dict]:
    global _terminal_bench_data_cache
    with _terminal_bench_data_lock:
        if _terminal_bench_data_cache is not None:
            return _terminal_bench_data_cache
        subset = _terminal_bench_subset_dir()
        result: dict[str, dict] = {}
        data_file = subset / "data.jsonl"
        if data_file.exists():
            for line in data_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)
                task_id = task.get("task_id") or task.get("instance_id") or task.get("name")
                if task_id:
                    result[str(task_id)] = task
        elif subset.exists():
            for task_dir in sorted(path for path in subset.iterdir() if path.is_dir()):
                instruction = ""
                instruction_path = task_dir / "instruction.md"
                if instruction_path.exists():
                    instruction = instruction_path.read_text(errors="replace")
                result[task_dir.name] = {"task_id": task_dir.name, "instruction": instruction}
        _terminal_bench_data_cache = result
        return result


def _terminal_bench_get_paths(task_id: str) -> dict[str, str]:
    task_dir = _traces_dir() / task_id
    return {
        "traj_path": str(task_dir / f"{task_id}.traj.json"),
        "atif_path": str(task_dir / f"{task_id}.atif.json"),
        "result_path": str(task_dir / "result.json"),
        "test_stdout_path": str(task_dir / "test-stdout.txt"),
        "test_stderr_path": str(task_dir / "test-stderr.txt"),
        "pane_path": str(task_dir / "terminus_2.pane"),
        "report_path": str(task_dir / "result.json"),
        "test_output_path": str(task_dir / "test-stdout.txt"),
        "output_path": RESULTS_DIR / f"{task_id}.traj.json",
    }


def _terminal_bench_extract_task_description(task_id: str) -> str:
    task = _load_terminal_bench_data().get(task_id, {})
    instruction = task.get("instruction") or task.get("question") or task.get("prompt") or ""
    if instruction:
        return instruction
    task_dir = _terminal_bench_subset_dir() / task_id
    instruction_path = task_dir / "instruction.md"
    if instruction_path.exists():
        return instruction_path.read_text(errors="replace")
    return f"(task description unavailable for task_id={task_id!r})"


def _terminal_bench_extract_info(traj_path: str) -> dict:
    path = Path(traj_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("info", {})
    except json.JSONDecodeError:
        return {}


def _terminal_bench_build_evidence_anchor(paths: dict[str, str]) -> dict:
    info = _terminal_bench_extract_info(paths["traj_path"])
    stdout = Path(paths["test_stdout_path"]).read_text(errors="replace") if Path(paths["test_stdout_path"]).exists() else ""
    stderr = Path(paths["test_stderr_path"]).read_text(errors="replace") if Path(paths["test_stderr_path"]).exists() else ""
    anchor = {
        "exit_status": str(info.get("exit_status", "unknown")),
        "reward": str(info.get("reward", "")),
        "exception": _clean_excerpt(json.dumps(info.get("exception"), ensure_ascii=False) if info.get("exception") else ""),
        "verifier_stdout_tail": _clean_excerpt(stdout[-3000:]) if stdout else "",
        "verifier_stderr_tail": _clean_excerpt(stderr[-3000:]) if stderr else "",
    }
    return {k: v for k, v in anchor.items() if v}


def _terminal_bench_run_analysis(
    task_id: str,
    failure_category: str,
    model_name: str,
    config: dict,
    impl_doc: str,
    logger: logging.Logger,
) -> dict | None:
    paths = _terminal_bench_get_paths(task_id)
    evidence_anchor = _terminal_bench_build_evidence_anchor(paths)
    task_description = _terminal_bench_extract_task_description(task_id)
    sanitized_traj_path = write_sanitized_artifact(paths["traj_path"], _sanitized_traj_path(task_id))
    htir_bundle = compile_terminal_bench_htir(
        task_id=task_id,
        failure_category=failure_category,
        paths=paths,
        task_description=task_description,
    )
    htir_path = write_bundle(htir_bundle, _htir_output_path(task_id))

    agent_config = _agent_config_from_analysis_config(config)
    agent_config["output_path"] = paths["output_path"]
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    model = LitellmTextbasedModel(
        model_name=model_name,
        observation_template=model_config.get("observation_template", ""),
        format_error_template=model_config.get("format_error_template", ""),
        action_regex=model_config.get("action_regex", ""),
        model_kwargs=model_config.get("model_kwargs", {}),
        cost_tracking="ignore_errors",
    )
    env = _make_analysis_environment(env_config)
    agent = DefaultAgent(model, env, **agent_config)

    logger.info(f"Running Terminal-Bench analysis for {task_id} ({failure_category})")
    try:
        result = agent.run(
            instance_id=task_id,
            failure_category=failure_category,
            traj_path=str(sanitized_traj_path),
            raw_traj_path=paths["traj_path"],
            result_path=paths["result_path"],
            atif_path=paths["atif_path"],
            test_stdout_path=paths["test_stdout_path"],
            test_stderr_path=paths["test_stderr_path"],
            pane_path=paths["pane_path"],
            impl_doc=impl_doc,
            task_description=task_description,
            **_agent_source_context("terminal_bench"),
            htir_path=str(htir_path),
        )
    except Exception as e:
        logger.error(f"Agent raised exception for {task_id}: {e}")
        return None

    submission = result.get("submission", "")
    parsed = parse_submission(submission)
    if parsed is None:
        logger.warning(f"Could not parse JSON from submission for {task_id}")
        return {
            "instance_id": task_id,
            "failure_category": failure_category,
            "exit_status": result.get("exit_status", "unknown"),
            "api_calls": agent.n_calls,
            "failure_manifestation": "Analysis agent failed to produce valid JSON",
            "failure_reason": f"Submission was: {submission}",
            "agent_design_issue": "unknown",
            "affected_component": "unknown",
            "evidence_anchor": evidence_anchor,
            "htir_path": str(htir_path),
            "_parse_error": True,
        }

    parsed.setdefault("instance_id", task_id)
    parsed.setdefault("failure_category", failure_category)
    return _enrich_analysis_output(parsed, evidence_anchor, htir_bundle, htir_path)



# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch failure analysis for SWE, GAIA, AppWorld, and Terminal-Bench traces"
    )
    parser.add_argument("--mode", choices=["swe", "gaia", "appworld", "terminal_bench"], default="swe",
                        help="Analysis mode: swe (default), gaia, appworld, or terminal_bench")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Analysis model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Maximum number of instances to analyze")
    parser.add_argument("--instance-id", "-i", dest="instance_id", default=None,
                        help="Only analyze this specific instance")
    parser.add_argument("--category", "-c",
                        choices=["empty_patch", "unresolved", "error"], default=None,
                        help="Only analyze instances of this failure category")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed instances (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-analyze already-completed instances")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel analysis workers (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running anything")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--traces-dir", type=Path, default=None,
                        help="Override traces directory")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="Override output JSONL path")
    parser.add_argument("--instance-ids-file", type=Path, default=None,
                        help="File with one instance_id per line to analyze")
    parser.add_argument("--eval-results", type=Path, default=None,
                        help="Override eval results JSON path")
    # SWE-bench only
    parser.add_argument("--logs-dir", type=Path, default=None,
                        help="[swe] Override harness logs directory")
    # GAIA only
    parser.add_argument("--gaia-subset", type=Path, default=None,
                        help="[gaia] GAIA subset dir containing data.jsonl")
    parser.add_argument("--appworld-subset", type=Path, default=None,
                        help="[appworld] AppWorld subset dir containing data.jsonl")
    parser.add_argument("--terminal-bench-subset", type=Path, default=None,
                        help="[terminal_bench] Terminal-Bench local dataset/subset dir")
    parser.add_argument("--agent-source-dir", type=Path, default=None,
                        help="Source directory for the task agent version that produced these traces")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_analysis")

    # Apply path overrides
    global _TRACES_DIR_OVERRIDE, _ALL_RESULTS_PATH_OVERRIDE, _EVAL_RESULTS_PATH_OVERRIDE
    global _LOGS_DIR_OVERRIDE, _GAIA_SUBSET_DIR, _APPWORLD_SUBSET_DIR, _TERMINAL_BENCH_SUBSET_DIR
    global _AGENT_SOURCE_DIR_OVERRIDE
    if args.traces_dir:
        _TRACES_DIR_OVERRIDE = args.traces_dir.resolve()
        logger.info(f"Traces dir override: {_TRACES_DIR_OVERRIDE}")
    if args.output_file:
        _ALL_RESULTS_PATH_OVERRIDE = args.output_file.resolve()
    elif args.traces_dir:
        # Derive output filename from traces-dir name so each run gets its own file
        run_id = args.traces_dir.resolve().name  # e.g. "gaia_train_gpt_4o_original"
        _ALL_RESULTS_PATH_OVERRIDE = RESULTS_DIR / f"{run_id}_analysis.jsonl"
    else:
        # Last-resort fallback: at least separate by mode
        _ALL_RESULTS_PATH_OVERRIDE = RESULTS_DIR / f"{args.mode}_all_results.jsonl"
    logger.info(f"Output file: {_ALL_RESULTS_PATH_OVERRIDE}")
    if args.eval_results:
        _EVAL_RESULTS_PATH_OVERRIDE = args.eval_results.resolve()
        logger.info(f"Eval results override: {_EVAL_RESULTS_PATH_OVERRIDE}")
    if args.logs_dir:
        _LOGS_DIR_OVERRIDE = args.logs_dir.resolve()
        logger.info(f"Logs dir override: {_LOGS_DIR_OVERRIDE}")
    if args.gaia_subset:
        _GAIA_SUBSET_DIR = args.gaia_subset.resolve()
        logger.info(f"GAIA subset dir: {_GAIA_SUBSET_DIR}")
    if args.appworld_subset:
        _APPWORLD_SUBSET_DIR = args.appworld_subset.resolve()
        logger.info(f"AppWorld subset dir: {_APPWORLD_SUBSET_DIR}")
    if args.terminal_bench_subset:
        _TERMINAL_BENCH_SUBSET_DIR = args.terminal_bench_subset.resolve()
        logger.info(f"Terminal-Bench subset dir: {_TERMINAL_BENCH_SUBSET_DIR}")
    if args.agent_source_dir:
        _AGENT_SOURCE_DIR_OVERRIDE = args.agent_source_dir.resolve()
        logger.info(f"Agent source dir: {_AGENT_SOURCE_DIR_OVERRIDE}")

    # Select mode-specific functions
    is_gaia = args.mode == "gaia"
    is_appworld = args.mode == "appworld"
    is_terminal_bench = args.mode == "terminal_bench"
    if is_gaia:
        _run_one = _gaia_run_analysis
        mode_label = "GAIA"
    elif is_appworld:
        _run_one = _appworld_run_analysis
        mode_label = "AppWorld"
    elif is_terminal_bench:
        _run_one = _terminal_bench_run_analysis
        mode_label = "Terminal-Bench"
    else:
        _run_one = run_analysis
        mode_label = "SWE-bench"

    # Load config and impl doc
    config = load_config(args.mode)
    source_context = _agent_source_context(args.mode)
    if is_gaia:
        impl_doc = f"(open_deep_research agent source for this run: {source_context['agent_source_dir']})"
    elif is_appworld:
        impl_doc = f"(appworld_agent source for this run: {source_context['agent_source_dir']})"
    elif is_terminal_bench:
        impl_doc = f"(harbor terminus-2 source root for this run: {source_context['agent_source_dir']})"
    else:
        impl_doc = load_impl_doc(logger) + f"\n\nCurrent analyzed source dir: {source_context['agent_source_dir']}"

    # Load failed instances
    all_failed = load_failed_instances()
    logger.info(f"Total failed instances: {len(all_failed)}")

    # Build candidate list
    explicit_ids: list[str] = []
    if args.instance_id:
        explicit_ids.append(args.instance_id)
    if args.instance_ids_file:
        ids_from_file = [l.strip() for l in args.instance_ids_file.read_text().splitlines() if l.strip()]
        explicit_ids.extend(ids_from_file)

    if explicit_ids:
        for iid in explicit_ids:
            if iid not in all_failed:
                logger.warning(f"{iid} not in failed list, using category 'regressed'")
                all_failed[iid] = "regressed"
        candidates = {iid: all_failed[iid] for iid in explicit_ids}
    else:
        candidates = dict(all_failed)

    if args.category:
        candidates = {k: v for k, v in candidates.items() if v == args.category}
        logger.info(f"Filtered to category '{args.category}': {len(candidates)} instances")

    if args.resume:
        completed = load_completed_ids()
        before = len(candidates)
        candidates = {k: v for k, v in candidates.items() if k not in completed}
        logger.info(f"Resuming: skipped {before - len(candidates)} already-completed instances")

    if args.limit is not None:
        candidates = dict(list(candidates.items())[:args.limit])

    logger.info(
        f"Will analyze {len(candidates)} instances | mode={args.mode} | "
        f"model={args.model} | workers={args.workers}"
    )

    if args.dry_run:
        if is_gaia:
            get_p = _gaia_get_paths
        elif is_appworld:
            get_p = _appworld_get_paths
        elif is_terminal_bench:
            get_p = _terminal_bench_get_paths
        else:
            get_p = get_paths
        print(f"\n{'='*60}")
        print(f"DRY RUN ({mode_label}): Would analyze {len(candidates)} instances")
        print(f"Model:   {args.model}")
        print(f"Workers: {args.workers}")
        print(f"Output:  {_all_results_path()}")
        print(f"Source:  {source_context['agent_source_dir']}")
        print(f"{'='*60}")
        for iid, cat in candidates.items():
            paths = get_p(iid)
            traj_exists = Path(paths["traj_path"]).exists()
            print(f"  [{cat:12s}] {iid} (traj={'OK' if traj_exists else 'MISSING'})")
        return

    if not candidates:
        logger.info("No instances to analyze. Done.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0
    total_api_calls = 0
    calls_lock = threading.Lock()

    def _analyze_one(iid: str, cat: str) -> tuple[str, bool, int]:
        thread_logger = logging.getLogger(f"run_analysis.{iid}")
        thread_logger.info(f"[{cat}] starting ...")
        try:
            result = _run_one(iid, cat, args.model, config, impl_doc, thread_logger)
            if result:
                result = _attach_analysis_run_metadata(result, args.mode)
                append_result(result)
                thread_logger.info(f"[{cat}] ✓ done")
                return iid, True, result.get("api_calls", 0)
            else:
                thread_logger.warning(f"[{cat}] ✗ no result")
                return iid, False, 0
        except Exception as e:
            thread_logger.error(f"[{cat}] ✗ exception: {e}", exc_info=True)
            return iid, False, 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_analyze_one, iid, cat): iid for iid, cat in candidates.items()}
            for future in as_completed(futures):
                iid, ok, n_calls = future.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                with calls_lock:
                    total_api_calls += n_calls
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Partial results saved.")

    print(f"\n{'='*60}")
    print(f"{mode_label} Analysis complete: {success_count} succeeded, {fail_count} failed")
    print(f"Total API calls:  {total_api_calls}")
    print(f"Results: {_all_results_path()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

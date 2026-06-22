from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OperatorDefinition:
    family: str
    summary: str
    target_defect_classes: tuple[str, ...]
    target_metrics: tuple[str, ...]
    regression_risks: tuple[str, ...]
    static_checks: tuple[str, ...]
    rollback_conditions: tuple[str, ...]
    swe_allowed_paths: tuple[str, ...]
    gaia_allowed_paths: tuple[str, ...]
    appworld_allowed_paths: tuple[str, ...]
    validation_thresholds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFECT_CLASSES = (
    "termination",
    "tool_affordance",
    "parsing",
    "verification",
    "protocol",
    "context",
    "orchestration",
    "state_mutation",
    "completion_without_effect",
)

OPERATOR_FAMILIES = (
    "guardrail",
    "prompt",
    "protocol",
    "final_output_validator",
    "large_observation_guardrail",
    "artifact_hygiene_guardrail",
    "completion_effect_guard",
    "targeted_verification_prompt",
    "bounded_tool_search",
    "tool_affordance",
    "parser",
    "verification",
    "context",
    "state",
    "memory",
    "orchestration",
    "instrumentation",
    "gate",
)

DEFECT_CLASS_ALIASES = {
    "editing": "tool_affordance",
    "file_editing": "tool_affordance",
    "tool_api": "tool_affordance",
    "tool/api": "tool_affordance",
    "submission": "protocol",
    "protocol/submission": "protocol",
    "completion-without-effect": "completion_without_effect",
    "completion_without_state_change": "completion_without_effect",
    "complete_task_without_effect": "completion_without_effect",
    "no_effect_completion": "completion_without_effect",
    "answer_extraction": "parsing",
    "parsing/schema": "parsing",
    "schema": "parsing",
    "observability": "context",
}

OPERATOR_FAMILY_ALIASES = {
    "evaluator": "gate",
    "evaluator_gate": "gate",
    "evaluator/gate": "gate",
    "answer_validator": "final_output_validator",
    "final_answer_validator": "final_output_validator",
    "output_validator": "final_output_validator",
    "submission_validator": "final_output_validator",
    "observation_truncation": "large_observation_guardrail",
    "output_truncation": "large_observation_guardrail",
    "large_output_guardrail": "large_observation_guardrail",
    "artifact_hygiene": "artifact_hygiene_guardrail",
    "patch_hygiene": "artifact_hygiene_guardrail",
    "completion_effect": "completion_effect_guard",
    "state_effect_guard": "completion_effect_guard",
    "verification_prompt": "targeted_verification_prompt",
    "targeted_verification": "targeted_verification_prompt",
    "bounded_search": "bounded_tool_search",
    "bounded_tool_use": "bounded_tool_search",
    "prompt_guidance": "prompt",
    "prompt/guidance": "prompt",
    "parser_schema": "parser",
    "parser/schema": "parser",
    "state_session": "state",
    "state/session": "state",
}


OPERATOR_REGISTRY: dict[str, OperatorDefinition] = {
    "guardrail": OperatorDefinition(
        family="guardrail",
        summary="Block invalid, repetitive, or unsafe behavior before it propagates through the harness.",
        target_defect_classes=("termination", "protocol", "state_mutation", "completion_without_effect"),
        target_metrics=("loop_rate", "empty_patch_rate", "invalid_submission_rate", "completion_without_effect_rate"),
        regression_risks=(
            "Over-blocking legitimate retries or recovery loops.",
            "Blocking a valid finalization path because the heuristic is too strict.",
        ),
        static_checks=("syntax", "changed-files", "submission-shape"),
        rollback_conditions=(
            "New regressions on previously solved instances.",
            "The guardrail blocks valid actions without reducing the target defect.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/agents/default.py",
            "src/minisweagent/environments/local.py",
            "src/minisweagent/models/litellm_model.py",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/agent.py",
            "src/open_deep_research/config.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.01, "max_regressions": 1, "max_cost_ratio": 1.15},
    ),
    "prompt": OperatorDefinition(
        family="prompt",
        summary="Change task-facing guidance or output instructions without changing tool semantics.",
        target_defect_classes=("context", "protocol", "parsing"),
        target_metrics=("invalid_submission_rate", "accuracy", "resolved_rate"),
        regression_risks=(
            "Broad prompt guidance can dilute decisive task evidence.",
            "Restrictive instructions can hurt tasks that were already solved by the previous policy.",
        ),
        static_checks=(),
        rollback_conditions=(
            "Token cost or step count rises without improving target failures.",
            "The prompt change creates new format or protocol violations.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=("src/open_deep_research/prompts.py",),
        appworld_allowed_paths=("src/appworld_agent/prompts.py",),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.25},
    ),
    "protocol": OperatorDefinition(
        family="protocol",
        summary="Tighten required workflow, finalization, or handoff contracts before accepting outputs.",
        target_defect_classes=("protocol", "verification", "termination", "completion_without_effect"),
        target_metrics=("invalid_submission_rate", "verification_effort_ratio", "resolved_rate", "completion_without_effect_rate"),
        regression_risks=(
            "Protocol becomes too strict and blocks productive short-horizon tasks.",
            "Agents spend more budget satisfying ceremony than solving the task.",
        ),
        static_checks=("syntax", "submission-shape"),
        rollback_conditions=(
            "Verification effort increases without reducing the target defect.",
            "Solved/correct rate drops on the held-out validation split.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/agents/default.py",
            "src/minisweagent/environments/local.py",
            "src/minisweagent/config/benchmarks/swebench.yaml",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/agent.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/prompts.py",
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
        ),
        validation_thresholds={"delta_min": 0.01, "max_regressions": 2, "max_cost_ratio": 1.2},
    ),
    "final_output_validator": OperatorDefinition(
        family="final_output_validator",
        summary="Validate the final answer, patch, or completion payload before accepting it as a benchmark submission.",
        target_defect_classes=("parsing", "protocol", "completion_without_effect"),
        target_metrics=("invalid_submission_rate", "empty_patch_rate", "exact_match", "completion_without_effect_rate"),
        regression_risks=(
            "A strict validator can reject semantically valid outputs with harmless formatting differences.",
            "A permissive validator can hide task-agent defects instead of repairing the finalization contract.",
        ),
        static_checks=("syntax", "submission-shape"),
        rollback_conditions=(
            "Correct outputs are newly rejected by the validator.",
            "Invalid or empty final outputs do not decrease on the validation split.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/agents/default.py",
            "src/minisweagent/environments/local.py",
            "src/minisweagent/environments/docker.py",
            "src/minisweagent/config/benchmarks/swebench.yaml",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/scripts/reformulator.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.1},
    ),
    "large_observation_guardrail": OperatorDefinition(
        family="large_observation_guardrail",
        summary="Bound oversized observations or command outputs before they cause context loss or termination.",
        target_defect_classes=("context", "termination", "tool_affordance"),
        target_metrics=("large_observation_rate", "context_overflow_rate", "error_rate", "loop_rate"),
        regression_risks=(
            "Over-truncation can remove decisive evidence from the next agent step.",
            "A command blocklist can reject legitimate diagnostic commands if the heuristic is too broad.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Context overflow or oversized observation failures do not decrease.",
            "Previously solved tasks regress because relevant output is hidden.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/environments/local.py",
            "src/minisweagent/environments/docker.py",
            "src/minisweagent/config/benchmarks/swebench.yaml",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/browser.py",
            "src/open_deep_research/tools.py",
            "src/open_deep_research/prompts.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.15},
    ),
    "artifact_hygiene_guardrail": OperatorDefinition(
        family="artifact_hygiene_guardrail",
        summary="Prevent temporary files, helper scripts, binary outputs, or broad staged changes from entering final artifacts.",
        target_defect_classes=("protocol", "state_mutation", "completion_without_effect"),
        target_metrics=("invalid_submission_rate", "artifact_pollution_rate", "empty_patch_rate"),
        regression_risks=(
            "The hygiene check can reject legitimate new source files if path rules are too broad.",
            "Prompt-only hygiene can be ignored unless paired with a final artifact validator.",
        ),
        static_checks=("syntax", "changed-files", "submission-shape"),
        rollback_conditions=(
            "Artifact-polluted submissions still appear after the repair.",
            "Valid source-only changes are newly blocked by the hygiene rule.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/environments/local.py",
            "src/minisweagent/environments/docker.py",
            "src/minisweagent/config/benchmarks/swebench.yaml",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/scripts/reformulator.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.1},
    ),
    "completion_effect_guard": OperatorDefinition(
        family="completion_effect_guard",
        summary="Require evidence of the expected task-state effect before accepting a completion signal.",
        target_defect_classes=("completion_without_effect", "state_mutation", "protocol"),
        target_metrics=("completion_without_effect_rate", "state_mutation_rate", "task_success_rate", "resolved_rate"),
        regression_risks=(
            "A state-effect guard can block legitimate read-only tasks if the completion contract is not domain-aware.",
            "A weak effect heuristic can create false confidence without improving benchmark outcomes.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Completion-without-effect failures do not decrease on validation.",
            "Read-only or answer-only tasks regress because the guard requires mutation when none is expected.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/agents/default.py",
            "src/minisweagent/environments/local.py",
            "src/minisweagent/environments/docker.py",
            "src/minisweagent/config/benchmarks/swebench.yaml",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/agent.py",
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/scripts/reformulator.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.01, "max_regressions": 1, "max_cost_ratio": 1.15},
    ),
    "targeted_verification_prompt": OperatorDefinition(
        family="targeted_verification_prompt",
        summary="Add narrowly scoped verification guidance that checks the actual failure path without broad behavior changes.",
        target_defect_classes=("verification", "protocol"),
        target_metrics=("verification_effort_ratio", "resolved_rate", "accuracy"),
        regression_risks=(
            "Extra verification can consume budget on easy tasks.",
            "Broad verification language can distract the agent from the task-specific signal.",
        ),
        static_checks=(),
        rollback_conditions=(
            "Verification effort rises without improving solved/correct outcomes.",
            "Previously solved short-horizon tasks regress from over-verification.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=("src/open_deep_research/prompts.py",),
        appworld_allowed_paths=("src/appworld_agent/prompts.py",),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.2},
    ),
    "bounded_tool_search": OperatorDefinition(
        family="bounded_tool_search",
        summary="Guide or constrain exploratory tool use so searches are targeted, bounded, and inspectable.",
        target_defect_classes=("context", "tool_affordance", "termination"),
        target_metrics=("repeated_search_rate", "large_observation_rate", "context_overflow_rate", "loop_rate"),
        regression_risks=(
            "Too much bounding can prevent discovery on tasks that require broader exploration.",
            "Search guidance can become domain-specific and fail to transfer across repositories or tools.",
        ),
        static_checks=(),
        rollback_conditions=(
            "Repeated or unbounded search behavior does not decrease.",
            "Validation accuracy drops because the agent stops exploring too early.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/tools.py",
            "src/open_deep_research/browser.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/prompts.py",
            "src/appworld_agent/core.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.2},
    ),
    "tool_affordance": OperatorDefinition(
        family="tool_affordance",
        summary="Repair brittle edit/search/file/API affordances so the agent can act safely.",
        target_defect_classes=("tool_affordance", "context", "state_mutation"),
        target_metrics=("syntax_error_rate", "repeated_search_rate", "repeated_api_call_rate"),
        regression_risks=(
            "Changing tooling can silently alter working behaviors on already-solved tasks.",
            "A narrow affordance fix can create a new dependency on a specific repository or API layout.",
        ),
        static_checks=("syntax", "changed-files"),
        rollback_conditions=(
            "Tool-use failures or repeated lookup behaviors do not decrease.",
            "The candidate changes files outside the declared tool surface.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/config/benchmarks/swebench.yaml",
            "src/minisweagent/agents/default.py",
            "src/minisweagent/environments/local.py",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/tools.py",
            "src/open_deep_research/browser.py",
            "src/open_deep_research/prompts.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.01, "max_regressions": 2, "max_cost_ratio": 1.25},
    ),
    "parser": OperatorDefinition(
        family="parser",
        summary="Normalize, validate, or recover structured outputs produced by the task agent.",
        target_defect_classes=("parsing", "protocol"),
        target_metrics=("invalid_submission_rate", "error_rate", "exact_match"),
        regression_risks=(
            "A permissive parser can hide task-agent defects instead of fixing them.",
            "A strict parser can reject semantically valid outputs with harmless formatting differences.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Format errors do not decrease on the validation split.",
            "Correct outputs are newly rejected by the parser.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=("src/open_deep_research/scripts/reformulator.py",),
        appworld_allowed_paths=("src/appworld_agent/core.py", "src/appworld_agent/official_react_adapter.py"),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.1},
    ),
    "verification": OperatorDefinition(
        family="verification",
        summary="Add stronger targeted checking before final answers or submissions are accepted.",
        target_defect_classes=("verification", "protocol", "completion_without_effect"),
        target_metrics=("verification_effort_ratio", "resolved_rate", "accuracy", "completion_without_effect_rate"),
        regression_risks=(
            "Extra verification consumes too much budget on easy tasks.",
            "A weak checker provides false confidence and does not actually reduce failures.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Verification cost grows without increasing solved/correct outcomes.",
            "The added checker introduces new false negatives or false positives.",
        ),
        swe_allowed_paths=(
            "src/minisweagent/config/benchmarks/swebench.yaml",
            "src/minisweagent/agents/default.py",
        ),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/scripts/reformulator.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.25},
    ),
    "context": OperatorDefinition(
        family="context",
        summary="Expose, retain, or select more useful within-run evidence for future decisions.",
        target_defect_classes=("context", "termination"),
        target_metrics=("repeated_search_rate", "loop_rate", "accuracy"),
        regression_risks=(
            "More context can dilute the decisive signal or increase token cost.",
            "A context-specific heuristic can overfit one benchmark branch.",
        ),
        static_checks=(),
        rollback_conditions=(
            "Token cost or step count rises without improving target failures.",
            "The candidate adds broad context clutter instead of selective evidence.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=(
            "src/open_deep_research/prompts.py",
            "src/open_deep_research/config.py",
        ),
        appworld_allowed_paths=("src/appworld_agent/prompts.py",),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.3},
    ),
    "state": OperatorDefinition(
        family="state",
        summary="Constrain session, database, filesystem, or external-app mutations.",
        target_defect_classes=("state_mutation", "completion_without_effect"),
        target_metrics=("collateral_damage_rate", "assertion_failure_rate", "repeated_api_call_rate", "completion_without_effect_rate"),
        regression_risks=(
            "A mutation guard can block legitimate multi-step state changes.",
            "State isolation can mask bugs that only appear across realistic sessions.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Collateral or assertion failures do not decrease.",
            "The new state policy prevents valid state-changing tasks from completing.",
        ),
        swe_allowed_paths=("src/minisweagent/environments/local.py",),
        gaia_allowed_paths=("src/open_deep_research/agent.py",),
        appworld_allowed_paths=("src/appworld_agent/core.py", "src/appworld_agent/official_react_adapter.py"),
        validation_thresholds={"delta_min": 0.01, "max_regressions": 1, "max_cost_ratio": 1.15},
    ),
    "memory": OperatorDefinition(
        family="memory",
        summary="Store accepted and rejected repairs with applicability conditions for later reuse.",
        target_defect_classes=("context", "verification", "protocol"),
        target_metrics=("repeated_bad_repair_rate", "iterations_to_pass_gate"),
        regression_risks=(
            "Stale memories bias the planner toward outdated fixes.",
            "Incorrect retrieval can suppress useful new repair directions.",
        ),
        static_checks=(),
        rollback_conditions=(
            "The same rejected repair is proposed again.",
            "Retrieved memories are unrelated to the current defect cluster.",
        ),
        swe_allowed_paths=(),
        gaia_allowed_paths=(),
        appworld_allowed_paths=(),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.0},
    ),
    "orchestration": OperatorDefinition(
        family="orchestration",
        summary="Repair manager-worker coordination, delegation, scheduling, or answer handoff logic.",
        target_defect_classes=("orchestration", "context"),
        target_metrics=("delegation_failure_rate", "accuracy", "repeated_search_rate"),
        regression_risks=(
            "A coordination rule may over-constrain a stronger base model.",
            "More orchestration can increase latency and token cost quickly.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Wrong-answer or delegation-loop failures do not decrease.",
            "Validation cost blows up because the coordination policy is too chatty.",
        ),
        swe_allowed_paths=("src/minisweagent/agents/default.py",),
        gaia_allowed_paths=(
            "src/open_deep_research/agent.py",
            "src/open_deep_research/prompts.py",
        ),
        appworld_allowed_paths=(
            "src/appworld_agent/core.py",
            "src/appworld_agent/official_react_adapter.py",
            "src/appworld_agent/prompts.py",
        ),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 2, "max_cost_ratio": 1.3},
    ),
    "instrumentation": OperatorDefinition(
        family="instrumentation",
        summary="Improve trace capture and attribution metadata used by the repair pipeline.",
        target_defect_classes=("context", "verification", "state_mutation", "completion_without_effect"),
        target_metrics=("trace_coverage", "missing_evidence_rate", "attribution_agreement", "completion_without_effect_rate"),
        regression_risks=(
            "Instrumentation can perturb timing-sensitive runs or increase storage cost.",
            "Over-detailed traces can make attribution noisier rather than clearer.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "The trace still cannot support root-cause attribution.",
            "Instrumentation changes alter task-agent behavior on validation runs.",
        ),
        swe_allowed_paths=("src/minisweagent/agents/default.py", "src/minisweagent/environments/local.py"),
        gaia_allowed_paths=("src/open_deep_research/agent.py",),
        appworld_allowed_paths=("src/appworld_agent/core.py", "src/appworld_agent/official_react_adapter.py"),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.1},
    ),
    "gate": OperatorDefinition(
        family="gate",
        summary="Repair acceptance gates, validation criteria, or post-run scoring interfaces.",
        target_defect_classes=("verification", "protocol", "completion_without_effect"),
        target_metrics=("collateral_regression_rate", "resolved_rate", "accuracy", "completion_without_effect_rate"),
        regression_risks=(
            "A gate change can report improvement by changing acceptance rather than behavior.",
            "A narrow gate can overfit one benchmark's scoring convention.",
        ),
        static_checks=("syntax",),
        rollback_conditions=(
            "Reported gains come only from scoring or formatting changes.",
            "Collateral regressions rise after the gate change.",
        ),
        swe_allowed_paths=("src/minisweagent/config/benchmarks/swebench.yaml",),
        gaia_allowed_paths=("src/open_deep_research/scripts/reformulator.py",),
        appworld_allowed_paths=("src/appworld_agent/core.py", "src/appworld_agent/official_react_adapter.py"),
        validation_thresholds={"delta_min": 0.0, "max_regressions": 1, "max_cost_ratio": 1.05},
    ),
}


SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


def load_operator_registry() -> dict[str, dict[str, Any]]:
    return {name: definition.to_dict() for name, definition in OPERATOR_REGISTRY.items()}


def _text_blob(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in (
            "affected_component",
            "defect_class",
            "failure_category",
            "failure_manifestation",
            "failure_reason",
            "agent_design_issue",
            "recommended_operator_family",
            "fix_scope",
            "candidate_responsible_steps",
            "responsible_steps",
            "implicated_harness_layers",
            "defect_labels",
        )
    ).lower()


def normalize_defect_class(defect_class: str | None) -> str | None:
    if not defect_class:
        return None
    normalized = str(defect_class).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = DEFECT_CLASS_ALIASES.get(normalized, normalized)
    return normalized if normalized in DEFECT_CLASSES else None


def normalize_operator_family(family: str | None) -> str | None:
    if not family:
        return None
    normalized = str(family).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = OPERATOR_FAMILY_ALIASES.get(normalized, normalized)
    return normalized if normalized in OPERATOR_FAMILIES else None


def infer_defect_class(record: dict[str, Any]) -> str:
    explicit = normalize_defect_class(record.get("defect_class"))
    if explicit:
        return explicit

    text = _text_blob(record)
    component = str(record.get("affected_component", "")).lower()
    category = str(record.get("failure_category", "")).lower()

    if (
        "repet" in text
        or "loop" in text
        or "limit" in text
        or "timeout" in text
        or (component in {"main_loop", "error_handling"} and category == "empty_patch")
    ):
        return "termination"
    if component in {"answer_extraction", "format_parsing"} or any(
        token in text for token in ("json", "schema", "parse", "format", "malformed", "answer extraction", "final_answer")
    ):
        return "parsing"
    if component in {"testing", "validator"} or any(token in text for token in ("test", "verification", "validator", "checker")):
        return "verification"
    if any(
        token in text
        for token in (
            "completion_without_effect",
            "completion without effect",
            "complete_task without",
            "complete task without",
            "no task-state",
            "no task state",
            "no database mutation",
            "no db mutation",
            "completed but no effect",
            "missing required effect",
        )
    ):
        return "completion_without_effect"
    if (
        component in {"state_guard", "api_execution"}
        or any(token in text for token in ("unsafe mutation", "collateral", "unexpected state", "state mutation"))
    ):
        return "state_mutation"
    if any(token in text for token in ("submission", "submit", "complete_task", "final answer", "handoff", "patch")):
        return "protocol"
    if component in {"system_prompt", "prompt"} or any(token in text for token in ("context", "truncat", "missing evidence", "prompt")):
        return "context"
    if any(token in text for token in ("delegate", "subagent", "search_agent", "manager", "worker", "handoff")):
        return "orchestration"
    if (
        component in {"tool_use", "file_handling", "web_search", "file_editing"}
        or any(token in text for token in ("sed", "indentation", "syntaxerror", "api", "file path", "browser", "tool"))
    ):
        return "tool_affordance"
    return "verification" if category == "unresolved" else "protocol"


def recommend_operator_family(record: dict[str, Any]) -> str:
    text = _text_blob(record)
    explicit = normalize_operator_family(record.get("recommended_operator_family"))
    if explicit:
        return explicit

    defect_class = infer_defect_class(record)
    component = str(record.get("affected_component", "")).lower()

    if "memory" in text:
        return "memory"
    if "instrument" in text or ("trace" in text and any(token in text for token in ("missing", "coverage", "observability"))):
        return "instrumentation"
    if "gate" in text or "acceptance" in text or "scoring" in text:
        return "gate"
    if any(token in text for token in ("artifact", "git add -a", "git add --all", "binary", "helper script", "test artifact")):
        return "artifact_hygiene_guardrail"
    if any(token in text for token in ("large output", "huge output", "ls -lar", "ls -la r", "context overflow", "contextwindow")):
        return "large_observation_guardrail"
    if any(token in text for token in ("bounded search", "unbounded search", "repeated search", "search loop")):
        return "bounded_tool_search"
    if any(token in text for token in ("empty patch", "invalid submission", "malformed submission", "final output", "final answer format")):
        return "final_output_validator"
    if "prompt" in text or record.get("fix_scope") in {"prompt_additive", "prompt_restrictive"}:
        return "prompt"
    if defect_class == "completion_without_effect":
        return "completion_effect_guard" if any(token in text for token in ("state", "effect", "mutation", "complete_task")) else "verification"
    if defect_class in {"termination", "protocol"} and ("loop" in text or "block" in text or "guard" in text):
        return "guardrail"
    if defect_class == "tool_affordance":
        return "tool_affordance"
    if defect_class == "parsing":
        return "parser"
    if defect_class == "verification":
        return "targeted_verification_prompt" if component in {"system_prompt", "prompt", "testing", "validator"} else "verification"
    if defect_class == "protocol":
        return "protocol"
    if defect_class == "context":
        if component in {"web_search", "file_handling", "tool_use"}:
            return "tool_affordance"
        return "context"
    if defect_class == "orchestration":
        return "orchestration"
    if defect_class == "state_mutation":
        return "guardrail" if any(token in text for token in ("block", "prevent", "reject unsafe")) else "state"
    return "protocol"


def infer_severity(record: dict[str, Any]) -> str:
    text = _text_blob(record)
    if any(token in text for token in ("syntaxerror", "indentationerror", "timeout", "regression", "fabricat", "hallucin")):
        return "high"
    if any(token in text for token in ("failed test", "wrong answer", "empty patch", "submission")):
        return "medium"
    return "low"


def operator_allowed_paths(mode: str, family: str) -> list[str]:
    family = normalize_operator_family(family) or family
    definition = OPERATOR_REGISTRY[family]
    if mode == "swe":
        allowed = definition.swe_allowed_paths
    elif mode == "gaia":
        allowed = definition.gaia_allowed_paths
    elif mode == "terminal_bench":
        allowed = (
            "harbor/src/harbor/agents/terminus_2/...",
            "run_terminal_bench_entry.py",
        )
    else:
        allowed = definition.appworld_allowed_paths
    return list(allowed)


def format_operator_registry_for_prompt(mode: str) -> str:
    lines = []
    for definition in OPERATOR_REGISTRY.values():
        allowed = operator_allowed_paths(mode, definition.family)
        allowed_text = ", ".join(allowed) if allowed else "(pipeline-only / memory-only)"
        lines.append(
            f"- {definition.family}: targets {', '.join(definition.target_defect_classes)} | "
            f"metrics={', '.join(definition.target_metrics)} | allowed_paths={allowed_text}"
        )
    return "\n".join(lines)


def thresholds_for_family(family: str) -> dict[str, float]:
    family = normalize_operator_family(family) or family
    return dict(OPERATOR_REGISTRY[family].validation_thresholds)

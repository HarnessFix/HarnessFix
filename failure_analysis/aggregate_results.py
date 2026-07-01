#!/usr/bin/env python3
"""Aggregate individual failure analyses into a clustered improvement plan.

Reads failure_analysis/results/all_results.jsonl, uses an LLM to:
  1. Cluster similar agent_design_issues into issue groups
  2. Rank groups by frequency and impact
  3. Produce a concrete, code-level improvement plan
  4. Emit a machine-readable implementation spec for the modify stage

Outputs:
  - improvement_plans/improvement_plan.md  (default; pipeline overrides with --output)
  - improvement_plans/improvement_plan.json

Usage:
  .venv/bin/python3 failure_analysis/aggregate_results.py
  .venv/bin/python3 failure_analysis/aggregate_results.py --model openai/gpt-5-mini
  .venv/bin/python3 failure_analysis/aggregate_results.py --force   # overwrite existing
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agent_framework" / "src"))

from failure_analysis.consolidation import (
    consolidate_diagnoses,
    enrich_spec_with_clusters,
    format_clusters_for_prompt,
)
from failure_analysis.harness_memory import (
    default_memory_root,
    format_memories_for_prompt,
    retrieve_relevant_memories,
)
from failure_analysis.operator_registry import (
    format_operator_registry_for_prompt,
    normalize_defect_class,
    normalize_operator_family,
)
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
import litellm

ALL_RESULTS_PATH = Path(__file__).parent / "results" / "all_results.jsonl"
IMPROVEMENT_PLANS_DIR = REPO_ROOT / "improvement_plans"
IMPROVEMENT_PLAN_PATH = IMPROVEMENT_PLANS_DIR / "improvement_plan.md"
IMPROVEMENT_SPEC_PATH = IMPROVEMENT_PLANS_DIR / "improvement_plan.json"

DEFAULT_MODEL = "openai/gpt-5-mini"
MODEL_KWARGS = {
    "temperature": 1,
    "stream": False,
    "timeout": 300,
    "max_tokens": 4096,
    "drop_params": True,
    "api_base": (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LITELLM_API_BASE")
        or "https://api.openai.com/v1"
    ),
    # api_key is read from OPENAI_API_KEY environment variable (set in .env)
}

TASK_AGENT_SRC_SWE = str(REPO_ROOT / "task_agent" / "mini-swe-agent" / "src" / "minisweagent")
TASK_AGENT_SRC_GAIA = str(REPO_ROOT / "task_agent" / "open_deep_research" / "src" / "open_deep_research")
TASK_AGENT_SRC_APPWORLD = str(REPO_ROOT / "task_agent" / "appworld_agent" / "src" / "appworld_agent")
TASK_AGENT_SRC_TERMINAL_BENCH = str(REPO_ROOT / "task_agent" / "terminal_bench_agent" / "harbor" / "src" / "harbor" / "agents" / "terminus_2")

# Legacy alias for backward compat
TASK_AGENT_SRC = TASK_AGENT_SRC_SWE

# Aggregation sees many records at once, so it must not inline full traces/HTIR.
# The detailed artifacts remain on disk and are referenced by path.
AGG_EVIDENCE_ANCHOR_CHARS = int(os.environ.get("HARNESSFIX_AGG_EVIDENCE_ANCHOR_CHARS", "1200"))
AGG_TEXT_FIELD_CHARS = int(os.environ.get("HARNESSFIX_AGG_TEXT_FIELD_CHARS", "900"))
AGG_PREV_CONTEXT_CHARS = int(os.environ.get("HARNESSFIX_AGG_PREV_CONTEXT_CHARS", "30000"))
AGG_MEMORY_LIMIT = int(os.environ.get("HARNESSFIX_AGG_MEMORY_LIMIT", "8"))
AGG_AGENT_STEP_LIMIT = int(os.environ.get("HARNESSFIX_AGG_AGENT_STEP_LIMIT", "40"))
AGG_AGENT_COST_LIMIT = float(os.environ.get("HARNESSFIX_AGG_AGENT_COST_LIMIT", "5.0"))


def load_results(results_path: Path) -> list[dict]:
    results = []
    for line in results_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def compact_text(value: object, limit: int) -> str:
    """Keep aggregate prompts bounded while preserving local evidence pointers."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    head = max(limit // 2, 1)
    tail = max(limit - head, 1)
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n"
        f"...[{omitted} chars omitted in aggregate prompt; inspect referenced artifacts for full context]...\n"
        f"{text[-tail:]}"
    )


def compact_json(value: object, limit: int) -> str:
    """Compact structured evidence without dropping all field names."""
    if isinstance(value, dict):
        per_field = max(300, limit // max(len(value), 1))
        compacted = {key: compact_text(item, per_field) for key, item in value.items()}
        return compact_text(compacted, limit)
    if isinstance(value, list):
        selected = value[:5]
        if len(value) > len(selected):
            selected.append(f"...[{len(value) - len(selected)} additional items omitted]")
        return compact_text(selected, limit)
    return compact_text(value, limit)


def _record_source_dir(record: dict) -> str:
    source = record.get("agent_source_dir")
    if not source and isinstance(record.get("analysis_config"), dict):
        source = record["analysis_config"].get("agent_source_dir")
    return str(source or "")


def format_results_for_llm(results: list[dict]) -> str:
    """Format all results compactly for the LLM prompt."""
    lines = []
    for r in results:
        defect_class = normalize_defect_class(r.get("defect_class")) or r.get("defect_class", "?")
        operator_family = (
            normalize_operator_family(r.get("recommended_operator_family"))
            or r.get("recommended_operator_family", "?")
        )
        evidence = r.get("evidence_anchor", {}) or {}
        evidence_str = compact_json(evidence, AGG_EVIDENCE_ANCHOR_CHARS) or "(none)"
        evidence_spans = compact_json(r.get("evidence_spans", []), AGG_TEXT_FIELD_CHARS)
        source_dir = _record_source_dir(r)
        lines.append(
            f"[{r.get('instance_id', '?')}] "
            f"cat={r.get('failure_category', '?')} "
            f"exit={r.get('exit_status', '?')} "
            f"component={r.get('affected_component', '?')} "
            f"defect={defect_class} "
            f"operator={operator_family} "
            f"source_dir={source_dir or '?'}\n"
            f"  manifestation: {compact_text(r.get('failure_manifestation', ''), AGG_TEXT_FIELD_CHARS)}\n"
            f"  failure_reason: {compact_text(r.get('failure_reason', ''), AGG_TEXT_FIELD_CHARS)}\n"
            f"  evidence_anchor: {evidence_str}\n"
            f"  evidence_spans: {evidence_spans}\n"
            f"  design_issue: {compact_text(r.get('agent_design_issue', ''), AGG_TEXT_FIELD_CHARS)}\n"
        )
    return "\n".join(lines)


# ── SWE-Bench mode prompts ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior software engineer analyzing failure patterns in an LLM-based coding agent system (mini-swe-agent).

Your job is to:
1. Use the provided harness-layer buckets as coarse organization only
2. Within each layer bucket, cluster the individual failure analyses into semantic groups of similar root causes
3. For each semantic cluster, synthesize the common pattern and propose concrete code-level improvements
4. Output a detailed, actionable improvement plan in Markdown

The agent system codebase is at:
  {src_dir}

Key files:
  agents/default.py          — main loop (DefaultAgent.run/step/query/execute_actions)
  exceptions.py              — LimitsExceeded, Timeout, FormatError
  environments/docker.py     — command execution, timeout, subshell
  models/litellm_model.py           — tool call action parsing
  config/benchmarks/swebench.yaml   — runtime config (step_limit=250, timeout=60s, cost_limit=$3)
""".format(src_dir=TASK_AGENT_SRC_SWE)

USER_PROMPT_TEMPLATE = """\
Below are {n} structured failure analyses from a mini-swe-agent evaluation run on SWE-Bench Verified.
Each entry shows: instance_id, failure category, exit status, affected component, failure manifestation,
evidence anchors extracted from raw artifacts, and agent design issue.

Failure distribution:
{distribution}

--- ALL FAILURE ANALYSES ---
{analyses}
--- END ---
{val_regression_section}{prev_plan_section}
## Your Task

Produce a comprehensive improvement plan in Markdown with the following structure:

# Agent Improvement Plan — Round 1 Analysis

## Executive Summary
(2-3 sentences: overall failure patterns, what % of failures each issue type causes)

## Issue Clusters

For each cluster (ordered by frequency/impact, most important first):

### Cluster N: <Short Name>
- **Affected component**: <component>
- **Frequency**: X instances (Y% of failures)
- **Failure categories**: empty_patch / unresolved / error (breakdown)
- **Evidence anchors**: summarize the shared raw evidence pattern (commands, terminal observations, failing tests, report signals)
- **Root cause**: (2-3 sentences: what design decision causes this class of failures)
- **Representative instances**: (list 2-3 instance IDs with one-line description)
- **Proposed fix**:
  - Target file: `<relative path in minisweagent/>`
  - Current behavior: (describe what the code currently does)
  - New behavior: (describe what it should do instead)
  - Implementation sketch: (pseudocode or concrete code snippet showing the change)

## Implementation Priority

| Priority | Cluster | Estimated Impact | Implementation Difficulty |
|----------|---------|-----------------|--------------------------|
| P0 | ... | ... | ... |
| P1 | ... | ... | ... |
| ...

## Detailed Implementation Plan

For each cluster (same order as above), provide:

### Fix N: <Name>
**Files to modify**: list of files
**Step-by-step implementation**:
1. (specific change 1)
2. (specific change 2)
...
**Expected outcome**: (what failure mode this eliminates or reduces)
**Test**: (how to verify the fix works)

## Notes for the Modify Agent
(Any cross-cutting concerns, order of implementation, backward compatibility notes, etc.)
"""


SPEC_SYSTEM_PROMPT = """\
You are converting an already-written agent improvement plan into a machine-readable implementation spec.

Return exactly one fenced code block:

```json
{ ... }
```

The JSON object MUST follow this schema:
{
  "plan_metadata": {
    "round_label": "Round 1 Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  },
  "edit_budget": {
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 2,
    "allowed_paths": ["src/minisweagent/..."],
    "forbidden_paths": ["src/minisweagent/..."],
    "active_files": ["src/minisweagent/..."],
    "must_inspect_files": ["src/minisweagent/..."],
    "do_not_edit_until_inspected": true,
    "rationale": "why this budget is appropriate"
  },
  "fixes": [
    {
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["src/minisweagent/..."],
      "target_symbols": ["DefaultAgent.query"],
      "active_files": ["src/minisweagent/..."],
      "must_inspect_files": ["src/minisweagent/..."],
      "do_not_edit_until_inspected": true,
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": ["fix_0"],
      "regression_risks": ["what could break"],
      "must_not_change": ["public API, message schema, etc."]
    }
  ]
}

Rules:
- Output valid JSON only inside the fenced block.
- Every fix must map to concrete files and behavior deltas already implied by the plan.
- Prefer narrow file lists. Do not introduce broad refactors that the plan did not justify.
- `allowed_paths` should be the minimal set needed for the proposed fixes.
- `active_files` should name the real files on the active execution path that the fix is expected to change.
- `must_inspect_files` should include every active file the modify agent must read before editing; include target files and nearby finalizer/parser/prompt/config files needed to prove the path is active.
- Set `do_not_edit_until_inspected` to true unless the plan is intentionally documentation-only.
- `forbidden_paths` should protect unrelated benchmark, eval, and pipeline files unless the plan explicitly requires them.
- If a fix cannot be implemented safely under the recommended budget, say so explicitly in `risk_level` and `regression_risks`.
"""


# ── GAIA mode prompts ──────────────────────────────────────────────────────────

GAIA_SYSTEM_PROMPT = """\
You are a senior software engineer analyzing failure patterns in an LLM-based research agent system (open_deep_research).

Your job is to:
1. Use the provided harness-layer buckets as coarse organization only
2. Within each layer bucket, cluster the individual failure analyses into semantic groups of similar root causes
3. For each semantic cluster, synthesize the common pattern and propose concrete code-level improvements
4. Output a detailed, actionable improvement plan in Markdown followed by a machine-readable JSON spec

The agent system codebase is at:
  {src_dir}

Architecture:
- **Manager CodeAgent** (agent.py): Top-level orchestrator with visualizer + TextInspectorTool, max_steps=12, planning_interval=4
- **Search ToolCallingAgent** (agent.py): Sub-agent with web search + browser tools, max_steps=20
- Each GAIA question is augmented with AUGMENTED_QUESTION_PREFIX (prompts.py) before running

Key files — only propose changes to these:
  src/open_deep_research/config.py       — step limits, browser config (PRIMARY target: parameter tuning)
  src/open_deep_research/prompts.py      — AUGMENTED_QUESTION_PREFIX (PRIMARY target: highest leverage for GAIA)
  src/open_deep_research/agent.py        — create_agent_team() — manager + search agent setup
  src/open_deep_research/tools.py        — WEB_TOOLS list (GoogleSearchTool + browser navigation)
  src/open_deep_research/browser.py      — SimpleTextBrowser factory

IMMUTABLE FILES — never propose changes to these:
  run_gaia_entry.py                                    — pipeline entry point (FROZEN)
  src/open_deep_research/scoring.py                    — scoring logic (FROZEN)
  src/open_deep_research/scripts/gaia_scorer.py        — GAIA scoring (FROZEN)
  src/open_deep_research/scripts/reformulator.py       — answer reformulation (treat as read-only unless explicitly required)

## CRITICAL: Distinguishing "premature giving up" from "genuinely hard tasks"

The failure analyses include a field `gave_up_prematurely`. When this is true, it means the agent
had REMAINING steps and resources but reported "Unable to determine" anyway. This is a DIFFERENT
failure mode from a task that genuinely exhausted all options.

For `gave_up_prematurely=true` failures: the fix is to improve PERSISTENCE GUIDANCE, NOT to add
more "when to give up" instructions. Adding give-up guidance to already-quick-to-quit agents
will cause regressions on tasks the baseline could solve.
""".format(src_dir=TASK_AGENT_SRC_GAIA)

GAIA_USER_PROMPT_TEMPLATE = """\
Below are {n} structured failure analyses from an open_deep_research agent evaluation run on GAIA benchmark.
Each entry shows: instance_id, failure category, exit status, affected component, failure manifestation,
gave_up_prematurely flag, and agent design issue.

Failure distribution:
{distribution}

--- ALL FAILURE ANALYSES ---
{analyses}
--- END ---
{val_regression_section}{prev_plan_section}
## CRITICAL RULES — Read Before Writing Any Fix

### Rule 0: Minimum Intervention Principle
**Fix only the specific failure pattern you observed. Do NOT change behavior for task types that are
already working correctly.**

Before proposing any fix, ask: "Does this change ONLY affect the broken behavior, or does it also
change how the agent handles currently-passing tasks?" If it affects both, find a more targeted
intervention. A fix that helps 5 failing tasks but breaks 3 passing tasks is a net loss.

The `fix_scope` field in each failure analysis tells you:
- `additive_only` / `config_param` — safe, low regression risk
- `prompt_additive` — moderate risk, review carefully
- `prompt_restrictive` — high risk, avoid or heavily justify

**Prefer `additive_only` and `config_param` fixes whenever possible.**
Only propose `prompt_restrictive` fixes if: (a) the failure pattern appears in ≥20% of failures
AND (b) you can show the restriction only activates for the failing task type.

### Rule 1: Premature giving up pattern
If many failures have `gave_up_prematurely=true`, the agent is already too quick to quit.
Do NOT add any "When Information Cannot Be Found" or "Unable to determine" guidance — this will
make the problem WORSE. Instead, add PERSISTENCE guidance.

### Rule 2: Prompt change regression risk
Every sentence you add to AUGMENTED_QUESTION_PREFIX applies to ALL tasks including ones the agent
already solves correctly. For each proposed prompt change, explicitly ask:
"On a simple Level-1 task that the baseline already passes in 3 steps, does this new instruction
change anything? Could it cause the agent to over-verify, refuse a valid answer, or behave
differently on a task it was handling correctly?"

### Rule 3: Data completeness guidance
"Do not proceed with partial data" → causes regression (agent gives up on partial results)
"Gather as much data as possible before concluding" → safe additive framing

### Rule 4: Conflicting instructions
New guidance must not contradict the existing instruction
"I know for a fact that you have access to all the relevant tools to solve it and find the correct answer".

## Your Task

Produce a comprehensive improvement plan in Markdown followed by a JSON spec block with the following structure:

# Agent Improvement Plan — GAIA Analysis

## Executive Summary
(2-3 sentences: overall failure patterns, what % are premature giving up vs genuine failures)

## Issue Clusters

For each cluster (ordered by frequency/impact, most important first):

### Cluster N: <Short Name>
- **Affected component**: <component>
- **Frequency**: X instances (Y% of failures)
- **Gave up prematurely**: X of Y instances in this cluster had gave_up_prematurely=true
- **Fix scope**: (additive_only / prompt_additive / prompt_restrictive / config_param)
- **Failure categories**: empty_patch / unresolved / error (breakdown)
- **Root cause**: (2-3 sentences: what design decision causes this class of failures)
- **Representative instances**: (list 2-3 instance IDs with one-line description)
- **Proposed fix**:
  - Target file: `<relative path under src/open_deep_research/>`
  - Current behavior: (describe what the code currently does)
  - New behavior: (describe what it should do instead)
  - **Impact on passing tasks**: (explicitly state which task types are NOT affected by this change
    and why — e.g. "This only adds a new file path lookup; tasks without attached files are unchanged")
  - **Regression risk assessment**: (could this change cause agents to give up earlier, over-verify,
    or refuse valid answers on tasks the baseline already handles correctly?)
  - Implementation sketch: (pseudocode or concrete code snippet showing the change)

## Implementation Priority

| Priority | Cluster | Estimated Impact | Regression Risk | Implementation Difficulty |
|----------|---------|-----------------|-----------------|--------------------------|
| P0 | ... | ... | low/medium/high | ... |

## Detailed Implementation Plan

For each cluster (same order as above), provide:

### Fix N: <Name>
**Files to modify**: list files under `src/open_deep_research/` only
**Step-by-step implementation**:
1. (specific change 1)
2. (specific change 2)
...
**Expected outcome**: (what failure mode this eliminates or reduces)
**Regression safeguard**: (what must NOT change to avoid breaking currently-passing tasks)

## Notes for the Modify Agent
(Cross-cutting concerns, order of implementation, and any prompt changes that must be balanced
with persistence guidance to avoid regression)

After the Markdown plan, output the JSON spec block:

```json
{{
  "plan_metadata": {{
    "round_label": "GAIA Round Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  }},
  "edit_budget": {{
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 2,
    "allowed_paths": ["src/open_deep_research/..."],
    "forbidden_paths": [
      "run_gaia_entry.py",
      "src/open_deep_research/scoring.py",
      "src/open_deep_research/scripts/gaia_scorer.py"
    ],
    "rationale": "why this budget is appropriate"
  }},
  "fixes": [
    {{
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["src/open_deep_research/..."],
      "target_symbols": ["AUGMENTED_QUESTION_PREFIX"],
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": [],
      "regression_risks": ["specific scenario where this could cause regression"],
      "must_not_change": ["what existing behavior must be preserved"]
    }}
  ]
}}
```
"""

GAIA_SPEC_SYSTEM_PROMPT = """\
You are converting an already-written open_deep_research improvement plan into a machine-readable implementation spec.

Return exactly one fenced code block:

```json
{ ... }
```

The JSON object MUST follow this schema:
{
  "plan_metadata": {
    "round_label": "GAIA Round Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  },
  "edit_budget": {
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 2,
    "allowed_paths": ["src/open_deep_research/..."],
    "forbidden_paths": [
      "run_gaia_entry.py",
      "src/open_deep_research/scoring.py",
      "src/open_deep_research/scripts/gaia_scorer.py"
    ],
    "rationale": "why this budget is appropriate"
  },
  "fixes": [
    {
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["src/open_deep_research/..."],
      "target_symbols": ["AUGMENTED_QUESTION_PREFIX"],
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": [],
      "regression_risks": ["specific scenario where this could cause regression"],
      "must_not_change": ["what existing behavior must be preserved"]
    }
  ]
}

Rules:
- Output valid JSON only inside the fenced block.
- All paths in `allowed_paths` and `target_files` MUST use `src/open_deep_research/` prefix.
- NEVER include `run_gaia_entry.py`, `scoring.py`, or `gaia_scorer.py` in target_files.
- `regression_risks` must be non-empty for any fix that modifies `prompts.py` — describe exactly
  which currently-passing task types could be harmed.
- If the plan proposes adding "give up" or "unable to determine" guidance to prompts.py, flag
  `risk_level` as "high" and add a regression risk entry describing the premature-giving-up risk.
"""


# ── AppWorld mode prompts ─────────────────────────────────────────────────────

APPWORLD_SYSTEM_PROMPT = """\
You are a senior software engineer analyzing failure patterns in an AppWorld task agent system.

Your job is to:
1. Use the provided harness-layer buckets as coarse organization only
2. Within each layer bucket, cluster the AppWorld failure analyses into recurring semantic harness defects
3. For each semantic cluster, synthesize the common pattern and propose concrete code-level repairs
4. Output a detailed, actionable improvement plan in Markdown followed by a machine-readable JSON spec

The AppWorld agent codebase is at:
  {src_dir}

Key files:
  src/appworld_agent/core.py     — runner, Docker environment, completion protocol, evaluation handoff
  src/appworld_agent/prompts.py  — system prompt, instance prompt, action regex, observation template

Critical AppWorld risks:
- unintended state mutation / collateral damage
- failing to call complete_task() correctly
- wrong API assumptions or missing login flows
- weak answer extraction for answer-seeking tasks
- excessive repeated API exploration without progress
""".format(src_dir=TASK_AGENT_SRC_APPWORLD)

APPWORLD_USER_PROMPT_TEMPLATE = """\
Below are {n} structured failure analyses from an AppWorld task-agent evaluation run.

Failure distribution:
{distribution}

--- ALL FAILURE ANALYSES ---
{analyses}
--- END ---
{val_regression_section}{prev_plan_section}
## Your Task

Produce a comprehensive improvement plan in Markdown followed by a JSON spec block.

# Agent Improvement Plan — AppWorld Analysis

## Executive Summary
(2-3 sentences: dominant defect clusters, especially completion/protocol issues, API misuse, and collateral damage)

## Issue Clusters

For each cluster:
- **Affected component**
- **Frequency**
- **Failure categories**
- **Root cause**
- **Representative instances**
- **Proposed fix**
  - Target file under `src/appworld_agent/`
  - Current behavior
  - New behavior
  - Impact on passing tasks
  - Regression risk assessment

## Implementation Priority

## Detailed Implementation Plan

For each fix:
- files to modify
- step-by-step implementation
- expected outcome
- regression safeguard

After the Markdown plan, output the JSON spec block with this schema:

```json
{{
  "plan_metadata": {{
    "round_label": "AppWorld Round Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  }},
  "edit_budget": {{
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 3,
    "allowed_paths": ["src/appworld_agent/..."],
    "forbidden_paths": ["run_appworld_entry.py", "../eval/**", "../failure_analysis/**"],
    "rationale": "why this budget is appropriate"
  }},
  "fixes": [
    {{
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["src/appworld_agent/..."],
      "target_symbols": ["symbol"],
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": [],
      "regression_risks": ["specific scenario where this could cause regression"],
      "must_not_change": ["what existing behavior must be preserved"]
    }}
  ]
}}
```
"""

APPWORLD_SPEC_SYSTEM_PROMPT = """\
You are converting an AppWorld improvement plan into a machine-readable implementation spec.

Return exactly one fenced JSON code block.

The JSON object MUST follow this schema:
{
  "plan_metadata": {
    "round_label": "AppWorld Round Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  },
  "edit_budget": {
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 3,
    "allowed_paths": ["src/appworld_agent/..."],
    "forbidden_paths": ["run_appworld_entry.py", "../eval/**", "../failure_analysis/**"],
    "rationale": "why this budget is appropriate"
  },
  "fixes": [
    {
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["src/appworld_agent/..."],
      "target_symbols": ["symbol"],
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": [],
      "regression_risks": ["specific scenario where this could cause regression"],
      "must_not_change": ["what existing behavior must be preserved"]
    }
  ]
}

Rules:
- All editable paths must use `src/appworld_agent/` prefix.
- Do not target `run_appworld_entry.py`, evaluator scripts, or failure-analysis pipeline files unless the plan explicitly requires it.
- `regression_risks` must explicitly mention collateral damage risk for any fix that changes completion or state-mutation behavior.
- Keep file scope narrow and avoid broad refactors.
"""


# ── Terminal-Bench mode prompts ───────────────────────────────────────────────

TERMINAL_BENCH_SYSTEM_PROMPT = """\
You are a senior software engineer analyzing failure patterns in Harbor Terminus-2 running Terminal-Bench 2.0.

Your job is to use the provided harness-layer buckets as coarse organization, cluster the diagnoses inside each bucket into semantic root-cause groups, and propose scoped improvements to the local Harbor Terminus-2 task agent/harness path.
The default experiment path must remain Harbor `terminus-2`; do not propose switching to claude-code, codex, gemini-cli, openhands, mini-swe-agent, or other Harbor agents.

Relevant source roots:
  {src_dir}
  {repo_root}/task_agent/terminal_bench_agent

Focus on terminal command strategy, tmux/shell interaction, parser behavior, context management, completion/termination, verifier interpretation, Docker environment setup, and HarnessFix adapter observability.
""".format(src_dir=TASK_AGENT_SRC_TERMINAL_BENCH, repo_root=REPO_ROOT)

TERMINAL_BENCH_USER_PROMPT_TEMPLATE = """\
Below are {n} structured failure analyses from Harbor Terminus-2 on Terminal-Bench 2.0.

Failure distribution:
{distribution}

--- ALL FAILURE ANALYSES ---
{analyses}
--- END ---
{val_regression_section}{prev_plan_section}
## Your Task

Produce a comprehensive improvement plan in Markdown followed by a JSON spec block.

The plan should prioritize the smallest scoped changes that improve Terminal-Bench failures while preserving the official Terminus-2 initial-agent identity.
The modify agent edits a copied `task_agent/terminal_bench_agent` root, so JSON paths must be relative to that root. Editable source should normally be `harbor/src/harbor/agents/terminus_2/...`; adapter-only fixes may use `run_terminal_bench_entry.py` when the defect is observability/format conversion.

After the Markdown plan, output the JSON spec block with this schema:

```json
{{
  "plan_metadata": {{
    "round_label": "Terminal-Bench Round Analysis",
    "summary": "short summary",
    "global_strategy": "how the modify agent should prioritize changes"
  }},
  "edit_budget": {{
    "recommended_budget": "low|medium|high",
    "max_files_to_modify": 3,
    "allowed_paths": ["harbor/src/harbor/agents/terminus_2/...", "run_terminal_bench_entry.py"],
    "forbidden_paths": ["harbor/src/harbor/agents/installed/...", "../eval/**", "../failure_analysis/**"],
    "rationale": "why this budget is appropriate"
  }},
  "fixes": [
    {{
      "id": "fix_1",
      "title": "short name",
      "priority": "P0|P1|P2",
      "target_files": ["harbor/src/harbor/agents/terminus_2/..."],
      "target_symbols": ["symbol"],
      "problem_statement": "current behavior and failure mode",
      "required_behavior_delta": "what must change",
      "implementation_steps": ["step 1", "step 2"],
      "tests": ["verification step"],
      "risk_level": "low|medium|high",
      "dependencies": [],
      "regression_risks": ["specific scenario where this could cause regression"],
      "must_not_change": ["do not route to non-Terminus agents"]
    }}
  ]
}}
```
"""

TERMINAL_BENCH_SPEC_SYSTEM_PROMPT = """\
You are converting a Terminal-Bench/Harbor Terminus-2 improvement plan into a machine-readable implementation spec.

Return exactly one fenced JSON code block. All fixes must preserve the default initial agent as Harbor `terminus-2`; never target claude-code, codex, gemini-cli, openhands, mini-swe-agent, or other Harbor installed agents.

Allowed target paths should use `harbor/src/harbor/agents/terminus_2/...` for Terminus-2 changes or `run_terminal_bench_entry.py` for HarnessFix adapter changes, relative to a copied `task_agent/terminal_bench_agent` root.
"""


SPEC_USER_PROMPT_TEMPLATE = """\
Convert the following Markdown improvement plan into the required JSON implementation spec.

Context:
- Task agent source root: {src_dir}
- Total analyzed failures: {n}
- Component distribution:
{distribution}

Markdown plan:
--- PLAN START ---
{plan_text}
--- PLAN END ---
"""


def call_llm(model: str, system: str, user: str) -> str:
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **MODEL_KWARGS,
    )
    return response.choices[0].message.content or ""


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _analysis_source_dirs(results: list[dict]) -> list[str]:
    dirs: list[Path] = []
    for record in results:
        source = _record_source_dir(record)
        if source:
            dirs.append(Path(source).expanduser().resolve())
    return _ordered_unique([str(path) for path in dirs])


def _source_roots_for_mode(mode: str, results: list[dict] | None = None) -> list[str]:
    dynamic_roots = _analysis_source_dirs(results or [])
    if mode == "gaia":
        return _ordered_unique(dynamic_roots + [TASK_AGENT_SRC_GAIA])
    if mode == "appworld":
        return _ordered_unique(dynamic_roots + [
            TASK_AGENT_SRC_APPWORLD,
            str(REPO_ROOT / "task_agent" / "appworld_agent" / "appworld_official_agents"),
        ])
    if mode == "terminal_bench":
        return _ordered_unique(dynamic_roots + [
            str(REPO_ROOT / "task_agent" / "terminal_bench_agent"),
            TASK_AGENT_SRC_TERMINAL_BENCH,
        ])
    return _ordered_unique(dynamic_roots + [TASK_AGENT_SRC_SWE])


def _candidate_source_file(root: Path, rel_parts: tuple[str, ...]) -> str:
    return str(root.joinpath(*rel_parts))


def _key_files_for_mode(mode: str, results: list[dict] | None = None) -> list[str]:
    dynamic_roots = [Path(root) for root in _analysis_source_dirs(results or [])]
    files: list[str] = []
    if mode == "gaia":
        for root in dynamic_roots:
            files.extend([
                _candidate_source_file(root, ("config.py",)),
                _candidate_source_file(root, ("prompts.py",)),
                _candidate_source_file(root, ("agent.py",)),
                _candidate_source_file(root, ("tools.py",)),
                _candidate_source_file(root, ("browser.py",)),
                _candidate_source_file(root, ("scripts", "reformulator.py")),
            ])
        root = REPO_ROOT / "task_agent" / "open_deep_research"
        files.extend([
            str(root / "src" / "open_deep_research" / "config.py"),
            str(root / "src" / "open_deep_research" / "prompts.py"),
            str(root / "src" / "open_deep_research" / "agent.py"),
            str(root / "src" / "open_deep_research" / "tools.py"),
            str(root / "src" / "open_deep_research" / "browser.py"),
            str(root / "src" / "open_deep_research" / "scripts" / "reformulator.py"),
        ])
        return _ordered_unique(files)
    if mode == "appworld":
        for root in dynamic_roots:
            files.extend([
                _candidate_source_file(root, ("core.py",)),
                _candidate_source_file(root, ("prompts.py",)),
                _candidate_source_file(root, ("official_react_adapter.py",)),
            ])
        root = REPO_ROOT / "task_agent" / "appworld_agent"
        files.extend([
            str(root / "src" / "appworld_agent" / "core.py"),
            str(root / "src" / "appworld_agent" / "prompts.py"),
            str(root / "src" / "appworld_agent" / "official_react_adapter.py"),
        ])
        return _ordered_unique(files)
    if mode == "terminal_bench":
        for root in dynamic_roots:
            files.extend([
                _candidate_source_file(root, ("harbor", "src", "harbor", "agents", "terminus_2", "terminus_2.py")),
                _candidate_source_file(root, ("harbor", "src", "harbor", "agents", "terminus_2", "templates", "terminus-json-plain.txt")),
                _candidate_source_file(root, ("run_terminal_bench_entry.py",)),
                _candidate_source_file(root, ("harbor", "src", "harbor", "trial", "trial.py")),
                _candidate_source_file(root, ("harbor", "src", "harbor", "verifier")),
            ])
        root = REPO_ROOT / "task_agent" / "terminal_bench_agent"
        files.extend([
            str(root / "harbor" / "src" / "harbor" / "agents" / "terminus_2" / "terminus_2.py"),
            str(root / "harbor" / "src" / "harbor" / "agents" / "terminus_2" / "templates" / "terminus-json-plain.txt"),
            str(root / "run_terminal_bench_entry.py"),
            str(root / "harbor" / "src" / "harbor" / "trial" / "trial.py"),
            str(root / "harbor" / "src" / "harbor" / "verifier"),
        ])
        return _ordered_unique(files)
    for root in dynamic_roots:
        files.extend([
            _candidate_source_file(root, ("agents", "default.py")),
            _candidate_source_file(root, ("config", "benchmarks", "swebench.yaml")),
            _candidate_source_file(root, ("models", "litellm_model.py")),
            _candidate_source_file(root, ("models", "utils", "actions_text.py")),
            _candidate_source_file(root, ("environments", "local.py")),
            _candidate_source_file(root, ("environments", "docker.py")),
        ])
    root = REPO_ROOT / "task_agent" / "mini-swe-agent"
    files.extend([
        str(root / "src" / "minisweagent" / "agents" / "default.py"),
        str(root / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml"),
        str(root / "src" / "minisweagent" / "models" / "litellm_model.py"),
        str(root / "src" / "minisweagent" / "models" / "utils" / "actions_text.py"),
        str(root / "src" / "minisweagent" / "environments" / "local.py"),
        str(root / "src" / "minisweagent" / "environments" / "docker.py"),
    ])
    return _ordered_unique(files)


def _aggregate_context_dir(output_path: Path) -> Path:
    safe_name = output_path.stem.replace("/", "__")
    return Path(__file__).parent / "results" / "aggregate_context" / safe_name


def _write_aggregate_context_files(
    *,
    context_dir: Path,
    operator_str: str,
    cluster_str: str,
    memory_str: str,
    analyses_str: str,
    distribution_str: str,
    val_regression_section: str,
    prev_plan_section: str,
) -> dict[str, Path]:
    context_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "operator_registry": context_dir / "operator_registry.txt",
        "layer_buckets": context_dir / "layer_buckets.txt",
        "memory": context_dir / "memory.txt",
        "analyses": context_dir / "analyses.txt",
        "distribution": context_dir / "distribution.txt",
        "val_regressions": context_dir / "val_regressions.txt",
        "previous_context": context_dir / "previous_context.txt",
    }
    files["operator_registry"].write_text(operator_str + "\n")
    files["layer_buckets"].write_text(cluster_str + "\n")
    files["memory"].write_text(memory_str + "\n")
    files["analyses"].write_text(analyses_str + "\n")
    files["distribution"].write_text(distribution_str + "\n")
    files["val_regressions"].write_text((val_regression_section or "(none)").strip() + "\n")
    files["previous_context"].write_text((prev_plan_section or "(none)").strip() + "\n")
    return files


AGGREGATE_AGENT_SYSTEM_TEMPLATE = """\
You are an expert agent-harness repair planner. You synthesize benchmark failure analyses into a scoped repair plan.

You are operating in a local shell. You MAY and SHOULD inspect source code before writing the plan. Use exactly one
`<mswea_bash_command>` block per turn.

Important constraints:
- The files in AGGREGATE_CONTEXT_DIR contain compact analysis records, coarse harness-layer buckets, memory, and operator registry.
- The buckets are coarse layer-only organization, not final root-cause clusters. You must cluster semantically inside each layer.
- Ground fixes in source code you inspect, not only in high-level summaries.
- Do not modify any files. This stage only writes the final plan/spec to /tmp and submits it.
- Preserve the JSON implementation spec schema exactly. The fenced ```json block must be included after the Markdown plan.
- Keep target_files and allowed_paths narrow and compatible with the operator registry.

Response format every turn:
THOUGHT: concise reasoning
<mswea_bash_command>command</mswea_bash_command>
"""


AGGREGATE_AGENT_INSTANCE_TEMPLATE = """\
Create the improvement plan and machine-readable JSON implementation spec for mode={{mode}}.

Context files:
- Distribution: {{distribution_path}}
- Coarse harness-layer buckets: {{cluster_path}}
- Compact analyses: {{analyses_path}}
- Typed operator registry: {{operator_path}}
- Harness memory: {{memory_path}}
- Val regressions: {{val_regression_path}}
- Previous iteration context: {{previous_context_path}}

Source roots you may inspect:
{{source_roots}}

Recommended source files to inspect before planning:
{{key_files}}

First steps:
1. Read the distribution, layer buckets, operator registry, memory, and enough compact analyses to understand the failure patterns.
2. Inspect the relevant source files for the highest-frequency layer buckets and likely fix targets.
3. Write the final Markdown plan followed by one fenced ```json block to `/tmp/aggregate_plan.md`.
4. Submit `/tmp/aggregate_plan.md`.

The plan must use the same high-level instructions as this system prompt:
--- MODE-SPECIFIC PLANNING PROMPT START ---
{{mode_system_prompt}}
--- MODE-SPECIFIC PLANNING PROMPT END ---

The output JSON object must include plan_metadata, edit_budget, and fixes. Each fix must include:
id, title, priority, target_files, target_symbols, problem_statement, required_behavior_delta,
implementation_steps, tests, risk_level, dependencies, regression_risks, must_not_change.
Each fix should also include active_files, must_inspect_files, and do_not_edit_until_inspected=true;
the deterministic spec enrichment step will add missing values from the operator registry.

When finished, first write the complete response to `/tmp/aggregate_plan.md`, then submit with:
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/aggregate_plan.md
"""


def call_aggregate_agent(
    *,
    model: str,
    mode: str,
    mode_system_prompt: str,
    context_files: dict[str, Path],
    output_path: Path,
    results: list[dict] | None = None,
) -> str:
    source_roots = "\n".join(f"- {root}" for root in _source_roots_for_mode(mode, results))
    key_files = "\n".join(f"- {item}" for item in _key_files_for_mode(mode, results))
    model = LitellmTextbasedModel(
        model_name=model,
        observation_template="""
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
<output>
{{ output.output }}
</output>
""",
        format_error_template="You must provide exactly one <mswea_bash_command> block.",
        action_regex=r"<mswea_bash_command>(.*?)</mswea_bash_command>",
        model_kwargs=MODEL_KWARGS,
        cost_tracking="ignore_errors",
    )
    env = LocalEnvironment(
        env={
            "AGGREGATE_CONTEXT_DIR": str(context_files["analyses"].parent),
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
        },
        timeout=120,
    )
    traj_path = output_path.with_suffix(output_path.suffix + ".aggregate_agent.traj.json")
    agent = DefaultAgent(
        model,
        env,
        output_path=traj_path,
        step_limit=AGG_AGENT_STEP_LIMIT,
        cost_limit=AGG_AGENT_COST_LIMIT,
        system_template=AGGREGATE_AGENT_SYSTEM_TEMPLATE,
        instance_template=AGGREGATE_AGENT_INSTANCE_TEMPLATE,
    )
    result = agent.run(
        mode=mode,
        distribution_path=str(context_files["distribution"]),
        cluster_path=str(context_files["layer_buckets"]),
        analyses_path=str(context_files["analyses"]),
        operator_path=str(context_files["operator_registry"]),
        memory_path=str(context_files["memory"]),
        val_regression_path=str(context_files["val_regressions"]),
        previous_context_path=str(context_files["previous_context"]),
        source_roots=source_roots,
        key_files=key_files,
        mode_system_prompt=mode_system_prompt,
    )
    submission = result.get("submission", "")
    if not submission or len(submission) < 200:
        raise ValueError(f"Aggregate agent returned empty or very short submission: {submission!r}")
    return submission


def extract_json_block(text: str) -> dict:
    marker = "```json"
    start = text.find(marker)
    if start == -1:
        raise ValueError("LLM output missing ```json block with implementation spec")
    end = text.find("```", start + len(marker))
    if end == -1:
        raise ValueError("LLM output has unterminated ```json block")
    json_block = text[start + len(marker):end].strip()
    spec = json.loads(json_block)
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    """Validate required top-level fields for the implementation spec."""
    required_top = {"plan_metadata", "edit_budget", "fixes"}
    missing = required_top - set(spec)
    if missing:
        raise ValueError(f"Implementation spec missing keys: {sorted(missing)}")

    budget = spec["edit_budget"]
    for key in ("recommended_budget", "max_files_to_modify", "allowed_paths", "forbidden_paths", "rationale"):
        if key not in budget:
            raise ValueError(f"Implementation spec edit_budget missing key: {key}")
    budget.setdefault("active_files", [])
    budget.setdefault("must_inspect_files", [])
    budget.setdefault("do_not_edit_until_inspected", True)

    fixes = spec["fixes"]
    if not isinstance(fixes, list) or not fixes:
        raise ValueError("Implementation spec must contain a non-empty fixes list")

    for idx, fix in enumerate(fixes, start=1):
        for key in (
            "id", "title", "priority", "target_files", "target_symbols",
            "problem_statement", "required_behavior_delta", "implementation_steps",
            "tests", "risk_level", "dependencies", "regression_risks", "must_not_change",
        ):
            if key not in fix:
                raise ValueError(f"Fix #{idx} missing key: {key}")
        fix.setdefault("active_files", [])
        fix.setdefault("must_inspect_files", [])
        fix.setdefault("do_not_edit_until_inspected", True)


def split_plan_and_spec(text: str) -> tuple[str, dict]:
    """Split LLM output into markdown plan and JSON spec."""
    marker = "```json"
    start = text.find(marker)
    if start == -1:
        raise ValueError("LLM output missing ```json block with implementation spec")
    end = text.find("```", start + len(marker))
    if end == -1:
        raise ValueError("LLM output has unterminated ```json block")

    plan_text = text[:start].rstrip() + "\n"
    json_block = text[start + len(marker):end].strip()
    spec = json.loads(json_block)
    validate_spec(spec)
    return plan_text, spec


def generate_spec_from_plan(model: str, plan_text: str, distribution_str: str, n: int,
                            mode: str = "swe", source_root_override: str | None = None) -> dict:
    if mode == "gaia":
        spec_system = GAIA_SPEC_SYSTEM_PROMPT
        src_dir = TASK_AGENT_SRC_GAIA
    elif mode == "appworld":
        spec_system = APPWORLD_SPEC_SYSTEM_PROMPT
        src_dir = TASK_AGENT_SRC_APPWORLD
    elif mode == "terminal_bench":
        spec_system = TERMINAL_BENCH_SPEC_SYSTEM_PROMPT
        src_dir = TASK_AGENT_SRC_TERMINAL_BENCH
    else:
        spec_system = SPEC_SYSTEM_PROMPT
        src_dir = TASK_AGENT_SRC_SWE
    if source_root_override:
        src_dir = source_root_override
        spec_system += (
            "\n\nCurrent analyzed source root for this run. Prefer this over static original paths: "
            + source_root_override
        )
    user_prompt = SPEC_USER_PROMPT_TEMPLATE.format(
        src_dir=src_dir,
        n=n,
        distribution=distribution_str,
        plan_text=plan_text,
    )
    response_text = call_llm(model, spec_system, user_prompt)
    if not response_text or len(response_text) < 40:
        raise ValueError(f"Spec generation returned empty or too short response: {response_text!r}")
    return extract_json_block(response_text)


def load_val_regression_analyses(path: Path) -> list[dict]:
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate failure analyses into improvement plan")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["swe", "gaia", "appworld", "terminal_bench"], default="swe",
                        help="Agent system mode: swe, gaia, appworld, or terminal_bench (default: swe)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing plan file")
    parser.add_argument(
        "--output", "-o", type=Path, default=IMPROVEMENT_PLAN_PATH,
        help="Output path for improvement plan (default: failure_analysis/improvement_plan.md)",
    )
    parser.add_argument(
        "--spec-output", type=Path, default=IMPROVEMENT_SPEC_PATH,
        help="Output path for machine-readable implementation spec JSON",
    )
    parser.add_argument(
        "--val-analyses", type=Path, default=None,
        help="Path to val regression analyses JSONL from a previous iteration",
    )
    parser.add_argument(
        "--prev-plan", type=Path, default=None,
        help="Path to previous iteration's improvement plan (for refinement context)",
    )
    parser.add_argument(
        "--prev-iteration-report", type=Path, default=None,
        help="Path to previous iteration report JSON with plan/diff/train/val/promotion outcomes",
    )
    parser.add_argument(
        "--results-file", type=Path, default=ALL_RESULTS_PATH,
        help="Path to failure analysis JSONL to aggregate",
    )
    parser.add_argument(
        "--direct-llm", action="store_true",
        help="Use the legacy single-call LLM planner instead of the agentic source-reading planner",
    )
    args = parser.parse_args()

    # Select prompts based on mode
    if args.mode == "gaia":
        active_system_prompt = GAIA_SYSTEM_PROMPT
        active_user_template = GAIA_USER_PROMPT_TEMPLATE
    elif args.mode == "appworld":
        active_system_prompt = APPWORLD_SYSTEM_PROMPT
        active_user_template = APPWORLD_USER_PROMPT_TEMPLATE
    elif args.mode == "terminal_bench":
        active_system_prompt = TERMINAL_BENCH_SYSTEM_PROMPT
        active_user_template = TERMINAL_BENCH_USER_PROMPT_TEMPLATE
    else:
        active_system_prompt = SYSTEM_PROMPT
        active_user_template = USER_PROMPT_TEMPLATE

    output_path = args.output
    spec_output_path = args.spec_output

    if output_path.exists() and spec_output_path.exists() and not args.force:
        print(f"{output_path.name} already exists ({output_path.stat().st_size} bytes).")
        print("Use --force to regenerate.")
        return

    print(f"Loading results from {args.results_file} ...")
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results.")
    clusters = consolidate_diagnoses(results, mode=args.mode, min_frequency=1)

    # Print distribution
    dist = Counter(r.get("affected_component") for r in results)
    dist_by_cat = Counter(r.get("failure_category") for r in results)
    dist_by_defect = Counter(normalize_defect_class(r.get("defect_class")) or "unknown" for r in results)
    print("\nComponent distribution:")
    for k, v in dist.most_common():
        print(f"  {k}: {v}")
    print("\nCategory distribution:")
    for k, v in dist_by_cat.most_common():
        print(f"  {k}: {v}")
    print("\nDefect distribution:")
    for k, v in dist_by_defect.most_common():
        print(f"  {k}: {v}")

    analysis_source_dirs = _analysis_source_dirs(results)
    source_dirs = _source_roots_for_mode(args.mode, results)
    current_source_dirs = analysis_source_dirs or source_dirs
    print("\nAnalysis source roots:")
    for source_dir in current_source_dirs:
        print(f"  {source_dir}")
    fallback_source_dirs = [item for item in source_dirs if item not in current_source_dirs]
    if fallback_source_dirs:
        print("\nFallback source roots available for reference:")
        for source_dir in fallback_source_dirs:
            print(f"  {source_dir}")

    distribution_str = "\n".join(
        f"  {k}: {v} instances" for k, v in dist.most_common()
    ) + "\n" + "\n".join(
        f"  {k}: {v} instances" for k, v in dist_by_cat.most_common()
    ) + "\n" + "\n".join(
        f"  defect:{k}: {v} instances" for k, v in dist_by_defect.most_common()
    )
    if source_dirs:
        distribution_str += "\n" + "\n".join(f"  source:{item}" for item in source_dirs)
    if source_dirs:
        active_system_prompt += (
            "\n\nCurrent analyzed source roots for this run. Prefer these over any static original paths above:\n"
            + "\n".join(f"- {item}" for item in source_dirs)
        )
    analyses_str = format_results_for_llm(results)
    cluster_str = format_clusters_for_prompt(clusters)
    operator_str = format_operator_registry_for_prompt(args.mode)
    memory_root = default_memory_root(REPO_ROOT)
    memory_entries = retrieve_relevant_memories(memory_root, args.mode, clusters, limit=AGG_MEMORY_LIMIT)
    memory_str = format_memories_for_prompt(memory_entries)

    # Optional: val regression context from the previous iteration.
    val_regression_section = ""
    if args.val_analyses and args.val_analyses.exists():
        val_results = load_val_regression_analyses(args.val_analyses)
        val_str = format_results_for_llm(val_results)
        val_regression_section = f"""
--- VAL SET REGRESSIONS ({len(val_results)} instances that your PREVIOUS improvements BROKE) ---
These instances were solved by the original agent but failed after your modifications.
You MUST ensure your new plan does not cause these regressions.
{val_str}
--- END VAL REGRESSIONS ---
"""
        print(f"Loaded {len(val_results)} val regression analyses from {args.val_analyses}")

    # Optional: previous plan context
    prev_plan_section = ""
    if args.prev_plan and args.prev_plan.exists():
        prev_plan_text = compact_text(args.prev_plan.read_text(), AGG_PREV_CONTEXT_CHARS)
        prev_plan_section = f"""
--- PREVIOUS IMPROVEMENT PLAN (for reference — refine, do not repeat mistakes) ---
{prev_plan_text}
--- END PREVIOUS PLAN ---
"""
        print(f"Loaded previous plan from {args.prev_plan}")

    if args.prev_iteration_report and args.prev_iteration_report.exists():
        prev_report_text = compact_text(args.prev_iteration_report.read_text(), AGG_PREV_CONTEXT_CHARS)
        prev_plan_section += f"""
--- PREVIOUS ITERATION REPORT (outcome, edits, train/val effects, promotion decision) ---
Use this as mandatory context. Preserve improvements that worked, explain and repair regressions/errors,
and do not repeat edits that caused the recorded side effects.
{prev_report_text}
--- END PREVIOUS ITERATION REPORT ---
"""
        print(f"Loaded previous iteration report from {args.prev_iteration_report}")

    user_prompt = f"""
Typed operator registry:
{operator_str}

Coarse harness-layer buckets (not final root-cause clusters):
{cluster_str}

Treat these buckets as a starting point. You must perform the finer semantic clustering inside each layer bucket using the individual evidence records and analyses below. Do not assume defect/operator distributions are final cluster labels.

Harness memory (accepted/rejected repairs with outcomes):
{memory_str}

""" + active_user_template.format(
        n=len(results),
        distribution=distribution_str,
        analyses=analyses_str,
        val_regression_section=val_regression_section,
        prev_plan_section=prev_plan_section,
    )

    print(f"\nPreparing aggregate planner context (mode={args.mode}) ...")
    print(f"Prompt/context size: ~{len(user_prompt)//1000}K chars")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_response_path = output_path.with_suffix(output_path.suffix + ".raw.txt")
    context_files = _write_aggregate_context_files(
        context_dir=_aggregate_context_dir(output_path),
        operator_str=operator_str,
        cluster_str=cluster_str,
        memory_str=memory_str,
        analyses_str=analyses_str,
        distribution_str=distribution_str,
        val_regression_section=val_regression_section,
        prev_plan_section=prev_plan_section,
    )
    if args.direct_llm:
        print(f"Calling legacy direct LLM planner ({args.model}) ...")
        response_text = call_llm(args.model, active_system_prompt, user_prompt)
    else:
        print(f"Calling aggregate planning agent ({args.model}) ...")
        response_text = call_aggregate_agent(
            model=args.model,
            mode=args.mode,
            mode_system_prompt=active_system_prompt,
            context_files=context_files,
            output_path=output_path,
            results=results,
        )

    if not response_text or len(response_text) < 200:
        print(f"ERROR: aggregate planner returned empty or very short response: {response_text!r}")
        sys.exit(1)

    raw_response_path.write_text(response_text)

    try:
        plan_text, spec = split_plan_and_spec(response_text)
        print("Parsed Markdown plan and JSON spec from a single response.")
    except Exception as e:
        print(f"WARNING: failed to parse implementation spec from aggregate response: {e}")
        print(f"Saved raw aggregate response to {raw_response_path}")
        plan_text = response_text.strip() + "\n"
        print(f"Calling LLM ({args.model}) again to convert plan into JSON spec ...")
        spec = generate_spec_from_plan(
            args.model,
            plan_text=plan_text,
            distribution_str=distribution_str,
            n=len(results),
            mode=args.mode,
            source_root_override=current_source_dirs[0] if current_source_dirs else None,
        )

    spec = enrich_spec_with_clusters(spec, clusters, args.mode)

    output_path.write_text(plan_text)
    spec_output_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    clusters_path = output_path.with_suffix(".clusters.json")
    clusters_path.write_text(json.dumps(clusters, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved improvement plan to {output_path}")
    print(f"Saved implementation spec to {spec_output_path}")
    print(f"Saved cluster records to {clusters_path}")
    print(f"Saved raw aggregate response to {raw_response_path}")
    print(f"Plan size: {len(plan_text)} chars")
    print(f"Fixes in spec: {len(spec['fixes'])}")
    print("\nPlan preview:")
    print(plan_text)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print a bounded HTIR projection for failure-analysis agents.

The full HTIR bundle can be very large for long SWE traces. This script keeps
the attribution-critical pieces visible without feeding the complete graph back
into the analysis model context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clip(value: Any, limit: int = 900) -> Any:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= limit else text[: limit - 3] + "..."
    if isinstance(value, list):
        return [_clip(item, limit) for item in value]
    if isinstance(value, dict):
        return {str(key): _clip(item, limit) for key, item in value.items()}
    return value


def _select_steps(steps: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(steps) <= limit:
        return steps
    head_count = min(3, max(1, limit // 3))
    tail_count = max(1, limit - head_count)
    selected = steps[:head_count] + steps[-tail_count:]
    omitted = len(steps) - len(selected)
    marker = {
        "step_id": "...",
        "role": "omitted",
        "summary": f"{omitted} intermediate AgentTraceStep records omitted",
    }
    return selected[:head_count] + [marker] + selected[head_count:]


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "step_id": step.get("step_id"),
        "role": step.get("role"),
        "execution_status": step.get("execution_status"),
        "source_refs": step.get("source_refs") or step.get("source_ref"),
        "legacy_event_ids": step.get("legacy_event_ids"),
        "summary": step.get("summary"),
    }
    if step.get("request_summary"):
        keep["request_summary"] = step.get("request_summary")
    if step.get("response_summary"):
        keep["response_summary"] = step.get("response_summary")
    if step.get("action_summary"):
        keep["action_summary"] = step.get("action_summary")
    return {key: value for key, value in keep.items() if value not in (None, "", [], {})}


def _compact_facet(facet: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_and_context": facet.get("data_and_context", {}),
        "control_flow": facet.get("control_flow", {}),
        "artifact_state": facet.get("artifact_state", {}),
        "harness_layer": facet.get("harness_layer", {}),
        "evidence_refs": (facet.get("evidence_refs") or [])[:4],
    }


def build_compact_bundle(bundle: dict[str, Any], step_limit: int) -> dict[str, Any]:
    steps = bundle.get("agent_trace_steps") or []
    selected_steps = _select_steps(steps, step_limit)
    selected_step_ids = {
        step.get("step_id")
        for step in selected_steps
        if isinstance(step, dict) and step.get("step_id") and step.get("step_id") != "..."
    }
    facets = bundle.get("node_facets") or {}
    selected_facets = {
        step_id: _compact_facet(facet)
        for step_id, facet in facets.items()
        if step_id in selected_step_ids and isinstance(facet, dict)
    }
    graph = bundle.get("base_graph") or {}
    return {
        "mode": bundle.get("mode"),
        "instance_id": bundle.get("instance_id"),
        "failure_category": bundle.get("failure_category"),
        "htir_version": bundle.get("htir_version"),
        "stats": bundle.get("stats"),
        "outcome_anchors": bundle.get("outcome_anchors") or [],
        "agent_trace_steps": [_compact_step(step) for step in selected_steps],
        "base_graph": {
            "edge_families": graph.get("edge_families"),
            "anchor_links": graph.get("anchor_links"),
            "step_node_count": graph.get("step_node_count"),
        },
        "node_facets": selected_facets,
        "views": {
            "failure": (bundle.get("views") or {}).get("failure", [])[:8],
            "causal": (bundle.get("views") or {}).get("causal", [])[:12],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("htir_path", type=Path)
    parser.add_argument("--step-limit", type=int, default=16)
    parser.add_argument("--char-limit", type=int, default=45000)
    args = parser.parse_args()

    bundle = json.loads(args.htir_path.read_text(errors="replace"))
    compact = _clip(build_compact_bundle(bundle, args.step_limit))
    text = json.dumps(compact, ensure_ascii=False, indent=2)
    if args.char_limit > 0 and len(text) > args.char_limit:
        text = text[: args.char_limit - 80] + "\n... [compact HTIR output truncated]\n"
    print(text)


if __name__ == "__main__":
    main()

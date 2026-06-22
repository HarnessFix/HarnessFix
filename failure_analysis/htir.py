from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any


NODE_TYPES = (
    "TaskSpecRecord",
    "ContextAssemblyEvent",
    "ModelInvocationEvent",
    "ParserEvent",
    "ToolCallEvent",
    "ToolResultRecord",
    "GuardrailEvent",
    "ArtifactRecord",
    "StateDeltaRecord",
    "VerificationEvent",
    "OrchestrationEvent",
    "SubmissionEvent",
    "ExceptionEvent",
)

# Backward-compatible alias for older scripts that imported EVENT_TYPES.
EVENT_TYPES = NODE_TYPES


def _clean_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _load_optional_text(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(errors="replace")


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


CONTEXT_MATCH_MIN_CHARS = 40
CONTEXT_MATCH_MAX_CHARS = 240
CONTEXT_MATCH_MAX_SOURCES = 6
CONTEXT_MATCH_TYPES = {"TaskSpecRecord", "ContextAssemblyEvent", "ModelInvocationEvent", "ToolResultRecord"}


def _flatten_text(value: Any, *, limit: int = 8000) -> str:
    if value in (None, {}, []):
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _fingerprint_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _context_probe(text: str) -> str:
    clean = _fingerprint_text(text)
    if len(clean) < CONTEXT_MATCH_MIN_CHARS:
        return ""
    if len(clean) <= CONTEXT_MATCH_MAX_CHARS:
        return clean
    head = clean[:CONTEXT_MATCH_MAX_CHARS]
    if " " in head:
        head = head.rsplit(" ", 1)[0]
    return head.strip()


def _request_context_text_from_attrs(attrs: dict[str, Any]) -> str:
    parts = []
    for key in ("_request_context_text", "request_message", "request_summary"):
        value = attrs.get(key)
        if value not in (None, {}, [], ""):
            parts.append(_flatten_text(value))
    return " ".join(part for part in parts if part)


def _node_context_output_text(node: dict[str, Any]) -> str:
    attrs = node.get("attributes", {}) or {}
    node_type = node.get("type")
    if node_type == "ModelInvocationEvent":
        values = [attrs.get("response_summary"), node.get("summary"), attrs.get("response_message")]
    elif node_type == "ToolResultRecord":
        values = [node.get("summary"), attrs.get("observation")]
    elif node_type in {"TaskSpecRecord", "ContextAssemblyEvent"}:
        values = [node.get("summary")]
    else:
        values = [node.get("summary")]
    for value in values:
        text = _flatten_text(value)
        if len(_fingerprint_text(text)) >= CONTEXT_MATCH_MIN_CHARS:
            return text
    return ""


def _infer_request_context_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Link model requests to prior trace outputs that are copied into their context.

    The benchmark adapters store API-facing request_messages for model calls. This
    pass does a conservative substring/probe match from each current request back
    to previous task/context/model/tool-result nodes. It records candidate context
    sources as `context-dependency` edges; downstream attribution still decides
    which of these candidates are actually responsible.
    """
    existing = {
        (str(edge.get("from")), str(edge.get("to")), str(edge.get("relation")))
        for edge in edges
    }
    prior_sources: list[dict[str, Any]] = []
    for node in sorted(nodes, key=_node_sort_key):
        node_id = _node_id(node)
        node_type = node.get("type")
        if node_type == "ModelInvocationEvent":
            attrs = node.get("attributes", {}) or {}
            request_text = _fingerprint_text(_request_context_text_from_attrs(attrs))
            if request_text:
                matches: list[dict[str, Any]] = []
                for source in reversed(prior_sources):
                    probe = source.get("probe", "")
                    if not probe or probe not in request_text:
                        continue
                    edge_key = (source["node_id"], node_id, "context-dependency")
                    if edge_key in existing:
                        continue
                    matches.append(source)
                    if len(matches) >= CONTEXT_MATCH_MAX_SOURCES:
                        break
                for source in reversed(matches):
                    attrs = {
                        "match_type": "request_contains_prior_output",
                        "source_type": source.get("type"),
                        "source_step_id": source.get("step_id"),
                        "matched_text": _clean_text(source.get("probe", ""), 180),
                    }
                    _add_edge(edges, source["node_id"], node_id, "context-dependency", attrs)
                    existing.add((source["node_id"], node_id, "context-dependency"))
        if node_type in CONTEXT_MATCH_TYPES:
            output_text = _node_context_output_text(node)
            probe = _context_probe(output_text)
            if probe:
                prior_sources.append({
                    "node_id": node_id,
                    "type": node_type,
                    "step_id": node.get("step_id"),
                    "source_ref": node.get("source_ref"),
                    "summary": node.get("summary"),
                    "probe": probe,
                })


def _node(event_id: str, node_type: str, step_id: int, component: str, summary: str, source_ref: str,
          attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "node_id": event_id,
        "type": node_type,
        "step_id": step_id,
        "component": component,
        "summary": _clean_text(summary, 260),
        "source_ref": source_ref,
        "attributes": attributes or {},
    }


_event = _node  # Compatibility for code paths that still use event terminology.


def _add_edge(
    edges: list[dict[str, Any]],
    from_id: str,
    to_id: str,
    relation: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    edge = {"from": from_id, "to": to_id, "relation": relation}
    if attributes:
        edge["attributes"] = attributes
    edges.append(edge)


def _append_context_node(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    parent_id: str,
    step_id: int,
    summary: str,
    source_ref: str,
    attributes: dict[str, Any] | None = None,
) -> str:
    node_id = f"evt_{len(nodes):03d}"
    nodes.append(_event(node_id, "ContextAssemblyEvent", step_id, "prompt", summary, source_ref, attributes))
    _add_edge(edges, parent_id, node_id, "context-dependency")
    return node_id


def _append_parser_node(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    parent_id: str,
    step_id: int,
    summary: str,
    source_ref: str,
    attributes: dict[str, Any] | None = None,
) -> str:
    node_id = f"evt_{len(nodes):03d}"
    nodes.append(_event(node_id, "ParserEvent", step_id, "parser", summary, source_ref, attributes))
    _add_edge(edges, parent_id, node_id, "control-flow")
    return node_id


def _command_paths(command: str) -> list[str]:
    return sorted(set(re.findall(r"[\w./-]+\.(?:py|txt|md|json|yaml|yml|patch)", command)))


def _looks_like_edit(command: str) -> bool:
    return any(token in command for token in ("sed -i", "cat <<", "cat >", "python3 <<", "apply_patch", "perl -0pi"))


def _looks_like_verification(command: str) -> bool:
    return any(token in command for token in ("pytest", "py_compile", "unittest", "tox", "runtests.py", "nose"))


def _looks_like_submission(command: str) -> bool:
    return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command


def _looks_like_exception(output: str, exception_info: str) -> bool:
    text = f"{output}\n{exception_info}".lower()
    return any(token in text for token in ("traceback", "syntaxerror", "indentationerror", "timeout", "exception"))


def _extract_evaluator_anchor(source_ref: str, summary: str, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs = attributes or {}
    digest = hashlib.sha1(f"{source_ref}\n{summary}".encode("utf-8")).hexdigest()[:10]
    return {
        "anchor_id": f"anchor_{digest}",
        "source_ref": source_ref,
        "summary": _clean_text(summary, 260),
        "attributes": attrs,
    }


def _swe_component_for_command(command: str) -> str:
    lower = command.lower()
    if _looks_like_submission(command):
        return "submitter"
    if _looks_like_verification(command):
        return "validator"
    if _looks_like_edit(command):
        return "tool"
    if "git diff" in lower or "git add" in lower or "git checkout" in lower:
        return "controller"
    return "tool"


def _gaia_component_for_step(step: dict[str, Any]) -> str:
    text = json.dumps(step, ensure_ascii=False).lower()
    if "final_answer" in text or "prediction" in text:
        return "submitter"
    if "search" in text or "browser" in text or "visittool" in text:
        return "tool"
    if "textinspector" in text or "file" in text:
        return "tool"
    if "plan" in text or "managed agent" in text:
        return "orchestrator"
    return "controller"


def _appworld_component_for_command(command: str) -> str:
    lower = command.lower()
    if "complete_task" in lower:
        return "submitter"
    if "show_api_doc" in lower or "show_api_descriptions" in lower:
        return "tool"
    if any(token in lower for token in ("login(", "requester", "apis.")):
        return "tool"
    if "print(" in lower:
        return "controller"
    return "tool"


def _gaia_context_summary(question: str, true_answer: str, predicted_answer: str) -> str:
    parts = [f"question={_clean_text(question, 120)}"]
    if true_answer:
        parts.append("expected_answer_available=true")
    if predicted_answer:
        parts.append("prediction_available=true")
    return "Initial GAIA research context assembled from question, files, and answer-evaluation metadata. " + "; ".join(parts)


def _gaia_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _json_compact(message.get("content") or message, 1200)
    return _json_compact(message, 1200)


def _gaia_request_summary(messages: list[Any]) -> str:
    if not messages:
        return ""
    tail = messages[-3:]
    parts = []
    for message in tail:
        if isinstance(message, dict):
            role = message.get("role", "?")
            content = _gaia_message_content(message)
            parts.append(f"{role}: {content}")
        else:
            parts.append(_gaia_message_content(message))
    prefix = f"request_messages={len(messages)}; " if len(messages) > len(tail) else ""
    return _clean_text(prefix + " | ".join(parts), 900)


def _gaia_component_for_model_call(call: dict[str, Any]) -> str:
    component = str(call.get("component") or "")
    if component == "reformulator":
        return "submitter"
    if "search" in component or "inspector" in component or "visualizer" in component or "file_context" in component:
        return "tool"
    if component == "manager":
        return "controller"
    return "controller"


def _gaia_manager_model_calls(traj: dict[str, Any]) -> list[dict[str, Any]]:
    calls = traj.get("model_calls") or []
    return [call for call in calls if isinstance(call, dict) and call.get("component") == "manager"]


def _build_views(nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
                 evaluator_anchors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    overview_types = {"TaskSpecRecord", "ModelInvocationEvent", "SubmissionEvent", "ExceptionEvent"}
    overview = [node for node in nodes if node["type"] in overview_types]
    failure = [node for node in nodes if node["type"] in {"ExceptionEvent", "VerificationEvent"}][-8:]
    if not failure and evaluator_anchors:
        failure = [node for node in nodes if node["type"] in {"SubmissionEvent", "ToolResultRecord"}][-8:]
    artifact = [node for node in nodes if node["type"] in {"ArtifactRecord", "VerificationEvent", "SubmissionEvent"}]
    context = [node for node in nodes if node["type"] in {"TaskSpecRecord", "ContextAssemblyEvent", "ModelInvocationEvent", "ToolResultRecord"}]
    control = [
        node for node in nodes
        if node["type"] in {"ParserEvent", "GuardrailEvent", "ToolCallEvent", "SubmissionEvent", "ExceptionEvent", "OrchestrationEvent"}
    ]
    causal = []
    if failure:
        failure_ids = {node["event_id"] for node in failure}
        linked_ids = set(failure_ids)
        for edge in reversed(edges):
            if edge["to"] in linked_ids:
                linked_ids.add(edge["from"])
            if len(linked_ids) >= 12:
                break
        causal = [node for node in nodes if node["event_id"] in linked_ids]
    harness_view: dict[str, list[str]] = {}
    for node in nodes:
        harness_view.setdefault(node["component"], []).append(node["event_id"])
    return {
        "overview": overview[-12:],
        "failure": failure,
        "artifact": artifact[-12:],
        "context": context[-12:],
        "control": control[-12:],
        "causal": causal[-12:],
        "harness": harness_view,
        "evaluator_anchors": evaluator_anchors or [],
    }


HTIR_VERSION = "paper_v2_compatible"

EDGE_FAMILY_MAP = {
    "temporal": "temporal",
    "context-dependency": "data-and-context-flow",
    "tool-invocation": "artifact-state",
    "artifact-dependency": "artifact-state",
    "test-dependency": "artifact-state",
    "causal": "control-flow",
    "control-flow": "control-flow",
}

COMPONENT_LAYER_MAP = {
    "prompt": ("Context and Memory",),
    "controller": ("Lifecycle and Orchestration",),
    "orchestrator": ("Lifecycle and Orchestration",),
    "parser": ("Tool Interface", "Lifecycle and Orchestration"),
    "tool": ("Tool Interface", "Execution Environment and Sandbox"),
    "validator": ("Verification and Evaluation",),
    "submitter": ("Lifecycle and Orchestration", "Verification and Evaluation"),
}

EVENT_LAYER_MAP = {
    "TaskSpecRecord": ("Context and Memory",),
    "ContextAssemblyEvent": ("Context and Memory",),
    "ModelInvocationEvent": ("Lifecycle and Orchestration",),
    "ParserEvent": ("Tool Interface", "Lifecycle and Orchestration"),
    "ToolCallEvent": ("Tool Interface",),
    "ToolResultRecord": ("Tool Interface", "Observability"),
    "GuardrailEvent": ("Lifecycle and Orchestration", "Verification and Evaluation"),
    "ArtifactRecord": ("Execution Environment and Sandbox", "Observability"),
    "StateDeltaRecord": ("Execution Environment and Sandbox", "Tool Interface", "Observability"),
    "VerificationEvent": ("Verification and Evaluation",),
    "OrchestrationEvent": ("Lifecycle and Orchestration",),
    "SubmissionEvent": ("Lifecycle and Orchestration", "Verification and Evaluation"),
    "ExceptionEvent": ("Lifecycle and Orchestration", "Observability"),
}

MUTATING_API_METHODS = {"post", "put", "patch", "delete"}


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("node_id") or node.get("event_id") or "")


def _node_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    try:
        step_id = int(node.get("step_id", 0) or 0)
    except (TypeError, ValueError):
        step_id = 0
    return step_id, _node_id(node)


def _json_compact(value: Any, limit: int = 500) -> str:
    if value in (None, {}, []):
        return ""
    if isinstance(value, str):
        return _clean_text(value, limit)
    return _clean_text(json.dumps(value, ensure_ascii=False, sort_keys=True), limit)


def _paper_edge_family(relation: str | None) -> str:
    return EDGE_FAMILY_MAP.get(str(relation or ""), "control-flow")


def _event_layers(node: dict[str, Any]) -> list[str]:
    layers = set(COMPONENT_LAYER_MAP.get(str(node.get("component", "")), ()))
    layers.update(EVENT_LAYER_MAP.get(str(node.get("type", "")), ()))
    attrs = node.get("attributes", {}) or {}
    text = f"{node.get('summary', '')} {_json_compact(attrs)}".lower()
    if "complete_task" in text or node.get("type") == "SubmissionEvent":
        layers.update({"Lifecycle and Orchestration", "Verification and Evaluation"})
    if "effect_delta" in attrs or node.get("type") == "StateDeltaRecord":
        layers.update({"Execution Environment and Sandbox", "Observability"})
    if "exception" in text or "traceback" in text:
        layers.add("Observability")
    return sorted(layers) or ["Lifecycle and Orchestration"]


def _api_call_summary(call: dict[str, Any]) -> str:
    method = str(call.get("method") or "?").upper()
    url = str(call.get("url") or call.get("api") or "?")
    data = call.get("data")
    data_keys = sorted(data.keys()) if isinstance(data, dict) else []
    key_text = f" data_keys={data_keys}" if data_keys else ""
    return f"{method} {url}{key_text}"


def _effect_delta_from_extra(extra: dict[str, Any]) -> dict[str, Any]:
    effect_delta = extra.get("effect_delta") if isinstance(extra, dict) else None
    return effect_delta if isinstance(effect_delta, dict) else {}


def _effect_summary(effect_delta: dict[str, Any]) -> str:
    if not effect_delta:
        return ""
    api_calls = effect_delta.get("api_calls", []) or []
    mutating_calls = effect_delta.get("mutating_api_calls", []) or effect_delta.get("mutating_calls", []) or []
    db_changes = effect_delta.get("db_changes", []) or []
    parts = []
    if api_calls:
        parts.append(f"api_calls={len(api_calls)}")
    if mutating_calls:
        parts.append(f"mutating_api_calls={len(mutating_calls)}")
    if db_changes:
        apps = sorted({str(change.get("app") or change.get("name") or "unknown") for change in db_changes if isinstance(change, dict)})
        parts.append(f"db_changes={len(db_changes)} apps={apps}")
    if effect_delta.get("completed"):
        parts.append("completed=true")
    if effect_delta.get("has_task_state_change") is False:
        parts.append("has_task_state_change=false")
    return "; ".join(parts)


def _experiment_task_dirs(traj_path: str | Path | None, task_id: str) -> list[Path]:
    if not traj_path:
        return []
    root = Path(traj_path).parent / "appworld_experiments" / "outputs"
    if not root.exists():
        return []
    return sorted(path for path in root.glob(f"*/tasks/{task_id}") if path.is_dir())


def _read_jsonl_limited(path: Path, max_lines: int = 80) -> list[Any]:
    if not path.exists():
        return []
    rows: list[Any] = []
    try:
        for line in path.read_text(errors="replace").splitlines()[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(line)
    except OSError:
        return []
    return rows


def _appworld_run_effect_evidence(traj_path: str | Path | None, task_id: str) -> dict[str, Any]:
    task_dirs = _experiment_task_dirs(traj_path, task_id)
    if not task_dirs:
        return {}
    task_dir = task_dirs[-1]
    api_calls = [item for item in _read_jsonl_limited(task_dir / "logs" / "api_calls.jsonl", max_lines=200) if isinstance(item, dict)]
    mutating_calls = [
        call for call in api_calls
        if str(call.get("method", "")).lower() in MUTATING_API_METHODS
    ]
    db_changes: list[dict[str, Any]] = []
    db_dir = task_dir / "dbs"
    if db_dir.exists():
        for db_path in sorted(db_dir.glob("*.jsonl")):
            if db_path.name in {"api_docs.jsonl"}:
                continue
            rows = _read_jsonl_limited(db_path, max_lines=200)
            if not rows:
                continue
            operations: dict[str, int] = {}
            samples: list[str] = []
            for row in rows:
                if isinstance(row, list) and row:
                    sql = str(row[0])
                else:
                    sql = str(row)
                op = sql.split(None, 1)[0].upper() if sql.strip() else "UNKNOWN"
                operations[op] = operations.get(op, 0) + 1
                if len(samples) < 3:
                    samples.append(_clean_text(sql, 160))
            db_changes.append(
                {
                    "app": db_path.stem,
                    "statement_count": len(rows),
                    "operations": operations,
                    "sample_statements": samples,
                    "source_ref": str(db_path),
                }
            )
    return {
        "source_task_dir": str(task_dir),
        "api_call_count": len(api_calls),
        "api_calls": api_calls[-40:],
        "mutating_api_calls": mutating_calls[-40:],
        "db_changes": db_changes,
        "has_task_state_change": bool(db_changes),
    }


def _append_appworld_run_effect_nodes(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    traj_path: str | Path | None,
    task_id: str,
    final_step_id: int,
) -> dict[str, Any]:
    effect_evidence = _appworld_run_effect_evidence(traj_path, task_id)
    if not effect_evidence:
        return {}
    source_ref = effect_evidence.get("source_task_dir", "appworld_experiments")
    api_calls = effect_evidence.get("api_calls", []) or []
    mutating_calls = effect_evidence.get("mutating_api_calls", []) or []
    if api_calls:
        api_id = f"evt_{len(nodes):03d}"
        nodes.append(
            _event(
                api_id,
                "ToolResultRecord",
                final_step_id,
                "tool",
                "AppWorld API log evidence: " + "; ".join(_api_call_summary(call) for call in api_calls[-8:]),
                f"{source_ref}/logs/api_calls.jsonl",
                {
                    "api_call_count": effect_evidence.get("api_call_count", len(api_calls)),
                    "mutating_api_call_count": len(mutating_calls),
                    "api_calls_tail": api_calls[-20:],
                    "mutating_api_calls_tail": mutating_calls[-20:],
                },
            )
        )
        if len(nodes) > 1:
            _add_edge(edges, nodes[-2]["event_id"], api_id, "tool-invocation")
    db_changes = effect_evidence.get("db_changes", []) or []
    if db_changes:
        state_id = f"evt_{len(nodes):03d}"
        nodes.append(
            _event(
                state_id,
                "StateDeltaRecord",
                final_step_id,
                "tool",
                "AppWorld database effect evidence: " + _effect_summary(effect_evidence),
                f"{source_ref}/dbs/*.jsonl",
                {"effect_delta": effect_evidence},
            )
        )
        if len(nodes) > 1:
            _add_edge(edges, nodes[-2]["event_id"], state_id, "causal")
    elif api_calls:
        state_id = f"evt_{len(nodes):03d}"
        nodes.append(
            _event(
                state_id,
                "StateDeltaRecord",
                final_step_id,
                "tool",
                "AppWorld API calls were observed, but no database mutation log was appended.",
                f"{source_ref}/dbs/*.jsonl",
                {"effect_delta": effect_evidence},
            )
        )
        if len(nodes) > 1:
            _add_edge(edges, nodes[-2]["event_id"], state_id, "causal")
    return effect_evidence


def _group_nodes_by_step(nodes: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for node in sorted(nodes, key=_node_sort_key):
        try:
            step_id = int(node.get("step_id", 0) or 0)
        except (TypeError, ValueError):
            step_id = 0
        grouped.setdefault(step_id, []).append(node)
    return grouped


def _step_role(step_nodes: list[dict[str, Any]]) -> str:
    types = {node.get("type") for node in step_nodes}
    if "TaskSpecRecord" in types or "ContextAssemblyEvent" in types:
        return "task_context"
    if "SubmissionEvent" in types:
        return "submission"
    if "VerificationEvent" in types:
        return "verification"
    if "ToolCallEvent" in types or "ToolResultRecord" in types:
        return "agent_action"
    return "agent_reasoning"


def _step_request_message(step_nodes: list[dict[str, Any]]) -> str:
    preferred = ["TaskSpecRecord", "ModelInvocationEvent", "ToolCallEvent", "SubmissionEvent"]
    parts = []
    for node_type in preferred:
        for node in step_nodes:
            if node.get("type") != node_type:
                continue
            attrs = node.get("attributes", {}) or {}
            if node_type == "ModelInvocationEvent" and attrs.get("request_summary"):
                parts.append(f"{_node_id(node)} {node_type} request: {attrs.get('request_summary', '')}")
            else:
                parts.append(f"{_node_id(node)} {node_type}: {node.get('summary', '')}")
    return _clean_text(" | ".join(parts), 900)


def _step_response_message(step_nodes: list[dict[str, Any]]) -> str:
    preferred = ["ModelInvocationEvent", "ToolResultRecord", "ExceptionEvent", "VerificationEvent", "StateDeltaRecord", "ArtifactRecord", "ParserEvent"]
    parts = []
    for node_type in preferred:
        for node in step_nodes:
            if node.get("type") != node_type:
                continue
            attrs = node.get("attributes", {}) or {}
            if node_type == "ModelInvocationEvent" and not attrs.get("response_summary"):
                continue
            extra = _effect_summary(attrs.get("effect_delta", {}))
            suffix = f" ({extra})" if extra else ""
            summary = attrs.get("response_summary") if node_type == "ModelInvocationEvent" else node.get("summary", "")
            parts.append(f"{_node_id(node)} {node_type}: {summary}{suffix}")
    return _clean_text(" | ".join(parts), 900)


def _step_execution_status(step_nodes: list[dict[str, Any]]) -> str:
    if any(node.get("type") == "ExceptionEvent" for node in step_nodes):
        return "exception"
    if any(node.get("type") == "SubmissionEvent" for node in step_nodes):
        return "submitted"
    if any(node.get("type") == "VerificationEvent" for node in step_nodes):
        return "verified"
    if any(node.get("type") == "ToolResultRecord" for node in step_nodes):
        return "observed"
    return "recorded"


def _step_external_effect(step_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    effect_deltas = []
    state_nodes = []
    for node in step_nodes:
        attrs = node.get("attributes", {}) or {}
        effect_delta = attrs.get("effect_delta")
        if isinstance(effect_delta, dict) and effect_delta:
            effect_deltas.append(effect_delta)
        if node.get("type") in {"StateDeltaRecord", "ArtifactRecord"}:
            state_nodes.append(_node_id(node))
    mutating_calls = []
    db_changes = []
    api_calls = []
    for delta in effect_deltas:
        api_calls.extend(delta.get("api_calls", []) or [])
        mutating_calls.extend(delta.get("mutating_api_calls", []) or delta.get("mutating_calls", []) or [])
        db_changes.extend(delta.get("db_changes", []) or [])
    return {
        "has_external_effect": bool(state_nodes or db_changes or mutating_calls),
        "state_nodes": state_nodes,
        "api_call_count": len(api_calls),
        "mutating_api_call_count": len(mutating_calls),
        "db_change_count": len(db_changes),
        "summary": _effect_summary({"api_calls": api_calls, "mutating_api_calls": mutating_calls, "db_changes": db_changes})
        or ("legacy_state_delta_present" if state_nodes else ""),
    }


def _build_agent_trace_steps(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for step_id, step_nodes in _group_nodes_by_step(nodes).items():
        source_refs = [node.get("source_ref") for node in step_nodes if node.get("source_ref")]
        legacy_ids = [_node_id(node) for node in step_nodes]
        steps.append(
            {
                "step_id": f"step_{step_id:03d}",
                "numeric_step_id": step_id,
                "node_id": f"step_{step_id:03d}",
                "type": "AgentTraceStep",
                "role": _step_role(step_nodes),
                "request_message": _step_request_message(step_nodes),
                "response_message": _step_response_message(step_nodes),
                "execution_status": _step_execution_status(step_nodes),
                "external_effect": _step_external_effect(step_nodes),
                "source_ref": "; ".join(source_refs[:6]),
                "legacy_event_ids": legacy_ids,
                "attributes": {
                    "node_count": len(step_nodes),
                    "node_types": sorted({str(node.get("type")) for node in step_nodes}),
                    "components": sorted({str(node.get("component")) for node in step_nodes}),
                },
            }
        )
    return steps


def _legacy_node_to_step(nodes: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in nodes:
        try:
            step_id = int(node.get("step_id", 0) or 0)
        except (TypeError, ValueError):
            step_id = 0
        mapping[_node_id(node)] = f"step_{step_id:03d}"
    return mapping


def _build_base_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], outcome_anchors: list[dict[str, Any]]) -> dict[str, Any]:
    node_to_step = _legacy_node_to_step(nodes)
    step_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        src = node_to_step.get(str(edge.get("from", "")))
        dst = node_to_step.get(str(edge.get("to", "")))
        if not src or not dst or src == dst:
            continue
        family = _paper_edge_family(edge.get("relation"))
        key = (src, dst, family)
        item = step_edges.setdefault(
            key,
            {"from": src, "to": dst, "family": family, "legacy_relations": [], "legacy_edges": []},
        )
        item["legacy_relations"].append(edge.get("relation"))
        item["legacy_edges"].append(edge)
    anchor_links = []
    candidate_steps = sorted({node_to_step.get(_node_id(node)) for node in nodes[-8:] if node_to_step.get(_node_id(node))})
    for anchor in outcome_anchors:
        for step_id in candidate_steps[-4:]:
            anchor_links.append(
                {
                    "from": step_id,
                    "to": anchor.get("anchor_id"),
                    "family": "outcome-anchor",
                    "relation": "candidate-explains-outcome",
                }
            )
    return {
        "nodes": sorted(set(node_to_step.values())),
        "edges": list(step_edges.values()),
        "outcome_anchors": [anchor.get("anchor_id") for anchor in outcome_anchors],
        "anchor_links": anchor_links,
        "edge_families": ["temporal", "data-and-context-flow", "control-flow", "artifact-state"],
        "attributes": {"legacy_node_count": len(nodes), "legacy_edge_count": len(edges)},
    }


def _incoming_outgoing_edges(step_id: str, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_to_step = _legacy_node_to_step(nodes)
    incoming = []
    outgoing = []
    for edge in edges:
        src = node_to_step.get(str(edge.get("from", "")))
        dst = node_to_step.get(str(edge.get("to", "")))
        if dst == step_id:
            incoming.append(edge)
        if src == step_id:
            outgoing.append(edge)
    return incoming, outgoing


def _build_node_facets(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_nodes_by_step(nodes)
    facets: dict[str, Any] = {}
    for step in steps:
        numeric_step_id = int(step.get("numeric_step_id", 0) or 0)
        step_id = str(step["step_id"])
        step_nodes = grouped.get(numeric_step_id, [])
        incoming, outgoing = _incoming_outgoing_edges(step_id, nodes, edges)
        data_refs = [
            {
                "node_id": _node_id(node),
                "type": node.get("type"),
                "source_ref": node.get("source_ref"),
                "summary": node.get("summary"),
            }
            for node in step_nodes
            if node.get("type") in {"TaskSpecRecord", "ContextAssemblyEvent", "ModelInvocationEvent", "ToolResultRecord"}
        ]
        control_refs = [
            {
                "node_id": _node_id(node),
                "type": node.get("type"),
                "source_ref": node.get("source_ref"),
                "summary": node.get("summary"),
            }
            for node in step_nodes
            if node.get("type") in {"ParserEvent", "GuardrailEvent", "ToolCallEvent", "SubmissionEvent", "ExceptionEvent", "OrchestrationEvent"}
        ]
        artifact_refs = []
        for node in step_nodes:
            if node.get("type") not in {"ArtifactRecord", "StateDeltaRecord", "VerificationEvent", "SubmissionEvent", "ToolResultRecord"}:
                continue
            attrs = node.get("attributes", {}) or {}
            artifact_refs.append(
                {
                    "node_id": _node_id(node),
                    "type": node.get("type"),
                    "source_ref": node.get("source_ref"),
                    "summary": node.get("summary"),
                    "effect_delta": attrs.get("effect_delta"),
                    "returncode": attrs.get("returncode"),
                    "exception_info": _clean_text(str(attrs.get("exception_info", "")), 220),
                }
            )
        layers = sorted({layer for node in step_nodes for layer in _event_layers(node)})
        evidence_refs = [
            {
                "node_id": _node_id(node),
                "source_ref": node.get("source_ref"),
                "summary": node.get("summary"),
            }
            for node in step_nodes[:8]
        ]
        incoming_context_sources = [
            {
                "from_node_id": edge.get("from"),
                "to_node_id": edge.get("to"),
                "relation": edge.get("relation"),
                "attributes": edge.get("attributes", {}),
            }
            for edge in incoming
            if _paper_edge_family(edge.get("relation")) == "data-and-context-flow"
        ]
        facets[step_id] = {
            "data_and_context": {
                "evidence_refs": data_refs,
                "incoming_families": sorted({_paper_edge_family(edge.get("relation")) for edge in incoming}),
                "incoming_context_sources": incoming_context_sources,
                "request_message": step.get("request_message", ""),
            },
            "control_flow": {
                "status": step.get("execution_status"),
                "incoming_edges": incoming[-8:],
                "outgoing_edges": outgoing[-8:],
                "evidence_refs": control_refs,
            },
            "artifact_state": {
                "external_effect": step.get("external_effect", {}),
                "evidence_refs": artifact_refs,
            },
            "harness_layer": {
                "implicated_layers": layers,
                "component_evidence": sorted({str(node.get("component")) for node in step_nodes}),
                "event_type_evidence": sorted({str(node.get("type")) for node in step_nodes}),
            },
            "evidence_refs": evidence_refs,
        }
    return facets


def _finalize_paper_htir(bundle: dict[str, Any]) -> dict[str, Any]:
    graph = bundle.get("graph", {})
    nodes = graph.get("nodes") or graph.get("events") or []
    edges = graph.get("edges") or []
    outcome_anchors = bundle.get("outcome_anchors") or bundle.get("evaluator_anchors") or []
    _infer_request_context_edges(nodes, edges)
    steps = _build_agent_trace_steps(nodes)
    bundle["htir_version"] = HTIR_VERSION
    bundle["agent_trace_steps"] = steps
    bundle["outcome_anchors"] = outcome_anchors
    bundle["base_graph"] = _build_base_graph(nodes, edges, outcome_anchors)
    bundle["node_facets"] = _build_node_facets(nodes, edges, steps)
    for node in nodes:
        attrs = node.get("attributes")
        if isinstance(attrs, dict):
            attrs.pop("_request_context_text", None)
    bundle.setdefault("candidate_responsible_steps", [])
    return bundle


def _repetition_stats(commands: list[str]) -> dict[str, Any]:
    if not commands:
        return {"max_repetition_run": 0, "repeated_commands": []}
    normalized = [re.sub(r"\s+", " ", c.strip()) for c in commands]
    max_run = 1
    current_run = 1
    repeated: set[str] = set()
    for idx in range(1, len(normalized)):
        if normalized[idx] == normalized[idx - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
            if current_run >= 3:
                repeated.add(_clean_text(normalized[idx], 100))
        else:
            current_run = 1
    return {"max_repetition_run": max_run, "repeated_commands": sorted(repeated)}


def _swe_model_call_map(traj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    calls = traj.get("model_calls") or []
    mapped: dict[int, dict[str, Any]] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        try:
            call_index = int(call.get("call_index", 0) or 0)
        except (TypeError, ValueError):
            call_index = 0
        if call_index > 0:
            mapped[call_index] = call
    return mapped


def _swe_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _json_compact(message.get("content") or message, 1200)
    return _json_compact(message, 1200)


def _swe_request_summary(messages: list[Any]) -> str:
    if not messages:
        return ""
    tail = messages[-3:]
    parts = []
    for message in tail:
        if isinstance(message, dict):
            role = message.get("role", "?")
            parts.append(f"{role}: {_swe_message_content(message)}")
        else:
            parts.append(_swe_message_content(message))
    prefix = f"request_messages={len(messages)}; " if len(messages) > len(tail) else ""
    return _clean_text(prefix + " | ".join(parts), 900)


def compile_swe_htir(instance_id: str, failure_category: str, paths: dict[str, str],
                     task_description: str) -> dict[str, Any]:
    traj = _read_json(paths["traj_path"])
    report = _load_optional_json(paths.get("report_path"))
    test_output = _load_optional_text(paths.get("test_output_path"))
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evaluator_anchors: list[dict[str, Any]] = []
    commands: list[str] = []
    messages = traj.get("messages", [])
    model_calls = traj.get("model_calls") or []
    swe_model_calls = _swe_model_call_map(traj)

    events.append(
        _event(
            "evt_000",
            "TaskSpecRecord",
            0,
            "prompt",
            task_description,
            "traj.messages[1]",
            {"failure_category": failure_category, "trajectory_format": traj.get("trajectory_format")},
        )
    )

    last_agent_event = _append_context_node(
        events,
        edges,
        "evt_000",
        0,
        "Initial SWE task context assembled from benchmark prompt and harness instructions.",
        "traj.messages[0:2]",
        {"message_count": min(len(messages), 2), "model_call_count": len(model_calls)},
    )
    step_id = 0
    assistant_call_index = 0
    pending_tool_calls: list[str] = []
    for msg_idx, message in enumerate(messages[2:], start=2):
        role = message.get("role")
        if role == "assistant":
            step_id += 1
            assistant_call_index += 1
            model_call = swe_model_calls.get(assistant_call_index)
            request_messages = model_call.get("request_messages", []) if isinstance(model_call, dict) else []
            response_message = model_call.get("response_message", {}) if isinstance(model_call, dict) else {}
            response_summary = _swe_message_content(response_message) if response_message else ""
            agent_event_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    agent_event_id,
                    "ModelInvocationEvent",
                    step_id,
                    "controller",
                    response_summary or message.get("content", ""),
                    f"traj.model_calls[{assistant_call_index - 1}].response_message" if model_call else f"traj.messages[{msg_idx}]",
                    {
                        "role": role,
                        "model_call_index": assistant_call_index if model_call else None,
                        "request_message_count": len(request_messages),
                        "_request_context_text": _flatten_text(request_messages),
                        "request_summary": _swe_request_summary(request_messages),
                        "response_summary": response_summary,
                        "response_message": response_message,
                        "raw_response": model_call.get("raw_response") if isinstance(model_call, dict) else None,
                        "usage": model_call.get("usage") if isinstance(model_call, dict) else {},
                    } if model_call else {"role": role},
                )
            )
            _add_edge(edges, last_agent_event, agent_event_id, "temporal")
            last_agent_event = agent_event_id
            actions = message.get("extra", {}).get("actions", [])
            if actions:
                parser_id = _append_parser_node(
                    events,
                    edges,
                    agent_event_id,
                    step_id,
                    f"Parsed {len(actions)} action(s) from model output.",
                    f"traj.messages[{msg_idx}].extra.actions",
                    {"action_count": len(actions)},
                )
            else:
                parser_id = _append_parser_node(
                    events,
                    edges,
                    agent_event_id,
                    step_id,
                    "No executable action parsed from model output.",
                    f"traj.messages[{msg_idx}]",
                    {"action_count": 0},
                )
            for action_idx, action in enumerate(actions, start=1):
                command = action.get("command", "")
                commands.append(command)
                tool_call_id = f"evt_{len(events):03d}"
                component = _swe_component_for_command(command)
                events.append(
                    _event(
                        tool_call_id,
                        "ToolCallEvent",
                        step_id,
                        component,
                        command,
                        f"traj.messages[{msg_idx}].extra.actions[{action_idx - 1}]",
                        {"command": command},
                    )
                )
                pending_tool_calls.append(tool_call_id)
                _add_edge(edges, parser_id, tool_call_id, "control-flow")
                _add_edge(edges, agent_event_id, tool_call_id, "tool-invocation")
                if _looks_like_edit(command):
                    artifact_id = f"evt_{len(events):03d}"
                    events.append(
                        _event(
                            artifact_id,
                            "ArtifactRecord",
                            step_id,
                            "tool",
                            f"Potential file edit touching: {', '.join(_command_paths(command)) or '(unknown files)'}",
                            f"traj.messages[{msg_idx}]",
                            {"paths": _command_paths(command)},
                        )
                    )
                    _add_edge(edges, tool_call_id, artifact_id, "artifact-dependency")
                if _looks_like_verification(command):
                    verification_id = f"evt_{len(events):03d}"
                    events.append(
                        _event(
                            verification_id,
                            "VerificationEvent",
                            step_id,
                            "validator",
                            command,
                            f"traj.messages[{msg_idx}]",
                            {"command": command},
                        )
                    )
                    _add_edge(edges, tool_call_id, verification_id, "test-dependency")
                if _looks_like_submission(command):
                    submit_id = f"evt_{len(events):03d}"
                    events.append(
                        _event(
                            submit_id,
                            "SubmissionEvent",
                            step_id,
                            "submitter",
                            command,
                            f"traj.messages[{msg_idx}]",
                            {"command": command},
                        )
                    )
                    _add_edge(edges, tool_call_id, submit_id, "causal")
        elif role == "user":
            observation_id = f"evt_{len(events):03d}"
            output = message.get("content", "")
            extra = message.get("extra", {})
            events.append(
                _event(
                    observation_id,
                    "ToolResultRecord",
                    step_id,
                    "tool",
                    output,
                    f"traj.messages[{msg_idx}]",
                    {
                        "returncode": extra.get("returncode"),
                        "exception_info": extra.get("exception_info"),
                    },
                )
            )
            if pending_tool_calls:
                _add_edge(edges, pending_tool_calls.pop(0), observation_id, "tool-invocation")
            else:
                _add_edge(edges, last_agent_event, observation_id, "temporal")
            if _looks_like_exception(output, str(extra.get("exception_info", ""))):
                exception_id = f"evt_{len(events):03d}"
                events.append(
                    _event(
                        exception_id,
                        "ExceptionEvent",
                        step_id,
                        "controller",
                        extra.get("exception_info") or output,
                        f"traj.messages[{msg_idx}]",
                        {"returncode": extra.get("returncode")},
                    )
                )
                _add_edge(edges, observation_id, exception_id, "causal")

    if report:
        report_key = next(iter(report.keys()))
        report_body = report[report_key]
        evaluator_anchors.append(
            _extract_evaluator_anchor(
                "report.json",
                json.dumps(report_body, ensure_ascii=False),
                report_body,
            )
        )
    if test_output:
        verification_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                verification_id,
                "VerificationEvent",
                step_id + 1,
                "validator",
                test_output,
                "test_output.txt",
            )
        )

    stats = {
        "exit_status": traj.get("info", {}).get("exit_status"),
        "api_calls": traj.get("info", {}).get("model_stats", {}).get("api_calls"),
        "instance_cost": traj.get("info", {}).get("model_stats", {}).get("instance_cost"),
        "model_call_count": len(model_calls),
        "command_count": len(commands),
        **_repetition_stats(commands),
    }
    views = _build_views(events, edges, evaluator_anchors)
    return _finalize_paper_htir({
        "mode": "swe",
        "instance_id": instance_id,
        "graph": {"nodes": events, "events": events, "edges": edges},
        "evaluator_anchors": evaluator_anchors,
        "views": views,
        "stats": stats,
    })


def compile_gaia_htir(task_id: str, failure_category: str, traj_path: str,
                      question: str, true_answer: str, predicted_answer: str) -> dict[str, Any]:
    traj = _read_json(traj_path)
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evaluator_anchors: list[dict[str, Any]] = []
    search_commands: list[str] = []

    events.append(
        _event(
            "evt_000",
            "TaskSpecRecord",
            0,
            "prompt",
            question,
            "trajectory[0]",
            {"failure_category": failure_category, "true_answer": true_answer, "predicted_answer": predicted_answer},
        )
    )

    last_id = _append_context_node(
        events,
        edges,
        "evt_000",
        0,
        _gaia_context_summary(question, true_answer, predicted_answer),
        "trajectory.context",
        {"has_expected_answer": bool(true_answer), "has_prediction": bool(predicted_answer)},
    )
    manager_model_calls = _gaia_manager_model_calls(traj)
    used_trace_model_call_indexes: set[int] = set()
    for idx, step in enumerate(traj.get("trajectory", []), start=1):
        step_type = step.get("_type", step.get("type", "Step"))
        agent_event_id = f"evt_{len(events):03d}"
        component = _gaia_component_for_step(step)
        summary = json.dumps(step, ensure_ascii=False)
        model_call = manager_model_calls[idx - 1] if idx - 1 < len(manager_model_calls) else None
        if not model_call and isinstance(step, dict) and step.get("model_input_messages"):
            response_message = step.get("model_output_message") or {"role": "assistant", "content": step.get("model_output", "")}
            model_call = {
                "call_index": idx,
                "component": component,
                "request_messages": step.get("model_input_messages") or [],
                "response_message": response_message,
                "usage": step.get("token_usage") or {},
            }
        request_messages = model_call.get("request_messages", []) if isinstance(model_call, dict) else []
        response_message = model_call.get("response_message", {}) if isinstance(model_call, dict) else {}
        response_summary = _gaia_message_content(response_message) if response_message else ""
        events.append(
            _event(
                agent_event_id,
                "ModelInvocationEvent",
                idx,
                str(model_call.get("component") or component) if isinstance(model_call, dict) else component,
                response_summary or f"{step_type}: {summary}",
                f"traj.model_calls[{int(model_call.get('call_index', idx)) - 1}].response_message" if model_call and int(model_call.get('call_index', 0) or 0) > 0 else f"trajectory[{idx - 1}]",
                {
                    "step_type": step_type,
                    "model_call_index": model_call.get("call_index") if isinstance(model_call, dict) else None,
                    "request_message_count": len(request_messages),
                    "_request_context_text": _flatten_text(request_messages),
                    "request_summary": _gaia_request_summary(request_messages),
                    "response_summary": response_summary,
                    "response_message": response_message,
                    "usage": model_call.get("usage") if isinstance(model_call, dict) else step.get("token_usage"),
                },
            )
        )
        _add_edge(edges, last_id, agent_event_id, "temporal")
        last_id = agent_event_id

        tool_ids: list[str] = []
        for call_idx, tool_call in enumerate(step.get("tool_calls", []) or [], start=1):
            tool_text = json.dumps(tool_call, ensure_ascii=False)
            search_commands.append(tool_text)
            tool_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    tool_id,
                    "ToolCallEvent",
                    idx,
                    "tool",
                    tool_text,
                    f"trajectory[{idx - 1}].tool_calls[{call_idx - 1}]",
                )
            )
            tool_ids.append(tool_id)
            _add_edge(edges, agent_event_id, tool_id, "tool-invocation")
        if tool_ids:
            parser_id = _append_parser_node(
                events,
                edges,
                agent_event_id,
                idx,
                f"Parsed {len(tool_ids)} tool call(s) from GAIA agent step.",
                f"trajectory[{idx - 1}].tool_calls",
                {"tool_call_count": len(tool_ids), "step_type": step_type},
            )
            for tool_id in tool_ids:
                _add_edge(edges, parser_id, tool_id, "control-flow")
        elif "final_answer" in summary.lower() or "prediction" in summary.lower():
            _append_parser_node(
                events,
                edges,
                agent_event_id,
                idx,
                "Parsed final-answer content from GAIA agent step.",
                f"trajectory[{idx - 1}]",
                {"step_type": step_type},
            )
        if step.get("observations"):
            obs_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    obs_id,
                    "ToolResultRecord",
                    idx,
                    "tool",
                    str(step.get("observations")),
                    f"trajectory[{idx - 1}].observations",
                )
            )
            for tool_id in tool_ids:
                _add_edge(edges, tool_id, obs_id, "tool-invocation")
            if not tool_ids:
                _add_edge(edges, agent_event_id, obs_id, "temporal")
        if model_call and int(model_call.get("call_index", 0) or 0) > 0:
            used_trace_model_call_indexes.add(int(model_call.get("call_index", 0) or 0))
        if step.get("error"):
            ex_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    ex_id,
                    "ExceptionEvent",
                    idx,
                    "controller",
                    str(step.get("error")),
                    f"trajectory[{idx - 1}].error",
                )
            )
            _add_edge(edges, agent_event_id, ex_id, "causal")

    auxiliary_parent = last_id
    for call in traj.get("model_calls") or []:
        if not isinstance(call, dict):
            continue
        try:
            call_index = int(call.get("call_index", 0) or 0)
        except (TypeError, ValueError):
            call_index = 0
        if call_index <= 0 or call_index in used_trace_model_call_indexes:
            continue
        request_messages = call.get("request_messages", []) if isinstance(call.get("request_messages"), list) else []
        response_message = call.get("response_message", {}) if isinstance(call.get("response_message"), dict) else {}
        aux_id = f"evt_{len(events):03d}"
        aux_step_id = len(traj.get("trajectory", [])) + call_index
        events.append(
            _event(
                aux_id,
                "ModelInvocationEvent",
                aux_step_id,
                _gaia_component_for_model_call(call),
                _gaia_message_content(response_message) or f"GAIA model call component={call.get('component')}",
                f"traj.model_calls[{call_index - 1}].response_message",
                {
                    "model_call_index": call_index,
                    "component_label": call.get("component"),
                    "request_message_count": len(request_messages),
                    "_request_context_text": _flatten_text(request_messages),
                    "request_summary": _gaia_request_summary(request_messages),
                    "response_summary": _gaia_message_content(response_message),
                    "response_message": response_message,
                    "usage": call.get("usage"),
                    "request_options": call.get("request_options"),
                    "error": call.get("error"),
                },
            )
        )
        _add_edge(edges, auxiliary_parent, aux_id, "temporal")
        auxiliary_parent = aux_id
        if call.get("error"):
            ex_id = f"evt_{len(events):03d}"
            events.append(_event(ex_id, "ExceptionEvent", aux_step_id, "controller", str(call.get("error")), f"traj.model_calls[{call_index - 1}].error"))
            _add_edge(edges, aux_id, ex_id, "causal")
    last_id = auxiliary_parent

    submit_id = f"evt_{len(events):03d}"
    events.append(
        _event(
            submit_id,
            "SubmissionEvent",
            len(traj.get("trajectory", [])) + 1,
            "submitter",
            predicted_answer,
            "traj.info.prediction",
            {"prediction": predicted_answer},
        )
    )
    _add_edge(edges, last_id, submit_id, "causal")

    resolved = str(traj.get("info", {}).get("exit_status", "")).lower() == "solved"
    evaluator_anchors.append(
        _extract_evaluator_anchor(
            "traj.info",
            f"predicted={predicted_answer} | expected={true_answer}",
            {"resolved": resolved, "failure_category": failure_category},
        )
    )

    if traj.get("info", {}).get("iteration_limit_exceeded"):
        ex_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                ex_id,
                "ExceptionEvent",
                len(traj.get("trajectory", [])) + 1,
                "controller",
                "iteration_limit_exceeded",
                "traj.info.iteration_limit_exceeded",
            )
        )
        _add_edge(edges, submit_id, ex_id, "causal")

    stats = {
        "exit_status": traj.get("info", {}).get("exit_status"),
        "elapsed_seconds": traj.get("info", {}).get("elapsed_seconds"),
        "step_count": len(traj.get("trajectory", [])),
        "model_call_count": len(traj.get("model_calls") or []),
        **_repetition_stats(search_commands),
    }
    views = _build_views(events, edges, evaluator_anchors)
    return _finalize_paper_htir({
        "mode": "gaia",
        "instance_id": task_id,
        "graph": {"nodes": events, "events": events, "edges": edges},
        "evaluator_anchors": evaluator_anchors,
        "views": views,
        "stats": stats,
    })


def _terminal_component_for_tool(tool_name: str | None, command: str | None = None) -> str:
    text = f"{tool_name or ''} {command or ''}".lower()
    if any(token in text for token in ("pytest", "test", "verify", "reward")):
        return "validator"
    if any(token in text for token in ("submit", "complete", "task_complete")):
        return "submitter"
    if any(token in text for token in ("bash", "shell", "terminal", "tmux", "exec", "command")):
        return "tool"
    return "controller"


def _terminal_step_message(step: dict[str, Any]) -> str:
    message = step.get("message")
    if isinstance(message, str):
        return message
    if message is None:
        return ""
    return json.dumps(message, ensure_ascii=False)


def _terminal_model_call_map(traj: dict[str, Any]) -> dict[int, dict[str, Any]]:
    calls = traj.get("model_calls") or []
    mapped: dict[int, dict[str, Any]] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        try:
            call_index = int(call.get("call_index", 0) or 0)
        except (TypeError, ValueError):
            call_index = 0
        if call_index > 0:
            mapped[call_index] = call
    return mapped


def _terminal_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _json_compact(message.get("content") or message, 1200)
    return _json_compact(message, 1200)


def _terminal_request_summary(messages: list[Any]) -> str:
    if not messages:
        return ""
    tail = messages[-3:]
    parts = []
    for message in tail:
        role = message.get("role", "message") if isinstance(message, dict) else "message"
        parts.append(f"{role}: {_terminal_message_content(message)}")
    prefix = f"request_messages={len(messages)}; " if len(messages) > len(tail) else ""
    return prefix + " | ".join(parts)


def _terminal_tool_summary(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("tool_name") or tool_call.get("name") or tool_call.get("type") or "tool"
    args = tool_call.get("arguments") or tool_call.get("args") or tool_call.get("input") or tool_call
    return f"{name}: {_json_compact(args, 500)}"


def _terminal_observation_summary(observation: dict[str, Any]) -> str:
    if not observation:
        return ""
    results = observation.get("results") or []
    parts = []
    for result in results[:4]:
        if isinstance(result, dict):
            content = result.get("content") or result.get("text") or result
            parts.append(_json_compact(content, 500))
        else:
            parts.append(_json_compact(result, 500))
    return " | ".join(parts) or _json_compact(observation, 700)


def compile_terminal_bench_htir(task_id: str, failure_category: str, paths: dict[str, str],
                                task_description: str = "") -> dict[str, Any]:
    traj = _read_json(paths["traj_path"])
    atif = _load_optional_json(paths.get("atif_path"))
    result = _load_optional_json(paths.get("result_path"))
    stdout = _load_optional_text(paths.get("test_stdout_path"))
    stderr = _load_optional_text(paths.get("test_stderr_path"))
    pane = _load_optional_text(paths.get("pane_path"))
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evaluator_anchors: list[dict[str, Any]] = []
    commands: list[str] = []

    if not task_description:
        task_description = _clean_text(_terminal_step_message((atif.get("steps") or [{}])[0]), 900) if atif else ""
    events.append(
        _event(
            "evt_000",
            "TaskSpecRecord",
            0,
            "prompt",
            task_description or f"Terminal-Bench task {task_id}",
            "atif.steps[0]" if atif else "traj.info",
            {"failure_category": failure_category, "trajectory_format": traj.get("trajectory_format")},
        )
    )
    last_id = _append_context_node(
        events,
        edges,
        "evt_000",
        0,
        "Terminal-Bench task context assembled from Harbor Terminus-2 prompt, terminal state, and verifier metadata.",
        "terminal_bench.context",
        {"agent": "terminus-2", "has_atif": bool(atif)},
    )

    terminal_model_calls = _terminal_model_call_map(traj)
    assistant_call_index = 0
    for idx, step in enumerate(atif.get("steps", []) if atif else [], start=1):
        source = step.get("source") or step.get("role") or "agent"
        message = _terminal_step_message(step)
        node_type = "ModelInvocationEvent" if source in {"assistant", "agent"} else "ContextAssemblyEvent"
        component = "controller" if node_type == "ModelInvocationEvent" else "prompt"
        model_call = None
        if node_type == "ModelInvocationEvent":
            assistant_call_index += 1
            model_call = terminal_model_calls.get(assistant_call_index)
        request_messages = model_call.get("request_messages", []) if isinstance(model_call, dict) else []
        response_message = model_call.get("response_message", {}) if isinstance(model_call, dict) else {}
        response_summary = _terminal_message_content(response_message) if response_message else ""
        summary = response_summary or message
        agent_event_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                agent_event_id,
                node_type,
                idx,
                component,
                summary,
                f"traj.model_calls[{assistant_call_index - 1}].response_message" if model_call else f"atif.steps[{idx - 1}]",
                {
                    "source": source,
                    "atif_step_id": step.get("step_id"),
                    "model_call_index": assistant_call_index if model_call else None,
                    "request_message_count": len(request_messages),
                    "_request_context_text": _flatten_text(request_messages),
                    "request_summary": _terminal_request_summary(request_messages),
                    "response_summary": response_summary,
                    "response_message": response_message,
                } if model_call else {"source": source, "atif_step_id": step.get("step_id")},
            )
        )
        _add_edge(edges, last_id, agent_event_id, "temporal")
        last_id = agent_event_id

        tool_ids: list[str] = []
        for call_idx, tool_call in enumerate(step.get("tool_calls", []) or [], start=1):
            summary = _terminal_tool_summary(tool_call)
            commands.append(summary)
            tool_name = tool_call.get("tool_name") or tool_call.get("name") or tool_call.get("type")
            tool_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    tool_id,
                    "ToolCallEvent",
                    idx,
                    _terminal_component_for_tool(str(tool_name), summary),
                    summary,
                    f"atif.steps[{idx - 1}].tool_calls[{call_idx - 1}]",
                    {"tool_call": tool_call},
                )
            )
            tool_ids.append(tool_id)
            _add_edge(edges, agent_event_id, tool_id, "tool-invocation")
        if tool_ids:
            parser_id = _append_parser_node(
                events,
                edges,
                agent_event_id,
                idx,
                f"Parsed {len(tool_ids)} terminal tool call(s) from Terminus-2 step.",
                f"atif.steps[{idx - 1}].tool_calls",
                {"tool_call_count": len(tool_ids)},
            )
            for tool_id in tool_ids:
                _add_edge(edges, parser_id, tool_id, "control-flow")

        observation = step.get("observation") or {}
        obs_summary = _terminal_observation_summary(observation)
        if obs_summary:
            obs_id = f"evt_{len(events):03d}"
            events.append(
                _event(
                    obs_id,
                    "ToolResultRecord",
                    idx,
                    "tool",
                    obs_summary,
                    f"atif.steps[{idx - 1}].observation",
                    {"observation": observation},
                )
            )
            if tool_ids:
                for tool_id in tool_ids:
                    _add_edge(edges, tool_id, obs_id, "tool-invocation")
            else:
                _add_edge(edges, agent_event_id, obs_id, "temporal")
            if _looks_like_exception(obs_summary, ""):
                ex_id = f"evt_{len(events):03d}"
                events.append(_event(ex_id, "ExceptionEvent", idx, "controller", obs_summary, f"atif.steps[{idx - 1}].observation"))
                _add_edge(edges, obs_id, ex_id, "causal")

    info = traj.get("info", {}) or {}
    reward = info.get("reward")
    exception = info.get("exception") or result.get("exception_info")
    submit_id = f"evt_{len(events):03d}"
    events.append(
        _event(
            submit_id,
            "SubmissionEvent",
            max(1, len(atif.get("steps", []) if atif else [])) + 1,
            "submitter",
            f"Terminal-Bench run completed with reward={reward} exit_status={info.get('exit_status')}",
            "traj.info",
            {"reward": reward, "exit_status": info.get("exit_status"), "exception": exception},
        )
    )
    _add_edge(edges, last_id, submit_id, "causal")

    verifier_text = "\n".join(part for part in [stdout, stderr] if part)
    if verifier_text:
        verification_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                verification_id,
                "VerificationEvent",
                max(1, len(atif.get("steps", []) if atif else [])) + 2,
                "validator",
                verifier_text,
                "verifier/test-stdout.txt; verifier/test-stderr.txt",
            )
        )
        _add_edge(edges, submit_id, verification_id, "test-dependency")
    if exception:
        ex_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                ex_id,
                "ExceptionEvent",
                max(1, len(atif.get("steps", []) if atif else [])) + 2,
                "controller",
                json.dumps(exception, ensure_ascii=False) if isinstance(exception, dict) else str(exception),
                "result.json.exception_info",
            )
        )
        _add_edge(edges, submit_id, ex_id, "causal")

    evaluator_anchors.append(
        _extract_evaluator_anchor(
            "result.json/verifier_result",
            f"reward={reward}; exit_status={info.get('exit_status')}; stdout={_clean_text(stdout, 180)}; stderr={_clean_text(stderr, 180)}",
            {"reward": reward, "failure_category": failure_category, "exception": exception},
        )
    )

    stats = {
        "exit_status": info.get("exit_status"),
        "reward": reward,
        "api_calls": info.get("model_stats", {}).get("api_calls"),
        "instance_cost": info.get("model_stats", {}).get("instance_cost"),
        "step_count": len(atif.get("steps", []) if atif else []),
        "pane_chars": len(pane),
        **_repetition_stats(commands),
    }
    views = _build_views(events, edges, evaluator_anchors)
    return _finalize_paper_htir({
        "mode": "terminal_bench",
        "instance_id": task_id,
        "graph": {"nodes": events, "events": events, "edges": edges},
        "evaluator_anchors": evaluator_anchors,
        "views": views,
        "stats": stats,
    })


def _appworld_model_call_summary(message: Any) -> str:
    if isinstance(message, dict):
        return _json_compact(message.get("content") or message, 1200)
    return _json_compact(message, 1200)


def _appworld_request_summary(messages: list[Any]) -> str:
    if not messages:
        return ""
    tail = messages[-3:]
    parts = []
    for message in tail:
        role = message.get("role", "message") if isinstance(message, dict) else "message"
        parts.append(f"{role}: {_appworld_model_call_summary(message)}")
    prefix = f"request_messages={len(messages)}; " if len(messages) > len(tail) else ""
    return prefix + " | ".join(parts)


def _appworld_extract_code_blocks(content: str) -> list[str]:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content or "", flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return [block.strip() for block in blocks if block.strip()]
    return []


def _append_appworld_agent_step(
    events: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    step_id: int,
    last_agent_event: str,
    summary: str,
    source_ref: str,
    attributes: dict[str, Any] | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str], list[str]]:
    agent_event_id = f"evt_{len(events):03d}"
    events.append(
        _event(
            agent_event_id,
            "ModelInvocationEvent",
            step_id,
            "controller",
            summary,
            source_ref,
            attributes or {},
        )
    )
    _add_edge(edges, last_agent_event, agent_event_id, "temporal")
    pending_tool_calls: list[str] = []
    commands: list[str] = []
    actions = actions or []
    if actions:
        parser_id = _append_parser_node(
            events,
            edges,
            agent_event_id,
            step_id,
            f"Parsed {len(actions)} Python action block(s) from AppWorld agent output.",
            f"{source_ref}.actions",
            {"action_count": len(actions)},
        )
    else:
        parser_id = _append_parser_node(
            events,
            edges,
            agent_event_id,
            step_id,
            "No executable AppWorld action parsed from model output.",
            source_ref,
            {"action_count": 0},
        )
    for action_idx, action in enumerate(actions, start=1):
        command = str(action.get("command", ""))
        commands.append(command)
        tool_id = f"evt_{len(events):03d}"
        component = _appworld_component_for_command(command)
        events.append(
            _event(
                tool_id,
                "ToolCallEvent",
                step_id,
                component,
                command,
                f"{source_ref}.actions[{action_idx - 1}]",
                {"command": command},
            )
        )
        pending_tool_calls.append(tool_id)
        _add_edge(edges, parser_id, tool_id, "control-flow")
        _add_edge(edges, agent_event_id, tool_id, "tool-invocation")
        if any(token in command for token in ("write_file", "create_", "delete_", "send_", "pay_", "transfer_", "rate_")):
            state_id = f"evt_{len(events):03d}"
            events.append(_event(state_id, "StateDeltaRecord", step_id, "tool", command, source_ref))
            _add_edge(edges, tool_id, state_id, "causal")
        if "complete_task" in command:
            submit_id = f"evt_{len(events):03d}"
            events.append(_event(submit_id, "SubmissionEvent", step_id, "submitter", command, source_ref))
            _add_edge(edges, tool_id, submit_id, "causal")
    return agent_event_id, pending_tool_calls, commands


def compile_appworld_htir(task_id: str, failure_category: str, paths: dict[str, str],
                          problem_statement: str) -> dict[str, Any]:
    traj = _read_json(paths["traj_path"])
    eval_result = _load_optional_json(paths.get("result_path"))
    eval_report = _load_optional_text(paths.get("eval_report_path"))
    events: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    evaluator_anchors: list[dict[str, Any]] = []
    commands: list[str] = []
    messages = traj.get("messages", [])
    model_calls = traj.get("model_calls") or []

    events.append(
        _event(
            "evt_000",
            "TaskSpecRecord",
            0,
            "prompt",
            problem_statement,
            "traj.messages[1]",
            {"failure_category": failure_category},
        )
    )

    last_agent_event = _append_context_node(
        events,
        edges,
        "evt_000",
        0,
        "Initial AppWorld task context assembled from task instruction, app APIs, and harness policy.",
        "traj.messages[0:2]",
        {
            "message_count": min(len(messages), 2),
            "model_call_count": len(model_calls),
            "required_apps": traj.get("info", {}).get("required_apps", []),
        },
    )
    step_id = 0
    pending_tool_calls: list[str] = []

    if model_calls:
        for call_idx, call in enumerate(model_calls, start=1):
            if not isinstance(call, dict):
                continue
            step_id += 1
            response_message = call.get("response_message") if isinstance(call.get("response_message"), dict) else {}
            request_messages = call.get("request_messages") if isinstance(call.get("request_messages"), list) else []
            content = str(response_message.get("content") or "") if isinstance(response_message, dict) else ""
            actions = [{"command": command} for command in _appworld_extract_code_blocks(content)]
            last_agent_event, pending_tool_calls, step_commands = _append_appworld_agent_step(
                events,
                edges,
                step_id=step_id,
                last_agent_event=last_agent_event,
                summary=_appworld_model_call_summary(response_message),
                source_ref=f"traj.model_calls[{call_idx - 1}].response_message",
                attributes={
                    "model_call_index": call.get("call_index", call_idx),
                    "request_message_count": len(request_messages),
                    "_request_context_text": _flatten_text(request_messages),
                    "request_summary": _appworld_request_summary(request_messages),
                    "response_summary": _appworld_model_call_summary(response_message),
                    "response_message": response_message,
                    "usage": call.get("usage"),
                    "finish_reason": call.get("finish_reason"),
                },
                actions=actions,
            )
            commands.extend(step_commands)
            next_call = model_calls[call_idx] if call_idx < len(model_calls) and isinstance(model_calls[call_idx], dict) else None
            next_request = next_call.get("request_messages") if isinstance(next_call, dict) and isinstance(next_call.get("request_messages"), list) else []
            observation_messages = next_request[len(request_messages) + 1 :] if len(next_request) > len(request_messages) else []
            for obs_idx, obs_message in enumerate(observation_messages, start=1):
                if not isinstance(obs_message, dict) or obs_message.get("role") != "user":
                    continue
                obs_id = f"evt_{len(events):03d}"
                output = _appworld_model_call_summary(obs_message)
                events.append(
                    _event(
                        obs_id,
                        "ToolResultRecord",
                        step_id,
                        "tool",
                        output,
                        f"traj.model_calls[{call_idx}].request_messages[{len(request_messages) + obs_idx}]",
                    )
                )
                if pending_tool_calls:
                    _add_edge(edges, pending_tool_calls.pop(0), obs_id, "tool-invocation")
                else:
                    _add_edge(edges, last_agent_event, obs_id, "temporal")
                if _looks_like_exception(output, ""):
                    ex_id = f"evt_{len(events):03d}"
                    events.append(_event(ex_id, "ExceptionEvent", step_id, "controller", output, f"traj.model_calls[{call_idx}].request_messages"))
                    _add_edge(edges, obs_id, ex_id, "causal")
    else:
        for msg_idx, message in enumerate(messages[2:], start=2):
            role = message.get("role")
            if role == "assistant":
                step_id += 1
                actions = message.get("extra", {}).get("actions", [])
                last_agent_event, pending_tool_calls, step_commands = _append_appworld_agent_step(
                    events,
                    edges,
                    step_id=step_id,
                    last_agent_event=last_agent_event,
                    summary=message.get("content", ""),
                    source_ref=f"traj.messages[{msg_idx}]",
                    actions=actions,
                )
                commands.extend(step_commands)
            elif role == "user":
                obs_id = f"evt_{len(events):03d}"
                output = message.get("content", "")
                extra = message.get("extra", {})
                events.append(
                    _event(
                        obs_id,
                        "ToolResultRecord",
                        step_id,
                        "tool",
                        output,
                        f"traj.messages[{msg_idx}]",
                        {"returncode": extra.get("returncode"), "exception_info": extra.get("exception_info")},
                    )
                )
                if pending_tool_calls:
                    _add_edge(edges, pending_tool_calls.pop(0), obs_id, "tool-invocation")
                else:
                    _add_edge(edges, last_agent_event, obs_id, "temporal")
                if _looks_like_exception(output, str(extra.get("exception_info", ""))):
                    ex_id = f"evt_{len(events):03d}"
                    events.append(_event(ex_id, "ExceptionEvent", step_id, "controller", extra.get("exception_info") or output, f"traj.messages[{msg_idx}]"))
                    _add_edge(edges, obs_id, ex_id, "causal")

    if eval_result:
        evaluator_anchors.append(
            _extract_evaluator_anchor(
                "result.json",
                json.dumps(eval_result, ensure_ascii=False),
                eval_result,
            )
        )
    run_effect_evidence = _append_appworld_run_effect_nodes(
        events,
        edges,
        traj_path=paths.get("traj_path"),
        task_id=task_id,
        final_step_id=step_id + 1,
    )
    if eval_report:
        verification_id = f"evt_{len(events):03d}"
        events.append(
            _event(
                verification_id,
                "VerificationEvent",
                step_id + 1,
                "validator",
                eval_report,
                "eval_report.md",
            )
        )

    stats = {
        "exit_status": traj.get("info", {}).get("exit_status"),
        "agent_exit_status": traj.get("info", {}).get("agent_exit_status"),
        "pass_percentage": traj.get("info", {}).get("pass_percentage"),
        "required_apps": traj.get("info", {}).get("required_apps", []),
        "api_calls": traj.get("info", {}).get("model_stats", {}).get("api_calls") or len(model_calls),
        "model_call_count": len(model_calls),
        "effect_evidence": {
            "api_call_count": run_effect_evidence.get("api_call_count", 0) if run_effect_evidence else 0,
            "mutating_api_call_count": len(run_effect_evidence.get("mutating_api_calls", [])) if run_effect_evidence else 0,
            "db_change_count": len(run_effect_evidence.get("db_changes", [])) if run_effect_evidence else 0,
            "has_task_state_change": run_effect_evidence.get("has_task_state_change") if run_effect_evidence else None,
        },
        **_repetition_stats(commands),
    }
    views = _build_views(events, edges, evaluator_anchors)
    return _finalize_paper_htir({
        "mode": "appworld",
        "instance_id": task_id,
        "graph": {"nodes": events, "events": events, "edges": edges},
        "evaluator_anchors": evaluator_anchors,
        "views": views,
        "stats": stats,
    })


def render_view_summary(bundle: dict[str, Any], max_events_per_view: int | None = None) -> str:
    lines = [
        f"mode={bundle['mode']}",
        f"instance_id={bundle['instance_id']}",
        f"htir_version={bundle.get('htir_version', 'legacy')}",
        f"stats={json.dumps(bundle.get('stats', {}), ensure_ascii=False)}",
    ]

    steps = bundle.get("agent_trace_steps") or []
    facets = bundle.get("node_facets") or {}
    if steps:
        limit = max_events_per_view if max_events_per_view is not None else 10
        lines.append("\n[agent_trace_steps]")
        for step in steps[:limit]:
            step_id = step.get("step_id", step.get("node_id", "?"))
            facet = facets.get(step_id, {})
            layers = facet.get("harness_layer", {}).get("implicated_layers", [])
            artifact = facet.get("artifact_state", {}).get("external_effect", {})
            artifact_summary = artifact.get("summary") if isinstance(artifact, dict) else ""
            data_context = facet.get("data_and_context", {})
            context_sources = data_context.get("incoming_context_sources", []) or []
            lines.append(
                f"- {step_id} | role={step.get('role')} | status={step.get('execution_status')} | "
                f"layers={layers} | context_sources={len(context_sources)} | source={step.get('source_ref', '')}"
            )
            if context_sources:
                previews = []
                for source in context_sources[:2]:
                    attrs = source.get("attributes", {}) if isinstance(source, dict) else {}
                    previews.append(
                        f"{source.get('from_node_id')}:{_clean_text(str(attrs.get('matched_text', '')), 120)}"
                    )
                lines.append(f"  incoming_context: {' | '.join(previews)}")
            if step.get("request_message"):
                lines.append(f"  request: {_clean_text(step.get('request_message', ''), 320)}")
            if step.get("response_message"):
                lines.append(f"  response/effect: {_clean_text(step.get('response_message', ''), 320)}")
            if artifact_summary:
                lines.append(f"  artifact_state: {_clean_text(artifact_summary, 260)}")
        if len(steps) > limit:
            lines.append(f"- ... {len(steps) - limit} additional AgentTraceStep records omitted")

    anchors = bundle.get("outcome_anchors") or bundle.get("views", {}).get("evaluator_anchors") or bundle.get("evaluator_anchors", [])
    if max_events_per_view is not None:
        anchors = anchors[:max_events_per_view]
    if anchors:
        lines.append("\n[outcome_anchors]")
        for anchor in anchors:
            lines.append(f"- {anchor['anchor_id']} | source={anchor['source_ref']} | {anchor['summary']}")

    base_graph = bundle.get("base_graph") or {}
    if base_graph:
        edge_families: dict[str, int] = {}
        for edge in base_graph.get("edges", []):
            family = edge.get("family", "unknown")
            edge_families[family] = edge_families.get(family, 0) + 1
        lines.append("\n[base_graph]")
        lines.append(
            f"- step_nodes={len(base_graph.get('nodes', []))} edge_families="
            f"{json.dumps(edge_families, ensure_ascii=False, sort_keys=True)} "
            f"anchor_links={len(base_graph.get('anchor_links', []))}"
        )

    if facets:
        limit = max_events_per_view if max_events_per_view is not None else 8
        lines.append("\n[node_local_facets]")
        for step_id, facet in list(facets.items())[:limit]:
            layers = facet.get("harness_layer", {}).get("implicated_layers", [])
            control_status = facet.get("control_flow", {}).get("status")
            data_context = facet.get("data_and_context", {})
            data_count = len(data_context.get("evidence_refs", []) or [])
            context_source_count = len(data_context.get("incoming_context_sources", []) or [])
            artifact_count = len(facet.get("artifact_state", {}).get("evidence_refs", []) or [])
            lines.append(
                f"- {step_id} | data_refs={data_count} | context_sources={context_source_count} | "
                f"control_status={control_status} | artifact_refs={artifact_count} | harness_layers={layers}"
            )
        if len(facets) > limit:
            lines.append(f"- ... {len(facets) - limit} additional node facet records omitted")

    # Legacy views are retained as a compact fallback for existing prompts and tools.
    for view_name in ("failure", "causal"):
        nodes = bundle.get("views", {}).get(view_name, [])
        if max_events_per_view is not None:
            nodes = nodes[:max_events_per_view]
        if not nodes:
            continue
        lines.append(f"\n[legacy_{view_name}_view]")
        for node in nodes:
            lines.append(
                f"- {node.get('node_id', node.get('event_id', '?'))} | {node.get('type', '?')} | "
                f"component={node.get('component', '?')} | source={node.get('source_ref', '?')} | "
                f"{node.get('summary', '')}"
            )
    return "\n".join(lines)


def write_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    return out_path

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from failure_analysis.operator_registry import (
    SEVERITY_ORDER,
    infer_defect_class,
    infer_severity,
    load_operator_registry,
    normalize_defect_class,
    normalize_operator_family,
    operator_allowed_paths,
    recommend_operator_family,
)

CLUSTER_TEXT_FIELD_CHARS = 700
CLUSTER_EVIDENCE_CHARS = 800
CLUSTER_MAX_EVIDENCE_PER_CLUSTER = 3


def _compact_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= limit:
        return text
    head = max(limit // 2, 1)
    tail = max(limit - head, 1)
    omitted = len(text) - head - tail
    return f"{text[:head]}...[{omitted} chars omitted]...{text[-tail:]}"


def _record_summary(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key, ""))
        for key in ("failure_manifestation", "failure_reason", "agent_design_issue")
    ).strip()



def _default_layers(record: dict[str, Any]) -> list[str]:
    layers = record.get("implicated_harness_layers") or record.get("harness_layers") or []
    if isinstance(layers, str):
        layers = [layers]
    layers = [str(layer) for layer in layers if str(layer).strip()]
    if layers:
        return sorted(set(layers))
    component = str(record.get("affected_component") or "unknown")
    mapping = {
        "system_prompt": "Context and Memory",
        "prompt": "Context and Memory",
        "web_search": "Tool Interface",
        "tool_use": "Tool Interface",
        "file_handling": "Tool Interface",
        "file_editing": "Tool Interface",
        "api_execution": "Tool Interface",
        "state_guard": "Execution Environment and Sandbox",
        "testing": "Verification and Evaluation",
        "validator": "Verification and Evaluation",
        "answer_extraction": "Verification and Evaluation",
        "format_parsing": "Tool Interface",
        "main_loop": "Lifecycle and Orchestration",
        "error_handling": "Lifecycle and Orchestration",
    }
    return [mapping.get(component, component or "unknown")]

def normalize_diagnosis(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    normalized["defect_class"] = normalize_defect_class(normalized.get("defect_class")) or infer_defect_class(normalized)
    normalized["recommended_operator_family"] = (
        normalize_operator_family(normalized.get("recommended_operator_family")) or recommend_operator_family(normalized)
    )
    normalized["severity"] = normalized.get("severity") or infer_severity(normalized)
    normalized["confidence"] = normalized.get("confidence") or "medium"
    normalized["affected_component"] = normalized.get("affected_component") or "unknown"
    normalized["implicated_harness_layers"] = _default_layers(normalized)
    normalized["summary"] = _record_summary(normalized)
    return normalized


def _ordered_union(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _registry_union(registry: dict[str, dict[str, Any]], families: list[str], field: str) -> list[str]:
    values: list[str] = []
    for family in families:
        definition = registry.get(family)
        if not definition:
            continue
        values.extend(str(value) for value in definition.get(field, []) if value)
    return _ordered_union(values)


def consolidate_diagnoses(results: list[dict[str, Any]], mode: str, min_frequency: int = 1) -> list[dict[str, Any]]:
    """Build coarse layer buckets; leave semantic subclustering to aggregation LLM.

    The deterministic stage only groups by implicated harness layer. Each bucket
    retains the per-diagnosis defect/operator labels as distributions so the LLM
    can decide finer root-cause clusters from evidence rather than from a fixed
    rule key.
    """
    registry = load_operator_registry()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    normalized = [normalize_diagnosis(result) for result in results]

    for record in normalized:
        for layer in record.get("implicated_harness_layers", []) or [record["affected_component"]]:
            buckets[str(layer)].append(record)

    clusters: list[dict[str, Any]] = []
    for layer, items in buckets.items():
        if len(items) < min_frequency:
            continue
        severity_dist = Counter(item["severity"] for item in items)
        category_dist = Counter(item.get("failure_category", "unknown") for item in items)
        defect_dist = Counter(item["defect_class"] for item in items)
        operator_dist = Counter(item["recommended_operator_family"] for item in items)
        component_dist = Counter(item.get("affected_component", "unknown") for item in items)
        operator_families = [family for family, _ in operator_dist.most_common() if family in registry]
        representative = sorted(
            items,
            key=lambda item: (
                -SEVERITY_ORDER.get(item["severity"], 0),
                item.get("instance_id", ""),
            ),
        )
        evidence_patterns = []
        for item in representative:
            anchor = item.get("evidence_anchor", {}) or {}
            evidence_patterns.append(
                {
                    "instance_id": item.get("instance_id"),
                    "affected_component": item.get("affected_component"),
                    "defect_class": item.get("defect_class"),
                    "operator_family": item.get("recommended_operator_family"),
                    "summary": item.get("summary"),
                    "evidence_anchor": anchor,
                    "evidence_spans": item.get("evidence_spans", []),
                    "responsible_steps": item.get("responsible_steps", []),
                    "implicated_harness_layers": item.get("implicated_harness_layers", []),
                }
            )
        cluster_name = f"layer:{layer}"
        cluster_id = hashlib.sha1(cluster_name.encode("utf-8")).hexdigest()[:10]
        primary_defect = defect_dist.most_common(1)[0][0] if defect_dist else "unknown"
        primary_operator = operator_dist.most_common(1)[0][0] if operator_dist else "unknown"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "cluster_scope": "implicated_harness_layer",
                "affected_component": layer,
                "implicated_harness_layer": layer,
                "defect_class": primary_defect,
                "operator_family": primary_operator,
                "defect_distribution": dict(defect_dist.most_common()),
                "operator_distribution": dict(operator_dist.most_common()),
                "component_distribution": dict(component_dist.most_common()),
                "frequency": len(items),
                "severity_distribution": dict(severity_dist),
                "failure_categories": dict(category_dist),
                "representative_instances": [item.get("instance_id") for item in representative],
                "root_cause_hypothesis": representative[0].get("agent_design_issue", "") if representative else "",
                "evidence_patterns": evidence_patterns,
                "target_metrics": _registry_union(registry, operator_families, "target_metrics"),
                "regression_risks": _registry_union(registry, operator_families, "regression_risks"),
                "static_checks": _registry_union(registry, operator_families, "static_checks"),
                "rollback_conditions": _registry_union(registry, operator_families, "rollback_conditions"),
                "allowed_paths": _ordered_union(
                    [path for family in operator_families for path in operator_allowed_paths(mode, family)]
                ),
            }
        )

    clusters.sort(
        key=lambda cluster: (
            -cluster["frequency"],
            -sum(
                count * SEVERITY_ORDER.get(level, 0)
                for level, count in cluster["severity_distribution"].items()
            ),
            cluster["cluster_name"],
        )
    )
    return clusters


def format_clusters_for_prompt(clusters: list[dict[str, Any]], max_clusters: int | None = None) -> str:
    lines = []
    selected_clusters = clusters if max_clusters is None else clusters[:max_clusters]
    for cluster in selected_clusters:
        layer = cluster.get("implicated_harness_layer", cluster["affected_component"])
        lines.append(
            f"[{cluster['cluster_id']}] scope=layer_only layer={layer} "
            f"freq={cluster['frequency']} categories={json.dumps(cluster['failure_categories'], ensure_ascii=False)}"
        )
        lines.append(
            f"  defect_distribution: {json.dumps(cluster.get('defect_distribution', {}), ensure_ascii=False)}"
        )
        lines.append(
            f"  operator_distribution: {json.dumps(cluster.get('operator_distribution', {}), ensure_ascii=False)}"
        )
        lines.append(
            f"  component_distribution: {json.dumps(cluster.get('component_distribution', {}), ensure_ascii=False)}"
        )
        if cluster.get("target_metrics"):
            lines.append(f"  candidate_target_metrics: {', '.join(cluster['target_metrics'])}")
        lines.append(
            "  instruction: this is a coarse harness-layer bucket; split it into semantic "
            "root-cause clusters before proposing fixes."
        )
        lines.append(f"  root_cause_seed: {_compact_text(cluster['root_cause_hypothesis'], CLUSTER_TEXT_FIELD_CHARS)}")
        evidence_patterns = cluster["evidence_patterns"][:CLUSTER_MAX_EVIDENCE_PER_CLUSTER]
        for evidence in evidence_patterns:
            lines.append(
                f"  - {evidence['instance_id']}: component={evidence.get('affected_component')} "
                f"defect={evidence.get('defect_class')} operator={evidence.get('operator_family')} "
                f"steps={evidence.get('responsible_steps', [])} | "
                f"{_compact_text(evidence['summary'], CLUSTER_TEXT_FIELD_CHARS)} | "
                f"anchor={_compact_text(evidence['evidence_anchor'], CLUSTER_EVIDENCE_CHARS)}"
            )
        omitted = len(cluster["evidence_patterns"]) - len(evidence_patterns)
        if omitted > 0:
            lines.append(f"  - ... {omitted} additional evidence records omitted from aggregate prompt")
    return "\n".join(lines)


def _fix_context_record(fix: dict[str, Any]) -> dict[str, Any]:
    return {
        "affected_component": fix.get("affected_component") or fix.get("implicated_harness_layer") or "",
        "defect_class": fix.get("target_defect_class"),
        "recommended_operator_family": fix.get("operator_family"),
        "failure_manifestation": fix.get("problem_statement", ""),
        "failure_reason": fix.get("required_behavior_delta", ""),
        "agent_design_issue": " ".join(
            str(part)
            for part in (
                fix.get("title", ""),
                fix.get("problem_statement", ""),
                fix.get("required_behavior_delta", ""),
                json.dumps(fix.get("implementation_steps", []), ensure_ascii=False),
            )
            if part
        ),
        "fix_scope": fix.get("fix_scope", ""),
        "defect_labels": fix.get("defect_labels", []),
    }


def _ordered_intersection(first: list[str], second: list[str]) -> list[str]:
    second_set = set(second)
    return [item for item in first if item in second_set]


def _fix_active_files(fix: dict[str, Any], allowed_paths: list[str]) -> list[str]:
    target_files = [str(path) for path in fix.get("target_files", []) if path]
    active = _ordered_intersection(target_files, allowed_paths)
    if active:
        return active
    # Fall back to operator-allowed paths so the modify agent inspects the real
    # active surface before deciding whether target_files need adjustment.
    return list(allowed_paths)


def _merge_ordered(existing: Any, additions: list[str]) -> list[str]:
    values: list[str] = []
    if isinstance(existing, list):
        values.extend(str(item) for item in existing if item)
    values.extend(str(item) for item in additions if item)
    return _ordered_union(values)


def _cluster_for_fix(fix: dict[str, Any], clusters_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    cluster_ids: list[str] = []
    for key in ("cluster_id", "source_cluster_id"):
        value = fix.get(key)
        if isinstance(value, str):
            cluster_ids.append(value)
    value = fix.get("source_cluster_ids") or fix.get("cluster_ids")
    if isinstance(value, list):
        cluster_ids.extend(str(item) for item in value)
    for cluster_id in cluster_ids:
        if cluster_id in clusters_by_id:
            return clusters_by_id[cluster_id]
    return None


def enrich_spec_with_clusters(spec: dict[str, Any], clusters: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    registry = load_operator_registry()
    enriched = json.loads(json.dumps(spec))
    fixes = enriched.get("fixes", [])
    clusters_by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    for fix in fixes:
        cluster = _cluster_for_fix(fix, clusters_by_id)
        if cluster:
            fix.setdefault("cluster_id", cluster["cluster_id"])
            fix.setdefault("source_cluster_ids", [cluster["cluster_id"]])
            fix.setdefault(
                "preconditions",
                [f"Observed coarse harness-layer bucket {cluster['cluster_id']} recurs in failed traces."],
            )
            fix.setdefault("implicated_harness_layer", cluster.get("implicated_harness_layer"))
        context = _fix_context_record(fix)
        defect_class = normalize_defect_class(fix.get("target_defect_class")) or infer_defect_class(context)
        context["defect_class"] = defect_class
        operator_family = normalize_operator_family(fix.get("operator_family")) or recommend_operator_family(context)
        fix["target_defect_class"] = defect_class
        fix["operator_family"] = operator_family
        operator = registry[operator_family]
        fix.setdefault("target_metrics", list(operator["target_metrics"]))
        fix.setdefault("static_checks", list(operator["static_checks"]))
        fix.setdefault("rollback_conditions", list(operator["rollback_conditions"]))
        fix.setdefault("regression_risks", list(operator["regression_risks"]))
        allowed = operator_allowed_paths(mode, operator_family)
        active_files = _fix_active_files(fix, allowed)
        fix["active_files"] = _merge_ordered(fix.get("active_files"), active_files)
        fix["must_inspect_files"] = _merge_ordered(
            fix.get("must_inspect_files"),
            fix["active_files"] + [path for path in allowed if path not in fix["active_files"]],
        )
        fix.setdefault("do_not_edit_until_inspected", True)
        for path in allowed:
            if path not in enriched.setdefault("edit_budget", {}).setdefault("allowed_paths", []):
                enriched["edit_budget"]["allowed_paths"].append(path)
        budget = enriched.setdefault("edit_budget", {})
        budget["active_files"] = _merge_ordered(budget.get("active_files"), fix["active_files"])
        budget["must_inspect_files"] = _merge_ordered(budget.get("must_inspect_files"), fix["must_inspect_files"])
        budget.setdefault("do_not_edit_until_inspected", True)
    enriched.setdefault("plan_metadata", {})["cluster_count"] = len(clusters)
    enriched.setdefault("plan_metadata", {})["cluster_scope"] = "implicated_harness_layer"
    return enriched

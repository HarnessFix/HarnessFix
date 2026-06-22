from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_memory_root(repo_root: Path) -> Path:
    return repo_root / "failure_analysis" / "memory"


def _memory_file(memory_root: Path, outcome: str) -> Path:
    return memory_root / f"{outcome}_repairs.jsonl"


def load_memory_entries(memory_root: Path, outcome: str | None = None) -> list[dict[str, Any]]:
    outcomes = [outcome] if outcome else ["accepted", "rejected"]
    entries: list[dict[str, Any]] = []
    for item in outcomes:
        path = _memory_file(memory_root, item)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _cluster_label_sets(clusters: list[dict[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    components: set[str] = set()
    defects: set[str] = set()
    operators: set[str] = set()
    for cluster in clusters:
        components.add(str(cluster.get("affected_component", "")))
        components.add(str(cluster.get("implicated_harness_layer", "")))
        components.update(str(key) for key in (cluster.get("component_distribution") or {}).keys())
        defects.add(str(cluster.get("defect_class", "")))
        defects.update(str(key) for key in (cluster.get("defect_distribution") or {}).keys())
        operators.add(str(cluster.get("operator_family", "")))
        operators.update(str(key) for key in (cluster.get("operator_distribution") or {}).keys())
    return components, defects, operators


def _score_memory(entry: dict[str, Any], mode: str, clusters: list[dict[str, Any]]) -> int:
    score = 0
    if entry.get("mode") == mode:
        score += 2
    cluster_components, cluster_defects, cluster_operators = _cluster_label_sets(clusters)
    for component in entry.get("affected_components", []):
        if component in cluster_components:
            score += 2
    for defect in entry.get("defect_classes", []):
        if defect in cluster_defects:
            score += 2
    for operator in entry.get("operator_families", []):
        if operator in cluster_operators:
            score += 3
    return score


def retrieve_relevant_memories(memory_root: Path, mode: str, clusters: list[dict[str, Any]],
                               limit: int | None = None) -> list[dict[str, Any]]:
    scored = []
    for entry in load_memory_entries(memory_root):
        score = _score_memory(entry, mode, clusters)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda item: (-item[0], item[1].get("timestamp", "")), reverse=False)
    entries = [entry for _, entry in scored]
    return entries if limit is None else entries[:limit]


def format_memories_for_prompt(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "(no prior accepted/rejected repair memories)"
    lines = []
    for entry in entries:
        lines.append(
            f"- [{entry.get('outcome')}] version={entry.get('version')} "
            f"operators={', '.join(entry.get('operator_families', []))} "
            f"defects={', '.join(entry.get('defect_classes', []))}"
        )
        lines.append(f"  summary: {entry.get('summary', '')}")
        if entry.get("regression_causes"):
            lines.append(f"  regressions: {', '.join(entry['regression_causes'])}")
        if entry.get("changed_files"):
            lines.append(f"  changed_files: {', '.join(entry['changed_files'])}")
    return "\n".join(lines)


def build_memory_entry(*, mode: str, version: int, outcome: str, plan_path: Path, spec: dict[str, Any],
                       summary: str, changed_files: list[str], audit: dict[str, Any] | None,
                       gate: dict[str, Any] | None) -> dict[str, Any]:
    fixes = spec.get("fixes", [])
    defect_classes = sorted({fix.get("target_defect_class", "unknown") for fix in fixes})
    operator_families = sorted({fix.get("operator_family", "unknown") for fix in fixes})
    affected_components = sorted(
        {
            fix.get("problem_statement", "").split(":", 1)[0]
            for fix in fixes
            if fix.get("problem_statement")
        }
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "version": version,
        "outcome": outcome,
        "plan_path": str(plan_path),
        "summary": summary,
        "defect_classes": defect_classes,
        "operator_families": operator_families,
        "affected_components": affected_components,
        "changed_files": changed_files,
        "audit_passed": None if audit is None else audit.get("passed"),
        "gate_passed": None if gate is None else gate.get("passed"),
        "regression_causes": [] if gate is None else gate.get("failure_reasons", []),
        "improved_ids": [] if gate is None else gate.get("improved_ids", []),
        "regressed_ids": [] if gate is None else gate.get("regressed_ids", []),
    }


def store_memory_entry(memory_root: Path, entry: dict[str, Any]) -> Path:
    memory_root.mkdir(parents=True, exist_ok=True)
    path = _memory_file(memory_root, entry["outcome"])
    with path.open("a") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path

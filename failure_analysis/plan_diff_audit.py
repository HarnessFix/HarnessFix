#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import filecmp
from pathlib import Path
from typing import Any


def _all_relative_files(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".git" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.parts)
        and not path.name.endswith(".pyc")
    }


def _changed_files(original_dir: Path, candidate_dir: Path) -> list[str]:
    original_files = _all_relative_files(original_dir)
    candidate_files = _all_relative_files(candidate_dir)
    all_files = sorted(original_files | candidate_files)
    changed = []
    for rel_path in all_files:
        original_file = original_dir / rel_path
        candidate_file = candidate_dir / rel_path
        if not original_file.exists() or not candidate_file.exists():
            changed.append(rel_path)
            continue
        if not filecmp.cmp(original_file, candidate_file, shallow=False):
            changed.append(rel_path)
    return changed


def _path_allowed(rel_path: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        normalized = prefix.strip()
        if not normalized:
            continue
        if normalized.endswith("/..."):
            root = normalized[:-4].rstrip("/")
            if rel_path == root or rel_path.startswith(root + "/"):
                return True
            continue
        if normalized.endswith("/**"):
            root = normalized[:-3].rstrip("/")
            if rel_path == root or rel_path.startswith(root + "/"):
                return True
            continue
        if rel_path == normalized or rel_path.startswith(normalized.rstrip("/") + "/"):
            return True
    return False


def _syntax_check(candidate_dir: Path, changed_files: list[str]) -> list[str]:
    failures = []
    for rel_path in changed_files:
        if not rel_path.endswith(".py"):
            continue
        file_path = candidate_dir / rel_path
        try:
            compile(file_path.read_text(), str(file_path), "exec")
        except Exception as exc:
            failures.append(f"{rel_path}: {type(exc).__name__}: {exc}")
    return failures


def audit_candidate(original_dir: Path, candidate_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed_files = _changed_files(original_dir, candidate_dir)
    budget = spec.get("edit_budget", {})
    allowed_paths = list(budget.get("allowed_paths", []))
    forbidden_paths = list(budget.get("forbidden_paths", []))
    max_files = int(budget.get("max_files_to_modify", max(len(changed_files), 1)))

    forbidden_touches = [path for path in changed_files if _path_allowed(path, forbidden_paths)]
    out_of_scope = [path for path in changed_files if allowed_paths and not _path_allowed(path, allowed_paths)]
    fix_coverage = {}
    uncovered_fixes = []
    for fix in spec.get("fixes", []):
        target_files = fix.get("target_files", [])
        touched = [path for path in changed_files if _path_allowed(path, target_files)]
        covered = bool(touched)
        fix_coverage[fix.get("id", "unknown")] = {"covered": covered, "touched_files": touched}
        if not covered:
            uncovered_fixes.append(fix.get("id", "unknown"))

    syntax_failures = _syntax_check(candidate_dir, changed_files)
    violations = []
    if len(changed_files) > max_files:
        violations.append(f"changed_files_exceed_budget:{len(changed_files)}>{max_files}")
    if forbidden_touches:
        violations.append("forbidden_paths_touched")
    if out_of_scope:
        violations.append("changed_files_outside_allowed_paths")
    if uncovered_fixes and spec.get("fixes"):
        violations.append("uncovered_fix_targets")
    if syntax_failures:
        violations.append("syntax_failures")

    return {
        "passed": not violations,
        "changed_files": changed_files,
        "changed_file_count": len(changed_files),
        "max_files_to_modify": max_files,
        "forbidden_touches": forbidden_touches,
        "out_of_scope": out_of_scope,
        "syntax_failures": syntax_failures,
        "fix_coverage": fix_coverage,
        "violations": violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit plan-to-diff consistency for a candidate harness")
    parser.add_argument("--mode", choices=["swe", "gaia", "appworld", "terminal_bench"], required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    result = audit_candidate(args.original_dir, args.candidate_dir, spec)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

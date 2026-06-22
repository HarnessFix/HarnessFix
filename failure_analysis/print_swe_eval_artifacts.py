#!/usr/bin/env python3
"""Print bounded SWE-bench evaluation artifacts for failure-analysis agents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FAILURE_PATTERNS = (
    r"(?m)^FAILED\s+.*$",
    r"(?m)^ERROR\s+.*$",
    r"(?m)^E\s+.*$",
    r"(?m)^AssertionError.*$",
    r"(?m)^Traceback \(most recent call last\):.*$",
    r"(?m)^={3,}\s*(?:FAILURES|ERRORS|short test summary info).*={3,}\s*$",
)


def _clip(text: str, limit: int) -> str:
    text = text.replace("\x00", "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 80)] + "\n... [truncated]\n"


def _json_summary(value: Any, limit: int) -> str:
    return _clip(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), limit)


def _report_summary(path: Path, limit: int) -> str:
    if not path.exists():
        return "No eval report available"
    text = path.read_text(errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _clip(text, limit)

    keys = [
        "resolved",
        "tests_status",
        "fail_to_pass",
        "pass_to_pass",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "error",
    ]
    summary = {key: data[key] for key in keys if key in data}
    if not summary:
        summary = data
    return _json_summary(summary, limit)


def _unique_lines(lines: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        cleaned = line.rstrip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _context_around_matches(text: str, window: int, max_blocks: int) -> list[str]:
    lines = text.splitlines()
    selected: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for pattern in FAILURE_PATTERNS:
        for match in re.finditer(pattern, text):
            line_index = text[: match.start()].count("\n")
            start = max(0, line_index - window)
            end = min(len(lines), line_index + window + 1)
            if any(start <= prev_end and end >= prev_start for prev_start, prev_end in used_ranges):
                continue
            used_ranges.append((start, end))
            selected.append("\n".join(lines[start:end]))
            if len(selected) >= max_blocks:
                return selected
    return selected


def _test_output_summary(path: Path, limit: int) -> str:
    if not path.exists():
        return "No test output available"
    text = path.read_text(errors="replace")
    if len(text) <= limit:
        return text

    matches: list[str] = []
    for pattern in FAILURE_PATTERNS:
        matches.extend(match.group(0) for match in re.finditer(pattern, text))
    match_lines = _unique_lines(matches, 80)
    context_blocks = _context_around_matches(text, window=3, max_blocks=10)
    tail = "\n".join(text.splitlines()[-80:])

    pieces = [
        f"Raw test output length: {len(text)} chars",
        "",
        "Failure/error lines:",
        "\n".join(match_lines) if match_lines else "(none found)",
        "",
        "Context around first failure/error matches:",
        "\n\n---\n\n".join(context_blocks) if context_blocks else "(none found)",
        "",
        "Tail:",
        tail,
    ]
    return _clip("\n".join(pieces), limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--char-limit", type=int, default=45000)
    args = parser.parse_args()

    if args.report:
        print("## Eval Report")
        print(_report_summary(args.report, args.char_limit // 3))
    if args.test_output:
        if args.report:
            print()
        print("## Test Output")
        print(_test_output_summary(args.test_output, args.char_limit))


if __name__ == "__main__":
    main()

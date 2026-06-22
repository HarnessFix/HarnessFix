#!/usr/bin/env python3
"""Download Terminal-Bench 2.0 locally, then create HarnessFix splits.

This script clones a Terminal-Bench 2.0 source into data/terminal_bench_2_verified
by default and then invokes data/sample_terminal_bench.py. The source is kept local so the
benchmark files can be inspected or modified later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "terminal_bench_2_verified"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "terminal_bench_splits.json"
SOURCES = {
    "zai_verified": "https://huggingface.co/datasets/zai-org/terminal-bench-2-verified",
    "github": "https://github.com/harbor-framework/terminal-bench-2.git",
    "huggingface": "https://huggingface.co/datasets/harborframework/terminal-bench-2.0",
}


def _pinned_revision(source: str, manifest_path: Path) -> str | None:
    if source != "zai_verified":
        return None
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("source_commit")


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Terminal-Bench 2.0 and create HarnessFix splits")
    parser.add_argument("--source", choices=sorted(SOURCES), default="zai_verified")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--revision", help="Git revision to check out after cloning")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--copy-splits", action="store_true", help="Copy split task dirs instead of symlinking")
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()

    dest = args.dest.resolve()
    if dest.exists():
        raise SystemExit(f"Destination already exists: {dest}")

    clone_cmd = ["git", "clone"]
    if args.depth > 0:
        clone_cmd.extend(["--depth", str(args.depth)])
    clone_cmd.extend([SOURCES[args.source], str(dest)])
    _run(clone_cmd)

    revision = args.revision or _pinned_revision(args.source, args.manifest)
    if revision:
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision])
        _run(["git", "-C", str(dest), "checkout", "--detach", revision])

    if not args.skip_sampling:
        sample_cmd = [sys.executable, str(REPO_ROOT / "data" / "sample_terminal_bench.py"), "--source", str(dest)]
        if args.copy_splits:
            sample_cmd.append("--copy")
        _run(sample_cmd)


if __name__ == "__main__":
    main()

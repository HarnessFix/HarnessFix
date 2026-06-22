#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ -n "${AGENT_SRC_DIR:-}" ]]; then
  export PYTHONPATH="$AGENT_SRC_DIR/harbor/src${PYTHONPATH:+:$PYTHONPATH}"
  exec "$PYTHON_BIN" "$AGENT_SRC_DIR/run_terminal_bench_entry.py" "$@"
fi

export PYTHONPATH="$SCRIPT_DIR/terminal_bench_agent/harbor/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/terminal_bench_agent/run_terminal_bench_entry.py" "$@"

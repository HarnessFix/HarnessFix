#!/usr/bin/env bash
# run_gaia.sh — GAIA benchmark runner for open_deep_research agent
#
# Usage examples:
#   ./run_gaia.sh --subset data/gaia_train_60 --model openai/gpt-5-mini
#   ./run_gaia.sh --subset data/gaia_train_60 --filter "<task_id>" --output traces/smoke
#   ./run_gaia.sh --subset data/gaia_val_30 --output traces/gaia_val_run
#
# Enhanced agent injection: set AGENT_SRC_DIR to the enhanced src/ directory.
#   The pipeline sets this automatically for enhanced versions:
#   AGENT_SRC_DIR=task_agent/enhanced_odr_v1/src ./run_gaia.sh ...
#
# Output layout (matches run_gaia_entry.py):
#   <output>/<task_id>/<task_id>.traj.json
#   <output>/preds.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"

# Load .env from project root
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

TRACES_DIR="$PROJECT_ROOT/traces"

# ── Default parameters ────────────────────────────────────────────────────────
SUBSET=""
MODEL="openai/gpt-5-mini"
WORKERS=1
CONCURRENCY=2
OUTPUT_DIR_OVERRIDE=""
FILTER=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)       SUBSET="$2";              shift 2 ;;
    --model)        MODEL="$2";               shift 2 ;;
    --workers)      WORKERS="$2";             shift 2 ;;
    --concurrency)  CONCURRENCY="$2";         shift 2 ;;
    --output|-o)    OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --filter)       FILTER="$2";              shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$SUBSET" ]]; then
  echo "ERROR: --subset is required"
  exit 1
fi

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Support both relative (relative to PROJECT_ROOT) and absolute paths for --subset
if [[ "$SUBSET" != /* ]]; then
  SUBSET="$PROJECT_ROOT/$SUBSET"
fi

# Build output directory
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  if [[ "$OUTPUT_DIR_OVERRIDE" = /* ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_OVERRIDE"
  fi
  RUN_ID="$(basename "$OUTPUT_DIR")"
else
  MODEL_SHORT="${MODEL//\//-}"
  SUBSET_NAME="$(basename "$SUBSET")"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ID="gaia_${SUBSET_NAME}_${MODEL_SHORT}_${TIMESTAMP}"
  OUTPUT_DIR="$TRACES_DIR/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  GAIA Runner"
echo "  Run ID  : $RUN_ID"
echo "  Subset  : $SUBSET"
echo "  Model   : $MODEL"
echo "  Workers : $WORKERS"
echo "  Output  : $OUTPUT_DIR"
echo "=========================================="

# ── Inject enhanced agent via PYTHONPATH ──────────────────────────────────────
# If AGENT_SRC_DIR is set (by pipeline for enhanced versions), prepend it.
# This ensures `from open_deep_research.agent import create_agent_team`
# resolves to the enhanced version instead of the installed package.
if [[ -n "${AGENT_SRC_DIR:-}" ]]; then
  echo "  Enhanced: PYTHONPATH=$AGENT_SRC_DIR"
  export PYTHONPATH="${AGENT_SRC_DIR}:$SCRIPT_DIR/open_deep_research/src:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SCRIPT_DIR/open_deep_research/src:${PYTHONPATH:-}"
fi

# ── Build command ─────────────────────────────────────────────────────────────
ENTRY_SCRIPT="$SCRIPT_DIR/open_deep_research/run_gaia_entry.py"
if [[ -n "${AGENT_SRC_DIR:-}" ]]; then
  ENHANCED_ENTRY="$(cd "$AGENT_SRC_DIR/.." && pwd)/run_gaia_entry.py"
  if [[ -f "$ENHANCED_ENTRY" ]]; then
    ENTRY_SCRIPT="$ENHANCED_ENTRY"
    echo "  Enhanced entry: $ENTRY_SCRIPT"
  fi
fi

CMD=(
  "$PYTHON_BIN" "$ENTRY_SCRIPT"
  --subset   "$SUBSET"
  --model    "$MODEL"
  --workers  "$WORKERS"
  --concurrency "$CONCURRENCY"
  --output   "$OUTPUT_DIR"
)

[[ -n "$FILTER" ]] && CMD+=(--filter "$FILTER")

echo "Running: ${CMD[*]}"
echo ""

# ── Execute ───────────────────────────────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "Traces saved to: $OUTPUT_DIR"
echo "  preds.json   : $OUTPUT_DIR/preds.json"
echo "  traj files   : $OUTPUT_DIR/<task_id>/<task_id>.traj.json"

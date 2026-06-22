#!/usr/bin/env bash
# run_appworld.sh — AppWorld benchmark runner for the local appworld_agent
#
# Usage examples:
#   ./run_appworld.sh --subset data/appworld_train_90 --model openai/gpt-5-mini
#   ./run_appworld.sh --subset data/appworld_val_45 --output traces/appworld_val_run
#   AGENT_SRC_DIR=task_agent/enhanced_appworld_v1/src ./run_appworld.sh --subset data/appworld_train_90

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi

TRACES_DIR="$PROJECT_ROOT/traces"
SUBSET=""
MODEL="openai/gpt-5-mini"
WORKERS=1
CONCURRENCY=2
OUTPUT_DIR_OVERRIDE=""
FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)       SUBSET="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --workers)      WORKERS="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY="$2"; shift 2 ;;
    --output|-o)    OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --filter)       FILTER="$2"; shift 2 ;;
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

if [[ "$SUBSET" != /* ]]; then
  SUBSET="$PROJECT_ROOT/$SUBSET"
fi

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
  RUN_ID="appworld_${SUBSET_NAME}_${MODEL_SHORT}_${TIMESTAMP}"
  OUTPUT_DIR="$TRACES_DIR/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  AppWorld Runner"
echo "  Run ID  : $RUN_ID"
echo "  Subset  : $SUBSET"
echo "  Model   : $MODEL"
echo "  Backend : official_simplified_react_code"
echo "  Workers : $WORKERS"
echo "  Output  : $OUTPUT_DIR"
echo "=========================================="

if [[ -n "${AGENT_SRC_DIR:-}" ]]; then
  echo "  Enhanced: PYTHONPATH=$AGENT_SRC_DIR"
  export PYTHONPATH="${AGENT_SRC_DIR}:$SCRIPT_DIR/appworld_agent/appworld_official_agents:${PYTHONPATH:-}"
else
  export PYTHONPATH="$SCRIPT_DIR/appworld_agent/src:$SCRIPT_DIR/appworld_agent/appworld_official_agents:${PYTHONPATH:-}"
fi

ENTRY_SCRIPT="$SCRIPT_DIR/appworld_agent/run_appworld_entry.py"
CMD=(
  "$PYTHON_BIN" "$ENTRY_SCRIPT"
  --subset "$SUBSET"
  --model "$MODEL"
  --workers "$WORKERS"
  --concurrency "$CONCURRENCY"
  --output "$OUTPUT_DIR"
)

[[ -n "$FILTER" ]] && CMD+=(--filter "$FILTER")

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "Traces saved to: $OUTPUT_DIR"
echo "  preds.json   : $OUTPUT_DIR/preds.json"
echo "  traj files   : $OUTPUT_DIR/<task_id>/<task_id>.traj.json"

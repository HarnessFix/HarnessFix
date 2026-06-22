#!/usr/bin/env bash
# run_swebench.sh - 封装 mini-extra swebench，自动将 trace 保存到项目 traces/ 目录
#
# 用法示例：
#   ./run_swebench.sh --subset verified --split test --model openai/gpt-5-mini --workers 4
#   ./run_swebench.sh --subset verified --filter "sqlfluff__sqlfluff-1625" --model openai/gpt-5-mini
#   ./run_swebench.sh --subset "$(pwd)/data/verified_val_50" --split test --output traces/swe_val_run
#
# 输出路径格式（未指定 --output 时）：<project_root>/traces/<subset>_<model_short>_<YYYYMMDD_HHMMSS>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env from project root if it exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$PROJECT_ROOT/.env"
  set +a
fi
TRACES_DIR="$PROJECT_ROOT/traces"

# -------- 默认参数 --------
SUBSET="lite"
SPLIT="dev"
MODEL="openai/gpt-5-mini"
WORKERS=1
FILTER=""
SLICE=""
REDO=""
COST_LIMIT="1"
OUTPUT_DIR_OVERRIDE=""
RERUN_STATUS=""
EXTRA_ARGS=()
CONFIG_OVERRIDES=()  # -c 覆盖项，统一收集后与 swebench.yaml 一起注入
CONFIG_FILE="swebench.yaml"
MODEL_CLASS=""

# -------- 解析参数 --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)      SUBSET="$2";  shift 2 ;;
    --split)       SPLIT="$2";   shift 2 ;;
    --model)       MODEL="$2";   shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --filter)      FILTER="$2";  shift 2 ;;
    --slice)       SLICE="$2";   shift 2 ;;
    --cost-limit)  COST_LIMIT="$2"; shift 2 ;;
    --output|-o)   OUTPUT_DIR_OVERRIDE="$2"; shift 2 ;;
    --redo-existing) REDO="--redo-existing"; shift ;;
    --rerun-status) RERUN_STATUS="$2"; shift 2 ;;
    --config)   EXTRA_ARGS+=("--config" "$2"); shift 2 ;;
    --environment-class) EXTRA_ARGS+=("--environment-class" "$2"); shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# -------- 构造 run_id 和输出路径 --------
# model 中的 / 替换为 - 避免路径问题
MODEL_SHORT="${MODEL//\//-}"
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
  # 支持相对路径（相对于 PROJECT_ROOT）和绝对路径
  if [[ "$OUTPUT_DIR_OVERRIDE" = /* ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_OVERRIDE"
  fi
  RUN_ID="$(basename "$OUTPUT_DIR")"
else
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ID="${SUBSET}_${MODEL_SHORT}_${TIMESTAMP}"
  OUTPUT_DIR="$TRACES_DIR/$RUN_ID"
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  Run ID : $RUN_ID"
echo "  Output : $OUTPUT_DIR"
echo "=========================================="

# -------- 处理 --rerun-status：从 preds.json 删除指定退出状态的实例 --------
if [[ -n "$RERUN_STATUS" ]]; then
  PREDS="$OUTPUT_DIR/preds.json"
  if [[ ! -f "$PREDS" ]]; then
    echo "Warning: $PREDS 不存在，无法处理 --rerun-status"
  else
    REMOVED=$(python3 - "$OUTPUT_DIR" "$RERUN_STATUS" "$PREDS" <<'PYEOF'
import json, sys
from pathlib import Path

output_dir, target_status, preds_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

# 从 traj 文件中找出指定状态的 instance_id
bad_ids = set()
for traj in output_dir.glob("*/*.traj.json"):
    try:
        data = json.loads(traj.read_text())
        if data.get("info", {}).get("exit_status") == target_status:
            bad_ids.add(data.get("instance_id", traj.parent.name))
    except Exception:
        pass

if not bad_ids:
    print(f"未找到 exit_status={target_status} 的实例")
    sys.exit(0)

# 从 preds.json 删除这些 instance
preds = json.loads(preds_path.read_text())
for iid in bad_ids:
    preds.pop(iid, None)
preds_path.write_text(json.dumps(preds, indent=2))
print("\n".join(sorted(bad_ids)))
PYEOF
    )
    echo "  已从 preds.json 移除 (${RERUN_STATUS}):"
    echo "$REMOVED" | sed 's/^/    /'
  fi
fi

# -------- 从 model_registry.json 自动设置 API base / api key / model_kwargs_override --------
REGISTRY="$SCRIPT_DIR/model_registry.json"
if [[ -f "$REGISTRY" ]]; then
  # 读取 api_base、api_key_env 和 model_kwargs_override（用 shell-safe 形式输出，避免空格问题）
  eval "$(python3 -c "
import json, shlex
reg = json.load(open('$REGISTRY'))
entry = reg.get('$MODEL', {})
base = entry.get('api_base', '')
api_key_env = entry.get('api_key_env', '')
config_name = entry.get('config_name', '')
model_class = entry.get('model_class', '')
cost_tracking = entry.get('cost_tracking', '')
overrides = entry.get('model_kwargs_override', {})
if base:
    print('API_BASE_FROM_REGISTRY=' + shlex.quote(json.dumps(base)))
if api_key_env:
    print('API_KEY_ENV_FROM_REGISTRY=' + shlex.quote(api_key_env))
if config_name:
    print('CONFIG_NAME_FROM_REGISTRY=' + shlex.quote(config_name))
if model_class:
    print('MODEL_CLASS_FROM_REGISTRY=' + shlex.quote(model_class))
if cost_tracking:
    print('COST_TRACKING_FROM_REGISTRY=' + shlex.quote(cost_tracking))
for k, v in overrides.items():
    # Output JSON literals so mini-swe-agent parses bool/int/str correctly.
    print('MKWARG_' + k + '=' + shlex.quote(json.dumps(v)))
" 2>/dev/null)"
  if [[ -n "${CONFIG_NAME_FROM_REGISTRY:-}" ]]; then
    echo "  Config: $CONFIG_NAME_FROM_REGISTRY (from model_registry.json)"
    CONFIG_FILE="$CONFIG_NAME_FROM_REGISTRY"
  fi
  if [[ -n "${MODEL_CLASS_FROM_REGISTRY:-}" ]]; then
    echo "  Model Class: $MODEL_CLASS_FROM_REGISTRY (from model_registry.json)"
    MODEL_CLASS="$MODEL_CLASS_FROM_REGISTRY"
  fi
  # 始终把 registry 路径注入，确保 litellm 能查到模型定价
  CONFIG_OVERRIDES+=("model.litellm_model_registry=$REGISTRY")
  if [[ -n "${API_BASE_FROM_REGISTRY:-}" ]]; then
    echo "  API Base: $API_BASE_FROM_REGISTRY (from model_registry.json)"
    CONFIG_OVERRIDES+=("model.model_kwargs.api_base=$API_BASE_FROM_REGISTRY")
  fi
  if [[ -n "${API_KEY_ENV_FROM_REGISTRY:-}" ]]; then
    if [[ -n "${!API_KEY_ENV_FROM_REGISTRY:-}" ]]; then
      echo "  API key: \$$API_KEY_ENV_FROM_REGISTRY is set (from environment via model_registry.json)"
    else
      echo "  Warning: env var $API_KEY_ENV_FROM_REGISTRY is not set; model may fail authentication"
    fi
  fi
  if [[ -n "${COST_TRACKING_FROM_REGISTRY:-}" ]]; then
    echo "  model.cost_tracking=$COST_TRACKING_FROM_REGISTRY (from model_registry.json)"
    CONFIG_OVERRIDES+=("model.cost_tracking=$COST_TRACKING_FROM_REGISTRY")
  fi
  # 应用 model_kwargs_override 中的所有覆盖
  for var in $(compgen -v MKWARG_); do
    key="${var#MKWARG_}"
    val="${!var}"
    echo "  model_kwargs.$key=$val (from model_registry.json)"
    CONFIG_OVERRIDES+=("model.model_kwargs.$key=$val")
  done
fi

# -------- 收集其他 -c 覆盖项 --------
[[ -n "$COST_LIMIT" ]] && CONFIG_OVERRIDES+=("agent.cost_limit=$COST_LIMIT")

# 注意：一旦使用 -c，默认的 swebench.yaml 不会自动加载，必须显式指定
if [[ ${#CONFIG_OVERRIDES[@]} -gt 0 ]]; then
  EXTRA_ARGS+=("-c" "$CONFIG_FILE")
  for override in "${CONFIG_OVERRIDES[@]}"; do
    EXTRA_ARGS+=("-c" "$override")
  done
elif [[ "$CONFIG_FILE" != "swebench.yaml" ]]; then
  EXTRA_ARGS+=("-c" "$CONFIG_FILE")
fi

[[ -n "$MODEL_CLASS" ]] && EXTRA_ARGS+=(--model-class "$MODEL_CLASS")

# -------- 组装命令 --------
CMD=(
  mini-extra swebench
  --subset "$SUBSET"
  --split  "$SPLIT"
  --model  "$MODEL"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR"
)

[[ -n "$FILTER" ]] && CMD+=(--filter "$FILTER")
[[ -n "$SLICE"  ]] && CMD+=(--slice  "$SLICE")
[[ -n "$REDO"   ]] && CMD+=("$REDO")
CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "执行: ${CMD[*]}"
echo ""

# -------- 运行 --------
"${CMD[@]}"

echo ""
echo "Traces 已保存到: $OUTPUT_DIR"
echo "  traj 文件位于: $OUTPUT_DIR/<instance_id>/<instance_id>.traj.json"

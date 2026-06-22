from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RAW_PROVIDER_RESPONSE_KEYS = {
    "raw_response",
    "api_response",
    "completion_response",
    "raw_completion",
}

PROVIDER_METADATA_KEYS = {
    "id",
    "created",
    "model",
    "object",
    "system_fingerprint",
    "service_tier",
    "usage",
    "logprobs",
}

MESSAGE_PROVIDER_KEYS = {
    "provider_specific_fields",
    "annotations",
}

EXTRA_DUPLICATE_KEYS = {
    "timestamp",
    "cost",
}

INFO_METADATA_KEYS = {
    "config",
    "mini_version",
    "model_id",
}


def _looks_like_provider_response(parent: dict[str, Any]) -> bool:
    return (
        parent.get("object") == "chat.completion"
        or "choices" in parent
        or "system_fingerprint" in parent
        or ("usage" in parent and "model" in parent)
    )


def _drop_key(key: str, value: Any, parent: dict[str, Any], path: tuple[str, ...]) -> bool:
    if key in RAW_PROVIDER_RESPONSE_KEYS:
        return True
    if path and path[-1] == "extra" and key == "response":
        return True
    if path == ("info",) and key in INFO_METADATA_KEYS:
        return True
    if path and path[-1] == "extra" and key in EXTRA_DUPLICATE_KEYS:
        return True
    if key == "content" and isinstance(parent.get("extra"), dict) and "raw_output" in parent["extra"]:
        return True
    if key in MESSAGE_PROVIDER_KEYS:
        return True
    if key in {"tool_calls", "function_call"} and value in (None, [], {}):
        return True
    if _looks_like_provider_response(parent) and key in PROVIDER_METADATA_KEYS | {"choices"}:
        return True
    return False


def sanitize_for_prompt(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Remove provider/runtime metadata while preserving semantic trace content.

    The sanitized artifact is what analysis agents should inspect. It keeps user
    tasks, assistant reasoning, commands, observations, submissions, and evaluator
    signals, but removes raw chat-completion payloads and duplicate bookkeeping.
    """
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _drop_key(key, item, value, path):
                continue
            cleaned = sanitize_for_prompt(item, path + (key,))
            if cleaned in (None, {}, []):
                continue
            sanitized[key] = cleaned
        return sanitized
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := sanitize_for_prompt(item, path)) not in (None, {}, [])
        ]
    return value


def sanitized_trace_output_path(results_dir: Path, run_label: str, instance_id: str) -> Path:
    safe_name = instance_id.replace("/", "__")
    return results_dir / "sanitized_traces" / run_label / f"{safe_name}.traj.json"


def write_sanitized_artifact(input_path: str | Path, output_path: str | Path) -> Path:
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(in_path.read_text())
    out_path.write_text(json.dumps(sanitize_for_prompt(data), indent=2, ensure_ascii=False) + "\n")
    return out_path

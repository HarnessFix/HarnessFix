from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SENSITIVE_LM_ARGUMENT_KEYS = {"api_key"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.]+", re.IGNORECASE),
)


def redact_secret_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def redact_lm_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in SENSITIVE_LM_ARGUMENT_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_lm_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_lm_payload(item) for item in value]
    if isinstance(value, str):
        return redact_secret_text(value)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"role": "log_parse_error", "content": line})
            continue
        rows.append(item if isinstance(item, dict) else {"value": item})
    return rows


def read_official_messages(task_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(task_dir / "official_logs" / "logger.jsonl")


def message_from_lm_output(output: dict[str, Any]) -> dict[str, Any]:
    choices = output.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        if isinstance(message, dict):
            return redact_lm_payload(message)
    return {"role": "assistant", "content": ""}


def read_official_model_calls(task_dir: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(task_dir / "official_logs" / "lm_calls.jsonl")
    model_calls: list[dict[str, Any]] = []
    for row in rows:
        call_input = row.get("input") if isinstance(row, dict) else None
        call_output = row.get("output") if isinstance(row, dict) else None
        if not isinstance(call_input, dict) or not isinstance(call_output, dict):
            continue
        request_messages = call_input.get("messages") or []
        if not isinstance(request_messages, list):
            request_messages = []
        response_message = message_from_lm_output(call_output)
        usage = call_output.get("usage") if isinstance(call_output.get("usage"), dict) else {}
        model_calls.append(
            {
                "call_index": len(model_calls) + 1,
                "request_messages": redact_lm_payload(request_messages),
                "response_message": response_message,
                "model": call_input.get("model") or call_output.get("model"),
                "usage": redact_lm_payload(usage),
                "finish_reason": (call_output.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(call_output.get("choices"), list) and call_output.get("choices")
                else None,
                "source_ref": f"official_logs/lm_calls.jsonl:{len(model_calls) + 1}",
            }
        )
    return model_calls


def messages_from_model_calls(model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not model_calls:
        return []
    messages = list(model_calls[-1].get("request_messages", []) or [])
    response_message = model_calls[-1].get("response_message")
    if isinstance(response_message, dict):
        messages.append(response_message)
    return messages

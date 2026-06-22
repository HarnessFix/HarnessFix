from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "Authorization",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.]+", re.IGNORECASE),
)


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str in SENSITIVE_KEYS:
                result[key_str] = "<redacted>"
            else:
                result[key_str] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return _redact_text(str(value))


def _request_message(message: dict[str, Any]) -> dict[str, Any]:
    """Approximate the API-facing chat item by removing mini-SWE local metadata."""
    return _jsonable({key: copy.deepcopy(value) for key, value in message.items() if key != "extra"})


def _usage_from_response(response: Any) -> Any:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def _model_from_response(response: Any) -> str:
    if isinstance(response, dict):
        model = response.get("model")
        if isinstance(model, str):
            return model
    return ""


def build_model_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build normalized request/response call records from a mini-SWE message trace."""
    calls: list[dict[str, Any]] = []
    for msg_idx, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
        response = extra.get("response") if isinstance(extra, dict) else None
        call_index = len(calls) + 1
        item: dict[str, Any] = {
            "call_index": call_index,
            "component": "controller",
            "model": _model_from_response(response),
            "request_messages": [
                _request_message(prior)
                for prior in messages[:msg_idx]
                if isinstance(prior, dict) and prior.get("role") != "exit"
            ],
            "response_message": _jsonable(message),
            "usage": _jsonable(_usage_from_response(response)),
            "source_ref": f"traj.messages[{msg_idx}]",
        }
        if response is not None:
            item["raw_response"] = _jsonable(response)
        calls.append(item)
    return calls


def build_chat_trace_summary(messages: list[dict[str, Any]], model_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": "mini-swe-agent.messages",
        "message_count": len(messages),
        "model_call_count": len(model_calls),
        "request_response_granularity": "assistant_turn_chat_history",
        "request_messages_drop_local_extra": True,
        "raw_provider_payload_available": any("raw_response" in call for call in model_calls),
        "sensitive_fields_redacted": True,
    }

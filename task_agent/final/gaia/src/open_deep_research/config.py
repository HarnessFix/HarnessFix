"""Constants and configuration for open_deep_research agent.

This file is the PRIMARY target for modify_agent — tweak parameters here
before touching agent.py, prompts.py, tools.py, or browser.py.

All values can be overridden by the modify_agent when generating enhanced versions.
"""

import os

# ── Browser configuration ──────────────────────────────────────────────────────

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": USER_AGENT},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

# ── Text inspection ────────────────────────────────────────────────────────────

# Maximum characters passed to TextInspectorTool
TEXT_LIMIT = 100_000

# ── Agent step limits ──────────────────────────────────────────────────────────

# Manager CodeAgent: max number of steps before stopping
DEFAULT_MAX_STEPS_MANAGER = 12

# Search ToolCallingAgent: max steps per sub-task
DEFAULT_MAX_STEPS_SEARCH = 20

# Planning interval: how often the manager re-plans (every N steps)
DEFAULT_PLANNING_INTERVAL = 4

# ── Model defaults ─────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "openai/gpt-5-mini"

# Max output tokens for the model
DEFAULT_MAX_TOKENS = 4096

# ── Verbosity ─────────────────────────────────────────────────────────────────

# 0 = silent, 1 = step summaries, 2 = full output
AGENT_VERBOSITY = 2

# ── LiteLLM role remapping ─────────────────────────────────────────────────────

# Required for some LiteLLM providers that don't support tool-call roles natively
CUSTOM_ROLE_CONVERSIONS = {"tool-call": "assistant", "tool-response": "user"}

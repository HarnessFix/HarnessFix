"""Agent factory for open_deep_research.

Public API (signature MUST NOT change — pipeline depends on it):
    create_agent_team(model_id: str) -> CodeAgent

The returned manager CodeAgent uses a ToolCallingAgent sub-agent for web search,
mirroring the architecture from smolagents/examples/open_deep_research/run.py.

modify_agent may edit the body of create_agent_team() to change configuration,
but must never change the function signature.
"""

from __future__ import annotations

import os

from smolagents import CodeAgent, ToolCallingAgent

from .config import (
    AGENT_VERBOSITY,
    CUSTOM_ROLE_CONVERSIONS,
    DEFAULT_MAX_STEPS_MANAGER,
    DEFAULT_MAX_STEPS_SEARCH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PLANNING_INTERVAL,
    TEXT_LIMIT,
)
from .browser import build_browser
from .tools import build_web_tools
from .scripts.text_inspector_tool import TextInspectorTool
from .scripts.visual_qa import visualizer
from .tracing import TracingLiteLLMModel, current_model_trace_recorder


def create_agent_team(model_id: str) -> CodeAgent:
    """Create and return the manager CodeAgent with a search sub-agent.

    Args:
        model_id: LiteLLM-compatible model ID (e.g. "openai/gpt-4o").
                  API credentials are read from environment variables:
                  OPENAI_API_KEY / OPENAI_API_BASE (or LITELLM_API_KEY / LITELLM_API_BASE).

    Returns:
        smolagents.CodeAgent configured as manager with a web-search sub-agent.
        Call agent.run(question_string) to get the answer.
    """
    # ── Build model ────────────────────────────────────────────────────────────
    model_kwargs: dict = {
        "model_id": model_id,
        "custom_role_conversions": CUSTOM_ROLE_CONVERSIONS,
        "max_tokens": DEFAULT_MAX_TOKENS,
    }
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_API_KEY")
    if api_base:
        model_kwargs["api_base"] = api_base
    if api_key:
        model_kwargs["api_key"] = api_key

    trace_recorder = current_model_trace_recorder()
    manager_model = TracingLiteLLMModel(**model_kwargs, trace_component="manager", trace_recorder=trace_recorder)
    search_model = TracingLiteLLMModel(**model_kwargs, trace_component="search_agent", trace_recorder=trace_recorder)

    # ── Build browser & search tools ───────────────────────────────────────────
    browser = build_browser()
    web_tools = build_web_tools(model=search_model, browser=browser)

    # ── Build search sub-agent ─────────────────────────────────────────────────
    text_webbrowser_agent = ToolCallingAgent(
        model=search_model,
        tools=web_tools,
        max_steps=DEFAULT_MAX_STEPS_SEARCH,
        verbosity_level=AGENT_VERBOSITY,
        planning_interval=DEFAULT_PLANNING_INTERVAL,
        name="search_agent",
        description=(
            "A team member that will search the internet to answer your question.\n"
            "Ask him for all your questions that require browsing the web.\n"
            "Provide him as much context as possible, in particular if you need to "
            "search on a specific timeframe!\n"
            "And don't hesitate to provide him with a complex search task, like finding "
            "a difference between two webpages.\n"
            "Your request must be a real sentence, not a google search! "
            'Like "Find me this information (...)" rather than a few keywords.'
        ),
        provide_run_summary=True,
    )
    # Append file-inspection guidance to the search agent's task prompt
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += (
        "\nYou can navigate to .txt online files.\n"
        "If a non-html page is in another format, especially .pdf or a Youtube video, "
        "use tool 'inspect_file_as_text' to inspect it.\n"
        "Additionally, if after some searching you find out that you need more information "
        "to answer the question, you can use `final_answer` with your request for "
        "clarification as argument to request for more information."
    )

    # ── Build manager CodeAgent ────────────────────────────────────────────────
    ti_tool = TextInspectorTool(
        TracingLiteLLMModel(**model_kwargs, trace_component="manager.text_inspector", trace_recorder=trace_recorder),
        TEXT_LIMIT,
    )
    manager_agent = CodeAgent(
        model=manager_model,
        tools=[visualizer, ti_tool],
        max_steps=DEFAULT_MAX_STEPS_MANAGER,
        verbosity_level=AGENT_VERBOSITY,
        additional_authorized_imports=["*"],
        planning_interval=DEFAULT_PLANNING_INTERVAL,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent

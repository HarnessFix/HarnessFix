"""Tool factories for open_deep_research.

build_web_tools() is the public factory called by agent.py.
It mirrors the WEB_TOOLS list from the upstream run.py / run_gaia.py.

modify_agent may add, remove, or swap tools here to improve search quality.
"""

from .config import TEXT_LIMIT
from .scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    VisitTool,
)
from .scripts.text_inspector_tool import TextInspectorTool

from smolagents import GoogleSearchTool


def build_web_tools(model, browser) -> list:
    """Build the list of web tools for the search ToolCallingAgent.

    Args:
        model: LiteLLMModel instance (used by TextInspectorTool)
        browser: SimpleTextBrowser instance from build_browser()

    Returns:
        List of tool instances matching the upstream WEB_TOOLS list.
    """
    return [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, TEXT_LIMIT),
    ]

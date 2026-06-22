"""Browser factory for open_deep_research.

Wraps the SimpleTextBrowser from the upstream scripts/text_web_browser.py.
modify_agent may adjust BROWSER_CONFIG in config.py.
"""

import os
from .config import BROWSER_CONFIG


def build_browser():
    """Build and return a SimpleTextBrowser instance.

    Returns:
        SimpleTextBrowser instance configured via BROWSER_CONFIG,
        or None if the dependency is unavailable.
    """
    from .scripts.text_web_browser import SimpleTextBrowser

    cfg = dict(BROWSER_CONFIG)
    # Inject SERPAPI key from environment if not already set
    if cfg.get("serpapi_key") is None:
        cfg["serpapi_key"] = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPER_API_KEY")

    # Ensure downloads folder exists
    downloads_folder = cfg.get("downloads_folder", "downloads_folder")
    os.makedirs(f"./{downloads_folder}", exist_ok=True)

    return SimpleTextBrowser(**cfg)

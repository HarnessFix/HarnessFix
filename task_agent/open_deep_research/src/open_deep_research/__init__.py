"""open_deep_research — local editable package for GAIA pipeline.

This package is intentionally kept as a pip-installable local package
so that modify_agent can edit the source files under src/ and the
changes take effect immediately (editable install via pip install -e .).
"""
from .agent import create_agent_team

__all__ = ["create_agent_team"]

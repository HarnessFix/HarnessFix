from __future__ import annotations

from jinja2 import Template


def render_template(template: str, allow_missing: bool = False, skip_fields=None, **kwargs) -> str:
    return Template(template).render(**kwargs)

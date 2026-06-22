from __future__ import annotations

import json
from pathlib import Path

FILE_LOGGER_WIDTH = 120
TERMINAL_LOGGER_WIDTH = 120


def maybe_create_parent_directory(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def read_file(file_path: str) -> str:
    return Path(file_path).read_text()


def write_file(content: str, file_path: str) -> None:
    maybe_create_parent_directory(file_path)
    Path(file_path).write_text(content)


def read_json(file_path: str):
    path = Path(file_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_json(data, file_path: str, silent: bool = False) -> None:
    maybe_create_parent_directory(file_path)
    Path(file_path).write_text(json.dumps(data, indent=2))


def write_jsonl(rows: list[dict], file_path: str, append: bool = False, silent: bool = False) -> None:
    maybe_create_parent_directory(file_path)
    mode = "a" if append else "w"
    with Path(file_path).open(mode) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

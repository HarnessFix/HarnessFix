#!/usr/bin/env python3
"""Create a tiny 5-task AppWorld smoke subset from appworld_val_45."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DIR = PROJECT_ROOT / "data" / "appworld_val_45"
SMOKE_DIR = PROJECT_ROOT / "data" / "appworld_smoke_5"


def main() -> None:
    data_file = VAL_DIR / "data.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing {data_file}. Run data/sample_appworld.py first.")
    tasks = [json.loads(line) for line in data_file.read_text().splitlines() if line.strip()]
    smoke_tasks = tasks[:5]
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "data.jsonl").write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in smoke_tasks) + "\n"
    )
    (SMOKE_DIR / "instance_ids.txt").write_text("\n".join(task["task_id"] for task in smoke_tasks) + "\n")
    print(f"Wrote {len(smoke_tasks)} tasks to {SMOKE_DIR}")


if __name__ == "__main__":
    main()

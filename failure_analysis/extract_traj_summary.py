#!/usr/bin/env python3
"""Extract a semantic trajectory summary for analysis agents.

Usage: python3 extract_traj_summary.py <traj_path> [--full]
"""

import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from failure_analysis.artifact_sanitizer import sanitize_for_prompt


def clip_text(value: object, limit: int = 1200) -> str:
    text = str(value).replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 80)] + "\n... [truncated]\n"


def select_head_tail(items: list, head: int = 5, tail: int = 12) -> list:
    if len(items) <= head + tail:
        return items
    omitted = len(items) - head - tail
    return items[:head] + [(None, f"... {omitted} intermediate items omitted ...")] + items[-tail:]


def extract_commands(messages: list) -> list[tuple[int, str]]:
    """Return [(step_number, command_str), ...] for all assistant messages with actions."""
    steps = []
    step_num = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            step_num += 1
            actions = msg.get("extra", {}).get("actions", [])
            for action in actions:
                if isinstance(action, dict):
                    cmd = action.get("command", str(action))
                else:
                    cmd = str(action)
                steps.append((step_num, cmd))
    return steps


def get_thought(msg: dict) -> str:
    """Extract THOUGHT section from assistant message content."""
    content = msg.get("content", "")
    if not isinstance(content, str):
        return ""
    match = re.search(r"THOUGHT:\s*(.*?)(?:<bash_code>|```|<mswea_bash_command>|$)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: first non-empty line
    for line in content.split("\n"):
        line = line.strip()
        if line and not line.startswith("```") and not line.startswith("<"):
            return line
    return content


def detect_loops(commands: list[tuple[int, str]], threshold: int = 3) -> list[tuple[str, int]]:
    """Return commands repeated >= threshold times."""
    counter = Counter(cmd for _, cmd in commands)
    return [(cmd, count) for cmd, count in counter.items() if count >= threshold]


def main(traj_path: str, full: bool = False, char_limit: int = 30000):
    path = Path(traj_path)
    if not path.exists():
        print(f"ERROR: File not found: {traj_path}")
        sys.exit(1)

    data = sanitize_for_prompt(json.loads(path.read_text()))
    info = data.get("info", {})
    messages = data.get("messages", [])
    model_stats = info.get("model_stats", {})

    # --- Basic stats ---
    exit_status = info.get("exit_status", "unknown")
    api_calls = model_stats.get("api_calls", 0)
    cost = model_stats.get("instance_cost", 0.0)
    submission = info.get("submission", "")

    print("=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"exit_status : {exit_status}")
    print(f"api_calls   : {api_calls}")
    print(f"cost        : ${cost:.4f}")
    print(f"total_msgs  : {len(messages)}")
    print()

    # --- All commands list ---
    commands = extract_commands(messages)
    visible_commands = commands if full else select_head_tail(commands)
    suffix = "" if full or len(visible_commands) == len(commands) else f"; showing {len(visible_commands) - 1}"
    print(f"--- COMMANDS ({len(commands)} total{suffix}) ---")
    for step, cmd in visible_commands:
        if step is None:
            print(f"  {cmd}")
            continue
        print(f"  step {step:3d}: {clip_text(cmd, 1200)}")
    print()

    # --- Loop detection ---
    loops = detect_loops(commands, threshold=3)
    if loops:
        print("--- REPEATED COMMANDS (possible loops) ---")
        for cmd, count in sorted(loops, key=lambda x: -x[1]):
            print(f"  x{count}: {cmd}")
        print()

    # --- All steps detail ---
    print("--- ASSISTANT STEPS (THOUGHT + command) ---")
    assistant_msgs = [
        (i, msg) for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    indexed_msgs = list(enumerate(assistant_msgs, start=1))
    visible_msgs = indexed_msgs if full else select_head_tail(indexed_msgs)
    for item in visible_msgs:
        if item[0] is None:
            print(f"  {item[1]}")
            continue
        step_num, (msg_idx, msg) = item
        thought = clip_text(get_thought(msg), 900)
        actions = msg.get("extra", {}).get("actions", [])
        cmds = []
        for action in actions:
            if isinstance(action, dict):
                cmds.append(action.get("command", str(action)))
            else:
                cmds.append(str(action))
        print(f"  [Step {step_num}] {thought}")
        for cmd in cmds:
            print(f"    CMD: {clip_text(cmd, 1200)}")
    print()

    # --- Exit / submission info ---
    print("--- EXIT / SUBMISSION ---")
    # Find exit message
    exit_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "exit":
            exit_msg = msg
            break

    if exit_msg:
        content = exit_msg.get("content", "")
        print(f"exit_message: {clip_text(content, 2000)}")

    if submission:
        print(f"submission: {clip_text(submission, 12000)}")
    else:
        print("submission: (empty)")
    print("=" * 60)


if __name__ == "__main__":
    full = "--full" in sys.argv[2:]
    char_limit = 30000
    raw_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg == "--full":
            continue
        if arg == "--char-limit":
            try:
                char_limit = int(sys.argv[i + 2])
            except (IndexError, ValueError):
                print("ERROR: --char-limit requires an integer")
                sys.exit(1)
            skip_next = True
            continue
        raw_args.append(arg)
    if len(raw_args) != 1:
        print(f"Usage: {sys.argv[0]} <traj_path> [--full] [--char-limit N]")
        sys.exit(1)
    buffer = StringIO()
    with redirect_stdout(buffer):
        main(raw_args[0], full=full, char_limit=char_limit)
    output = buffer.getvalue()
    if char_limit > 0 and len(output) > char_limit:
        output = output[: max(0, char_limit - 80)] + "\n... [trajectory summary truncated]\n"
    print(output, end="")

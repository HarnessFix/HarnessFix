#!/usr/bin/env python3
"""Run an exploration agent to deeply understand the task_agent codebase.

The agent reads source files, traces execution paths, and produces a detailed
implementation document saved to failure_analysis/task_agent_impl_doc.md.

This document is later injected into the system_template of the failure
analysis agent so it has precise knowledge of what the task agent actually does.

Usage:
  .venv/bin/python3 failure_analysis/explore_task_agent.py
  .venv/bin/python3 failure_analysis/explore_task_agent.py --model openai/gpt-5-mini
  .venv/bin/python3 failure_analysis/explore_task_agent.py --force   # re-run even if doc exists
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "agent_framework" / "src"))

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

IMPL_DOC_PATH = Path(__file__).parent / "task_agent_impl_doc.md"
DEFAULT_MODEL = "openai/gpt-5-mini"
TRAJ_OUTPUT_PATH = Path(__file__).parent / "results" / "explore_task_agent.traj.json"

TASK_AGENT_SRC = str(REPO_ROOT / "task_agent" / "mini-swe-agent" / "src" / "minisweagent")
SWEBENCH_CONFIG = str(REPO_ROOT / "task_agent" / "mini-swe-agent" / "src" / "minisweagent" / "config" / "benchmarks" / "swebench.yaml")

SYSTEM_TEMPLATE = """\
You are a senior software engineer doing a thorough code review of an LLM agent system.
Your response MUST contain EXACTLY ONE `<mswea_bash_command>` block per turn.
THOUGHT: Your reasoning here
<mswea_bash_command>your_command_here</mswea_bash_command>
"""

INSTANCE_TEMPLATE = """\
Explore the mini-swe-agent task agent codebase at:
  {src_dir}

And produce a comprehensive implementation details document about how this agent system works.

## Your Goal

Write a detailed technical document (Markdown) that covers every implementation detail
a failure analyst would need to correctly diagnose WHY a particular agent run failed.
The analyst will be looking at failure traces and needs to know exactly what the code does
so they can attribute failures to the right design component.

## Key Files to Explore (explore ALL of these, plus anything else you discover)

- `{src_dir}/agents/default.py` — main agent loop, step(), query(), execute_actions(), limits checking
- `{src_dir}/exceptions.py` — exception types (LimitsExceeded, Timeout, FormatError, InterruptAgentFlow)
- `{src_dir}/environments/docker.py` — command execution, timeout handling, subshell behavior
- `{src_dir}/environments/local.py` — local execution (used by analysis agent)
- `{src_dir}/models/litellm_model.py` — tool-call based action parsing (used by task agent)
- `{src_dir}/models/utils/actions_text.py` — regex-based action extraction, FormatError raising
- `{src_dir}/models/utils/actions_toolcall.py` — tool-call action parsing
- `{src_dir}/config/benchmarks/swebench.yaml` — the ACTUAL config used in this experiment
- `{config_file}` — confirm the exact values

## Required Sections in Your Document

Structure your output document with these sections:

### 1. Main Agent Loop
- Exact flow of DefaultAgent.run() → step() → query() → execute_actions()
- When and how limits are checked (before query, not after)
- What happens when step_limit is hit vs cost_limit is hit
- How n_calls is incremented and when
- What "exit" role message means and when it's added

### 2. Limit Enforcement
- Exact condition: `0 < step_limit <= n_calls OR 0 < cost_limit <= cost`
- What LimitsExceeded exception carries (exit_status, submission fields)
- What submission value is when LimitsExceeded (empty string)
- No best-effort commit: agent just stops

### 3. Timeout Handling
- What triggers a Timeout (command takes >60s in Docker)
- How Timeout propagates (exception from environment.execute())
- Does Timeout = exit? What is in the exit message?
- Difference between command timeout and API timeout

### 4. FormatError Handling
- When does FormatError occur (model returns no valid tool call)
- Does FormatError consume a step? (check n_calls increment timing)
- What is the format_error_template and what gets sent back to the model
- How many FormatErrors can accumulate before limit is hit?

### 5. Command Execution (Docker Environment)
- Each command runs in a FRESH subshell via bash -c
- cd, export, env vars do NOT persist between commands
- Observation rendering: exact output template and whether any command output elision occurs
- How returncode is captured and shown to the model
- What happens on non-zero returncode (shown but agent continues)

### 6. Submission Detection
- Exact string that triggers submission: `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- Where this is detected (in environment.execute()? or in model parsing?)
- What gets captured as the "submission" (stdout after the echo command?)
- Why submission must be TWO separate commands (patch creation + echo+cat)

### 7. Action Parsing (Tool calling via swebench.yaml config)
- How the bash tool is defined and passed to the model (BASH_TOOL in actions_toolcall.py)
- How parse_toolcall_actions extracts the command from the tool call response
- What happens when the model returns no tool call (FormatError)
- How the command string is extracted from the tool call arguments

### 8. Actual Config Values (from swebench.yaml)
- step_limit, cost_limit, timeout exact values
- system_prompt content (key constraints on model behavior)
- instance_template structure
- observation_template (exactly what the model sees after each command)
- model_name used in the experiment

### 9. Known Design Gaps (synthesize from your reading)
List specific code locations where these gaps exist:
- No loop/repetition detection (where would you add it?)
- No best-effort commit on LimitsExceeded (what code would need changing?)
- Subshell isolation causing state loss (where is this in docker.py?)
- sed fragility for file editing (mentioned in system prompt?)
- FormatError not counting as step (exact code line)

## Submission

After reading all files and drafting the document, write it to a file and submit:

```
cat > /tmp/task_agent_impl_doc.md << 'DOCEOF'
# Task Agent Implementation Details
... your full document here ...
DOCEOF
```

Then submit with:

`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat /tmp/task_agent_impl_doc.md`
""".format(
    src_dir=TASK_AGENT_SRC,
    config_file=SWEBENCH_CONFIG,
)

MODEL_KWARGS = {
    "temperature": 0.0,
    "stream": False,
    "timeout": 300,
    "max_retries": 0,
    "drop_params": True,
    "api_base": "https://api.openai.com/v1",
    # api_key is read from OPENAI_API_KEY environment variable (set in .env)
}


def main():
    parser = argparse.ArgumentParser(description="Explore task_agent codebase and produce impl doc")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Re-run even if doc already exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("explore_task_agent")

    if IMPL_DOC_PATH.exists() and not args.force:
        logger.info(f"Doc already exists at {IMPL_DOC_PATH}. Use --force to re-run.")
        print(f"Doc already exists: {IMPL_DOC_PATH}")
        print(f"Size: {IMPL_DOC_PATH.stat().st_size} bytes")
        return

    logger.info(f"Running exploration agent with model={args.model}")

    model = LitellmTextbasedModel(
        model_name=args.model,
        action_regex="<mswea_bash_command>(.*?)</mswea_bash_command>",
        model_kwargs=MODEL_KWARGS,
        cost_tracking="ignore_errors",
    )
    env = LocalEnvironment(env={
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
    })
    agent = DefaultAgent(
        model,
        env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        step_limit=30,
        cost_limit=2.0,
        output_path=TRAJ_OUTPUT_PATH,
    )

    result = agent.run()
    submission = result.get("submission", "")

    if not submission or len(submission) < 100:
        logger.error(f"Submission too short or empty: {submission!r}")
        print("ERROR: Agent did not produce a valid document.")
        sys.exit(1)

    IMPL_DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMPL_DOC_PATH.write_text(submission)

    logger.info(f"Saved implementation doc to {IMPL_DOC_PATH} ({len(submission)} chars)")
    print(f"\nSaved: {IMPL_DOC_PATH}")
    print(f"Size: {len(submission)} chars")
    print(f"Traj: {TRAJ_OUTPUT_PATH}")
    print("\nGenerated doc:")
    print(submission)


if __name__ == "__main__":
    main()

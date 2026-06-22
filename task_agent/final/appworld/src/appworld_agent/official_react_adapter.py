from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from appworld_agent.core import (
    AppWorldDockerEnvironment,
    AppWorldTask,
    DEFAULT_COST_LIMIT,
    DEFAULT_STEP_LIMIT,
    evaluate_appworld_task,
    resolve_appworld_root,
    write_eval_artifacts,
)
from appworld_agent.official_trace import (
    SENSITIVE_LM_ARGUMENT_KEYS,
    messages_from_model_calls,
    read_official_messages,
    read_official_model_calls,
)


VENDORED_AGENTS_ROOT = Path(__file__).resolve().parents[2] / "appworld_official_agents"
OFFICIAL_REACT_PROMPT = (
    VENDORED_AGENTS_ROOT
    / "appworld_agents"
    / "prompts"
    / "react_code_agent"
    / "instructions.txt"
)


if str(VENDORED_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_AGENTS_ROOT))


@dataclass
class _Supervisor:
    first_name: str
    last_name: str
    email: str
    phone_number: str


@dataclass
class _OfficialTask:
    id: str
    instruction: str
    supervisor: _Supervisor
    app_descriptions: dict[str, str]


class _HarnessWorld:
    """Small adapter exposing the subset of AppWorld used by the official ReAct agent."""

    def __init__(
        self,
        *,
        task: AppWorldTask,
        env: AppWorldDockerEnvironment,
        task_dir: Path,
    ) -> None:
        self.task_id = task.task_id
        self.task = _OfficialTask(
            id=task.task_id,
            instruction=task.instruction,
            supervisor=_Supervisor(**task.supervisor),
            app_descriptions=_load_app_descriptions(env),
        )
        self._env = env
        self._completed = False
        self.output_logs_directory = str(task_dir / "official_logs")
        self.output_misc_directory = str(task_dir / "official_misc")
        Path(self.output_logs_directory).mkdir(parents=True, exist_ok=True)
        Path(self.output_misc_directory).mkdir(parents=True, exist_ok=True)

    def batch_execute(self, code_blocks: list[str]) -> list[str]:
        outputs: list[str] = []
        for code in code_blocks:
            try:
                payload = self._env.execute_persistent(code)
                output = payload.get("output", "")
                if payload.get("exception_info"):
                    output = (output + "\n" + payload["exception_info"]).strip()
                if payload.get("completed"):
                    self._completed = True
                outputs.append(output)
            except Exception as exc:
                submission = _submission_from_exception(exc)
                if submission is not None:
                    self._completed = True
                    outputs.append("__TASK_COMPLETED__")
                    continue
                outputs.append(traceback.format_exc())
        return outputs

    def task_completed(self) -> bool:
        return self._completed


def _model_config(model_id: str) -> dict[str, Any]:
    normalized_model = model_id.split("/", 1)[-1].lower()
    temperature = 1.0 if normalized_model.startswith("gpt-5") else 0.0
    config: dict[str, Any] = {
        "name": model_id,
        "client_name": "litellm",
        "api_type": "chat_completions",
        "temperature": temperature,
        "seed": 100,
        "drop_reasoning_content": False,
        "cost_per_token": {
            "input_cache_hit": 0.0,
            "input_cache_miss": 0.0,
            "input_cache_write": 0.0,
            "output": 0.0,
        },
        "retry_after_n_seconds": 15,
        "use_cache": False,
        "max_retries": 5,
    }
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("LITELLM_API_BASE")
    if api_base:
        config["base_url"] = api_base
    return config


def _build_official_react_agent(model_id: str):
    from appworld_agents.code.simplified.react_code_agent import SimplifiedReActCodeAgent

    return SimplifiedReActCodeAgent(
        model_config=_model_config(model_id),
        appworld_config={"random_seed": 100, "raise_on_extra_parameters": True},
        logger_config={"color": False, "verbose": False},
        usage_tracker_config={
            "max_cost_overall": 1000,
            "max_cost_per_task": DEFAULT_COST_LIMIT,
            "max_output_tokens_per_task": 100000,
        },
        prompt_file_path=str(OFFICIAL_REACT_PROMPT),
        ignore_multiple_calls=True,
        max_prompt_length=None,
        max_output_length=60000,
        max_steps=DEFAULT_STEP_LIMIT,
        log_lm_calls=True,
        skip_if_finished=False,
    )


def _load_app_descriptions(env: AppWorldDockerEnvironment) -> dict[str, str]:
    script = """
import json
import os

os.environ["APPWORLD_ROOT"] = "/workspace/appworld_root"
from appworld.common.path_store import path_store
from appworld.environment import AppWorld

path_store.update_root("/workspace/appworld_root")
world = AppWorld(task_id={task_id!r}, experiment_name={experiment_name!r})
try:
    print(json.dumps(world.task.app_descriptions))
finally:
    world.close()
""".format(task_id=env.task.task_id, experiment_name=env.experiment_name)
    result = env._execute_python(script)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout.strip())


def _submission_from_exception(exc: Exception) -> str | None:
    if type(exc).__name__ == "Submitted":
        return "__submitted__"
    args = getattr(exc, "args", ())
    payload = args[0] if args else None
    if isinstance(payload, dict):
        extra = payload.get("extra", {})
        if extra.get("exit_status") == "Submitted":
            return str(extra.get("submission", ""))
    return None


def run_official_react_task(
    task: AppWorldTask,
    model_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    from appworld_agents.code.simplified.agent import ExecutionIO

    task_dir = output_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    traj_path = task_dir / f"{task.task_id}.traj.json"
    appworld_root = resolve_appworld_root()
    start_time = time.time()
    env: AppWorldDockerEnvironment | None = None
    prediction = ""
    error_text = ""

    try:
        env = AppWorldDockerEnvironment(task=task, output_dir=task_dir, appworld_root=appworld_root)
        agent = _build_official_react_agent(model_id)
        world = _HarnessWorld(task=task, env=env, task_dir=task_dir)
        execution_outputs = []
        agent.initialize(world)  # type: ignore[arg-type]
        for _ in range(agent.max_steps):
            agent.step_number += 1
            execution_inputs, usage, status = agent.next_execution_inputs_usage_and_status(
                execution_outputs
            )
            if status.failed:
                error_text = status.message
                break
            output_texts = world.batch_execute([item.content for item in execution_inputs])
            execution_outputs = [
                ExecutionIO(content=output_text, metadata=execution_input.metadata)
                for execution_input, output_text in zip(execution_inputs, output_texts, strict=True)
            ]
            agent.usage_tracker.add(task.task_id, usage)
            if world.task_completed() or agent.usage_tracker.exceeded(task.task_id):
                break

        if world.task_completed():
            prediction = "__TASK_COMPLETED__"
            eval_result = evaluate_appworld_task(task.task_id, task_dir, appworld_root, env.experiment_name)
        else:
            eval_result = {
                "success": False,
                "pass_percentage": 0,
                "passes": [],
                "failures": [{"requirement": "task_not_completed", "trace": error_text}],
                "error": error_text,
            }
    except Exception as exc:
        error_text = traceback.format_exc()
        eval_result = {
            "success": False,
            "pass_percentage": 0,
            "passes": [],
            "failures": [{"requirement": "official_react_error", "trace": error_text}],
            "error": str(exc),
        }
    finally:
        if env is not None:
            env.cleanup()

    elapsed = round(time.time() - start_time, 2)
    final_exit_status = (
        "solved"
        if eval_result.get("success")
        else ("error" if eval_result.get("error") else "unsolved")
    )
    logger_messages = read_official_messages(task_dir)
    model_calls = read_official_model_calls(task_dir)
    messages = messages_from_model_calls(model_calls) or logger_messages
    chat_trace = {
        "source": "official_logs/lm_calls.jsonl",
        "message_count": len(messages),
        "model_call_count": len(model_calls),
        "request_response_granularity": "message_level_chat_completion",
        "raw_provider_payload_available": True,
        "sensitive_fields_redacted": sorted(SENSITIVE_LM_ARGUMENT_KEYS),
    }
    traj_data = {
        "instance_id": task.task_id,
        "info": {
            "exit_status": final_exit_status,
            "agent_exit_status": "OfficialReActCompleted" if prediction else "OfficialReActStopped",
            "prediction": prediction,
            "elapsed_seconds": elapsed,
            "model_id": model_id,
            "agent_backend": "official_simplified_react_code",
            "required_apps": task.required_apps,
            "difficulty": task.difficulty,
            "eval_success": eval_result.get("success", False),
            "pass_percentage": eval_result.get("pass_percentage", 0),
            "agent_error": error_text,
            "chat_trace": chat_trace,
        },
        "messages": messages,
        "model_calls": model_calls,
        "official_logger_messages": logger_messages,
        "trajectory_format": "appworld-official-simplified-react-adapter",
    }
    traj_path.write_text(json.dumps(traj_data, indent=2, ensure_ascii=False))
    write_eval_artifacts(task_dir, eval_result)
    return {
        "task_id": task.task_id,
        "prediction": prediction,
        "exit_status": final_exit_status,
        "score": float(eval_result.get("pass_percentage", 0)) / 100.0,
        "success": bool(eval_result.get("success", False)),
    }

"""Harbor wrapper for running local mini-SWE-agent against task environments."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.trajectory_utils import format_trajectory_json


REPO_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_MINI_SWE_AGENT_SRC = REPO_ROOT / "task_agent" / "mini-swe-agent" / "src"


def _resolve_mini_swe_agent_src(src_path: str | Path | None = None) -> Path:
    path = Path(src_path) if src_path else DEFAULT_MINI_SWE_AGENT_SRC
    if not path.is_absolute():
        path = REPO_ROOT / path
    if (path / "src" / "minisweagent").is_dir():
        path = path / "src"
    path = path.resolve()
    if not (path / "minisweagent").is_dir():
        raise FileNotFoundError(f"mini-swe-agent source not found under {path}")
    return path


def _ensure_local_mini_swe_agent_importable(src_path: str | Path | None = None) -> Path:
    source = _resolve_mini_swe_agent_src(src_path)
    loaded = sys.modules.get("minisweagent")
    loaded_file_raw = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file_raw:
        loaded_file = Path(loaded_file_raw).resolve()
        try:
            loaded_file.relative_to(source)
        except ValueError:
            for name in list(sys.modules):
                if name == "minisweagent" or name.startswith("minisweagent."):
                    del sys.modules[name]

    src = str(source)
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    return source


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _usage_from_message(message: dict[str, Any]) -> dict[str, Any]:
    extra = message.get("extra") or {}
    response = extra.get("response") or {}
    return response.get("usage") or {}


def _tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_arguments": raw}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _metrics_from_usage(
    usage: dict[str, Any],
    *,
    total_cost: float | None,
    total_completion_tokens: int,
) -> Metrics | None:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    )

    if prompt_tokens is None and completion_tokens is None and cached_tokens is None:
        return None

    cost_usd = None
    if (
        total_cost is not None
        and completion_tokens
        and total_completion_tokens > 0
    ):
        cost_usd = total_cost * (completion_tokens / total_completion_tokens)

    extra = None
    if reasoning_tokens:
        extra = {"reasoning_tokens": reasoning_tokens}

    return Metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost_usd,
        extra=extra,
    )


def convert_mini_swe_agent_to_atif(
    trajectory_data: dict[str, Any],
    session_id: str,
) -> Trajectory:
    info = trajectory_data.get("info") or {}
    config = info.get("config") or {}
    model_config = config.get("model") or {}
    messages = trajectory_data.get("messages") or []
    model_stats = info.get("model_stats") or {}

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = _usage_from_message(message)
        total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        total_completion_tokens += int(usage.get("completion_tokens") or 0)
        total_cached_tokens += int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        )
        total_reasoning_tokens += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or 0
        )

    total_cost = model_stats.get("instance_cost")
    if not isinstance(total_cost, (int, float)):
        total_cost = None

    steps: list[Step] = []
    last_agent_step_idx: int | None = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "exit":
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id and last_agent_step_idx is not None:
                result = ObservationResult(
                    source_call_id=str(tool_call_id),
                    content=_normalize_content(message.get("content")),
                )
                step = steps[last_agent_step_idx]
                if step.observation is None:
                    step.observation = Observation(results=[])
                step.observation.results.append(result)
            continue

        if role not in {"system", "user", "assistant"}:
            continue

        source = "agent" if role == "assistant" else role
        kwargs: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": source,
            "message": _normalize_content(message.get("content")),
        }

        if source == "agent":
            usage = _usage_from_message(message)
            tool_calls = []
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                call_id = str(tool_call.get("id") or f"tool-{len(tool_calls) + 1}")
                tool_calls.append(
                    ToolCall(
                        tool_call_id=call_id,
                        function_name=str(function.get("name") or ""),
                        arguments=_tool_arguments(function.get("arguments")),
                    )
                )
            if tool_calls:
                kwargs["tool_calls"] = tool_calls

            content = _normalize_content(message.get("content"))
            if content:
                kwargs["reasoning_content"] = content

            metrics = _metrics_from_usage(
                usage,
                total_cost=total_cost,
                total_completion_tokens=total_completion_tokens,
            )
            if metrics is not None:
                kwargs["metrics"] = metrics
            last_agent_step_idx = len(steps)

        steps.append(Step(**kwargs))

    if not steps:
        steps.append(Step(step_id=1, source="system", message=""))

    final_extra = (
        {"total_reasoning_tokens": total_reasoning_tokens}
        if total_reasoning_tokens
        else None
    )
    return Trajectory(
        schema_version="ATIF-v1.2",
        session_id=session_id,
        agent=Agent(
            name=AgentName.MINI_SWE_AGENT.value,
            version=str(info.get("mini_version") or "unknown"),
            model_name=str(model_config.get("model_name") or "unknown"),
            extra={"original_format": trajectory_data.get("trajectory_format")},
        ),
        steps=steps,
        final_metrics=FinalMetrics(
            total_prompt_tokens=total_prompt_tokens or None,
            total_completion_tokens=total_completion_tokens or None,
            total_cached_tokens=(total_cached_tokens if (total_prompt_tokens or total_completion_tokens or total_cached_tokens) else None),
            total_cost_usd=total_cost,
            total_steps=len(steps),
            extra=final_extra,
        ),
    )


def convert_and_save_trajectory(
    source_path: Path,
    target_path: Path,
    session_id: str,
) -> None:
    trajectory_data = json.loads(source_path.read_text())
    trajectory = convert_mini_swe_agent_to_atif(trajectory_data, session_id)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(format_trajectory_json(trajectory.to_json_dict()))


class HarborMiniEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30


class HarborMiniEnvironment:
    """Synchronous mini-SWE environment that delegates commands to Harbor."""

    def __init__(
        self,
        harbor_environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        *,
        config: dict[str, Any] | None = None,
        mini_swe_agent_src: str | Path | None = None,
    ) -> None:
        _ensure_local_mini_swe_agent_importable(mini_swe_agent_src)
        from minisweagent.exceptions import Submitted

        self._submitted_exc = Submitted
        self._harbor_environment = harbor_environment
        self._loop = loop
        self.config = HarborMiniEnvironmentConfig(**(config or {}))

    def execute(self, action: dict[str, Any], cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        future = asyncio.run_coroutine_threadsafe(
            self._harbor_environment.exec(
                command,
                cwd=cwd or self.config.cwd or None,
                env=self.config.env or None,
                timeout_sec=timeout or self.config.timeout,
            ),
            self._loop,
        )
        try:
            result = future.result()
            output = {
                "output": result.stdout or "",
                "returncode": result.return_code,
                "exception_info": result.stderr or "",
            }
        except Exception as exc:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise self._submitted_exc(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        import platform

        return {
            **self.config.model_dump(),
            **platform.uname()._asdict(),
            **os.environ,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


class MiniSweAgent(BaseInstalledAgent):
    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        cost_limit: float = 0,
        step_limit: int = 0,
        command_timeout: int = 30,
        api_base: str | None = None,
        model_info: dict[str, Any] | None = None,
        mini_swe_agent_src: str | Path | None = None,
        execution_mode: str = "container_cli",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self.cost_limit = cost_limit
        self.step_limit = step_limit
        self.command_timeout = command_timeout
        self.api_base = api_base
        self.model_info = model_info or {}
        self.mini_swe_agent_src = _resolve_mini_swe_agent_src(mini_swe_agent_src)
        self.execution_mode = execution_mode

    @staticmethod
    def name() -> str:
        return AgentName.MINI_SWE_AGENT.value

    async def install(self, environment: BaseEnvironment) -> None:
        _ensure_local_mini_swe_agent_importable(self.mini_swe_agent_src)
        await environment.exec(
            "mkdir -p " + shlex.quote(str(EnvironmentPaths.agent_dir)),
            user="root",
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.model_name is None or "/" not in self.model_name:
            raise ValueError("mini-swe-agent requires model_name in provider/model_name format")

        remote_trajectory_path = EnvironmentPaths.agent_dir / "mini-swe-agent.trajectory.json"
        cmd_for_log = (
            "mini-swe-agent "
            "--yolo "
            f"--model={self.model_name} "
            f"--cost-limit {self.cost_limit} "
            "--exit-immediately "
            f"--output {remote_trajectory_path} "
            f"--task {shlex.quote(instruction)}"
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "mini-swe-agent.command.txt").write_text(cmd_for_log + "\n")

        if self.execution_mode == "container_cli":
            env = {"MSWEA_CONFIGURED": "true"}
            for key in ("MSWEA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY"):
                if os.environ.get(key):
                    env[key] = os.environ[key]
            await environment.exec(command=cmd_for_log, env=env)
            return

        if self.execution_mode != "host_bridge":
            raise ValueError(f"Unknown mini-swe-agent execution_mode: {self.execution_mode}")

        loop = asyncio.get_running_loop()
        trajectory_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        await asyncio.to_thread(
            self._run_sync,
            instruction,
            environment,
            loop,
            trajectory_path,
        )
        self.populate_context_post_run(context)

    def _run_sync(
        self,
        instruction: str,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        trajectory_path: Path,
    ) -> None:
        _ensure_local_mini_swe_agent_importable(self.mini_swe_agent_src)
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.models import get_model
        from minisweagent.utils.serialize import recursive_merge

        config = recursive_merge(
            get_config_from_spec(builtin_config_dir / "mini.yaml"),
            {
                "agent": {
                    "mode": "yolo",
                    "confirm_exit": False,
                    "cost_limit": self.cost_limit,
                    "step_limit": self.step_limit,
                    "output_path": trajectory_path,
                },
                "environment": {
                    "timeout": self.command_timeout,
                },
                "model": {
                    "model_name": self.model_name,
                    "cost_tracking": "ignore_errors",
                    "model_kwargs": ({"api_base": self.api_base} if self.api_base else {}),
                },
            },
        )
        model = get_model(config=config.get("model", {}))
        mini_env = HarborMiniEnvironment(
            environment,
            loop,
            config=config.get("environment", {}),
            mini_swe_agent_src=self.mini_swe_agent_src,
        )
        agent = DefaultAgent(model, mini_env, **config.get("agent", {}))
        agent.run(instruction)

    def populate_context_post_run(self, context: AgentContext) -> None:
        source_path = self.logs_dir / "mini-swe-agent.trajectory.json"
        if not source_path.exists():
            return
        try:
            trajectory_data = json.loads(source_path.read_text())
            trajectory = convert_mini_swe_agent_to_atif(
                trajectory_data,
                self.logs_dir.parent.name,
            )
            (self.logs_dir / "trajectory.json").write_text(
                format_trajectory_json(trajectory.to_json_dict())
            )
        except Exception:
            return

        final_metrics = trajectory.final_metrics
        if final_metrics is not None:
            context.n_input_tokens = final_metrics.total_prompt_tokens
            context.n_output_tokens = final_metrics.total_completion_tokens
            context.n_cache_tokens = final_metrics.total_cached_tokens
            context.cost_usd = final_metrics.total_cost_usd
        context.metadata = {
            "trajectory_format": trajectory_data.get("trajectory_format"),
            "mini_version": (trajectory_data.get("info") or {}).get("mini_version"),
            "mini_swe_agent_src": str(self.mini_swe_agent_src),
        }

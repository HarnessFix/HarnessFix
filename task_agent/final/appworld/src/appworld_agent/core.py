from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minisweagent.exceptions import Submitted

DEFAULT_STEP_LIMIT = 50
DEFAULT_COST_LIMIT = 10.0
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_APPWORLD_ROOT = Path("appworld_root")
DEFAULT_APPWORLD_IMAGE = "appworld-agent-pypi:latest"


@dataclass
class AppWorldTask:
    task_id: str
    instruction: str
    supervisor: dict[str, str]
    required_apps: list[str]
    difficulty: str

    @property
    def problem_statement(self) -> str:
        return (
            f"My name is: {self.supervisor['first_name']} {self.supervisor['last_name']}. "
            f"My personal email is {self.supervisor['email']} and phone number is {self.supervisor['phone_number']}.\n\n"
            f"Task: {self.instruction}"
        )


def resolve_appworld_root() -> Path:
    env_root = os.environ.get("APPWORLD_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    if DEFAULT_APPWORLD_ROOT.exists():
        return DEFAULT_APPWORLD_ROOT
    raise FileNotFoundError(
        "AppWorld root not found. Set APPWORLD_ROOT to a directory that contains data/."
    )


def load_subset(subset_dir: Path) -> list[AppWorldTask]:
    data_file = subset_dir / "data.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"data.jsonl not found in {subset_dir}")
    tasks: list[AppWorldTask] = []
    for line in data_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        tasks.append(
            AppWorldTask(
                task_id=raw["task_id"],
                instruction=raw["instruction"],
                supervisor=raw["supervisor"],
                required_apps=raw.get("required_apps", []),
                difficulty=str(raw.get("difficulty", "")),
            )
        )
    return tasks


def load_preds(output_dir: Path) -> dict[str, str]:
    preds_file = output_dir / "preds.json"
    if not preds_file.exists():
        return {}
    try:
        return json.loads(preds_file.read_text())
    except json.JSONDecodeError:
        return {}


def save_preds(output_dir: Path, preds: dict[str, str]) -> None:
    preds_file = output_dir / "preds.json"
    tmp_file = preds_file.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(preds, ensure_ascii=False, indent=2))
    tmp_file.rename(preds_file)


def _extract_submission(command: str, output: str) -> str:
    answer_match = re.search(r"complete_task\(\s*answer\s*=\s*(.+?)\s*\)", command, re.DOTALL)
    if answer_match:
        value = answer_match.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value
    return output.strip() or command.strip()


def _docker_run(cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


class AppWorldDockerEnvironment:
    def __init__(
        self,
        *,
        task: AppWorldTask,
        output_dir: Path,
        appworld_root: Path,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.task = task
        self.output_dir = output_dir
        self.appworld_root = appworld_root
        self.timeout_seconds = timeout_seconds
        self.experiment_name = f"appworld_agent_{task.task_id}_{uuid.uuid4().hex[:8]}"
        self.container_name = f"appworld_{task.task_id}_{uuid.uuid4().hex[:8]}"
        self.image = os.environ.get("APPWORLD_AGENT_IMAGE", DEFAULT_APPWORLD_IMAGE)
        self.experiments_dir = output_dir / "appworld_experiments"
        self._persistent_process: subprocess.Popen | None = None
        self._persistent_extra_output: list[str] = []
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._start_container()

    def _start_container(self) -> None:
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "-v",
            f"{self.appworld_root / 'data'}:/workspace/appworld_root/data:ro",
            "-v",
            f"{self.experiments_dir}:/workspace/appworld_root/experiments",
            "-e",
            "APPWORLD_ROOT=/workspace/appworld_root",
            self.image,
            "sleep",
            "infinity",
        ]
        result = _docker_run(cmd, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start AppWorld container: {result.stderr or result.stdout}")
        preflight = _docker_run(
            [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-lc",
                "python3 - <<'PY'\nfrom appworld.environment import AppWorld\nprint('APPWORLD_IMPORT_OK')\nPY",
            ],
            timeout=60,
        )
        if preflight.returncode != 0 or "APPWORLD_IMPORT_OK" not in (preflight.stdout or ""):
            raise RuntimeError(
                "AppWorld environment bootstrap failed inside container. "
                f"stdout={preflight.stdout!r} stderr={preflight.stderr!r}"
            )

    def _execute_python(self, script: str) -> subprocess.CompletedProcess:
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        command = f"python3 -c \"import base64; exec(base64.b64decode('{encoded}').decode())\""
        return _docker_run(
            ["docker", "exec", self.container_name, "bash", "-lc", command],
            timeout=self.timeout_seconds,
        )

    def _start_persistent_process(self) -> None:
        if self._persistent_process is not None and self._persistent_process.poll() is None:
            return
        server_script = f"""
import base64
import json
import os
import sys
import traceback

os.environ["APPWORLD_ROOT"] = "/workspace/appworld_root"
from appworld.common.path_store import path_store
from appworld.environment import AppWorld

path_store.update_root("/workspace/appworld_root")
_appworld = AppWorld(
    task_id={self.task.task_id!r},
    experiment_name={self.experiment_name!r},
    max_interactions=1000,
    timeout_seconds={self.timeout_seconds},
)
print("__HARNESSFIX_READY__", flush=True)

try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("close"):
            break
        code = base64.b64decode(request["code"]).decode("utf-8")
        try:
            result = _appworld.execute(code)
            payload = {{
                "returncode": 0,
                "output": "" if result is None else str(result),
                "exception_info": "",
                "completed": bool(_appworld.task_completed()),
            }}
        except Exception:
            payload = {{
                "returncode": 1,
                "output": "",
                "exception_info": traceback.format_exc(),
                "completed": bool(_appworld.task_completed()),
            }}
        print("__HARNESSFIX_RESULT__" + json.dumps(payload), flush=True)
finally:
    _appworld.close()
"""
        encoded = base64.b64encode(server_script.encode("utf-8")).decode("ascii")
        command = f"python3 -u -c \"import base64; exec(base64.b64decode('{encoded}').decode())\""
        self._persistent_process = subprocess.Popen(
            ["docker", "exec", "-i", self.container_name, "bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._persistent_process.stdout is not None
        for line in self._persistent_process.stdout:
            line = line.rstrip("\n")
            if line == "__HARNESSFIX_READY__":
                return
            self._persistent_extra_output.append(line)
        raise RuntimeError("Persistent AppWorld process exited before becoming ready.")

    def execute_persistent(self, code: str) -> dict[str, Any]:
        self._start_persistent_process()
        process = self._persistent_process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent AppWorld process is not available.")
        request = {"code": base64.b64encode(code.encode("utf-8")).decode("ascii")}
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith("__HARNESSFIX_RESULT__"):
                return json.loads(line.removeprefix("__HARNESSFIX_RESULT__"))
            self._persistent_extra_output.append(line)
        extra = "\n".join(self._persistent_extra_output[-20:])
        raise RuntimeError(f"Persistent AppWorld process exited unexpectedly.\n{extra}")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        code = action.get("command", "")
        script = f"""
import os
import traceback

os.environ["APPWORLD_ROOT"] = "/workspace/appworld_root"
from appworld.common.path_store import path_store
from appworld.environment import AppWorld

path_store.update_root("/workspace/appworld_root")

_appworld = AppWorld(
    task_id={self.task.task_id!r},
    experiment_name={self.experiment_name!r},
    max_interactions=1000,
    timeout_seconds={self.timeout_seconds},
)

try:
    _result = _appworld.execute({code!r})
    if _result is not None:
        print(_result)
    if _appworld.task_completed():
        print("__TASK_COMPLETED__")
except Exception:
    traceback.print_exc()
    raise
finally:
    _appworld.close()
"""
        result = self._execute_python(script)
        output = (result.stdout or "") + (f"\n{result.stderr}" if result.stderr else "")
        payload = {
            "output": output.replace("__TASK_COMPLETED__", "").strip(),
            "returncode": result.returncode,
            "exception_info": "" if result.returncode == 0 else (result.stderr or output),
        }
        if "__TASK_COMPLETED__" in output and payload["returncode"] == 0:
            submission = _extract_submission(code, payload.get("output", ""))
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )
        return payload

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "required_apps": ",".join(self.task.required_apps),
            "difficulty": self.task.difficulty,
            **kwargs,
        }

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": {
                        "task_id": self.task.task_id,
                        "experiment_name": self.experiment_name,
                        "appworld_root": str(self.appworld_root),
                        "image": self.image,
                        "timeout_seconds": self.timeout_seconds,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def cleanup(self) -> None:
        if self._persistent_process is not None and self._persistent_process.poll() is None:
            try:
                if self._persistent_process.stdin is not None:
                    self._persistent_process.stdin.write(json.dumps({"close": True}) + "\n")
                    self._persistent_process.stdin.flush()
                self._persistent_process.wait(timeout=5)
            except Exception:
                self._persistent_process.kill()
        _docker_run(["docker", "rm", "-f", self.container_name], timeout=30)



def evaluate_appworld_task(task_id: str, output_dir: Path, appworld_root: Path, experiment_name: str) -> dict[str, Any]:
    eval_container = f"appworld_eval_{task_id}_{uuid.uuid4().hex[:8]}"
    image = os.environ.get("APPWORLD_AGENT_IMAGE", DEFAULT_APPWORLD_IMAGE)
    experiments_dir = output_dir / "appworld_experiments"
    script = f"""
import json
import os

os.environ["APPWORLD_ROOT"] = "/workspace/appworld_root"
from appworld.common.path_store import path_store
from appworld.evaluator import evaluate_task

path_store.update_root("/workspace/appworld_root")

tracker = evaluate_task(
    task_id={task_id!r},
    experiment_name={experiment_name!r},
    suppress_errors=True,
    save_report=True,
)

result = {{
    "success": tracker.success,
    "pass_percentage": getattr(tracker, "pass_percentage", 100 if tracker.success else 0),
    "passes": [dict(item) for item in getattr(tracker, "passes", [])],
    "failures": [dict(item) for item in getattr(tracker, "failures", [])],
}}
print(json.dumps(result))
"""
    try:
        start = _docker_run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                eval_container,
                "-v",
                f"{appworld_root / 'data'}:/workspace/appworld_root/data:ro",
                "-v",
                f"{experiments_dir}:/workspace/appworld_root/experiments",
                "-e",
                "APPWORLD_ROOT=/workspace/appworld_root",
                image,
                "sleep",
                "60",
            ],
            timeout=60,
        )
        if start.returncode != 0:
            raise RuntimeError(start.stderr or start.stdout)
        encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
        command = f"python3 -c \"import base64; exec(base64.b64decode('{encoded}').decode())\""
        result = _docker_run(
            ["docker", "exec", eval_container, "bash", "-lc", command],
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout.strip())
    finally:
        _docker_run(["docker", "rm", "-f", eval_container], timeout=30)


def write_eval_artifacts(output_dir: Path, eval_result: dict[str, Any]) -> None:
    failures = eval_result.get("failures", [])
    passes = eval_result.get("passes", [])
    persisted = dict(eval_result)
    persisted["pass_count"] = len(passes)
    persisted["fail_count"] = len(failures)
    lines = [
        "## AppWorld Evaluation Result",
        "",
        f"Status: {'PASS' if eval_result.get('success') else 'FAIL'}",
        f"Pass percentage: {eval_result.get('pass_percentage', 0)}",
        f"Pass count: {len(passes)}",
        f"Fail count: {len(failures)}",
        "",
        "### Failed requirements",
    ]
    if failures:
        for failure in failures:
            lines.append(f"- {failure.get('requirement', str(failure))}")
    else:
        lines.append("- None")
    (output_dir / "eval_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "result.json").write_text(json.dumps(persisted, indent=2, ensure_ascii=False) + "\n")

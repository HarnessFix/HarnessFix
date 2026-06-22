import re
from pathlib import PurePosixPath

MAX_OBSERVATION_CHARS = 12000
_HEAD_CHARS = 6000
_TAIL_CHARS = 6000

_FORBIDDEN_COMMAND_PATTERNS = (
    (re.compile(r"\bgit\s+add\s+(?:-A|--all|\.)\b"), "Do not use `git add -A`, `git add --all`, or `git add .`; create patch.txt with `git diff HEAD -- <explicit source files> > patch.txt`."),
    (re.compile(r"\bgit\s+commit\b"), "Do not commit inside the task container; submit a patch only."),
    (re.compile(r"\bls\b[^\n;&|]*-[A-Za-z]*R[A-Za-z]*\b"), "Recursive `ls -R`/`ls -laR` output is too large. Use targeted `find . -maxdepth ...`, `rg --files`, or `sed -n` commands."),
    (re.compile(r"\bfind\s+/"), "Do not search from filesystem root. Search the repository with bounded predicates such as `find . -maxdepth 3 -name ...`."),
)

_FORBIDDEN_PATH_PARTS = {"test", "tests", "testing", "mytestapp", "__pycache__"}
_FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".so", ".o", ".a", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz")
_FORBIDDEN_BASENAME_PREFIXES = ("test_", "check_", "repro", "reproduce", "tmp", "temp")
_FORBIDDEN_BASENAME_CONTAINS = ("repro", "scratch")


def reject_forbidden_command(command: str) -> dict | None:
    compact = " ".join((command or "").split())
    for pattern, reason in _FORBIDDEN_COMMAND_PATTERNS:
        if pattern.search(compact):
            return {
                "output": "Command rejected by HarnessFix guardrail.\n" + reason,
                "returncode": 2,
                "exception_info": "",
                "extra": {"guardrail": "forbidden_command", "reason": reason},
            }
    return None


def truncate_observation_output(text: str) -> str:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    omitted = len(text) - _HEAD_CHARS - _TAIL_CHARS
    if omitted < 0:
        omitted = len(text) - MAX_OBSERVATION_CHARS
    return (
        text[:_HEAD_CHARS]
        + "\n[HarnessFix output truncated: {} characters omitted. Use a narrower command, write large output to a file, then inspect small slices.]\n".format(omitted)
        + text[-_TAIL_CHARS:]
    )


def truncate_output_dict(output: dict) -> dict:
    if isinstance(output.get("output"), str):
        output = dict(output)
        output["output"] = truncate_observation_output(output["output"])
    return output


def _diff_paths(submission: str) -> list[str]:
    paths: list[str] = []
    for line in submission.splitlines():
        if line.startswith("diff --git "):
            pieces = line.split()
            if len(pieces) >= 4:
                for raw in pieces[2:4]:
                    if raw.startswith("a/") or raw.startswith("b/"):
                        path = raw[2:]
                        if path != "/dev/null":
                            paths.append(path)
        elif line.startswith("+++ ") or line.startswith("--- "):
            raw = line[4:].strip()
            if raw.startswith("a/") or raw.startswith("b/"):
                path = raw[2:]
                if path != "/dev/null":
                    paths.append(path)
    return sorted(set(paths))


def _looks_forbidden_path(path: str) -> str | None:
    posix = path.replace("\\", "/")
    parts = [p for p in PurePosixPath(posix).parts if p not in (".", "")]
    lowered_parts = [p.lower() for p in parts]
    basename = lowered_parts[-1] if lowered_parts else ""
    if any(part in _FORBIDDEN_PATH_PARTS for part in lowered_parts):
        return "patch modifies test/helper directories"
    if basename.startswith(_FORBIDDEN_BASENAME_PREFIXES):
        return "patch includes a helper/reproduction-looking file"
    if any(token in basename for token in _FORBIDDEN_BASENAME_CONTAINS):
        return "patch includes a helper/reproduction-looking file"
    if basename.endswith(_FORBIDDEN_SUFFIXES):
        return "patch includes binary/build artifacts"
    if basename in {"patch.txt", "tmp.patch", "output.txt", "log.txt"}:
        return "patch includes generated artifacts"
    return None


def extract_submission(output: str, returncode: int) -> tuple[bool, str]:
    if returncode != 0:
        return False, ""
    lines = output.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if line.strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            return True, "".join(lines[idx + 1:])
    return False, ""


def validate_submission(submission: str) -> tuple[bool, str]:
    if not submission.strip():
        return False, "Submission is empty. Create patch.txt from explicit modified source files, verify it, then submit it."
    if "diff --git" not in submission:
        return False, "Submission does not look like a unified git diff. Submit `cat patch.txt`, where patch.txt was created with `git diff HEAD -- <explicit source files> > patch.txt`."
    paths = _diff_paths(submission)
    if not paths:
        return False, "Could not identify changed paths in the submitted diff."
    bad = []
    for path in paths:
        reason = _looks_forbidden_path(path)
        if reason:
            bad.append(f"{path} ({reason})")
    if bad:
        return False, "Submission rejected because it includes non-source artifacts: " + "; ".join(bad[:8])
    if "new file mode" in submission:
        new_file_paths = []
        current_path = ""
        for line in submission.splitlines():
            if line.startswith("diff --git "):
                pieces = line.split()
                current_path = pieces[3][2:] if len(pieces) >= 4 and pieces[3].startswith("b/") else ""
            elif line.startswith("new file mode") and current_path:
                new_file_paths.append(current_path)
        bad_new = [p for p in new_file_paths if _looks_forbidden_path(p)]
        if bad_new:
            return False, "Submission rejected because it creates helper/test artifacts: " + ", ".join(bad_new[:8])
    return True, ""

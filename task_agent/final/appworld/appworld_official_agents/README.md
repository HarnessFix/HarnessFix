# Vendored AppWorld Agents

This directory contains a local, editable copy of the official AppWorld
`experiments` agent code needed for `simplified_react_code_agent`.

Source:

- Repository: `https://github.com/StonyBrookNLP/appworld`
- Local source used for vendoring: `/tmp/appworld_official/experiments`
- License: Apache-2.0, copied in `LICENSE`

Only the simplified ReAct Code Agent path is vendored:

- `appworld_agents/code/common/`
- `appworld_agents/code/simplified/agent.py`
- `appworld_agents/code/simplified/language_model.py`
- `appworld_agents/code/simplified/react_code_agent.py`
- `appworld_agents/prompts/react_code_agent/instructions.txt`

`appworld/` is a minimal host-side compatibility shim. The real AppWorld
environment still runs inside the existing Docker image through the HarnessFix
adapter in `task_agent/appworld_agent/src/appworld_agent/official_react_adapter.py`.

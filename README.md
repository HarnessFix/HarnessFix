# HarnessFix

HarnessFix is a trace-guided framework for diagnosing failures in LLM agent trajectories and repairing the agent harness that caused them.

This repository contains the code artifact for the paper:

> From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws

The public artifact keeps the source code, repair pipeline, analysis utilities, and the initial/final harness snapshots used by the paper. It intentionally excludes private API keys, local paths, raw benchmark data, model traces, downloaded files, logs, and generated evaluation outputs.

## What Is Included

```text
failure_analysis/              HTIR construction, failure diagnosis, consolidation, repair memory, validation helpers
enhancement_implementation/    Modify-agent prompts/configs for scoped harness repair
task_agent/                    Benchmark task agents and runner scripts
task_agent/final/              Final repaired harness snapshots used as paper artifacts
data/                          Dataset sampling scripts, not the datasets themselves
eval/                          Lightweight evaluators for GAIA, AppWorld, and Terminal-Bench outputs
run_pipeline_*.py              End-to-end closed-loop pipelines
```

Final repaired harness snapshots:

- `task_agent/final/swe`
- `task_agent/final/gaia`
- `task_agent/final/appworld`
- `task_agent/final/terminal_bench`

Initial harness snapshots are kept at:

- `task_agent/mini-swe-agent`
- `task_agent/open_deep_research`
- `task_agent/appworld_agent`
- `task_agent/terminal_bench_agent`

## Setup

```bash
git clone <repo-url> HarnessFix
cd HarnessFix
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with your own credentials. Do not commit `.env`.

For SWE-Bench runs, install the vendored agent package:

```bash
.venv/bin/python -m pip install -e task_agent/mini-swe-agent
```

For GAIA runs:

```bash
.venv/bin/python -m pip install -e task_agent/open_deep_research
```

Some benchmarks require additional official setup:

- SWE-Bench: Docker and the official SWE-Bench evaluation dependencies.
- Terminal-Bench: Docker and Terminal-Bench task data.
- GAIA: HuggingFace access for the gated dataset and a web-search key.
- AppWorld: an installed AppWorld root and Docker image.

## Model Configuration

`task_agent/model_registry.json` contains public OpenAI-compatible examples. If you use another provider, update `api_base`, `api_key_env`, and model names there, or set compatible LiteLLM environment variables in `.env`.

## Data Preparation

The repository does not include benchmark data. Use the scripts under `data/` after you have access to the official datasets.

Examples:

```bash
.venv/bin/python data/sample_swebench.py
.venv/bin/python data/sample_gaia.py
.venv/bin/python data/sample_terminal_bench.py
APPWORLD_TASK_CACHE=/path/to/appworld/task_cache.json .venv/bin/python data/sample_appworld.py
```

## Running Pipelines

SWE-Bench:

```bash
.venv/bin/python run_pipeline_swe.py \
  --model openai/gpt-5-mini \
  --analysis-model openai/gpt-5-mini \
  --max-iterations 1
```

GAIA:

```bash
.venv/bin/python run_pipeline_gaia.py \
  --model openai/gpt-5-mini \
  --analysis-model openai/gpt-5-mini \
  --max-iterations 1
```

Terminal-Bench:

```bash
.venv/bin/python run_pipeline_terminal_bench.py \
  --model openai/gpt-5-mini \
  --analysis-model openai/gpt-5-mini \
  --max-iterations 1
```

AppWorld:

```bash
APPWORLD_ROOT=/path/to/appworld_root \
.venv/bin/python run_pipeline_appworld.py \
  --model openai/gpt-5-mini \
  --analysis-model openai/gpt-5-mini \
  --max-iterations 1
```

These commands assume that the corresponding sampled datasets already exist under `data/`.

## Artifact Notes

- `traces/`, `logs/`, `results/`, `failure_analysis/results/`, and `failure_analysis/memory/` are generated at runtime and ignored by Git.
- Large raw trajectories and benchmark outputs were excluded from the public repository.
- The code uses public placeholder paths and OpenAI-compatible endpoints; configure your local environment through `.env` and `task_agent/model_registry.json`.

## Citation

If you use this artifact, please cite the accompanying paper once citation metadata is available.

# Evaluation Helpers

This directory contains lightweight evaluators for benchmark outputs that can be scored directly from saved predictions and task metadata:

- `eval_gaia.py`: scores GAIA final-answer predictions.
- `eval_appworld.py`: summarizes AppWorld task-goal completion outputs.
- `eval_terminal_bench.py`: summarizes Terminal-Bench result outputs.

SWE-Bench Verified is not evaluated by a lightweight script in this directory. The SWE pipeline uses the official SWE-Bench evaluation harness because each instance must be resolved by applying the submitted patch and running the benchmark's repository-level tests. See `run_pipeline_swe.py` for the `swebench.harness.run_evaluation` invocation.

# Experiment Details

This directory preserves the detailed experimental design and result analysis used in the HarnessFix paper. The paper may contain a shortened version for space; these notes keep the complete setup, tables, and interpretation.

## Contents

- `rq1_effectiveness.md`: end-to-end benchmark comparison and adaptation/repair token statistics.
- `rq2_failure_diagnosis.md`: human-audited diagnosis evaluation and HTIR representation ablation.
- `rq3_ablation.md`: task-performance ablation of the main design choices.
- `rq4_transfer.md`: cross-model transfer details and cross-benchmark repair-source sensitivity checks.
- `model_configuration.md`: default model settings and cross-model provider settings used in auxiliary/transfer runs.

## Shared Setup

We evaluate HarnessFix on four agent harnesses and benchmarks:

| Benchmark | Domain | Initial harness H0 | Sampled tasks | Split |
| --- | --- | --- | ---: | --- |
| GAIA | Open-ended research QA | open-deep-research | 150 of 166 public-ground-truth developer questions | 60/30/60 |
| SWE-Bench Verified | Repository-level software repair | mini-swe-agent | 250 of 500 instances | 100/50/100 |
| AppWorld | Stateful application automation | simplified ReAct code agent | 225 of 750 tasks | 90/45/90 |
| Terminal-Bench 2.0 Verified | Terminal command-line workflows | Harbor Terminus-2 | 85 of 89 tasks | 34/17/34 |

Unless otherwise stated, experiments use `gpt-5-mini` as the default task model and report averages over three independent runs. Test scores are task-success percentages on held-out test tasks. Offline adaptation/repair token counts exclude task-execution tokens used to collect trajectories.

Training tasks are used for adaptation, including running the initial harness H0, collecting trajectories, and producing method-specific updates. Validation tasks are used for candidate selection or acceptance when a method has a validation gate. Test tasks are held out from adaptation and validation decisions and are used only for the final comparison.

Following standard benchmark practice, we measure task completion rate within each benchmark. GAIA success means an exact-match final answer. SWE-Bench Verified success means a resolved repository instance. AppWorld reports Task Goal Completion. Terminal-Bench 2.0 Verified success means passing the verification tests for the task.

## Benchmark-Specific Notes

GAIA evaluates research workflows that require web search, file reading, document inspection, evidence retention, and final answer synthesis. SWE-Bench Verified evaluates repository-level software repair; each instance produces a trace, prediction, patch, and evaluator report. AppWorld evaluates application automation over controlled apps and APIs, with state-based and execution-based checks for missing or unintended side effects. Terminal-Bench 2.0 Verified evaluates command-line workflows in task-specific terminal environments, with tests over produced files, command effects, and final artifacts.

## Baselines

RQ1 compares HarnessFix with two baseline families. Human-designed harness baselines are benchmark-specific systems: DeepResearchAgent and MiroFlow for GAIA; OpenHands and Trae-Agent for SWE-Bench Verified; FullCodeRefl, IPFunCall, and CUGA for AppWorld; and OpenCode and OpenHands for Terminal-Bench 2.0 Verified. Self-evolution and repair baselines are GEPA, SCOPE, ReCreate, and Meta-Harness.

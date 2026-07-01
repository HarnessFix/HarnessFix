# Token Statistics and Amortization

This note preserves the detailed token-cost discussion for HarnessFix and the adaptation baselines. Counts report offline adaptation/repair tokens in millions and exclude task-execution tokens used to collect trajectories.

| Benchmark | GEPA | SCOPE | ReCreate | Meta-Harness | HarnessFix |
| --- | ---: | ---: | ---: | ---: | ---: |
| GAIA | 32.6 | 26.9 | 67.9 | 94.2 | 57.6 |
| SWE-Bench Verified | 28.1 | 25.6 | 57.2 | 82.9 | 48.9 |
| AppWorld | 22.7 | 18.8 | 44.1 | 74.6 | 37.2 |
| Terminal-Bench 2.0 Verified | 18.6 | 18.1 | 38.8 | 66.8 | 33.5 |

The variation follows the adaptation split size, the number of failed trajectories to diagnose, the length of retained traces, and the number of repair iterations. GAIA and SWE consume more adaptation tokens because their failures carry longer diagnostic context. AppWorld and Terminal-Bench consume fewer tokens because the retained traces or adaptation splits are smaller.

Prompt- and memory-oriented baselines such as GEPA and SCOPE are cheaper because they mainly update model-facing instructions or guideline memory. HarnessFix spends more tokens than these prompt-oriented baselines because it performs HTIR-based failure diagnosis, harness-layer mapping, implementation-anchor localization, scoped repair planning, and patch validation. ReCreate and Meta-Harness spend more tokens than HarnessFix in the reported comparison because they perform broader scaffold or harness search. This cost is paid offline during harness adaptation rather than during every future task execution.

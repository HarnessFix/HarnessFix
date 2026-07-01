# RQ3: Ablation Study

RQ3 asks whether HarnessFix's design choices contribute to task-performance gains. Values are held-out test task scores in percentages.

| Variant | GAIA | SWE | AppWorld | TB2 |
| --- | ---: | ---: | ---: | ---: |
| H0 | 43.3 | 45.3 | 36.7 | 17.6 |
| Prompt-only repair | 50.6 | 48.3 | 37.4 | 18.6 |
| w/o trace-grounded diagnosis | 51.1 | 50.7 | 38.1 | 21.6 |
| w/o scoped repair operators | 50.6 | 49.3 | 37.4 | 18.6 |
| w/o regression-aware acceptance | 55.6 | 53.3 | 39.3 | 24.5 |
| Full HarnessFix | 61.7 | 57.3 | 43.0 | 26.5 |

## Variants

Prompt-only repair converts diagnosed flaws into prompt updates without changing runtime harness mechanisms. This variant recovers only part of the full gain: it reaches 50.6 on GAIA, 48.3 on SWE-Bench Verified, 37.4 on AppWorld, and 18.6 on Terminal-Bench 2.0 Verified, trailing full HarnessFix by 11.1, 9.0, 5.6, and 7.9 points. Soft guidance can nudge model behavior, but it does not change runtime mechanisms such as tool schemas, lifecycle guards, observability instrumentation, or verification logic.

The `w/o trace-grounded diagnosis` variant removes responsible-step attribution and harness flaw diagnosis, deriving repair context from raw trajectory summaries. It gives intermediate improvements on every benchmark, but still trails full HarnessFix by 10.6 points on GAIA, 6.6 points on SWE-Bench Verified, 4.9 points on AppWorld, and 4.9 points on Terminal-Bench 2.0 Verified. Without trace-grounded diagnosis, the repair stage tends to address surface symptoms rather than the specific TraceStep and harness layer responsible for the failure.

The `w/o scoped repair operators` variant replaces layer-scoped repair with free-form harness editing. It reaches 50.6 on GAIA, 49.3 on SWE-Bench Verified, 37.4 on AppWorld, and 18.6 on Terminal-Bench 2.0 Verified, trailing full HarnessFix on all four benchmarks. Free-form edits in core orchestration or context-construction code can mix targeted fixes with broader, weakly motivated changes, which can introduce regressions and offset the intended improvement.

The `w/o regression-aware acceptance` variant accepts candidate changes without the validation-set target-flaw and regression checks. It is the closest ablation to full HarnessFix, confirming that diagnosis and repair often produce useful candidates. However, it still trails full HarnessFix on all four benchmarks: 55.6 vs. 61.7 on GAIA, 53.3 vs. 57.3 on SWE-Bench Verified, 39.3 vs. 43.0 on AppWorld, and 24.5 vs. 26.5 on Terminal-Bench 2.0 Verified. Regression-aware acceptance turns useful candidates into safe-to-accept harness changes.

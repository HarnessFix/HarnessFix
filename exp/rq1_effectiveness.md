# RQ1: Effectiveness

RQ1 asks whether HarnessFix improves task performance compared with the original harness, benchmark-specific human-designed harnesses, and self-evolution or repair baselines.

Test values are percentages. Parentheses in the Test column show HarnessFix's percentage-point gain over that row. Tokens are offline adaptation/repair costs in millions and exclude task-execution tokens used to collect trajectories; parentheses in the Tokens column show relative change against HarnessFix within the same benchmark. `/` denotes not applicable.

| Benchmark | Role | Method / harness | Test | Tokens (M) |
| --- | --- | --- | ---: | ---: |
| GAIA | Selected H0 | open-deep-research | 43.3 (+18.4) | / |
| GAIA | Human-designed | DeepResearchAgent | 52.8 (+8.9) | / |
| GAIA | Human-designed | MiroFlow | 58.3 (+3.4) | / |
| GAIA | Evolve/repair from H0 | H0 + GEPA | 46.7 (+15.0) | 32.6 (-43.4) |
| GAIA | Evolve/repair from H0 | H0 + SCOPE | 45.0 (+16.7) | 26.9 (-53.3) |
| GAIA | Evolve/repair from H0 | H0 + ReCreate | 52.8 (+8.9) | 67.9 (+17.9) |
| GAIA | Evolve/repair from H0 | H0 + Meta-Harness | 56.7 (+5.0) | 94.2 (+63.5) |
| GAIA | Evolve/repair from H0 | H0 + HarnessFix | 61.7 | 57.6 |
| SWE | Selected H0 | mini-swe-agent | 45.3 (+12.0) | / |
| SWE | Human-designed | OpenHands | 47.3 (+10.0) | / |
| SWE | Human-designed | Trae-Agent | 48.7 (+8.6) | / |
| SWE | Evolve/repair from H0 | H0 + GEPA | 46.7 (+10.6) | 28.1 (-42.5) |
| SWE | Evolve/repair from H0 | H0 + SCOPE | 48.3 (+9.0) | 25.6 (-47.6) |
| SWE | Evolve/repair from H0 | H0 + ReCreate | 51.7 (+5.6) | 57.2 (+17.0) |
| SWE | Evolve/repair from H0 | H0 + Meta-Harness | 54.7 (+2.6) | 82.9 (+69.5) |
| SWE | Evolve/repair from H0 | H0 + HarnessFix | 57.3 | 48.9 |
| AppWorld | Selected H0 | ReAct | 36.7 (+6.3) | / |
| AppWorld | Human-designed | FullCodeRefl | 35.6 (+7.4) | / |
| AppWorld | Human-designed | IPFunCall | 38.1 (+4.9) | / |
| AppWorld | Human-designed | CUGA | 41.1 (+1.9) | / |
| AppWorld | Evolve/repair from H0 | H0 + GEPA | 37.4 (+5.6) | 22.7 (-39.0) |
| AppWorld | Evolve/repair from H0 | H0 + SCOPE | 38.9 (+4.1) | 18.8 (-49.5) |
| AppWorld | Evolve/repair from H0 | H0 + ReCreate | 39.3 (+3.7) | 44.1 (+18.5) |
| AppWorld | Evolve/repair from H0 | H0 + Meta-Harness | 40.4 (+2.6) | 74.6 (+100.5) |
| AppWorld | Evolve/repair from H0 | H0 + HarnessFix | 43.0 | 37.2 |
| TB2 | Selected H0 | Harbor Terminus-2 | 17.6 (+8.9) | / |
| TB2 | Human-designed | OpenCode | 22.5 (+4.0) | / |
| TB2 | Human-designed | OpenHands | 18.6 (+7.9) | / |
| TB2 | Evolve/repair from H0 | H0 + GEPA | 19.6 (+6.9) | 18.6 (-44.5) |
| TB2 | Evolve/repair from H0 | H0 + SCOPE | 20.6 (+5.9) | 18.1 (-46.0) |
| TB2 | Evolve/repair from H0 | H0 + ReCreate | 21.6 (+4.9) | 38.8 (+15.8) |
| TB2 | Evolve/repair from H0 | H0 + Meta-Harness | 23.5 (+3.0) | 66.8 (+99.4) |
| TB2 | Evolve/repair from H0 | H0 + HarnessFix | 26.5 | 33.5 |

## Main Observations

Across three independent runs, held-out test standard deviations are small when reported in percentage points: 1.7 on GAIA, 1.7 on SWE-Bench Verified, 1.9 on AppWorld, and 2.9 on Terminal-Bench 2.0 Verified.

HarnessFix improves over the initial harness on all four benchmarks: +18.4 points on GAIA, +12.0 points on SWE-Bench Verified, +6.3 points on AppWorld, and +8.9 points on Terminal-Bench 2.0 Verified. The gains appear across open-ended research, repository patching, stateful application automation, and command-line workflows.

HarnessFix also outperforms every self-evolution and repair baseline. Compared with the strongest non-HarnessFix repair baseline in each benchmark, it gains +5.0 points on GAIA, +2.6 points on SWE-Bench Verified, +2.6 points on AppWorld, and +3.0 points on Terminal-Bench 2.0 Verified.

Prompt-oriented baselines such as GEPA and SCOPE improve H0 only modestly, consistent with their edit surface being limited to model-facing instructions or memory. ReCreate narrows the gap by adapting tools and workflow scaffolds, and Meta-Harness improves further by searching over code-level harness revisions. HarnessFix still performs best because it explicitly attributes failures to responsible TraceSteps, harness layers, and implementation anchors before generating scoped repairs.

The offline adaptation/repair token counts are higher for GAIA and SWE because their failed trajectories carry more diagnostic context. HarnessFix uses more tokens than prompt-oriented baselines, but remains below ReCreate and Meta-Harness in this accounting because its repair loop typically stops after 3 to 4 iterations.

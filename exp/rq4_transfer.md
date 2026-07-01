# RQ4: Cross-Model Transfer

RQ4 asks whether a harness repaired with one task model still improves performance when reused with different task models, and whether HarnessFix remains effective when the repair-source model changes. Values are three-run average percentages; Delta is the percentage-point improvement from the repaired harness.

## Cross-Model Transfer on GAIA

The repaired GAIA harness is selected using `gpt-5-mini` and reused without target-model-specific harness repair.

| Model | H0 | H0 + HarnessFix | Delta |
| --- | ---: | ---: | ---: |
| GPT-5 mini (repair source) | 43.3 | 61.7 | +18.4 |
| Claude Sonnet 4.5 | 66.7 | 72.2 | +5.5 |
| DeepSeek V3.2 | 58.9 | 66.7 | +7.8 |
| Qwen3.5 Plus | 63.3 | 72.8 | +9.5 |
| Gemini 3 Pro | 69.4 | 78.3 | +8.9 |

The source-model repair improves GAIA performance by +18.4 points. When reused with four additional target models, the same repaired harness improves every model: +9.5 points for Qwen3.5 Plus, +7.8 points for DeepSeek V3.2, +8.9 points for Gemini 3 Pro, and +5.5 points for Claude Sonnet 4.5. The transfer gains are smaller than the source-model gain but consistently positive, suggesting that the repaired mechanisms address harness-level failures shared across models.

## Repair-Source Sensitivity Across Benchmarks

We rerun the full repair loop using multiple repair-source models and evaluate H0 and H0 + HarnessFix on all four benchmark test splits. This setting checks whether the diagnosis and repair loop remains effective when the failed trajectories and candidate repairs are produced with source models other than `gpt-5-mini`.

| Benchmark | Repair-source model | H0 | H0 + HarnessFix | Delta |
| --- | --- | ---: | ---: | ---: |
| GAIA | gpt-5-mini | 43.3 | 61.7 | +18.4 |
| GAIA | Claude Sonnet 4.5 | 66.7 | 84.4 | +17.7 |
| GAIA | DeepSeek V3.2 | 58.9 | 75.6 | +16.7 |
| GAIA | Qwen3.5 Plus | 63.3 | 78.9 | +15.6 |
| GAIA | Gemini 3 Pro | 69.4 | 83.9 | +14.5 |
| SWE | gpt-5-mini | 45.3 | 57.3 | +12.0 |
| SWE | Claude Sonnet 4.5 | 61.3 | 70.7 | +9.4 |
| SWE | DeepSeek V3.2 | 58.7 | 70.3 | +11.6 |
| SWE | Qwen3.5 Plus | 62.7 | 72.3 | +9.6 |
| SWE | Gemini 3 Pro | 60.3 | 70.7 | +10.4 |
| AppWorld | gpt-5-mini | 36.7 | 43.0 | +6.3 |
| AppWorld | Claude Sonnet 4.5 | 61.1 | 70.4 | +9.3 |
| AppWorld | DeepSeek V3.2 | 56.3 | 62.2 | +5.9 |
| AppWorld | Qwen3.5 Plus | 53.3 | 61.9 | +8.6 |
| AppWorld | Gemini 3 Pro | 50.7 | 59.6 | +8.9 |
| TB2 | gpt-5-mini | 17.6 | 26.5 | +8.9 |
| TB2 | Claude Sonnet 4.5 | 34.3 | 47.1 | +12.7 |
| TB2 | DeepSeek V3.2 | 31.4 | 39.2 | +7.9 |
| TB2 | Qwen3.5 Plus | 42.2 | 52.9 | +10.7 |
| TB2 | Gemini 3 Pro | 46.1 | 55.9 | +9.8 |

Across repair-source models and benchmarks, HarnessFix improves the selected H0 by +5.9 to +18.4 points.

## Gemini 3 Flash GAIA Baseline Comparison

As a focused baseline check, we use Gemini 3 Flash as the repair-source model on GAIA and compare against the full baseline set. Parentheses show HarnessFix point gains over the corresponding row.

| Role | Method / harness | TCR |
| --- | --- | ---: |
| Selected H0 | open-deep-research | 48.9 (+17.8) |
| Human-designed | DeepResearchAgent | 57.2 (+9.5) |
| Human-designed | MiroFlow | 61.7 (+5.0) |
| Evolve/repair from H0 | H0 + GEPA | 52.8 (+13.9) |
| Evolve/repair from H0 | H0 + SCOPE | 51.1 (+15.6) |
| Evolve/repair from H0 | H0 + ReCreate | 58.3 (+8.4) |
| Evolve/repair from H0 | H0 + Meta-Harness | 61.1 (+5.6) |
| Evolve/repair from H0 | H0 + HarnessFix | 66.7 |

HarnessFix remains above the strongest human-designed harness and the strongest automated baseline in this Gemini 3 Flash repair-source setting. This indicates that the repair loop is not tied to the original `gpt-5-mini` source model: when failed trajectories and candidate repairs are produced by a different model, the remaining failures still expose harness-level issues that benefit from TraceStep-grounded diagnosis and scoped repair.

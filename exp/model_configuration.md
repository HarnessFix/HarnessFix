# Model Configuration

This note preserves model and provider settings that may be shortened in the paper for space.

## Default Setting

We use `gpt-5-mini` as the default model for all approaches and benchmarks. Because `gpt-5-mini` is a reasoning model, we fix temperature to `1` and use the default reasoning configuration: `reasoning={effort: medium}` and `text={verbosity: medium}`. Results are reported as averages over three independent runs unless otherwise stated.

## Cross-Model Transfer Settings

For the RQ4 cross-model transfer experiments, we follow the recommended inference settings for each model provider while reusing the same repaired GAIA harness.

| Model | Settings |
| --- | --- |
| DeepSeek V3.2 | `temperature=1.0`, `top_p=1.0` |
| Claude Sonnet 4.5 | default reasoning setting; not the high-reasoning configuration |
| Qwen3.5 Plus | thinking enabled; `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0` |
| Gemini 3 Pro | default `temperature=1.0`; medium thinking level |

Auxiliary experiments vary model strength to test whether harness repair changes task performance under weaker or stronger base models.

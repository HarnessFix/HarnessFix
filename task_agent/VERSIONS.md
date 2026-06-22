# Task Agent Version Notes

This document records the fixed initial versions used for the task-agent
repositories. These versions define the baseline harnesses used by HarnessFix.

## Version Selection Policy

To keep the experiments reproducible, each upstream repository was checked out
to a stable historical version at the start of the project. This avoids result
changes caused by ongoing upstream development. The selected versions generally
come from late 2025 to early 2026, with preference for the last stable release
before a major upstream rewrite.

---

## Repository Versions

### mini-swe-agent

| Field | Value |
|------|------|
| **Version** | `v1.17.5` |
| **Date** | 2026-01-30 |
| **Commit** | `68d367bf` |
| **Rationale** | The last stable version in the v1.x series. v2.0.0, released on 2026-02-10, was a major rewrite with 197 intervening commits and substantial structural changes. v1.17.5 was mature and stable for SWE-Bench integration and interfaces. |

```bash
cd task_agent/mini-swe-agent && git checkout v1.17.5
```

---

### OpenHands

| Field | Value |
|------|------|
| **Version** | `1.1.0` |
| **Date** | 2025-12-30 |
| **Commit** | `9885ddea3` |
| **Rationale** | A stable official release from the end of 2025. Compared with 1.0.0, released on 2025-12-15, it includes about two weeks of fixes while remaining more conservative than the 1.2.x line released around 2026-01-15. |

```bash
cd task_agent/OpenHands && git checkout 1.1.0
```

---

### SWE-Fixer

| Field | Value |
|------|------|
| **Version** | commit `ab0bffd` |
| **Date** | 2025-01-13 |
| **Notes** | This repository did not have tags and had a relatively informal maintenance history, with many commit messages such as "update". The merge commit from 2025-01-13 was selected as the baseline because it was the last clearly meaningful checkpoint in the repository history. |

```bash
cd task_agent/SWE-Fixer && git checkout ab0bffd
```

---

## Restore Commands

To restore the initial baseline versions from another checkout state, run:

```bash
cd task_agent
git -C mini-swe-agent checkout v1.17.5
git -C OpenHands checkout 1.1.0
git -C SWE-Fixer checkout ab0bffd
```

## Notes

- The upstream repositories were evaluated at fixed detached-HEAD versions. If you need to modify one of them directly, create a branch first with `git switch -c <branch-name>`.
- The initial SWE-Bench experiments used **mini-swe-agent** to run baseline evaluations and collect traces.

## HarnessFix SWE GPT-5 Mini Evolution Marker

For the SWE-Bench Verified GPT-5 mini evolution experiment recorded on
2026-05-25, the final local enhanced-agent marker is:

```text
task_agent/enhanced_swe_final_swe_evolve_gpt5mini_20260523
```

It was copied from:

```text
task_agent/enhanced_swe_v3_swe_evolve_gpt5mini_20260523
```

The final marker directory is ignored by git through `task_agent/enhanced_*/`.
Tracked experiment results are recorded in:

```text
exp_result/swe_gpt5mini_evolution_results.md
exp_result/swe_gpt5mini_evolution_results.json
```

## HarnessFix GAIA GPT-5 Mini Evolution Marker

For the GAIA GPT-5 mini manual evolution experiment recorded on 2026-05-28, the
final local enhanced-agent marker is:

```text
task_agent/enhanced_odr_final_gpt5mini_20260528
```

It was copied from:

```text
task_agent/enhanced_odr_gpt_5_mini_v3
```

The final marker directory is ignored by git through `task_agent/enhanced_*/`.
Tracked experiment results are recorded in:

```text
exp_result/gaia_gpt5mini_manual_evolution_results.md
exp_result/gaia_gpt5mini_manual_evolution_results.json
```

# Task Agent 初始版本说明

本文档记录各 task agent 仓库所使用的初始固定版本，作为 AgentSystemEvolve 项目的基线。

## 版本选择原则

为保证实验的可复现性，在项目启动时将各仓库 checkout 到一个稳定的历史版本，避免上游持续更新导致结果不一致。版本选择区间为 2025 年底至 2026 年初，优先选取大版本更新之前的最后稳定版。

---

## 各仓库版本

### mini-swe-agent

| 字段 | 内容 |
|------|------|
| **版本** | `v1.17.5` |
| **日期** | 2026-01-30 |
| **Commit** | `68d367bf` |
| **选择理由** | v1.x 系列最后一个稳定版本。v2.0.0（2026-02-10）为大版本重构，中间有 197 个 commits 的变化，结构改动较大。v1.17.5 在 SWE-Bench 集成和接口方面成熟稳定。 |

```bash
cd task_agent/mini-swe-agent && git checkout v1.17.5
```

---

### OpenHands

| 字段 | 内容 |
|------|------|
| **版本** | `1.1.0` |
| **日期** | 2025-12-30 |
| **Commit** | `9885ddea3` |
| **选择理由** | 2025 年底的稳定正式版本，相较于 1.0.0（2025-12-15）包含约两周的修复，比 1.2.x（2026-01-15）更保守稳定。 |

```bash
cd task_agent/OpenHands && git checkout 1.1.0
```

---

### SWE-Fixer

| 字段 | 内容 |
|------|------|
| **版本** | commit `ab0bffd` |
| **日期** | 2025-01-13 |
| **说明** | 该仓库无任何 tag，维护较为随意（commit 消息多为 "update"）。选取 2025-01-13 的 merge commit 作为基线，为仓库中最后一个有明确意义的提交点。 |

```bash
cd task_agent/SWE-Fixer && git checkout ab0bffd
```

---

## 恢复命令汇总

若需从其他状态恢复到初始基线版本，执行：

```bash
cd /volume/med-train/users/mzchen/lab/AgentSystemEvolve/task_agent
git -C mini-swe-agent checkout v1.17.5
git -C OpenHands checkout 1.1.0
git -C SWE-Fixer checkout ab0bffd
```

## 备注

- 当前各仓库均处于 detached HEAD 状态，如需在其上做修改，应先创建新分支：`git switch -c <branch-name>`
- PLAN.md 规划的首要任务是使用 **mini-swe-agent** 在 SWE-Bench Lite 上跑初始测试并收集 Traces

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

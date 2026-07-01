# Domain-Specific Harness Flaw Patterns

This note summarizes the concrete harness modifications selected by HarnessFix on each benchmark and the recurring ETCLOVG layers involved in each domain.

## Layer-Level Edit Summary

| Harness layer | GAIA | SWE | AppWorld | TB2 | Representative accepted edit |
| --- | --- | --- | --- | --- | --- |
| Execution | partial | primary | none | none | GAIA timeout worker; SWE exec guards |
| Tool Interface | primary | partial | partial | none | GAIA endpoint/files/audio support; AppWorld adapter output bound |
| Context/Memory | partial | primary | primary | primary | SWE submission protocol; AppWorld prompt overlay; TB2 solving protocol |
| Lifecycle | primary | primary | partial | primary | GAIA timeout subprocess; TB2 session-kill sanitizer |
| Observability | primary | partial | none | none | GAIA `unsolved` timeout traces; SWE bounded observations |
| Verification | none | primary | primary | primary | SWE patch validation; AppWorld read-only API check; TB2 `/tests` guidance |
| Governance | none | primary | partial | primary | SWE broad-command rejection; AppWorld app-API constraints; TB2 kill-command block |

Final modified files: GAIA 6, SWE 6, AppWorld 2, TB2 3. Final layers touched: GAIA 5, SWE 7, AppWorld 5, TB2 4.

## Domain Patterns

In open-ended research QA (GAIA), failures often concentrate in Tool Interface and Observability. Typical issues include missing API configuration, unsupported document formats, fragile media conversion, and traces that do not preserve enough evidence for later answer synthesis.

In repository-level software repair (SWE-Bench Verified), many failures involve the interface between tool execution, context construction, and verification evidence. The agent must issue shell commands and file edits, observe the right test output, and submit a patch under benchmark-specific constraints. The selected SWE repairs therefore span all seven ETCLOVG layers.

In stateful application automation (AppWorld), failures often combine Tool Interface, Lifecycle, Observability, and Verification concerns. The harness must validate API arguments, expose API errors, track artifact/state effects, and avoid accepting completion before the required side effect occurs.

Terminal-style command-line tasks (Terminal-Bench 2.0 Verified) place pressure on Execution Environment, Lifecycle, and Verification. The harness must preserve the terminal session, handle infrastructure failures, and prefer benchmark-provided tests over ad hoc checks.

These patterns show that harness repair is not only a prompt-optimization problem. Different benchmarks require different runtime mechanisms, and a change in one layer can affect adjacent layers. This is why HarnessFix uses scoped repair operators and validation-and-regression checks before accepting a candidate harness change.


## Method Edit-Scope Contrast

The score-level comparison in RQ1 leaves a complementary question open: why does HarnessFix improve more than self-evolution and repair baselines? A key difference is the edit surface and the evidence used to choose edits.

GEPA evolves prompt strings through reflective trial-and-error and Pareto-frontier selection. It does not modify tools, scaffolds, adapters, tracing hooks, lifecycle logic, or verification code. SCOPE evolves role-specific guideline memory that is appended to agent prompts; it is also confined to model-facing instruction changes and does not use held-out regression checks. ReCreate widens the surface to reusable scaffolds, procedures, tool scripts, and memory, but it is still less tied to concrete harness implementation anchors and does not enforce flaw-specific regression bounds. Meta-Harness is a stronger code-level harness baseline that searches over harness revisions from prior versions, task scores, and traces, but its optimization is more outcome-driven and less explicitly tied to responsible TraceSteps, harness layers, and implementation anchors.

HarnessFix generates flaw-specific harness patches from scoped repair operators. Candidate patches may touch prompts, tool interfaces, lifecycle hooks, observability instrumentation, verification logic, and governance policies, but they are accepted only when they improve the diagnosed target flaw without exceeding the validation-set regression bound.

These differences explain why prompt-only methods can miss failures caused by runtime mechanisms, and why broad harness search can underperform trace-grounded scoped repair: HarnessFix combines a broader editable surface with step-level attribution, implementation anchors, and validation against the target flaw.

# Invalidate candidate results when forcing source regeneration

## Context

`--force` regenerates candidate code but validation inference/evaluation skip
existing per-version results unconditionally. Train runners can also resume
per-instance results inside an existing output directory. Reusing a version
can therefore evaluate newly generated code using a previous candidate's
predictions, traces, or reports. SWE-Bench additionally caches per-instance
harness reports under its run-specific log directory.

## Decision

At the start of each positive candidate-version iteration, a real forced run
clears that candidate's code, train/validation traces, and evaluation reports.
Reuse the existing retry cleanup helpers for GAIA, AppWorld, and Terminal Bench;
add the corresponding SWE helper including its train/validation harness log
caches. Leave normal resume and dry-run behavior untouched. Do not clear the
current base's results or unrelated versions.

## Alternatives

Passing a force flag only to the validation wrapper would not clear the task
runner's internal resume files or SWE harness report cache. Disabling caching
entirely would remove useful interrupted-run recovery. A new version for every
rerun would be a larger change to the CLI's existing version/force contract.

## Consequences

Forced candidate regeneration cannot reuse candidate-level evidence from its
previous implementation. An interrupted forced run may discard prior candidate
outputs; this is consistent with requesting regeneration. This change is scoped
to candidate invalidation, not a redesign of baseline caching, concurrency, or
content-addressed experiment storage. Normal resume still assumes unchanged code.

## Validation

`python3 -m unittest discover -s tests -p 'test_force_candidate_cache.py'` on
Python 3.10+ exercises each real pipeline main with temporary candidate artifacts,
mocked dotenv and baseline work, and a deliberate stop before inference. All four
force subcases fail on the baseline; all three tests/twelve mode subcases pass
afterward. Tests verify candidate code, traces, reports, and SWE harness logs are
cleared, unrelated baseline output remains, and normal/dry runs preserve caches.
No model APIs, datasets, Docker jobs, or evaluation subprocesses run.

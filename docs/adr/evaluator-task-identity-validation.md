# Validate evaluator task identities before indexing or counting

## Context

Native evaluator loaders do not enforce unique nonempty string task IDs. GAIA
silently overwrites duplicate JSONL rows in its task dictionary, potentially
changing the chosen answer. AppWorld and Terminal Bench instead count the same
trace repeatedly: two rows with one ID report two resolved tasks. Terminal
Bench's rewards dictionary contains only one entry for those two resolutions.
Duplicated successes or failures can therefore change cohort weighting and
produce contradictory identity/count evidence.

## Decision

Use one shared identity validator across the three evaluators. Reject duplicate
IDs and IDs that are not nonempty strings before scoring. GAIA validates rows
before dictionary construction; AppWorld validates loaded rows and direct
`evaluate` calls; Terminal Bench validates its evaluation inputs, including text
ID lists and run-info fallback. Stop silently coercing numeric Terminal Bench
IDs or skipping malformed JSONL IDs. Preserve supported identity field aliases,
blank-line handling, and unique valid input behavior.

## Alternatives

Deduplicating silently would choose an arbitrary GAIA ground truth and conceal
incorrect dataset membership. Counting duplicates as independent trials conflicts
with per-task trace and reward maps, which identify only one result per ID.
Validation only in a downstream gate is too late: invalid reports can already
be consumed by users and other pipeline tools.

## Consequences

Malformed task cohorts exit unsuccessfully before writing a new result report.
The error names duplicate IDs or the nonempty-string requirement. Valid IDs are
not normalized, preserving exact matching to prediction and trace keys. Existing
output files from earlier invocations are not removed by this validation;
pipeline callers must separately enforce fresh report consumption. This change
does not address path confinement or evaluator reward semantics.

## Validation

`python3 -m unittest discover -s tests -p 'test_evaluator_task_ids.py'` exercises
real CLI argument parsing and main functions in separate Python subprocesses
with temporary JSONL, trace, text-ID, and run-info fixtures. Only dotenv and GAIA's
optional scorer import are stubbed; no external services are used. Baseline has
17 failing subcases, including duplicate reports showing resolved=2 for a single
unique ID in AppWorld/Terminal Bench, and GAIA silently reducing two rows to one.
All four tests pass after the change, including valid two-task count controls,
duplicate rejection in all sources, and blank/null/numeric ID rejection.

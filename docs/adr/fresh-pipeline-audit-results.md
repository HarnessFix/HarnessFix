# Require fresh plan audit output for every pipeline attempt

## Context

All four pipeline audit wrappers write to stable version-specific JSON paths.
When a rerun fails before writing its report (for example, an unreadable plan or
an import error), the wrapper reads any existing report. A previous passing
report can therefore approve a different candidate that was never audited.

## Decision

Remove the prior audit output immediately before invoking the audit subprocess
in each pipeline. If the subprocess does not produce a fresh report, retain the
existing fail-closed `audit_output_missing` result. Preserve rejection reports
from normal nonzero audit exits, including their specific scope violations.
Dry runs do not alter evidence or launch a subprocess.

## Alternatives

Checking only the subprocess status would lose useful normal rejection reports:
the audit uses exit code 1 for policy violations. Unique output files could
support concurrent attempts, but would change report naming and downstream
consumers; concurrent runs sharing one version remain unsupported by this fix.

## Consequences

A stale approval can no longer survive a failed audit invocation. The previous
report at that version's output location is replaced on real reruns, consistent
with successful runs already overwriting it. Failures to remove that report
propagate rather than silently trusting it. Report schemas remain unchanged.

## Validation

`python3 -m unittest discover -s tests -p 'test_fresh_pipeline_audit.py'` with
Python 3.10 or later runs four tests across SWE, GAIA, AppWorld, and Terminal
Bench. Baseline fails all four stale-approval subcases; the fix passes all tests.
Tests also verify fresh approval, fresh rejection details, and nonmutating dry
runs. dotenv loading is mocked before import and subprocess execution is mocked;
no credentials, model API calls, or benchmark datasets are involved.

## Gate and comparison coverage

Apply the same fresh-output requirement to subprocess-based train comparisons
and validation gates in GAIA, AppWorld, and Terminal Bench. SWE computes those
comparisons in-process, so it does not consume stale subprocess report files.
Missing train reports retain the existing neutral zero-change fallback; missing
validation reports always reject, including a zero-exit process that wrote no
report. Exit status alone is insufficient promotion evidence.

The expanded suite has eight tests. Before this extension, nine subcases fail
across the three modes: stale train improvements and stale gate approvals for
both zero and nonzero subprocess exits. All eight tests pass after the change,
including fresh positive/negative reports and dry-run report preservation.

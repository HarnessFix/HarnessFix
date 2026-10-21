# Require complete inputs for requested target validation

## Context
The validation CLI initializes target_pass to true and computes targets only when both trace arguments are present. Supplying a target plan without traces, or only one trace arm, silently bypasses target validation while resolved-count improvement can still emit PASS. A missing explicitly supplied plan is likewise silently replaced by an empty plan.

## Decision
Reject partial trace pairs, nonexistent trace directories and missing explicit plan files as input errors. Require a trace pair when the loaded plan requests target metrics. Preserve the existing resolved-count-only mode when neither traces nor explicit targets are supplied.

## Alternatives
Requiring traces on every invocation would break intentional resolved-only use. Silently ignoring partial inputs or plans changes the requested evaluation contract. Fabricating target observations would provide no evidence that the planned improvement occurred.

## Consequences
Misconfigured target validation exits with argparse status 2 before emitting a new gate report. Valid paired-trace behavior is unchanged. Directory existence does not guarantee complete, matching task cohorts; that separate evidence condition is not claimed here.

## Validation
Six stdlib subprocess tests cover explicit targets without traces, each partial pair, missing plan, nonexistent trace directories, retained resolved-only success, and valid paired target evaluation. Baseline has five failing assertions including false PASS output; candidate passes all tests. Synthetic local files only, with no model or benchmark execution.

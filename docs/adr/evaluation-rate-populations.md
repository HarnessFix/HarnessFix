# Separate evaluation and trace populations in outcome rates

## Context
Outcome counts come from evaluation reports, but rates divide them by the number
of available trajectory files. Losing or adding trace files changes reported
success/error rates even though the evaluation results are identical; rates can
exceed one when more tasks resolve than have saved traces.

## Decision
Use the report's declared total for evaluation-derived rates, falling back to
its outcome-ID union for legacy reports and finally trace count when neither is
available. Keep trace-derived diagnostics and average costs on their observed
trace population. Preserve an explicitly supplied GAIA accuracy.

## Alternatives
Capping rates at one conceals the denominator mismatch. Discarding untraced
successful evaluations loses valid evaluator evidence. Requiring complete traces
would be a larger run-admission policy change.

## Consequences
Trace retention no longer changes evaluated outcome rates. The existing
`total_instances` field still reports trace count for compatibility; the metrics
now use the correct population. Report totals are trusted, not authenticated or
fully schema validated. Missing trace diagnostics remain a separate limitation.

## Validation
Two stdlib tests cover all four modes with zero, one and three trace files for
an unchanged two-task report, plus a legacy report using outcome IDs. Thirteen
baseline assertions failed; all pass after the change. Fixtures are synthetic
and no model or benchmark runs are used.

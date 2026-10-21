# Require observed finite target metrics

## Context
Target comparisons substitute zero for missing metrics. Losing a candidate's
error-rate measurement therefore looks like eliminating every error. Nonfinite
values can also yield misleading improvement signals or corrupt JSON output.

## Decision
Require each requested metric on both arms to be a finite numeric observation;
raise a descriptive ValueError otherwise. Reject nonfinite individual and aggregate deltas as well.
Preserve true zero observations, metric direction and default target selection.

## Alternatives
Silently skipping unavailable targets changes the plan after evaluation.
Substituting zero fabricates evidence. Structured unavailable results would be
useful but require a broader report schema change; fail closed for now.

## Consequences
Invalid or incomplete metric inputs stop validation rather than minting an
improvement. Callers must collect the requested measurements or fix their plan.
This intentional compatibility restriction does not establish metric provenance
or validate every collector's source telemetry.

## Validation
Stdlib tests reproduced missing metric acceptance before the fix and now check
missing arms, nonfinite/non-numeric values, genuine zero and default accuracy.
All five tests, including aggregate overflow, pass locally; no benchmark or paid model execution is involved.

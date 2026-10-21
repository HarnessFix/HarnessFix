# ADR: Compare costs using the same metric and preserve measured zero

## Context

Validation reports call `compute_cost_ratio` to compare candidate resource usage
with the baseline. Its truthiness checks discard zero candidate measurements.
A cost decrease from 2 to 0 can therefore be reported as an increase when the
step count rises. Independently choosing step or API-call values for each arm
also divides different units: 10 baseline steps with 2 API calls versus 4
candidate API calls currently produces 0.4 instead of the comparable ratio 2.
The ratio is currently reported, not used to decide the validation gate.

## Decision

Choose the first comparable pair in the existing priority order: instance cost,
steps, then API calls. Both values must be present; the baseline must be positive
and the candidate nonnegative. A zero candidate numerator is valid. If no pair
has a usable denominator, return the existing neutral fallback of 1.0. Never
substitute a different metric or an invented denominator into either arm.

## Alternatives

Only changing the dollar-cost truthiness check leaves zero proxy values and
mixed units broken. Reporting an unavailable value or adding provenance to the
return type would distinguish neutral results from missing measurements, but
requires changing report consumers and is outside this focused correction.

## Consequences

Reports now reflect reductions to zero and use dimensionally comparable values.
Existing positive cost comparisons and step/API-call priority remain intact.
Zero baseline measurements cannot define a finite relative ratio, so another
shared metric is used or the neutral fallback is returned. Collectors that
already coerce missing cost data to zero still lose provenance; this function
cannot recover it and this change does not claim to solve missing telemetry.
Historical paper artifacts and gate behavior are unchanged.

## Validation

Nine standard-library unittest cases cover zero numerators at all three levels,
missing steps on each arm, cost precedence, zero denominators, unavailable
pairs, and None cost fallback. Before the fix the suite reports seven assertion
failures (including subtests); after the fix every case must pass. Run:
`python3 -m unittest discover -s tests -v`. No model calls or training are needed.

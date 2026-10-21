# Treat elapsed time as a minimization metric

## Context
GAIA reports `avg_elapsed_seconds`. When a repair plan targets it, the direction
lookup falls through to higher-is-better. A candidate taking 20 seconds instead
of 10 can therefore satisfy the target gate, while a 5-second candidate fails.

## Decision
Explicitly classify `avg_elapsed_seconds` as lower-is-better, alongside steps and
API calls. Keep the existing strict positive-delta threshold and aggregation.

## Alternatives
Changing the default for unknown metrics would affect unrelated target metrics.
Negating reported durations would make the raw measurement misleading. Neither
is needed for this known metric.

## Consequences
Plans explicitly targeting elapsed time now reward speedups and penalize
slowdowns. Ties remain unimproved. Default accuracy targets and the paper's
initial/final harness snapshots are unchanged. This does not establish model
quality, benchmark performance, or reproducibility of timing measurements.

## Validation
Standard-library unittest coverage checks speedups, slowdowns and ties, plus
subprocess runs of the GAIA validation gate with synthetic local evaluations and
traces. Both gate cases satisfy the resolved-count requirement, isolating the
target-metric decision. The tests reproduced five failing subtests before the
fix and pass afterward. No datasets, credentials or model calls are needed.

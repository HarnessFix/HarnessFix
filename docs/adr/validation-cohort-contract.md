# Require target validation evidence to match the selected cohort

## Context
Resolved-count gating intersects evaluation IDs with --val-ids, but target collectors read every trace and full report. An excluded task improving from 1000 to 0 API calls can mask the selected task regressing from 10 to 20 calls and produce PASS. Missing or duplicate traces also alter metric denominators.

## Decision
Reject empty validation cohorts. When traces are supplied, require both arms to contain exactly one canonical <id>/<id>.traj.json for each selected ID and no other trajectory files. Reject report ID arrays containing IDs outside the cohort and reject a declared total or total_instances that differs from the cohort size. Fields absent from reports remain optional.

## Alternatives
Filtering only traces would leave report-based rates and GAIA accuracy inconsistent. Copying or rewriting evidence risks losing its relationship to original measurements. Extending all collectors to support scoped outcome reconstruction is possible but larger; an explicit input restriction prevents mixed-cohort decisions now.

## Consequences
Callers using target metrics must supply evaluations and traces produced for their exact validation cohort. Broader reports remain supported in resolved-only mode with nonempty selected IDs. Existing matching cohorts work across all four modes. This is an intentional compatibility restriction, not proof that the supplied report's aggregate accuracy or individual measurements are truthful.

## Validation
Six stdlib subprocess tests cover contamination, missing/duplicate/noncanonical traces, out-of-cohort outcome arrays, declared-total mismatch, empty IDs, matching cohorts in all four modes, and broad reports in resolved-only mode. Baseline produced eight failing assertions; a further regression covers SWE total_instances mismatches; the fixed suite passes. Tests use synthetic local files and do not run benchmarks or model APIs.

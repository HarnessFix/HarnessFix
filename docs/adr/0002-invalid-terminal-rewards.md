# Reject invalid Terminal-Bench rewards per task

## Context
The Terminal-Bench evaluator converts rewards without handling invalid values.
A malformed string aborts the whole evaluation, while positive infinity counts
as success and nonfinite values can leak into the JSON report.

## Decision
Validate each selected non-null reward as a finite float. Classify malformed or
nonfinite rewards as task errors, record null for the invalid reward, and continue
with the remaining tasks. Retain the existing success threshold and finite
numeric-string compatibility. Apply this after either trajectory or result lookup.

## Alternatives
Aborting the whole run loses valid task results. Treating malformed rewards as
missing evidence conflates corrupt measurements with absent measurements. Clamping
infinity to a finite value could manufacture a successful result.

## Consequences
Invalid measurements cannot resolve a task or poison report serialization. Valid
finite rewards and missing-reward classification are unchanged. This does not
validate arbitrary malformed report structures or authenticate reward provenance.

## Validation
Standard-library tests exercise six invalid values from each input source, with a
valid task following each invalid task. Before the change they produced six failed
assertions and six exceptions; afterward both tests pass. Additional cases retain
zero/fractional failures, numeric-string success, missing evidence and explicit
error precedence. Strict JSON serialization checks exclude NaN/Infinity output.
No benchmark or model API is used; paper harness snapshots remain unchanged.

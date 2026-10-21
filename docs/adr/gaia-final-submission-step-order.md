# Place GAIA submission after auxiliary model calls

## Context

GAIA HTIR appends auxiliary calls such as answer reformulation after trajectory
steps, with numeric step IDs beyond the trajectory length. The submission is
then assigned trajectory length plus one. For a one-step trace with a second
reformulator call, submission is step 2 while its causal predecessor is step 3.
Sorted AgentTraceStep output presents the answer before the call producing it.

## Decision

Allocate the final submission step after the maximum step ID already emitted.
Associate the terminal iteration-limit exception with that same final step, as
before. Preserve model-call IDs, source references, and all existing edges.

## Alternatives

Renumbering every auxiliary event would affect their existing identities and
would require a broader chronological reconstruction. Leaving numeric order
inconsistent with the explicit final causal edge causes consumers to misorder
terminal evidence. Assigning only the final step after prior events is sufficient
for this defect.

## Consequences

Submissions and terminal iteration-limit evidence remain after auxiliary calls
in the compiled step order. Traces without auxiliary calls retain the existing
next-step numbering. This does not claim that auxiliary calls are fully aligned
with their real execution timestamps; their existing relative ordering remains.

## Validation

`python3 -m unittest discover -s tests -v` compiles a trace with a manager call,
reformulator call, and iteration-limit signal. Baseline puts submission before
reformulation and fails; the corrected result puts it last and keeps the terminal
exception in the same step. A second test preserves numbering without auxiliary
calls. Both tests pass without model calls or historical artifact modifications.

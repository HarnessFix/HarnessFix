# Preserve explicit absence of state changes in external-effect summaries

## Context

AppWorld compilation emits a StateDeltaRecord even for read-only API activity
without a database mutation log. Its structured evidence explicitly records
has_task_state_change=false. The step summary nevertheless treats the mere
presence of any StateDeltaRecord as proof of an external effect and reports true.
This contradicts the observation and can mislead completion-effect diagnosis.

## Decision

Keep all state-node references, but do not use a node explicitly marked as having
no task state change as standalone positive effect evidence. Preserve independent
positive database-change or mutating-call evidence and the existing fallback for
legacy nodes without the explicit negative flag. Avoid the legacy-positive
summary fallback when only explicitly negative state nodes are present.

## Alternatives

Removing negative StateDeltaRecords would discard useful evidence. Treating all
state nodes as unknown would change legacy behavior without a migration plan.
Letting the negative flag override separate positive evidence would hide actual
observations. Excluding only that node's implicit positive signal fixes the
contradiction while preserving the existing evidence combination rules.

## Consequences

Read-only API logs no longer produce a positive external-effect claim merely
because their absence of mutation is represented by a state node. References and
API counts remain available. This is not proof of absence of effects outside the
available logs, and mutating-call evidence retains its existing interpretation.

## Validation

`python3 -m unittest discover -s tests -v` compiles an AppWorld trace with a real
temporary GET-only API log and no database logs. Baseline incorrectly reports an
external effect; the fixed version reports false while retaining the count and
state reference. Three additional cases preserve explicit positive, legacy, and
database-change evidence. All four methods pass; no models or external calls run.

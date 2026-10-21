# Enforce compact printer output limits at small values

## Context

The compact HTIR CLI bounds context with character and step limits. Its character
truncation subtracts 80 without clamping: a limit below 80 produces a negative
slice, retaining almost the entire document. A one-step budget also selects both
a head and a tail step, exceeding the request. Zero or negative step limits are
silently treated as approximately two steps rather than a valid bounded request.

## Decision

Reject nonpositive step limits explicitly, including in the selection helper.
Allocate head steps without forcing a minimum so a one-step limit retains only
the final step. Preserve the omission marker separately from the actual step
count. For positive character limits, reserve the final print newline, clamp the
content prefix budget, and truncate the marker itself if necessary. Character
limit zero retains its existing unlimited behavior.

## Alternatives

Silently clamping invalid step limits to one hides configuration mistakes. Raising
on small positive character limits defeats a valid request for a small preview.
Removing the truncation marker entirely loses useful context when space permits.
Budgeting both content and marker supports every positive character limit.

## Consequences

Actual retained steps never exceed the positive step budget; a separate omission
record still signals skipped evidence. Entire stdout, including its newline,
fits the character budget. A one-character budget emits only a newline. Truncated
output remains a text preview rather than guaranteed valid JSON, matching the
existing CLI behavior. Negative character limits retain their existing unlimited
behavior; this change does not introduce a new policy for them.

## Validation

`python3 -m unittest discover -s tests -v` launches the actual CLI with character
limits 1, 20, 79, 80 and 120, unlimited characters, step limits 1, 2 and 3, and
invalid step limits 0 and -1. The original code fails six subcases; all four test
methods pass after the fix. Fixtures are local synthetic bundles; no models or
network calls are involved.

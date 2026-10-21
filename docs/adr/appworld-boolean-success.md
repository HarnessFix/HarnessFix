# Require a boolean AppWorld success result

## Context
AppWorld result producers persist a boolean success field. The evaluator instead
uses Python truthiness: a corrupted success value of the string "false" counts
as success, as do nonempty lists and mappings. This can inflate measured accuracy.

## Decision
When success is present, require a JSON boolean. Classify invalid types as task
errors and continue other tasks. Only true resolves a task; missing success keeps
its existing missing/unsolved handling.

## Alternatives
Coercing strings silently accepts an undocumented result format and cannot
reliably interpret arbitrary collections. Aborting the full evaluation would
lose other valid task results. Explicit per-task errors preserve the denominator.

## Consequences
Malformed success values cannot manufacture a passing result. Existing native
boolean producers remain compatible. External producers using numeric or string
flags must emit booleans. Other result fields are not fully schema validated.

## Validation
Two stdlib tests cover seven invalid types, a valid task following invalid input,
and genuine false/missing controls. Baseline failed all seven invalid cases; the
fixed tests pass without datasets, models or external services.

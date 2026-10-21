# Compare Terminal-Bench tool operations instead of invocation IDs

## Context
The repeated-command metric serializes entire ATIF tool-call objects. The local
ATIF schema requires a unique tool_call_id, so repeated operations with different
IDs evade loop detection. Global whitespace collapse also conflates distinct
quoted command arguments.

## Decision
Compare canonical JSON of function_name and arguments only, sorting mapping keys.
Ignore invocation IDs and extra bookkeeping. Preserve argument string whitespace
and retain the existing threshold of four consecutive equal operations.

## Alternatives
Removing only IDs still lets metadata differences hide repeated operations.
Comparing command text alone merges different tools and drops non-command
arguments. Normalizing whitespace inside argument strings changes their meaning.

## Consequences
Repeated operations are recognized across ordinary distinct ATIF call IDs, while
different functions and argument values remain distinct. This is still a loop
heuristic, not proof that a repetition is unwanted; legitimate polling can repeat.
The change expects the existing ATIF ToolCall schema, not arbitrary tool formats.

## Validation
Three stdlib tests construct real collector fixtures. Baseline fails unique-ID,
argument-order and quoted-whitespace cases. All tests pass after correction,
including a distinct-tool control. No benchmark, credentials or models are used.

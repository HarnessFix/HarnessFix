# Retain compiled request and response messages in compact HTIR

## Context

The HTIR compiler builds AgentTraceStep objects with request_message and
response_message fields. The compact projection selects request_summary and
response_summary instead. Consequently a freshly generated bundle loses both
message fields when shown through the compact tool, hiding the requested action
and observed response from failure analysis.

## Decision

Include nonempty request_message and response_message in the compact step's
selected fields. Preserve the existing summary fields for older or externally
produced bundles. Keep step selection, clipping, and facet projection unchanged.

## Alternatives

Renaming the compiler fields would change the full HTIR schema and its other
consumers. Falling back into legacy summary names would conceal which schema
was supplied. Preserving both recognized field names is a compatible projection
change with no modification to the generated graph or paper snapshots.

## Consequences

Compact analysis now has the request and response text already bounded by the
compiler. Empty messages remain omitted. Legacy bundles keep their current
representation. A bundle supplying both schemas may include both, but ordinary
compiler output contains only the canonical message fields.

## Validation

`python3 -m unittest discover -s tests -v` passes a real finalized HTIR node
through compact projection and compares both messages with compiler output.
The original projection fails this regression; all three tests pass after the
fix. Additional tests preserve legacy summaries and omit empty canonical fields.
No model calls, external services, or historical artifacts are required.

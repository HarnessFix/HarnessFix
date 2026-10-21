# Reject non-object memory records during loading

## Context

Repair memory is persisted as JSONL objects. The loader already ignores malformed
JSON and blank lines so one damaged row does not prevent using other repairs.
However valid JSON scalars, null, and arrays are accepted into its dictionary
result. Retrieval subsequently calls get on them and crashes before using the
valid neighboring repair records.

## Decision

Accept only decoded dictionary records at the JSONL boundary. Skip other JSON
values consistently with the existing malformed-line policy. Leave valid
records, their order, scoring, and append-only storage unchanged.

## Alternatives

Failing the entire file would discard useful context and contradict the existing
best-effort loader. Catching AttributeError in retrieval would hide unrelated
programming defects and expose other consumers to the wrong shape. Full schema
validation could be useful separately but would require a versioned schema and
compatibility decisions beyond this top-level contract fix.

## Consequences

A syntactically valid but incorrectly shaped line no longer stops memory
retrieval. Input files remain untouched. Dictionary fields are not fully schema
validated by this change, and skipped records receive no new diagnostics, matching
the existing behavior for malformed JSON.

## Validation

`python3 -m unittest discover -s tests -v` covers mixed scalar/null/array and valid
records through actual retrieval, malformed/blank lines, and missing files.
Baseline raises AttributeError in retrieval; all three methods pass after the
fix. Tests use temporary directories and no model calls or production memory.

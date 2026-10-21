# Require explicit selection when evaluation directories are ambiguous

## Context
The shared evaluation loader picks the first glob result from a directory.
Multiple reports can therefore select a stale or unrelated evaluation based on
filesystem iteration order, affecting validation decisions.

## Decision
Continue accepting a directory with one report, ignoring run metadata and
non-file entries. Reject directories containing multiple report files with a
message requesting an explicit file path. Explicit file selection is unchanged.

## Alternatives
Sorting and selecting first or choosing newest would be deterministic but would
still guess which experiment the caller intended. Merging reports would change
experiment identity and could double-count tasks.

## Consequences
Ambiguous callers must supply a report filename. Single-report workflows remain
compatible. No report is silently overwritten or selected by recency. This does
not validate the contents or provenance of the chosen report.

## Validation
Three stdlib tests cover ambiguity, explicit selection, single-report selection
with metadata/directories, and no available report. The ambiguity case failed
before the fix; all pass afterward. No dataset or model calls are required.

# Preserve invalid original trajectories during backfill

## Context
Backfill treats malformed or non-object original JSON as an empty dictionary.
When official logs exist, it overwrites that original evidence with a newly
constructed partial trace, destroying recoverable content and missing fields.

## Decision
Require the original to decode to a JSON object. Skip invalid originals with an
explicit reason and leave their bytes untouched, even under force. Valid object
backfills keep the existing field-preserving behavior.

## Alternatives
Replacing unreadable input silently loses evidence. Aborting the entire batch
would block unrelated valid traces. Recovering malformed JSON requires a separate
explicit repair workflow rather than a log enrichment operation.

## Consequences
Backfill cannot erase a malformed original while claiming successful enrichment.
Skipped files require inspection or repair. This does not make writes atomic or
fully validate dictionary field schemas. Force still replaces model-call fields
on otherwise valid original trajectories.

## Validation
Two stdlib tests cover four malformed/non-object originals and a valid object
with unrelated fields. Baseline overwrote all four invalid cases; after the fix
all cases pass with byte preservation. Official-log readers are mocked, with no
model execution, network calls or user traces.

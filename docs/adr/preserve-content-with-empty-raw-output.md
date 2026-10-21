# Preserve message content when raw output is unavailable

## Context

The prompt sanitizer promises to preserve observations while dropping duplicated
bookkeeping. It removes message content whenever the extra dictionary contains a
raw_output key, even when that value is null or empty. A message containing an
error explanation can consequently lose its only useful diagnostic text.

## Decision

Only use the existing raw-output replacement rule when raw_output is not null,
an empty string, or an empty collection. Preserve content otherwise. Keep the
existing behavior for populated raw output and metadata removal unchanged.

## Alternatives

Removing all deduplication would preserve evidence but increase context size for
every message. Requiring exact equality between raw and rendered output would
also retain wrapper text deliberately removed by the existing design. This fix
addresses only the unambiguous case where replacement content is absent.

## Consequences

Failure-analysis prompts retain error explanations when execution produced no
raw output. Populated outputs still use the existing compact representation.
This is not a general claim that every nonempty raw-output representation is
equivalent to message content; that broader policy remains unchanged.

## Validation

`python3 -m unittest discover -s tests -v` exercises null, empty string, empty
mapping and empty list outputs, populated-output deduplication, and absent
raw_output behavior. The original implementation fails all four empty-output
subcases; the corrected implementation passes all three test methods. No models,
network calls, or historical paper artifacts are involved.

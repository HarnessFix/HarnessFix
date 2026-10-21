# Distinguish task choices from provider completion choices

## Context

The recursive prompt sanitizer treats any dictionary containing choices as a
provider response. It then deletes choices and id. Ordinary multiple-choice
task records therefore lose their possible answers and identifying key before
failure analysis. The sanitizer's stated contract is to preserve task content.

## Decision

Require recognizable indexed completion-choice entries (message, delta, or text)
for choices alone to indicate a provider response. Keep explicit response object,
system fingerprint, and model-plus-usage indicators as before. Plain answer
lists and label/value option dictionaries no longer trigger provider stripping.

## Alternatives

Never recognizing untyped completion choices would retain unnecessary provider
payloads. Matching the key alone is the existing false positive. Requiring a
small structural signature preserves common completion formats without assuming
all choices fields are provider metadata.

## Consequences

Task choices and IDs survive sanitization. Typed responses and indexed chat,
streaming, and text-completion choices still lose bookkeeping. This heuristic
is not a comprehensive provider schema validator or a security redactor; an
untyped provider response lacking indexes may now retain more content.

## Validation

`python3 -m unittest discover -s tests -v` covers scalar task options, structured
label/value options, typed chat responses and untyped indexed message/delta/text
choices. The original implementation fails both task cases. All four methods
pass after the fix. Only synthetic artifacts are used, without model calls.

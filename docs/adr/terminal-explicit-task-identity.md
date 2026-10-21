# Preserve explicit task names containing double underscores

## Context
The output converter strips text after the first double underscore from every task identifier, including explicit task_name values. Two tasks named group__first and group__second therefore overwrite the same trace directory and prediction entry.

## Decision
Preserve the basename of explicit task strings or path/name/ref records. Apply trial suffix stripping only when inferring identity from a trial name because no explicit task identity exists.

## Alternatives
Rejecting all double underscores would exclude valid directory names. Retaining suffixes on every fallback trial would change established behavior for records that lack task identity. Explicit identity is the best available evidence.

## Consequences
Distinct explicit tasks remain distinct in prediction and trace artifacts. Existing simple trial fallback behavior remains compatible. This does not solve repeated trials of the same explicit task or redefine trial aggregation.

## Validation
Two stdlib tests fail on baseline and pass with this change. The integration test converts two explicit names sharing a prefix and checks both predictions and both trace paths. A unit case covers explicit path records and preserves fallback suffix removal. No benchmark or model calls are made.

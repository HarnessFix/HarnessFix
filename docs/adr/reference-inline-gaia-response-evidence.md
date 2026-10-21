# Reference inline GAIA response evidence at its actual location

## Context

The GAIA compiler supports trajectory steps that store model input and output
inline when no manager model-call record exists. It synthesizes a temporary call
for normalization, then emits a source reference to traj.model_calls anyway.
That array may be empty, so attribution consumers receive a nonexistent evidence
location despite the response being present in the trajectory step.

## Decision

Track the inline response location when constructing the temporary call. Emit
a reference to the populated model_output_message, the model_output fallback,
or the step itself if no output field exists. Keep normal recorded-call source
references and normalized message attributes unchanged.

## Alternatives

Persisting the synthetic call into the input would change original evidence.
Always pointing to the whole step is valid but less precise when the output
field is known. Removing the fallback would discard supported inline traces.
Tracking its actual source locally avoids all three problems.

## Consequences

Inline model invocation nodes point to real trajectory evidence. This does not
change source-reference policy for separately recorded model calls or reconstruct
missing outputs. Input files and historical artifacts remain untouched.

## Validation

`python3 -m unittest discover -s tests -v` compiles inline message and inline text
responses and a separately recorded manager call. Both inline cases fail on the
original source reference; all three pass after the fix. No model invocation or
external service is required.

# Preserve conversion when a Harbor trial has only configuration

## Context
Interrupted Harbor trials can contain config.json without result.json. The adapter has an explicit fallback for these trials, but passes two arguments to list.append. One incomplete trial therefore prevents conversion of every trial in the job.

## Decision
Append the fallback record as the same (trial, directory) tuple used by completed trials. Preserve the existing no-reward representation for incomplete trials.

## Alternatives
Skipping incomplete trials would hide attempted tasks. Aborting the job discards useful completed results. Neither is needed to correct the tuple construction.

## Consequences
Mixed completed and config-only trials produce predictions and trace metadata. This change does not invent a reward or claim incomplete tasks succeeded. Existing trial ordering and naming remain unchanged.

## Validation
A standard-library integration test constructs a completed trial and a config-only trial, then calls output conversion. Baseline raises TypeError at list.append; the candidate produces both traces and predictions with an empty reward for the incomplete trial. No Harbor execution or model calls are involved.

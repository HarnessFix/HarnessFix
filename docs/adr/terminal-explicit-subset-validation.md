# Reject missing explicitly selected Terminal-Bench subsets

## Context
The adapter interprets every nonexistent subset path as a request for the registered terminal-bench@2.0 dataset. A typo in an explicit --subset can therefore launch the full benchmark instead of the intended local subset, changing scope and potential execution cost.

## Decision
Distinguish omitted --subset from explicit selection. Preserve the default local-or-registered dataset fallback when omitted, but reject explicit paths that are not directories before launching Harbor.

## Alternatives
Rejecting a missing default would remove supported registry execution. Silently falling back for explicit input changes the user's requested experiment. Automatically creating the missing directory would conceal the typo rather than provide tasks.

## Consequences
Explicit invalid inputs exit with argparse status 2 before any Harbor launch. Users wanting the registered default can omit --subset. Existing valid local subset directories remain supported. This does not check dataset contents or validate task manifests.

## Validation
Two stdlib CLI tests intercept the runner boundary. The explicit-missing case fails on baseline because execution continues and passes after validation; the omitted-default case confirms registered fallback remains available. No subprocess benchmark or model calls are executed.

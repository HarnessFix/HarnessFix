# Preserve caller-relative Terminal-Bench paths

## Context
The adapter creates its output directory and checks the subset in the caller's current directory. Harbor subsequently runs with cwd set to the repository root, receiving the same relative strings. Runs started outside the repository therefore read a different subset, write elsewhere, or fail to locate output during conversion. A relative mini-swe-agent source path has the same mismatch.

## Decision
Resolve subset, output and optional agent source paths against the caller's working directory at the CLI boundary before creating output or launching Harbor.

## Alternatives
Changing Harbor's working directory could disrupt its repository-relative integrations. Resolving only output would leave the selected dataset and custom agent inconsistent. Requiring absolute user input would unnecessarily remove relative-path support.

## Consequences
Parent conversion and child Harbor use the same paths independent of launch directory. Absolute paths and the optional absent agent source retain their meaning. This does not validate datasets or execute benchmark tasks.

## Validation
A stdlib test invokes the CLI from a temporary external directory with relative subset, output and agent paths and intercepts the Harbor boundary. It fails before the fix and passes afterward, asserting that conversion receives the same absolute output path. No subprocess benchmark, model call or credentials are used.

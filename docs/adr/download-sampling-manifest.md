# Use the selected Terminal-Bench manifest throughout download and sampling

## Context

The downloader accepts `--manifest` and uses it to select the source revision.
It then invokes the sampler without that option, causing the sampler to read
its default manifest instead. A custom manifest can therefore select one source
revision while creating splits from another experiment's membership, or fail
because the default manifest is unavailable.

## Decision

Resolve the selected manifest relative to the invoking process's working
 directory once and pass that same absolute path both to revision lookup and
 to the sampler command. This preserves its meaning when the child command
 runs with the repository root as its working directory.

## Alternatives

Changing the sampler's global default would not respect per-invocation choices.
Passing a relative path unchanged would still select a different file when the
caller runs outside the repository. Duplicating the manifest into the repository
would add unnecessary state and stale-copy risk.

## Consequences

Custom split membership and pinned revisions now come from the same file.
Default manifest selection, explicit revisions, copying, and skip-sampling
behavior remain intact. This does not change dataset contents or fetch policy.

## Validation

`python3 -m unittest discover -s tests -p 'test_download_manifest.py'` runs three
standard-library tests with all commands mocked. Baseline fails both custom
manifest cases; candidate passes all three. Tests verify pinned revision and
sampling share the custom manifest, relative paths become absolute before the
child changes working directory, and skip-sampling invokes only cloning. No
network requests or model calls occur.

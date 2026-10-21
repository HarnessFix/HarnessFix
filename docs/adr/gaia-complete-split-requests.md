# Honor and validate all GAIA split level requests

## Context

The GAIA stratified sampler accepts independent train, validation, and test
count dictionaries. It iterates only training dictionary keys, so a level
requested only for validation or test silently contributes no tasks. It also
accepts negative counts, letting Python's negative slicing generate unexpected
counts and overlapping split members instead of rejecting the configuration.

## Decision

Iterate the sorted union of levels requested by all three split dictionaries.
A missing count means zero for that split. Reject any negative per-level count
before constructing slices. Continue rejecting requests larger than the level's
available pool. Empty requests and zero counts remain supported.

## Alternatives

Requiring all dictionaries to repeat every level would unnecessarily restrict
the existing independent-count API. Silently dropping levels or clamping
negative values hides a malformed experiment configuration. Sampling each split
independently would sacrifice the shared permutation's disjointness guarantee.

## Consequences

Validation-only and test-only requests now produce their requested tasks or
raise the existing insufficient-data error. Valid configurations with the same
levels retain their sorted traversal and random sequence. No output directories
are written until the sampler returns to main, so invalid requests fail before
split artifacts are created. This does not validate duplicate task identities
inside caller-provided source pools.

## Validation

`python3 -m unittest discover -s tests -p 'test_gaia_split_requests.py'` on Python
3.10+ runs six standard-library tests. Baseline has six assertion failures:
missing validation/test-only levels, missing pool detection for a validation-only
level, and negative counts across all split positions including test-only levels.
Candidate passes all six tests. Controls verify repeatable disjoint samples,
unchanged input lists, and empty/zero requests. dotenv is mocked before import;
no credentials, network datasets, or model calls are used.

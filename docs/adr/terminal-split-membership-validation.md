# Validate Terminal-Bench split membership before writing

## Context

The Terminal-Bench sampler validates overlap only against earlier splits, so a
manifest repeating a task within one split is accepted. Writing that split then
fails at its second copy or symlink, after deleting the previous output directory.
Random splitting also accepts negative counts from the CLI; Python's negative
slice indexes can produce overlapping train/validation tasks instead of the
requested disjoint subsets.

## Decision

Track every manifest task as it is encountered, rejecting duplicates both within
and across splits. Reject negative random-split counts before shuffling or
slicing. Both validations execute before main begins writing any split folders.
Keep zero counts valid, preserve valid manifest ordering, and retain seeded
sampling semantics for valid requests.

## Alternatives

Silently deduplicating a manifest would conceal an incorrect requested count and
alter the experiment. Clamping negative counts to zero would similarly hide
invalid configuration. Detecting duplicates in the writer would be too late to
preserve existing split outputs; membership validation belongs before writing.

## Consequences

Malformed split requests fail with explicit messages before destructive output
replacement. Valid datasets and seeds retain their previous output. This does
not make split writes transactional against unrelated disk or permission errors.

## Validation

`python3 -m unittest discover -s tests -p 'test_terminal_split_integrity.py'`
uses six standard-library tests. Baseline has six failing subcases: duplicate
membership in each of three manifest splits and a negative count in each of
three random splits. Candidate passes all tests. Additional cases verify valid
manifest order, existing cross-split overlap rejection, oversized requests,
zero counts, seed reproducibility, disjointness, and unchanged input lists.

## Source task identities

Task directory basenames are the manifest IDs and split destination names. Reject
repeated basenames during source discovery, before either manifest lookup or
random splitting: otherwise the manifest's dictionary silently picks one source
while random output can fail after deleting prior splits. Two additional tests
verify ambiguity is rejected and unique nested tasks remain sorted with hidden
tasks excluded. The duplicate-source test fails before this extension; all eight
tests pass afterward.

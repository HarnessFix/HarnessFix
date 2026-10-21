# Audit deleted Python files without compiling absent source

## Context

The plan-to-diff audit is called by the SWE, GAIA, AppWorld, and Terminal Bench
pipelines before evaluating a candidate. Its change inventory deliberately
includes additions, modifications, and deletions. The syntax stage attempts to
read every changed Python path from the candidate, so deleting an obsolete
Python module fails with FileNotFoundError even when the plan permits and targets
that deletion. Non-Python deletions do not have this accidental restriction.

## Decision

Skip absent candidate paths in the Python syntax stage. Keep deleted paths in
the complete change inventory, edit count, allowed/forbidden path checks, and
fix-target coverage. Existing candidate paths still undergo compilation; a
directory or dangling symlink replacing a Python file remains an error rather
than being accepted as a deletion.

## Alternatives

- Exclude deletions from the change inventory: rejected because it would bypass
  scope restrictions and edit budgets.
- Compile the original deleted source: rejected because it validates code that
  will not execute in the candidate and can reject removal of invalid code.
- Introduce a new operation-specific plan schema: unnecessary for this fix;
  existing path and count policies already apply to deletions.

## Consequences

Authorized module removal can pass the structural audit. That does not establish
runtime correctness: downstream evaluation must still detect broken imports or
behavior. Added and modified Python syntax errors continue to reject candidates.
The existing report schema and CLI exit semantics remain unchanged.

## Validation

Run `python3 -m unittest discover -s tests -p 'test_plan_diff_audit.py' -v`.
On base 9167a0b9a58748c73b56c3ee04fdc3437ba0c56e, six of seven tests fail because
deleted files produce spurious syntax failures. With the fix, all seven pass.
Cases cover permitted deletion and fix coverage, zero-file budget, forbidden and
out-of-scope deletions, invalid added/modified source alongside deletion, and a
directory replacing a Python file. Tests use temporary directories and the
standard library only, without model calls or benchmark snapshot changes.

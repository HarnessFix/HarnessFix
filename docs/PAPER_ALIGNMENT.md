# Paper Alignment

This copy of the project upgrades the original prototype toward the design in
`EvoHarnessPaper`.

Implemented alignment points:

- `failure_analysis/htir.py`
  - Compiles SWE and GAIA traces into a harness-aware trace bundle with
    overview, failure, artifact, causal, and harness views.
- `failure_analysis/operator_registry.py`
  - Defines typed harness repair operators, target metrics, static checks, and
    rollback conditions.
- `failure_analysis/consolidation.py`
  - Consolidates per-instance diagnoses into cluster-level defect records and
    enriches repair specs with typed operator metadata.
- `failure_analysis/plan_diff_audit.py`
  - Audits plan-to-diff consistency before validation.
- `failure_analysis/validation_metrics.py`
  - Computes process and target metrics for the regression-aware validation
    gate.
- `failure_analysis/harness_memory.py`
  - Stores accepted/rejected repair memories and retrieves relevant ones during
    planning.

Pipeline integration:

- `run_analysis.py`
  - Generates HTIR bundles before diagnosis and enriches diagnosis outputs with
    defect class, operator family, evidence spans, severity, and confidence.
- `aggregate_results.py`
  - Uses cluster records, typed operator registry, and repair memory as prompt
    context; writes a cluster JSON artifact and enriches plan specs.
- `run_pipeline_swe.py` / `run_pipeline_gaia.py`
  - Insert plan-to-diff audit before validation and record accepted/rejected
    repair memories after audit/gate.
- `run_pipeline_appworld.py`
  - Adds the third benchmark branch for AppWorld using the same
    analysis/aggregate/modify/audit/gate/memory structure.
- `check_val_gate.py`
  - Replaces resolved-count-only acceptance with target metric, regression, and
    cost checks.

Additional cleanup:

- Removed historical traces, eval outputs, logs, cached results, generated
  enhanced agent copies, and download leftovers from this repo copy.
- Added `.gitignore` rules so regenerated outputs stay out of version control.

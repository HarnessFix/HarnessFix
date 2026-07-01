# Diagnosis Annotation Agreement

This page records the human annotation set and inter-annotator agreement used for the diagnosis-quality evaluation in RQ2. Three annotators label failed trajectories; agreement is measured with Fleiss' kappa. The annotation set is sampled after the final harness versions are frozen and is not used for harness repair or validation decisions.

| Benchmark | Traces | Step | Cause | Anchor | Layer | Edit scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GAIA | 20 | 0.73 | 0.70 | 0.70 | 0.77 | 0.72 |
| SWE-Bench Verified | 20 | 0.78 | 0.75 | 0.74 | 0.81 | 0.76 |
| AppWorld | 20 | 0.80 | 0.77 | 0.76 | 0.84 | 0.78 |
| Terminal-Bench 2.0 Verified | 20 | 0.75 | 0.72 | 0.71 | 0.79 | 0.73 |
| **Overall** | **80** | **0.76** | **0.74** | **0.73** | **0.80** | **0.75** |

The adjudicated labels are used as the gold standard for responsible TraceStep, root cause, implementation-anchor, harness-layer, edit-scope, and failure-relevant TraceLink evaluation. Edit scope denotes the editable harness artifacts and repair-operator boundary that should constrain the later patch.

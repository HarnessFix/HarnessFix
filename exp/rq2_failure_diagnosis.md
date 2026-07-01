# RQ2: Failure Diagnosis

RQ2 asks whether HarnessFix produces effective failure diagnoses. We audit responsible TraceSteps, natural-language root causes, implementation anchors, ETCLOVG harness layers, and edit scopes. Here, edit scope means the editable harness artifacts and repair-operator boundary that should constrain the later patch.

## Human Annotation Protocol

After the final harness versions are frozen, the diagnosis-audit set samples 80 failed trajectories: 20 initial-harness failed test trajectories per benchmark. This set is not used for harness repair or validation decisions. Three annotators inspect the raw trace, harness code or prompts, task metadata, and evaluator feedback without seeing HarnessFix's predicted diagnosis records. They label responsible TraceSteps, root causes, relevant implementation anchors, implicated harness layers, edit scopes, and failure-relevant data-flow/control-flow links. Disagreements are adjudicated to produce the gold labels.


The adjudicated labels serve as the gold standard. The annotators reach substantial agreement overall: Fleiss' kappa is 0.76 for responsible TraceStep labels, 0.74 for root-cause labels, 0.80 for harness-layer labels, 0.73 for implementation anchors, and 0.75 for repair scope.

Detailed inter-annotator agreement is preserved in `docs/diagnosis_annotation.md`.

## Diagnosis Metrics


All values are percentages. All input variants use the same diagnosis model; only the representation provided to the model changes.

| Input representation | Step | Cause | Anchor | Layer | Operator |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw trace | 55.0 | 53.8 | 50.0 | 58.4 | 51.3 |
| Raw + data-flow | 70.0 | 68.8 | 65.0 | 73.2 | 66.3 |
| Raw + data/control | 77.5 | 75.0 | 72.5 | 79.4 | 73.8 |
| Full HTIR | 85.0 | 83.8 | 81.3 | 86.2 | 82.5 |

### Auxiliary link and consolidation metrics

The paper table reports the core diagnosis metrics. During the audit, we also tracked failure-relevant TraceLink precision and flaw-clustering quality; these auxiliary metrics were removed from the paper table for space.

| Input representation | Link precision | Cluster F1 |
| --- | ---: | ---: |
| Raw trace | 51.3 | 55.4 |
| Raw + temporal order | 56.9 | 60.8 |
| Raw + data-flow | 71.2 | 70.1 |
| Raw + data/control | 78.4 | 76.5 |
| Full HTIR | 84.7 | 82.4 |

The temporal-order-only variant was useful as an internal sanity check but was omitted from the paper because most of the diagnosis gain comes from explicit data-flow and control-flow structure rather than chronology alone.

## Interpretation

Full HTIR aligns well with human-adjudicated labels, reaching 85.0% responsible-step accuracy, 83.8% root-cause accuracy, 81.3% anchor accuracy, 86.2% harness-layer macro-F1, and 82.5% repair-operator accuracy. For fine-grained step-level localization, this is substantially above existing techniques that typically achieve around 45-52% step accuracy.

Adding data-flow and control-flow structure consistently improves the diagnosis metrics over raw traces. Data-flow links help track reused, transformed, or omitted information across TraceSteps. Control-flow links make controller behavior visible, such as completion guards, retry policies, validation gates, and termination decisions. Full HTIR adds TraceStep annotations and implementation anchors, yielding the best score on every reported metric.

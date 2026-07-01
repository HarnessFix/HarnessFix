# Motivational Study Details

This page preserves the collection, classification, and validation procedure for the motivational study summarized in the paper.

## Repository Collection

We collect open-source LLM-based agents according to four task categories: open-ended research, repository-level software engineering, application automation, and terminal-based workflows. For each category, repositories are ranked by GitHub stars. We select the top repositories that provide executable agent implementations and public development histories, then remove duplicates across categories and repositories without usable version records.

The resulting corpus contains 30 agent systems and approximately 57,780 development records, including issues, pull requests, commits, and release notes.

## Harness-Related Record Identification

Two authors first inspect 100 randomly sampled development records and summarize harness-related terms and decision criteria using the ETCLOVG taxonomy. The classification guide covers whether a record modifies or discusses harness artifacts such as tool specifications, prompt templates, orchestration code, configuration files, adapters, logging hooks, verification scripts, and governance or permission logic.

An LLM-based analysis agent then classifies each development record as harness-related or unrelated and provides supporting evidence. The authors inspect sampled predictions, revise the instructions and examples, and repeat this process until the classification behavior stabilizes. Applying the resulting classifier to the full dataset yields 26,174 harness-related records, accounting for approximately 45.3% of all collected development records.

As a final audit, two authors independently inspect a random sample of 300 classified records, and a third author resolves disagreements. The classifier achieves 97% precision and 94% recall on this sample.

## Layer Classification

For harness-related records, the analysis follows the ETCLOVG responsibility layers: Execution, Tool Interface, Context/Memory, Lifecycle, Observability, Verification, and Governance. A record can involve multiple layers when the described flaw or fix concerns several harness responsibilities.

Two authors apply open coding to a sample of harness-related records, compare the identified flaw patterns, and refine the definitions and examples for the seven layers. The LLM-based analysis agent assigns the remaining records using the resulting coding guide. Authors inspect random samples of the classifications, and a third author adjudicates ambiguous cases. The layer classification achieves 96% accuracy on the inspected sample.

The resulting layer distribution is: Lifecycle (25.4%), Tool Interface (21.5%), Observability (18.1%), Context/Memory (13.2%), Execution (9.3%), Verification (7.0%), and Governance (5.5%).

## Repair-Strategy Extraction

To identify repair strategies, the study selects 13,238 harness-related issues linked to concrete fixing commits or pull requests. Two authors inspect issue descriptions and associated code changes, then use card sorting to group modifications with similar targets and behavioral effects. Semantically equivalent cards are merged through open coding, disputed cases are adjudicated by a third author, and the resulting repair strategies are summarized as repair operators in the paper.

The operators cover recurring strategies such as sandbox-boundary tightening, tool-schema refinement, failure-relevant context preservation, lifecycle and finalization guards, trace instrumentation, intermediate and final validation, and governance or approval-policy enforcement. These strategies provide the empirical repair space used by HarnessFix's scoped repair operators.

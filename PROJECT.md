# Project Plan — Scaling Specialization in Dense LMs

## Goals

* Show that specialization/modularity rise with model size in dense LMs.
* Convert structure into compute savings (dynamic-k) at fixed quality.

## Phases

* **A — Measurement:** AS, SI, Q across a clean scaling suite.
* **B — Scaling Law:** Fit trends vs parameters; analyze variance across layers.
* **C — Efficiency:** Dynamic-k execution; causal ablations; throughput/quality trade-off.

## Deliverables

* Figures: AS/SI/Q vs size; monosemanticity histograms; community maps; throughput vs perplexity.
* Repro configs + scripts; concise write-up.

## Timeline (6 weeks)

W1: activation capture + AS; W2–3: SAEs + SI/Q; W4: dynamic-k; W5: ablations; W6: figures + paper draft.

## Risks & Mitigations

* Compute limits → sample activations, fewer layers, chunking.
* Instability in SAE training → smaller codebook, L1 sweep, early stopping.

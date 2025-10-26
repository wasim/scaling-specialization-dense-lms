# Scaling Specialization in Dense LMs — Project Overview

This repo explores whether dense transformers develop sparse, modular structure as they scale, and how to turn that structure into efficiency gains. Future Codex runs should keep these pillars in mind:

## Objectives

1. **Measure**: capture feed-forward activations across model sizes and tasks, logging activation fraction + participation ratio per layer.
2. **Explain**: train sparse autoencoders (SAEs) on cached activations to expose monosemantic features and co-activation communities.
3. **Exploit**: prototype dynamic-k execution (training-free + learned gates) and report throughput vs perplexity trade-offs.

### Guiding Hypothesis

Dense transformers, even without routers, acquire sparse, modular internal structure that becomes more specialized as model size grows. The project tests whether that specialization scales, whether SAEs can expose it cleanly, and whether selective execution can cash it out for real inference savings.

## Current State

- `sdlms.cli.sparsity` emits CSV/metadata under `artifacts/sparsity/`.
- Placeholder CLIs exist for SI/modularity, SAE training, and dynamic-k; they should load artifacts from phase 1.
- `tests/test_sparsity_cli.py` is the regression guard; add similar tests when implementing the remaining CLIs.

## Phase Plan (compact)

- **Phase A – Measure**: activation sparsity, specialization index, and modularity metrics across the Pythia suite (70M → 2.8B+, same data order). Use consistent probe tasks (IOI, toy arithmetic, POS/NER slice, induction sequences).
- **Phase B – Explain**: train SAEs per layer to extract monosemantic features, compute SI/Q scaling trends, and visualize co-activation communities.
- **Phase C – Exploit**: implement dynamic-k feed-forward execution—static top-k masks plus learned predictors—and evaluate throughput vs perplexity alongside causal ablations of specialized modules.

## Workflow Conventions

- Use `uv sync --all-groups` to create environments (Python 3.13 pinned via `.python-version`).
- Artifacts live under `artifacts/YYYYMMDD-*/` and notebooks read from there—avoid recomputing inside notebooks.
- Follow the principles in `AGENTS.md`: small surfaces, real loops, measured claims, fast deletions.

## Reading List

- Activation sparsity & TEAL masking (ICML/ICLR 2025).
- Emergent modularity in dense transformers (Findings-ACL 2023).
- Sparse autoencoders & Scaling Monosemanticity (Anthropic).
- Contextual/dynamic sparsity systems (DejaVu, ShadowLLM, DiffSkip).
- Foundational pieces: Toy Models of Superposition, FFNs as Key-Value Memories.

## Next Milestones

1. Implement `sdlms.cli.sae_train` using SAELens.
2. Define task probes (`data/probe_tasks.jsonl`) and update capture scripts.
3. Wire `sdlms.cli.si_modularity` + `dynamic_k` for scaling experiments.

Document progress here whenever major workflows land so future Codex sessions stay aligned.

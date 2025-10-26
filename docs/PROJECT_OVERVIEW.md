# Scaling Specialization in Dense LMs — Project Overview

This repo explores whether dense transformers develop sparse, modular structure as they scale, and how to turn that structure into efficiency gains. Future Codex runs should keep these pillars in mind:

## Objectives

1. **Measure**: capture feed-forward activations across model sizes and tasks, logging activation fraction + participation ratio per layer.
2. **Explain**: train sparse autoencoders (SAEs) on cached activations to expose monosemantic features and co-activation communities.
3. **Exploit**: prototype dynamic-k execution (training-free + learned gates) and report throughput vs perplexity trade-offs.

## Current State

- `sdlms.cli.sparsity` emits CSV/metadata under `artifacts/sparsity/`.
- Placeholder CLIs exist for SI/modularity, SAE training, and dynamic-k; they should load artifacts from phase 1.
- `tests/test_sparsity_cli.py` is the regression guard; add similar tests when implementing the remaining CLIs.

## Workflow Conventions

- Use `uv sync --all-groups` to create environments (Python 3.13 pinned via `.python-version`).
- Artifacts live under `artifacts/YYYYMMDD-*/` and notebooks read from there—avoid recomputing inside notebooks.
- Follow the principles in `AGENTS.md`: small surfaces, real loops, measured claims, fast deletions.

## Next Milestones

1. Implement `sdlms.cli.sae_train` using SAELens.
2. Define task probes (`data/probe_tasks.jsonl`) and update capture scripts.
3. Wire `sdlms.cli.si_modularity` + `dynamic_k` for scaling experiments.

Document progress here whenever major workflows land so future Codex sessions stay aligned.

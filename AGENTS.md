# Agents.md — Working Principles

This project leans on the spirit of [agents.md](https://agents.md/): build small, sharp systems that solve concrete problems, with a bias toward explicit reasoning over magical thinking. Our interpretation for `sdlms`:

1. **Ship real loops.** Every script should run end-to-end, touch real models, and leave reproducible artifacts in `artifacts/YYYYMMDD/…`.
2. **Keep the surface area tight.** Fewer abstractions, more clarity. Prefer plain `argparse`, typed helpers, and explicit tensors over framework-heavy wrappers.
3. **Measure before you claim.** Scripts default to saving CSV/JSON summaries; notebooks must load those artifacts instead of recomputing.
4. **Determinism first.** Seed PyTorch/NumPy, log device + dtype, and write smoke tests for the CLI entry points.
5. **Compost fast.** If scaffolds stop pulling their weight, delete them; entropy is the enemy.

Future work inherits these rules—if code reads like “AI slop”, it doesn’t merge.

## Solution architecture

- Core CLIs (sparsity, sae_train, si_modularity, dynamic_k) feed artifacts into notebooks and reports; keep each script a self-contained loop.
- Artifacts follow `artifacts/YYYYMMDD-<desc>/` convention.
- Future modules (e.g., gating predictors) should slot into `src/sdlms/cli/`.

## Next actions

- Implement `sdlms.cli.sae_train` with SAELens.
- Define probe task manifest (`data/probe_tasks.jsonl`) and update capture pipeline.
- Flesh out `sdlms.cli.si_modularity` to consume SAE outputs and produce SI/Q CSVs.
- Prototype dynamic-k evaluation and logging in `sdlms.cli.dynamic_k`.

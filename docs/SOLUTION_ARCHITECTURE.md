# Solution Architecture (Living Document)

## High-Level Flow

```
Raw text/tasks  ──>  sdlms.cli.sparsity  ┐
                        │                │
Activations (NPY/HDF5) ─┤──> SAE Training │──> SI & Modularity CLI ──> Reports/Plots
                        │                │
Dynamic-k CLI <─────────┘                └──> notebooks/*.ipynb
```

## Components

- **Capture & Metrics**
  - `sdlms.cli.sparsity`: collects FFN activations, logs sparsity/participation ratio.
  - Future: `run_capture.py` integration with probe tasks (IOI, arithmetic, POS, induction).

- **Representation Learning**
  - `sdlms.cli.sae_train`: trains SAEs per layer; outputs weights + diagnostics.
  - Artifact format: `artifacts/<stamp>/sae/layer=<name>/`.

- **Analysis & Reporting**
  - `sdlms.cli.si_modularity`: loads task-conditioned activations + SAE features, computes SI/Q metrics.
  - `notebooks/`: read-only notebooks that visualize artifacts (Plotly/Altair).

- **Efficiency Experiments**
  - `sdlms.cli.dynamic_k`: static and learned gating; logs throughput vs perplexity curves.
  - `src/sdlms/dynamick.py`: shared gating primitives.

- **Testing & CI**
  - `tests/`: CLI smoke tests (tiny models); expand as new scripts land.
  - `.github/workflows/ci.yml`: lint + import smoke.

## Data & Artifacts

- `data/probe_tasks.jsonl`: manifest for probe tasks (pending).
- `artifacts/`: timestamped runs; keep metadata JSONL + CSV outputs.
- Focus on reproducibility: notebooks never recompute heavy steps.

## Open Questions

1. Best format for activations? (NPY vs HDF5 vs HuggingFace dataset).
2. How to manage SAE hyperparameter sweeps reproducibly? (configs vs CLI args).
3. Dynamic-k benchmarking harness—local vs Slurm vs Kaggle.

Update this file as architecture evolves (new components, data flow changes, deployment targets).

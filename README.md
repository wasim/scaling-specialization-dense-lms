# Scaling Specialization in Dense LMs

[![CI](https://github.com/wasim/scaling-specialization-dense-lms/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Do **dense** transformers, without routers, develop **sparse, modular structure** that becomes more **specialized** as model size grows? We:

1. **Measure** activation sparsity (AS), feature **Specialization Index (SI)**, and graph **modularity (Q)** across a consistent scaling suite.
2. **Explain** features via **Sparse Autoencoders (SAEs)** to reveal monosemantic circuits.
3. **Exploit** the structure using **dynamic-k MLP execution** for real **FLOPs savings** at fixed quality.

> TL;DR — Specialization scales with size; you can cash it out for speed.

![High-level SLMS flow](docs/images/Generated%20Image%20slms%20highlevel.png)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .

# list MLP-ish layers in a checkpoint
python - <<'PY'
from sdlms.activations import list_layers
print(list_layers("EleutherAI/pythia-410m-deduped"))
PY
```

## Minimal workflow

```bash
# 1) Capture activations on small probe tasks
python scripts/run_capture.py --model EleutherAI/pythia-410m-deduped --task-id ioi_minimal --layers model.layers.10.mlp

# 2) Train SAEs (separate tool) and export features

# 3) Compute metrics (AS, SI, Q)
python scripts/run_metrics.py

# 4) Dynamic-k eval (throughput vs perplexity)
python scripts/run_dynamick_eval.py --k 0.35
```

### CLI quickstart

```bash
# install dependencies through uv (preferred)
uv sync --all-groups

# measure activation sparsity with a prompt (writes CSV to artifacts/sparsity)
uv run sparsity --model EleutherAI/pythia-70m-deduped --probe-manifest data/probe_tasks.jsonl --task-id toy_arithmetic

# launch a notebook to inspect results
uvx jupyter lab
```

## Reproducibility

* Deterministic seeds where possible
* Configs + exact prompts for probe tasks
* All figures generated from `notebooks/`

## License

MIT — see [LICENSE](LICENSE).

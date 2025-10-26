# Codex Morning Handoff

Welcome! Previous session (Oct 26, 2025) wrapped Step 1 of the plan: probe manifest + capture integration is live. Here’s how to get productive quickly.

## Repository State
- Manifest-driven capture in place: `data/probe_tasks.jsonl`, `scripts/run_capture.py`, `sdlms.cli.sparsity` w/ `--probe-manifest`.
- `src/sdlms/probe_tasks.py` handles prompt/dataset iteration.
- Visualization helper: `scripts/plot_sparsity.py` (bar chart CSV → PNG).
- Activations utilities now detect device (CUDA → MPS → CPU) and flatten activations.

## Immediate To-Do (Plan Step 2 primer)
1. Run multi-layer captures for each task to populate artifacts.
   - Example: `uv run python scripts/run_capture.py --model EleutherAI/pythia-70m-deduped --task-id ioi_minimal --layers transformer.h.0.mlp transformer.h.6.mlp transformer.h.12.mlp`
2. Expand plotting (per-layer comparisons, multi-task overlays).
3. Start `sdlms.cli.si_modularity`: define interface, load captures + SAE outputs, compute SI/Q metrics, emit CSV.
4. Add smoke tests mirroring `tests/test_sparsity_cli.py` for new CLIs.

## Environment Notes
- Python 3.13 via `uv`; run format/tests before committing: `uvx ruff check --fix .`, `uv run pytest`.
- GPU jobs via Kaggle (credentials already configured); use `uv run kaggle …`.
- Artifacts convention: `artifacts/YYYYMMDD-<desc>/`.

## Useful Commands
```
# list probe tasks
jq '.task_id' data/probe_tasks.jsonl

# quick capture + plot (tiny model)
uv run python scripts/run_capture.py --model hf-internal-testing/tiny-random-gpt2 --task-id toy_arithmetic --layers transformer.h.0.mlp transformer.h.1.mlp --no-save
uv run sparsity --model hf-internal-testing/tiny-random-gpt2 --probe-manifest data/probe_tasks.jsonl --task-id toy_arithmetic --layers transformer.h.0.mlp transformer.h.1.mlp --output-dir artifacts/sparsity-demo
uv run python scripts/plot_sparsity.py artifacts/sparsity-demo/<recent>_sparsity.csv

# launch tests
uv run pytest
```

## Gotchas
- Tiny models expose layer names as `transformer.h.N.mlp`; Pythia uses `model.layers.N.mlp`.
- Capture scripts flatten activations; if you need full shapes, adjust `collect_ffn_acts` accordingly.
- `matplotlib` plots save to PNG by default; no display backend needed.

Ping this document at next login to orient quickly.

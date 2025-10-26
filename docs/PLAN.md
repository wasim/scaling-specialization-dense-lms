# Execution Plan (Living Doc)

- [x] Implement `sdlms.cli.sae_train` (local sparse autoencoder training loop)
- [x] Create `data/probe_tasks.jsonl` and update activation capture pipeline
- [ ] Generate multi-layer capture + visualization sweeps for probe tasks
- [ ] Implement `sdlms.cli.si_modularity` end-to-end (reads activations + SAE outputs)
- [ ] Implement `sdlms.cli.dynamic_k` benchmarking loop
- [ ] Expand test suite with smoke tests for new CLIs
- [x] Prepare Kaggle environment (place `kaggle.json` in `~/.kaggle/`, verify CLI)
- [ ] Schedule scaling experiments on GPU (after local validation)

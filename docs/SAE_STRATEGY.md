# SAE Strategy Update — Use Pre-trained Models

**Date:** October 26, 2025  
**Discovery:** EleutherAI already trained SAEs on Pythia models

---

## Key Findings

### 1. EleutherAI `sparsify` Library
**Repo:** https://github.com/EleutherAI/sparsify  
**Install:** `pip install eai-sparsify`

**Features:**
- TopK activation (not L1 penalty) — Pareto improvement per Gao et al. 2024
- On-the-fly activation computation (no caching overhead)
- Pre-trained SAEs available on HuggingFace
- Distributed training support

**Usage:**
```python
from sparsify import Sae

# Load pre-trained
sae = Sae.load_from_hub("EleutherAI/sae-pythia-70m-32x", hookpoint="layers.10")

# Train custom
# python -m sparsify EleutherAI/pythia-160m --hookpoints "layers.3" "layers.6"
```

### 2. EleutherAI `SAELens` (Fork)
**Repo:** https://github.com/EleutherAI/SAELens (fork of jbloomAus/SAELens)

**Features:**
- L1-based training (older approach)
- More analysis/visualization tools
- Pre-trained GPT-2 SAEs available

**Usage:**
```python
from sae_lens import SparseAutoencoder
sae = SparseAutoencoder.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
```

---

## Recommended Strategy

### Phase 1: Use Pre-trained SAEs
1. Check HuggingFace for available Pythia SAE checkpoints
2. Load via `sparsify` library
3. Verify layer coverage for pythia-70m/410m/1.4b/6.9b
4. Skip training entirely if coverage sufficient

### Phase 2: Train Custom SAEs (if needed)
Only if pre-trained SAEs don't cover required layers:
```bash
python -m sparsify EleutherAI/pythia-410m-deduped \
  --hookpoints "layers.10" "layers.15" "layers.20" \
  --k 192 \
  --expansion_factor 32
```

### Phase 3: Extract Features
```python
# Get SAE latent activations
hidden_state = model(**inputs, output_hidden_states=True).hidden_states[layer]
features = sae.encode(hidden_state.flatten(0, 1))  # (batch*seq, d_sae)
```

---

## Impact on Pipeline

**Before:** 
Capture → Train SAEs → Compute SI/Q → Dynamic-k

**After:**
Capture → **Load Pre-trained SAEs** → Compute SI/Q → Dynamic-k

**Time Saved:** ~1 week (no SAE training/hyperparameter sweeps)

---

## Action Items

1. ✅ Update `pyproject.toml`: Add `eai-sparsify` to dependencies
2. ⚠️ Verify pre-trained SAE availability for Pythia models on HuggingFace
3. ⚠️ Update `si_modularity.py` to load SAEs via `sparsify.Sae.load_from_hub()`
4. ⚠️ Test feature extraction on pythia-70m layer 3

---

## Open Questions

1. **Coverage:** Which Pythia layers have pre-trained SAEs? (Check HuggingFace)
2. **Expansion factor:** Are 32× SAEs sufficient or do we need 8×/16× variants?
3. **Hookpoint naming:** Does `"layers.X"` match Pythia's naming convention?
4. **Feature stability:** Are TopK SAEs compatible with our SI/Q metrics?

---

*This changes Priority 1 from "train SAEs" to "load & validate pre-trained SAEs"*

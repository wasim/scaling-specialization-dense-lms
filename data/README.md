Tiny probe tasks only (keep it reproducible):
- IOI minimal set
- Toy 2–3 digit arithmetic
- Mini POS/NER slice (e.g., CoNLL sample)
- Induction sequences (ABAB…)

## Manifest (`probe_tasks.jsonl`)

- Each line is a JSON object with `task_id`, `domain`, `description`, optional `max_tokens`/`repeat`, and a `source` block.
- Supported sources:
  - `prompt`: single prompt repeated `repeat` times (default 1).
  - `prompt_list`: iterate through the list, optionally repeating the list.
  - `dataset`: lazily load via `datasets.load_dataset` with optional `config`, `split`, `text_key`, and `num_docs`.
- Keep prompts short and deterministic so CLI smoke tests can run on CPU with `hf-internal-testing/tiny-random-gpt2`.

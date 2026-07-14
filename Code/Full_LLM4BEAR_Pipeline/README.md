# LLM4BEAR Dataset Pipeline

End-to-end pipeline that builds the LLM4BEAR bundle dataset across three domains
(electronic, clothing, food). It runs in four stages:

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `stage1_egpo.py` | **EGPO** — Expert-Guided Prompt Optimization: beam-search refine the evaluator ("judge") prompts. |
| 2 | `stage2_best_prompts.py` | Score the refined prompts on held-out annotated bundles and pick the **best per (domain × personality) by ICC** vs human experts. |
| 3 | `stage3_refinement.py` | **BundleRefinement** — iteratively evaluate → diagnose → add/remove items to improve each bundle (the canonical `bundle_refinement_workflow`). |
| 4 | `stage4_make_dataset.py` | Assemble the final per-domain `(bundle, intent)` dataset from the refinement history. |

Shared code lives in `common.py` (OpenAI plumbing + parsing), `prompts.py` (prompt
templates), and `config.py` / `mock_configs.py` (all run parameters).

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env      # then put your OpenAI key in .env
```

`.env` (gitignored) holds your secrets:

```
OPENAI_API_KEY=sk-...
WANDB_KEY=            # optional, stage 1 only
```



## Running the full pipeline

Run the stages in order (each writes to `output/`):

```bash
./venv/bin/python stage1_egpo.py
./venv/bin/python stage2_best_prompts.py
./venv/bin/python stage3_refinement.py
./venv/bin/python stage4_make_dataset.py
```

### Mock (cheap) runs

A full run is expensive (thousands of bundles × many LLM calls × up to 20 iterations).
If you'd like a **lighter run of the full pipeline without spending too much money** on an
API key, pass `--mock` — every stage then reads its reduced parameters (fewer bundles,
fewer iterations, one domain) from **`mock_configs.py`** instead of the full defaults:

```bash
./venv/bin/python stage1_egpo.py          --mock
./venv/bin/python stage2_best_prompts.py  --mock
./venv/bin/python stage3_refinement.py    --mock
./venv/bin/python stage4_make_dataset.py
```

The **canonical (full) configuration lives in `config.py`**; the **reduced smoke-test
configuration lives in `mock_configs.py`**. Be careful to run the right version — `--mock`
selects `mock_configs.py`, no flag selects the full `config.py` values. Both you and any
reviewer can use `--mock` to verify the code runs before committing to a full run.

## Running stages individually

**Results from each stage have already been provided** (the `anon5159753/LLM4BEAR` repo ships
EGPO's `1_EGPO/final_prompts/*.pkl`, the refinement history pickles under `2_Bundle
Refinement/historic_bundle_refinement/`, etc.). Because of that, **you can run any stage on
its own without running the prior stages** — each stage reads the provided upstream results
by default:

- `stage2_best_prompts.py --export-selection` writes `output/selected_best_prompts.pkl`
  from the provided refined prompts (no API needed).
- `stage3_refinement.py` reads `output/selected_best_prompts.pkl` (or falls back to the
  provided repo prompts — see the commented block in `load_domain_prompts`).
- `stage4_make_dataset.py` reads the provided `*_no_graph_help_run.pkl` refinement outputs
  and needs only `numpy`/`pickle` — it runs with no API key.

## Output

`output/` contains:
- `final_prompts/Refined_*.pkl` — stage 1 refined prompts
- `selected_best_prompts.pkl` — stage 2 best-ICC selection (consumed by stage 3)
- `historical_bundle_changes_<domain>.pkl` — stage 3 refinement history
- `llm4bear_<domain>_bundles.pkl` / `llm4bear_<domain>_intents.pkl` — the final dataset
  (per-domain parallel lists; a record is `(bundles[k], intents[k])`).

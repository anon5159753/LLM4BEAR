# Non-LLM Baselines for BundleRec

Three **traditional (non-LLM)** bundle-recommendation baselines — **Freq (Apriori)**,
**BBPR**, **BYOB** — adapted to run on the **BundleRec** dataset (domains: clothing,
electronic, food), then scored by an **LLM-as-a-Judge** (1–5 intent coherence) and by
**structural metrics** (Precision / Recall / Jaccard vs. ground truth), so they sit
apples-to-apples beside LLM-generated bundle baselines.

`NOTES.md` has the full technical detail; this README is the overview, credit, and
how-to-run.

## Results (BundleRec, our 3 baselines × 3 domains)

| baseline | domain | precision | recall | jaccard | $\bar{s}$ |
|---|---|---|---|---|---|
| freq | clothing | 0.386 | 0.334 | 0.523 | 3.57 |
| bbpr | clothing | 0.356 | 0.314 | 0.505 | 3.33 |
| byob | clothing | 0.344 | 0.312 | 0.500 | 3.43 |
| freq | electronic | 0.279 | 0.214 | 0.460 | 3.12 |
| bbpr | electronic | 0.256 | 0.211 | 0.440 | 3.11 |
| byob | electronic | 0.261 | 0.216 | 0.462 | 3.07 |
| freq | food | 0.247 | 0.206 | 0.458 | 3.49 |
| bbpr | food | 0.189 | 0.177 | 0.433 | 3.34 |
| byob | food | 0.206 | 0.190 | 0.452 | 3.31 |

$\bar{s}$ = mean LLM-judge intent score (1–5). Full table incl. per-run bundle
counts: `judge_scores/nonllm_results.csv`.

## Credit / upstream sources
This work **adapts existing code** from the repositories below (our files adapt it to
BundleRec — please verify each repo's license before redistributing):

- **BundleRec dataset & baselines** — https://github.com/BundleRec/bundle_recommendation
  → the **Freq / Apriori** baseline and the dataset (`BundleRec_dataset/`).
- **BYOB — "Build Your Own Bundle"** — https://github.com/fuxiAIlab/BYOB
  → the **BBPR** models (`ItemBPRModel` / `BundleBPRModel`, vendored verbatim into
    `bbpr_baseline.py` under an attribution header) and the **BYOB** RL method.
- **LLM4BEAR** — https://github.com/anon5159753/LLM4BEAR
  → the LLM-as-a-Judge (`Comparing_Baselines.ipynb`) and a pre-processed copy of the
    BundleRec data (see below).

## The `LLM4BEAR/` folder (data source — fetched, not committed)
Not committed (gitignored). It is a **sparse checkout** of the LLM4BEAR repo, pulled
so you don't download everything:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/anon5159753/LLM4BEAR.git
cd LLM4BEAR
git sparse-checkout set "4_Bundle Generation"
```

It provides: `data/bundlerec/<domain>/` (a clean, judge-aligned **pre-processed copy
of the BundleRec dataset** — session→items, train/test split keyed by session ID,
titles, `test_ground_indices.pkl` ground truth); `final_prompts/` (the judge's domain
evaluator prompts); and `baselines/` + `evaluation/` (**the existing LLM baseline
outputs and their judge scores are also included**, if you want to run the full
non-LLM-vs-LLM comparison). This is still the **BundleRec dataset**, just
pre-organized — not a different dataset. The raw `BundleRec_dataset/` CSVs are used
only for item→category (Freq) and the `user_item_pretrain` corpus (BBPR/BYOB
embeddings).

## The baselines (what we adapted)
The upstream baselines were wired to other datasets (movielens/yoochoose) and, for
BBPR/BYOB, to a `byob` package that is **not present**. We retargeted them at BundleRec:

- **Freq** (`freq_baseline.py`) — keeps the Apriori/frequent-pattern concept, but
  replaces the original blind self-join candidate generation (which does not
  terminate at support 0.001) with equivalent **within-session itemset counting**
  (identical support/confidence semantics). Category-level.
- **BBPR** (`bbpr_baseline.py`) — **reuses `ItemBPRModel` + `BundleBPRModel`
  unchanged** (vendored verbatim from the BYOB repo, with an attribution header);
  adds a BundleRec adapter, two-stage Item-BPR → Bundle-BPR training, item-embedding
  pretraining on the interaction corpus, and session-context top-K generation.
- **BYOB** (`byob_baseline.py`) — **lightweight reimplementation** (SkipGram
  pretraining + greedy compatibility construction), since the original Ray/RLlib
  `BundleEnv` lives in the absent `byob` package. Keeps BYOB's two defining pieces
  (item2vec embedding + fixed-size bundle construction).
- **Judge** (`llm_as_a_judge.py`, `run_llm_judge.py`) — a **mechanical de-Colab** of
  LLM4BEAR's `Comparing_Baselines.ipynb`; the evaluation logic (intent inference +
  1–5 scoring) is unchanged.

## Hyperparameters used
Inherited from prior work that grid-searched these on the same BundleRec datasets
(adopted, not re-tuned):

- **Freq:** support = 0.001, confidence = 0.001.
- **BBPR:** embedding = 20, negative samples = 2, learning rate = 0.01, batch size =
  64, bundle size = 3. *(The inherited "initial bundle size 3 / neighbors 10" describe
  Pathak's seed+neighbor greedy and do **not** apply — the BBPR we run does top-K.)*
  The item-embedding pretraining stage uses a large batch (8192) for CPU feasibility.
- **BYOB:** embedding = 20, negative samples = 2, learning rate = 0.001, bundle size =
  3. SkipGram pretraining uses window 5, batch 4096.

## Shared generation rules (all three baselines)
Two deliberate adaptations to the BundleRec setting, applied identically:

1. **Candidate pool = the whole session's items (no cap).** The inherited "candidate
   item set = 20" was tuned for large-catalog datasets; here the maximum session
   length is **10 items** (never ≥ 20), so a cap of 20 is vacuous — each baseline
   selects from *all* items in the session.
2. **A 2-item session is automatically a bundle.** Both items are emitted with no
   selection step (a valid bundle needs ≥ 2 items). Sessions with ≥ 3 items are pruned
   to a size-3 bundle; sessions with < 2 items (which do not occur) yield an empty entry.

Intent labels (`bundle_intent.csv`) are **not** used — these baselines are
structural/collaborative; the judge infers its own intent at evaluation.

## Output schema
`output/<baseline>/<domain>.npy` — `{test_sid: {"bundle1": ["productN", ...]}}`, one
entry per test key (empty list where nothing was generated). `productN` is the 1-based
session position; the judge maps it to `session_items[N-1]`. This matches the existing
AICL `.npy` baselines, so our outputs drop straight into the judge.

## How to run
Prereqs: activate the venv (deps: numpy, pandas, torch, mlxtend, openai) and fetch
the `LLM4BEAR/` data (see the sparse-checkout command above). For the judge step,
provide an OpenAI key in a `.env` file at the repo root (gitignored, loaded
automatically):
```bash
echo "API_KEY=sk-..." > .env
```
(Alternatively `export API_KEY=sk-...` or `OPENAI_API_KEY=sk-...` in your shell.)

```bash
source venv/bin/activate.fish
```

**1. Generate the bundles** (offline, no API) — writes `output/<baseline>/<domain>.npy`:
```bash
python run_baselines.py --baselines all --domains all
```

**2. Score them with the LLM judge** (needs the API key) — writes the per-band pkls +
`summary.json` to `judge_scores/`; resume-safe, so re-runs skip completed work:
```bash
python run_llm_judge.py --baselines all --domains all      # add --force to re-score
```

**3. Eyeball the generated bundles** — see what each method produced, in judge-input
form, with session cross-checking:
```bash
python show_baseline_bundles.py --compare --domain clothing --key 65   # methods side by side
python show_baseline_bundles.py --baseline byob --domain food --limit 10
python show_baseline_bundles.py --validate                             # membership check, all 9
```

**4. Build the final results table** — merges structural metrics (Precision / Recall /
Jaccard, computed offline here) with the judge scores into
`judge_scores/nonllm_results.csv`:
```bash
python build_eval_report.py
```

(`structural_metrics.py` can also be run on its own for the offline P/R/Jaccard.)

## Files
**Source:** `adapter.py`, `freq_baseline.py`, `bbpr_baseline.py`, `byob_baseline.py`,
`run_baselines.py`, `show_baseline_bundles.py`, `structural_metrics.py`,
`llm_as_a_judge.py`, `run_llm_judge.py`, `build_eval_report.py`, `NOTES.md`, this
`README.md`.
**Fetched / generated (gitignored):** `LLM4BEAR/`, `output/`, `judge_scores/`, `venv/`,
`.env`.

The upstream reference code we adapted from (the original `old_*` folders) has been
removed; the BBPR model classes are vendored into `bbpr_baseline.py`, and all credit
is via the links in **Credit / upstream sources** above.

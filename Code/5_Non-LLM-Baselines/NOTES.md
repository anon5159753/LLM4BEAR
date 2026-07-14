# Non-LLM Baselines for BundleRec — Notes

Documents the data format found, the mapping used, assumptions, and every
deviation from the original algorithms, for the three reviewer-requested non-LLM
baselines (**Freq**, **BBPR**, **BYOB**) and the localized LLM-as-a-Judge.

---

## 1. Data format found

`BundleRec_dataset/<domain>/` ships as **CSV only** (12 files/domain, relational)
— there is **no `test_set.npy` / train-test split** in it. Sessions are stored
relationally: group `session_item.csv` by `session ID`; items are raw integer
`item ID`s, mapped to Amazon ASINs via `item_idx_mapping.csv`.

The test keys, the `|split|` title strings, the train/test/val split, and the
existing LLM baseline outputs live **only in the `anon5159753/LLM4BEAR` GitHub repo**
(cloned at runtime by the original Colab notebook). We fetched it (sparse
checkout of `4_Bundle Generation`).

## 2. Data source used (deviation from the brief, approved)

We build on the **LLM4BEAR package** (`LLM4BEAR/4_Bundle Generation/data/bundlerec/
<domain>/`) rather than reconstructing sessions from the raw CSVs, because it is a
self-consistent universe keyed by the exact test-session IDs the judge consumes:

| File | Content | Use |
|---|---|---|
| `session_items.npy` (968) | `{sid: "ASIN,ASIN,…"}` ordered | **bridge: test key → ordered item IDs** |
| `training_set.npy` (≈700) + `training_p_data.npy` | train sessions + positive bundles (`pN`) | train BBPR/BYOB |
| `test_set.npy` / `test_p_data.npy` (180) | test sessions + ground-truth bundles | generation targets + judge input |
| `item_titles.npy` | `{ASIN: title}` | display |

**Verified invariant:** for every key in every domain, `session_items` ASIN order
is length-identical and **position-aligned** with `test_set`'s `|split|` title
order (residual string diffs are cosmetic — stripped `"` quotes/HTML entities).
So position *N* in `session_items` == the `productN` the judge resolves. No
title-matching needed.

The **raw CSVs are used for one thing:** item → leaf category for Freq
(`item_categories.csv` joined to `item_idx_mapping.csv` → ASIN → leaf). 100%
coverage of test-session items in all domains.

## 3. Core assumptions

- **Session-as-bundle:** each user session / purchase list is treated as a
  positive bundle (per the brief).
- **No intent labels:** `bundle_intent.csv` is untouched metadata. All three
  baselines are structural/collaborative and consume interaction data only; the
  judge infers its own intent at evaluation.
- **Output convention:** `{test_sid: {"bundle1": ["productN", …]}}`, one entry
  per test key, empty list where nothing was generated (keys never dropped).
  `productN` is 1-based session position; the judge maps it to `session_items[N-1]`.

## 4. Candidate pool rule (all three baselines)

- **Pool = the whole test session's items. No cap.** Max session length is **10**
  across all domains (never ≥ 20), so the inherited "candidate set = 20" is
  **vacuous** here.
- **|S| < 2** → `[]` (does not occur; min session length is 2).
- **|S| == 2** → auto-bundle both items.
- **|S| ≥ 3** → select down to bundle size 3.

## 5. Cold-start findings (motivate the embedding choices)

- **User cold-start:** train/test sessions are disjoint and ~95–99% of test-session
  users never appear in training. A user-ID embedding is therefore untrained at
  generation time. → BBPR/BYOB use a **session-context proxy** at generation:
  context = mean of the session's item embeddings; score each candidate item
  against it. No user ID needed at test.
- **Item cold-start:** training on the bundle-sessions alone leaves ~90% of
  *clothing* test items with untrained embeddings (131/180 sessions fully cold).
  → item embeddings are **pretrained on `user_item_pretrain.csv`** (the large
  interaction corpus). Test-item coverage then rises to 83% (clothing) / 97%
  (electronic) / 94% (food); fully-cold sessions drop to 4 / 0 / 1.

## 6. Per-baseline details & deviations

### Freq (`freq_baseline.py`)
- Category-level frequent-pattern mining over training sessions (leaf-category
  token lists), then per test session pick the best realisable rule, map it to
  items (earliest unused position per category), fill to size 3 with the most
  corpus-frequent leaf, emit one bundle. No rule → honest empty.
- **Deviation (miner):** the original `old_apriori/apriori_bundle.py` generates
  candidates by blindly self-joining the full category set, which at support
  0.001 explodes to ~10¹² ops at pattern size 3 and does not terminate. Since any
  itemset with support > 0 must occur inside a session, we instead **count the
  size-2/3 category combinations that actually occur within each session**
  (≤ C(10,3)=120/session). Identical support/confidence semantics, milliseconds.
- **Deviation (patterns 2–3, not 2–5):** bundle size is 3, so a bundle realises
  ≤ 3 categories; larger patterns are irrelevant.
- **Known characteristic (kept, faithful):** leaf categories are the last node of
  the first Amazon category path, which is sometimes a **brand** (`Ray-Ban`,
  `Tommy Hilfiger`) or a **logistics tag** (`… International Shipping Available`)
  rather than a product type. This adds noise to Freq's groupings (e.g. grouping a
  wallet + boot via a shared shipping-tag leaf). This is a faithful reflection of
  frequency-based bundling on noisy category metadata and is left as-is by design.
- Hyperparameters: support = 0.001, confidence = 0.001 (inherited).

### BBPR (`bbpr_baseline.py`)
- Reuses `ItemBPRModel` + `BundleBPRModel` from `old_BBPR/baseline_bpr.py`
  (mean-pooled bundle embeddings) unchanged. Two-stage: **Item-BPR** on
  `user_item_pretrain` triplets → carry item embeddings forward → **Bundle-BPR**
  fine-tunes them on training bundle-sessions.
- **Generation = greedy top-K over the session pool** using the session-context
  proxy (score each item by `sigmoid(ctx · item_embed)`, take top 3).
- **Deviation (top-K vs seed+neighbor):** the runner's BBPR is top-K selection,
  **not** Pathak's seed+neighbor greedy. Consequently the inherited "initial
  bundle size = 3 / neighbors = 10" **do not apply** and are not used.
- Hyperparameters: embed = 20, negative samples = 2, lr = 0.01,
  **Bundle-BPR batch = 64** (directive). The item-embedding **pretraining** stage
  uses **batch 8192** (a separate item2vec-style step on 0.3–2M interactions;
  batch-64 there means ~100× more tiny batches and ~16 min/domain on CPU vs ~1 min).
  Torch threads capped at 8 to avoid oversubscription.

### BYOB (`byob_baseline.py`) — lightweight
- The original is Ray/RLlib PPO over a `BundleEnv` fed by a SkipGram embedding; the
  `byob` package (env/models/config) is absent locally. We keep the two defining
  pieces without Ray: **(1)** SkipGram/item2vec pretrained on `user_item_pretrain`;
  **(2)** greedy bundle construction to size 3 over the session pool, scoring
  incremental additions by embedding compatibility (deterministic stand-in for the
  RL policy that maximises bundle reward).
- **Deviation:** no Ray/RLlib PPO; greedy construction instead. SkipGram
  pretraining uses `main_vec.py`'s regime (window 5, batch 4096 for CPU
  feasibility; main_vec used 256). The inherited batch-64/lr-0.001 pertained to the
  RL stage we replace.
- Hyperparameters: embed = 20, negative samples = 2, bundle size = 3.

## 7. Output schema & storage

`output/<baseline>/<domain>.npy` — `{test_sid: {"bundle1": [...]}}` dict matching
the existing AICL `.npy` baselines (`4o-mini_bundle_res.npy`), so our outputs drop
straight into the judge. A `.json` mirror is written alongside.

## 8. Phase 2 — localized judge (`llm_as_a_judge.py`, `run_llm_judge.py`)

Mechanical de-Colab of `Comparing_Baselines.ipynb`; **evaluation logic unchanged**:
- removed `drive.mount` / `google.colab.userdata` / `!git clone` / `!pip` / autotime;
- API key from `os.environ` (`API_KEY`/`OPENAI_API_KEY`), loaded from a gitignored
  `.env` via a tiny built-in reader;
- `/content/...` paths repointed to the local repo;
- `single_request` / `openai_request` / `separate_consideration_scores` /
  `intent_evaluation_module` and the 1–5 intent-scoring loop kept verbatim
  (**async concurrency preserved** — all bundles per domain fire concurrently,
  batched 128);
- domain evaluator prompts loaded from `final_prompts/Refined_<band>_evaluator_
  <domain>_importance_no_experts.pkl` with the notebook's tuned indices
  (electronic [1,1,2], clothing [2,2,0], food [0,2,0]);
- judge model = OpenAI gpt-4o-mini, temperature 0, seed 42.

Our three baselines feed `make_bundle_strings` → `input_strings` →
`intent_evaluation_module` identically to gpt-4o-mini/claude/etc.

**Empty / fringe cases:** empties contribute no bundle string (consistent with the
original pipeline, which scores over produced bundles for every method). Our
baselines never emit 0/1-item bundles (size handling → ≥2 or empty). `run_llm_judge`
reports bundle count `n` next to each average so coverage differences (Freq ~90–95%
vs BBPR/BYOB 100%) are visible.

## 9. Verification (Phase 1)

- Non-empty rate: Freq 95.0 / 95.6 / 90.0%; BBPR & BYOB 100% (clothing/electronic/food).
- Session membership: **all 9 outputs clean** — every bundle item resolves within
  its session (`show_baseline_bundles.py --validate`).
- `make_bundle_strings` round-trip resolves `productN` → coherent in-session titles
  (offline, no API); judge live-smoke-tested end-to-end.

## 10. How to run

```fish
source venv/bin/activate.fish

# Phase 1 — generate bundles (offline, ~15-25 min full)
python run_baselines.py --baselines all --domains all

# Inspect / cross-check
python show_baseline_bundles.py --compare --domain clothing --key 65
python show_baseline_bundles.py --validate

# Phase 2 — judge (needs API key in .env)
python run_llm_judge.py --baselines all --domains all
```

## 11. Files

- **New:** `adapter.py`, `freq_baseline.py`, `bbpr_baseline.py`, `byob_baseline.py`,
  `run_baselines.py`, `show_baseline_bundles.py`, `llm_as_a_judge.py`,
  `run_llm_judge.py`, `NOTES.md`, `.gitignore`, `.env` (gitignored).
- **Reference only (unmodified):** `old_apriori/`, `old_BBPR/`, `old_BYOB/`,
  `old_LLM-as-a-Judge/Comparing_Baselines.ipynb`.
- **Fetched (gitignored):** `LLM4BEAR/`, `output/`, `judge_scores/`, `venv/`.

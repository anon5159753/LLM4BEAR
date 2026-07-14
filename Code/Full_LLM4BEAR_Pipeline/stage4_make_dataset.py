"""Stage 4 — Build the final LLM4BEAR bundle dataset.

Consumes the canonical BundleRefinement history pickles and, for every seed bundle,
selects the best refined version:

* if refinement *succeeded* (the bundle's final-iteration flag is ``False``), take the
  final-iteration bundle/intent;
* otherwise take the highest-scoring *valid* (>= 2 item) bundle across all iterations
  ("pseudo-refined").

Output: per-domain parallel lists ``llm4bear_<domain>_bundles.pkl`` and
``llm4bear_<domain>_intents.pkl`` written to ``output/``.

Ported from ``making_datasets.ipynb``. Fixes the original food size-filter bug and no
longer hardcodes the iteration count.
"""

import copy
import pickle

import config

# By default we read the canonical pre-shipped refinement outputs from the repo.
# To instead consume a fresh stage-3 run, point this at
#   config.OUTPUT_DIR / f"historical_bundle_changes_{domain}.pkl"
CANONICAL_PKL = (
    config.HISTORIC_REFINEMENT
    / "historical_bundle_changes_{domain}_complete_{domain}_no_graph_help_run.pkl"
)


def load_history(domain):
    """Load a refinement history pickle: (intents, bundle_items, bundle_indices,
    scores, min_scores, flags), each indexed [iteration][seed]."""
    path = str(CANONICAL_PKL).format(domain=domain)
    with open(path, "rb") as f:
        return pickle.load(f)


def invalidate_small_bundles(bundle_items, scores):
    """Zero the score of any bundle with fewer than 2 items across all iterations,
    so it can never be chosen as the 'best' pseudo-refined version.

    (The original notebook had a malformed food loop that only caught size-1 bundles;
    this applies the size 0 *and* 1 rule uniformly to every domain.)"""
    changed = copy.deepcopy(scores)
    n_iters = len(bundle_items)
    n_seeds = len(bundle_items[0])
    for i in range(n_iters):
        for j in range(n_seeds):
            if len(bundle_items[i][j]) <= 1:
                changed[i][j] = 0
    return changed


def build_domain_dataset(domain):
    """Return (bundles, intents) parallel lists for one domain."""
    intents, bundle_items, _indices, scores, _min_scores, flags = load_history(domain)
    changed_scores = invalidate_small_bundles(bundle_items, scores)
    n_iters = len(bundle_items)

    # Split seeds by whether refinement converged (final flag == False -> success).
    last_flags = flags[-1]
    success, unsuccess = [], []
    for i, flag in enumerate(last_flags):
        (unsuccess if flag else success).append(i)

    # Successful seeds: take the final refined bundle/intent.
    # Unsuccessful seeds: take the highest-scoring valid bundle across iterations.
    bundles = [[] for _ in range(len(success) + len(unsuccess))]
    ds_intents = [None] * (len(success) + len(unsuccess))

    for seed in success:
        bundles[seed] = bundle_items[-1][seed]
        ds_intents[seed] = intents[-1][seed]

    for seed in unsuccess:
        per_iter_scores = [changed_scores[j][seed] for j in range(n_iters)]
        best_iter = per_iter_scores.index(max(per_iter_scores))
        bundles[seed] = bundle_items[best_iter][seed]
        ds_intents[seed] = intents[best_iter][seed]

    return bundles, ds_intents


def main():
    out = config.ensure_output_dir()
    for domain in config.DOMAINS:
        bundles, intents = build_domain_dataset(domain)
        assert len(bundles) == len(intents)
        with open(out / f"llm4bear_{domain}_bundles.pkl", "wb") as f:
            pickle.dump(bundles, f)
        with open(out / f"llm4bear_{domain}_intents.pkl", "wb") as f:
            pickle.dump(intents, f)
        n_small = sum(1 for b in bundles if len(b) < 2)
        print(f"{domain:11s}: {len(bundles):5d} bundles  (<2 items: {n_small}) -> saved")

    print(f"\nDataset written to {out}")


if __name__ == "__main__":
    main()

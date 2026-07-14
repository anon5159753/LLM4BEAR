"""Entry point: score our non-LLM baseline bundles with the LLM-as-a-Judge.

Prereq: Phase 1 has been run (output/<baseline>/<domain>.npy exist) and an API key
is in .env (API_KEY) or the environment.

Usage:
    python run_llm_judge.py --baselines all --domains all
    python run_llm_judge.py --baselines freq --domains food
    python run_llm_judge.py --baselines all --domains all --force   # re-score everything

Saving mirrors Comparing_Baselines.ipynb (simple pickle dumps in judge_scores/):
  * per band, during scoring: intent_<baseline>_<domain>_<band>.pkl = [responses, verdicts, scores]
  * at the end: all_baseline_bundle_strings.pkl, all_baseline_bundle_sizes.pkl,
    all_baseline_bundle_stats.pkl, all_baseline_scores.pkl, and summary.json.

Resume: a (baseline, domain) whose 3 band pkls already exist is skipped unless --force.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle

import numpy as np

import llm_as_a_judge as judge

BASELINES = ["freq", "bbpr", "byob"]
DOMAINS = ["clothing", "electronic", "food"]


def _load_predictions(baseline, domain):
    path = os.path.join(judge.OUR_OUTPUT, baseline, f"{domain}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).tolist()


async def _run(baselines, domains, force):
    summary = {}
    all_strings, all_sizes, all_scores = {}, {}, {}
    for domain in domains:
        for baseline in baselines:
            name = f"{baseline}_{domain}"
            preds = _load_predictions(baseline, domain)
            if preds is None:
                print(f"!! missing output for {name} -- run run_baselines.py first")
                continue
            if not force and judge.band_pkls_exist(baseline, domain):
                print(f"== skip {name} (already scored; --force to redo) ==")
                continue
            print(f"\n=== judging {name} ({len(preds)} keys) ===")
            res = await judge.evaluate_baseline(baseline, domain, preds)
            summary[name] = {"band_scores": res["band_scores"], "average": res["average"], "n": res["n"]}
            all_strings[name] = res["bundle_strings"]
            all_sizes[name] = res["sizes"]
            all_scores[name] = {"band_scores": res["band_scores"], "average": res["average"]}
            print(f"    bands(bad/mid/good)={res['band_scores']} average={res['average']} n={res['n']}")
    return summary, all_strings, all_sizes, all_scores


def _dump(name, obj):
    with open(os.path.join(judge.EVAL_OUT, name), "wb") as f:
        pickle.dump(obj, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="all")
    ap.add_argument("--domains", default="all")
    ap.add_argument("--force", action="store_true", help="re-score even if band pkls exist")
    args = ap.parse_args()

    if not (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("Set API_KEY (or OPENAI_API_KEY), e.g. in .env")

    baselines = BASELINES if args.baselines == "all" else args.baselines.split(",")
    domains = DOMAINS if args.domains == "all" else args.domains.split(",")

    summary, all_strings, all_sizes, all_scores = asyncio.run(_run(baselines, domains, args.force))

    # end-of-run pickle dumps (notebook-style)
    if all_sizes:
        keys = list(all_sizes.keys())
        stats_tuple = judge.generate_bundle_stats(all_sizes)
        stats = dict(zip(
            ["avg_num_bundles_per_sess", "avg_num_items_per_bundle", "num_bundles_generated", "num_session"],
            stats_tuple))
        stats["keys"] = keys
        _dump("all_baseline_bundle_strings.pkl", all_strings)
        _dump("all_baseline_bundle_sizes.pkl", all_sizes)
        _dump("all_baseline_bundle_stats.pkl", stats)
        _dump("all_baseline_scores.pkl", all_scores)
        with open(os.path.join(judge.EVAL_OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if not summary:
        print("\nNothing new scored -- all requested runs already have scores "
              "(resume). Use --force to re-score.")
        return
    print("\n=== SUMMARY (1-5 average intent score) ===")
    for k, v in summary.items():
        print(f"  {k:20s} average={v['average']}  bands={v['band_scores']}  n={v['n']}")
    print(f"\nsaved pickles + summary.json -> {judge.EVAL_OUT}")


if __name__ == "__main__":
    main()

"""Combined evaluation report for the three non-LLM baselines (BundleRec).

Merges the offline structural metrics (Precision / Recall / Jaccard + bundle stats)
with the LLM-judge 1-5 intent averages (from judge_scores/summary.json) into one
9-row table (3 baselines x 3 domains) and writes judge_scores/nonllm_results.csv.

Self-contained -- our baselines only, no existing-LLM-baseline rows.

Run structural metrics any time (offline); the intent column is populated once
run_llm_judge.py has produced summary.json (shows blank/None otherwise).
"""

from __future__ import annotations

import json
import os

import pandas as pd

import structural_metrics as sm
from llm_as_a_judge import EVAL_OUT

BASELINES = ["freq", "bbpr", "byob"]
DOMAINS = ["clothing", "electronic", "food"]


def _intent_scores():
    path = os.path.join(EVAL_OUT, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)   # {"<baseline>_<domain>": {band_scores, average, n}}


def build():
    struct = sm.compute_all(BASELINES, DOMAINS)
    intent = _intent_scores()

    rows = []
    for domain in DOMAINS:
        for baseline in BASELINES:
            m = struct.get((baseline, domain))
            if m is None:
                continue
            iv = intent.get(f"{baseline}_{domain}", {})
            rows.append({
                "baseline": baseline,
                "domain": domain,
                "sessions": m["sessions"],
                "total_bundles": m["total_bundles"],
                "avg_bundles_per_sess": round(m["avg_bundles_per_sess"], 3),
                "avg_bundle_size": round(m["avg_bundle_size"], 3),
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "jaccard": round(m["jaccard"], 3),
                "intent_avg_score": (round(iv["average"], 3) if iv.get("average") is not None else None),
            })
    df = pd.DataFrame(rows)
    out = os.path.join(EVAL_OUT, "nonllm_results.csv")
    os.makedirs(EVAL_OUT, exist_ok=True)
    df.to_csv(out, index=False)
    return df, out


if __name__ == "__main__":
    df, out = build()
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(df.to_string(index=False))
    print(f"\nsaved -> {out}")

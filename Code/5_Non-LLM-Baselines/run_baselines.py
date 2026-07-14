"""Single core runner: generate judge-ready bundles for the non-LLM baselines.

Usage:
    python run_baselines.py --baselines all --domains all
    python run_baselines.py --baselines freq,bbpr --domains food

For each (baseline, domain) it loads the domain once, runs the baseline, enforces
one entry per test key (empty list where nothing was generated -- never drop
keys), and writes ``output/<baseline>/<domain>.npy`` (a dict
``{test_sid: {"bundle1": [...]}}`` matching the existing AICL npy baselines) plus
a ``.json`` mirror.

It prints a per-run summary with the NON-EMPTY BUNDLE RATE.  If any baseline comes
back mostly empty (< MIN_NONEMPTY_RATE), it flags a hard STOP -- a high empty rate
would make the judge comparison meaningless.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from adapter import DOMAINS, load_domain
from freq_baseline import run_freq
from bbpr_baseline import run_bbpr
from byob_baseline import run_byob

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
BASELINES = {"freq": run_freq, "bbpr": run_bbpr, "byob": run_byob}
MIN_NONEMPTY_RATE = 0.50


def _enforce_keys(result, domain):
    """One entry per test key, empty list default (don't drop keys)."""
    full = {}
    for s in domain.test_sessions:
        v = result.get(s.sid)
        full[s.sid] = v if v else []
    return full


def _summary(result):
    n = len(result)
    nonempty = [v for v in result.values() if v]
    sizes = [len(b) for v in nonempty for b in v.values()]
    return {
        "keys": n,
        "non_empty": len(nonempty),
        "non_empty_rate": len(nonempty) / n if n else 0.0,
        "mean_bundle_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "mean_bundles_per_key": (sum(len(v) for v in nonempty) / len(nonempty)) if nonempty else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="all")
    ap.add_argument("--domains", default="all")
    args = ap.parse_args()

    baselines = list(BASELINES) if args.baselines == "all" else args.baselines.split(",")
    domains = list(DOMAINS) if args.domains == "all" else args.domains.split(",")
    device = torch.device("cpu")

    flags = []
    print(f"baselines={baselines} domains={domains}\n")
    for dom in domains:
        print(f"### loading domain: {dom}")
        d = load_domain(dom)
        for bl in baselines:
            print(f"--- running {bl} on {dom} ---")
            result, _meta = BASELINES[bl](d, device) if bl != "freq" else BASELINES[bl](d)
            result = _enforce_keys(result, d)
            outdir = os.path.join(OUT_DIR, bl)
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f"{dom}.npy"), result, allow_pickle=True)
            with open(os.path.join(outdir, f"{dom}.json"), "w") as f:
                json.dump({str(k): v for k, v in result.items()}, f, indent=1)
            s = _summary(result)
            tag = ""
            if s["non_empty_rate"] < MIN_NONEMPTY_RATE:
                tag = "  <<< STOP: mostly empty, judge comparison unreliable"
                flags.append((bl, dom, s["non_empty_rate"]))
            print(f"    {bl}/{dom}: keys={s['keys']} non_empty={s['non_empty']} "
                  f"({s['non_empty_rate']*100:.1f}%) mean_bundle_size={s['mean_bundle_size']:.2f} "
                  f"mean_bundles/key={s['mean_bundles_per_key']:.2f}{tag}")
        print()

    print("=== DONE ===")
    if flags:
        print("!!! FLAGGED (non-empty rate < %.0f%%):" % (MIN_NONEMPTY_RATE * 100))
        for bl, dom, r in flags:
            print(f"    {bl}/{dom}: {r*100:.1f}%")
    else:
        print("All runs above the non-empty-rate threshold.")


if __name__ == "__main__":
    main()

"""Inspect generated bundles in the EXACT form fed to the LLM judge.

Loads output/<baseline>/<domain>.npy, runs it through the judge's own
make_bundle_strings (productN -> session titles), and prints the resolved
"Intent:\\nBundle Items:\\n1. <title>..." blocks -- i.e. precisely the strings
that go into intent_evaluation_module.  No API key needed.

Usage:
    python show_baseline_bundles.py --baseline bbpr --domain clothing
    python show_baseline_bundles.py --baseline freq --domain food --limit 5 --raw
    python show_baseline_bundles.py --compare --domain clothing --key 1
    python show_baseline_bundles.py --compare --domain clothing --limit 3
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import llm_as_a_judge as J

BASELINES = ["freq", "bbpr", "byob"]
DOMAINS = ["clothing", "electronic", "food"]


def _load(baseline, domain):
    path = os.path.join(J.OUR_OUTPUT, baseline, f"{domain}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).tolist()


def _session_items(domain):
    """{key: [title, ...]} -- the exact session the judge resolves productN against."""
    ts = np.load(os.path.join(J.DATA_DIR, domain, "test_set.npy"), allow_pickle=True).tolist()
    return {k: str(v).split("|split|") for k, v in ts.items()}


def _bundle_strings(preds, domain):
    """{key: [judge_input_string, ...]} for every non-empty key."""
    bs, _ = J.make_bundle_strings(os.path.join(J.DATA_DIR, domain) + os.sep, preds)
    return bs


def _validate(preds, sessions):
    """Cross-check every bundle item is within its session. Returns
    (n_bundles, n_items, violations) where violations lists (key, token, reason)."""
    n_bundles = n_items = 0
    violations = []
    for key, val in preds.items():
        if not isinstance(val, dict):
            continue
        sess_len = len(sessions.get(key, []))
        for bid, content in val.items():
            n_bundles += 1
            idxs = J.convert_products_to_indices(content)
            if len(idxs) != len(content):
                violations.append((key, content, "unparseable productN token"))
            for tok, idx in zip(content, idxs):
                n_items += 1
                if idx < 0 or idx >= sess_len:
                    violations.append((key, tok, f"index {idx} outside session (len {sess_len})"))
    return n_bundles, n_items, violations


def _print_session(sessions, key):
    items = sessions.get(key, [])
    print(f"  SESSION ({len(items)} items):")
    for i, t in enumerate(items):
        print(f"    product{i+1}. {t}")


def show_one(baseline, domain, limit=None, raw=False, keys=None):
    preds = _load(baseline, domain)
    if preds is None:
        print(f"!! no output for {baseline}/{domain} (run run_baselines.py first)")
        return
    sessions = _session_items(domain)
    bs = _bundle_strings(preds, domain)
    shown_keys = keys if keys else [k for k in bs if bs[k]]
    if limit:
        shown_keys = shown_keys[:limit]
    n_nonempty = sum(1 for k in bs if bs[k])
    nb, ni, viol = _validate(preds, sessions)
    print(f"\n{'#'*70}\n# {baseline.upper()} / {domain}  "
          f"({n_nonempty}/{len(bs)} keys non-empty) -- showing {len(shown_keys)}")
    print(f"# cross-check: {nb} bundles, {ni} items -- "
          f"{'ALL within session ✓' if not viol else f'{len(viol)} VIOLATIONS ✗'}\n{'#'*70}")
    for k in shown_keys:
        print(f"\n----- test key {k} -----")
        _print_session(sessions, k)
        if raw:
            print(f"  raw: {preds.get(k)}")
        strings = bs.get(k, [])
        if not strings:
            print("  (empty -- no bundle generated)")
        for s in strings:
            print("  BUNDLE ->")
            for line in s.splitlines():
                print(f"    {line}")
    if viol:
        print(f"\n!! {len(viol)} violations (first 10): {viol[:10]}")


def compare(domain, limit=None, keys=None):
    """Show the same key across all three baselines side by side, with the session."""
    sessions = _session_items(domain)
    loaded = {b: _load(b, domain) for b in BASELINES}
    strings = {b: (_bundle_strings(loaded[b], domain) if loaded[b] else {}) for b in BASELINES}
    if keys:
        shown = keys
    else:
        allkeys = sorted({k for b in BASELINES for k in strings[b] if strings[b].get(k)})
        shown = allkeys[:limit] if limit else allkeys
    for k in shown:
        print(f"\n{'='*70}\n=== {domain} | test key {k} ===\n{'='*70}")
        _print_session(sessions, k)
        for b in BASELINES:
            raw = loaded[b].get(k) if loaded[b] else None
            print(f"\n--- {b}  (raw: {raw}) ---")
            ss = strings[b].get(k, []) if strings[b] else []
            if not ss:
                print("  (empty)")
            for s in ss:
                for line in s.splitlines():
                    print(f"    {line}")


def validate_all():
    """Sweep every available baseline x domain and report the session-membership check."""
    print(f"{'baseline/domain':22s} {'bundles':>8s} {'items':>7s}  result")
    for domain in DOMAINS:
        sessions = _session_items(domain)
        for baseline in BASELINES:
            preds = _load(baseline, domain)
            if preds is None:
                print(f"{baseline+'/'+domain:22s} {'--':>8s} {'--':>7s}  (missing)")
                continue
            nb, ni, viol = _validate(preds, sessions)
            res = "ALL within session ✓" if not viol else f"{len(viol)} VIOLATIONS ✗"
            print(f"{baseline+'/'+domain:22s} {nb:>8d} {ni:>7d}  {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=BASELINES)
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--key", type=int, action="append", help="specific test key(s); repeatable")
    ap.add_argument("--raw", action="store_true", help="also print raw productN form")
    ap.add_argument("--compare", action="store_true", help="all baselines side by side")
    ap.add_argument("--validate", action="store_true", help="sweep all baseline x domain membership checks")
    args = ap.parse_args()

    if args.validate:
        validate_all()
        return
    if not args.domain:
        ap.error("--domain is required unless --validate")
    if args.compare:
        compare(args.domain, limit=args.limit, keys=args.key)
    else:
        if not args.baseline:
            ap.error("--baseline is required unless --compare/--validate")
        show_one(args.baseline, args.domain, limit=args.limit, raw=args.raw, keys=args.key)


if __name__ == "__main__":
    main()

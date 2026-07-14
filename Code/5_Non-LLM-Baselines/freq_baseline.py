"""Freq baseline (Apriori-style frequent-pattern mining) for BundleRec.

Category-level frequent patterns are mined over TRAINING sessions (rendered as
leaf-category token lists); bundles are then realised per TEST session by mapping
fired category rules back onto the specific session items.

DEVIATION from old_apriori/apriori_bundle.py (documented in NOTES.md): the
original miner generates candidates by blindly self-joining the full category set
(~400 tokens), which at support = 0.001 (threshold "appears in >=1 session")
explodes to ~10^12 operations at pattern size 3 and does not terminate.  Any
itemset with support > 0 must appear *inside* some session, so we instead count
the size-2 / size-3 category combinations that actually occur within each session
(<= C(10,3)=120 combos/session). This is identical frequent-itemset / support /
confidence semantics, computed in milliseconds.

Approved generation logic (see NOTES.md):
  * |S| < 2  -> [] (does not occur in this data)
  * |S| == 2 -> auto-bundle both items  {"bundle1": [product1, product2]}
  * |S| >= 3 -> pick the best rule whose category multiset is realisable by the
    session (>= 2 distinct positions), realise it to items (earliest unused
    position per category), fill to size 3 with the remaining item of the most
    corpus-frequent leaf category; emit one bundle.
  * no rule fires -> [] (honest empty).

Hyperparameters (inherited): support = 0.001, confidence = 0.001.
Patterns are mined at sizes 2 and 3 only: bundle size is 3, so a size-3 bundle
can realise at most 3 categories -- larger patterns are irrelevant here.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from adapter import positions_to_products

SUPPORT = 0.001
CONFIDENCE = 0.001
BUNDLE_SIZE = 3


def _multiset_subset(small: Counter, big: Counter) -> bool:
    return all(big[k] >= v for k, v in small.items())


def mine_rules(cat_sessions):
    """Return a list of rules: dict(cats=[...], size, support, conf), mined at
    category level over the training sessions by counting the size-2 / size-3
    category itemsets that actually occur within each session (see module
    docstring for why this replaces the non-scaling self-join miner)."""
    n = len(cat_sessions)
    singleton = Counter()
    for s in cat_sessions:
        for c in s:
            singleton[c] += 1

    # count sessions containing each itemset (multiset-aware, deduped per session)
    counts = Counter()
    for s in cat_sessions:
        cats = sorted(s)
        seen = set()
        for size in (2, 3):
            for combo in combinations(cats, size):  # combo is a sorted tuple
                if combo not in seen:
                    seen.add(combo)
                    counts[combo] += 1

    min_count = SUPPORT * n
    rules = []
    for combo, cnt in counts.items():
        if cnt < min_count:
            continue
        support = cnt / n
        denom = sum(singleton[c] for c in combo) or 1
        conf = cnt / denom
        if conf > CONFIDENCE:
            rules.append({"cats": list(combo), "size": len(combo),
                          "support": support, "conf": conf})
    # best first: prefer size 3, then higher confidence, then higher support
    rules.sort(key=lambda r: (r["size"] == 3, r["conf"], r["support"]), reverse=True)
    return rules


def _realise(rule_cats, cats, n):
    """Map a rule's categories onto earliest unused session positions."""
    used, positions = set(), []
    for c in rule_cats:
        pos = next((p for p in range(n) if p not in used and cats[p] == c), None)
        if pos is None:
            return None
        used.add(pos)
        positions.append(pos)
    return positions


def generate_bundle(session, idx2leaf, rules, cat_corpus_freq):
    cats = [idx2leaf[i] for i in session.items]
    n = len(cats)
    if n < 2:
        return []
    if n == 2:
        return {"bundle1": positions_to_products([0, 1])}

    sess_ms = Counter(cats)
    best = None  # (sortkey, positions)
    for r in rules:
        rc = Counter(r["cats"])
        if not _multiset_subset(rc, sess_ms):
            continue
        positions = _realise(r["cats"], cats, n)
        if positions is None or len(positions) < 2:
            continue
        key = (r["size"] == 3, r["conf"], r["support"])
        if best is None or key > best[0]:
            best = (key, positions)

    if best is None:
        return []

    positions = best[1][:BUNDLE_SIZE]
    if len(positions) < BUNDLE_SIZE:
        remaining = [p for p in range(n) if p not in positions]
        remaining.sort(key=lambda p: cat_corpus_freq.get(cats[p], 0), reverse=True)
        for p in remaining:
            positions.append(p)
            if len(positions) == BUNDLE_SIZE:
                break

    return {"bundle1": positions_to_products(sorted(positions))}


def run_freq(domain):
    """Return {test_sid: bundle_dict_or_empty_list} for every test key."""
    rules = mine_rules(domain.train_category_sessions())
    out = {}
    for s in domain.test_sessions:
        out[s.sid] = generate_bundle(s, domain.idx2leaf, rules, domain.cat_corpus_freq)
    return out, {"n_rules": len(rules)}


if __name__ == "__main__":
    import time
    from adapter import load_domain
    for dom in ["clothing", "electronic", "food"]:
        t = time.time()
        d = load_domain(dom)
        res, meta = run_freq(d)
        nonempty = sum(1 for v in res.values() if v)
        sizes = [len(next(iter(v.values()))) for v in res.values() if v]
        print(f"{dom}: rules={meta['n_rules']} keys={len(res)} "
              f"non_empty={nonempty} ({nonempty/len(res)*100:.1f}%) "
              f"mean_bundle_size={sum(sizes)/len(sizes):.2f} | {time.time()-t:.1f}s")
        ex = next(iter(res.items()))
        print(f"   example {ex}")

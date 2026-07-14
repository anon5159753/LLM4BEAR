"""Shared data adapter for the non-LLM BundleRec baselines (Freq, BBPR, BYOB).

Design (see NOTES.md for the full rationale):

  * Session composition + keys come from the fetched LLM4BEAR package
    (``LLM4BEAR/4_Bundle Generation/data/bundlerec/<domain>/``), which is keyed by
    the exact test-session IDs the judge consumes.  ``session_items.npy`` is the
    authoritative bridge: its ASIN order is position-aligned with the ``pN`` /
    ``productN`` convention (verified length-identical to ``test_set.npy`` titles).
  * Item -> leaf category (Freq only) comes from the raw BundleRec CSVs
    (``item_categories.csv`` joined to ``item_idx_mapping.csv`` -> ASIN).  100%
    coverage of test-session items in all three domains.
  * Item embeddings (BBPR / BYOB) are pretrained on ``user_item_pretrain.csv`` --
    the large user-item interaction corpus -- so test items are not cold-start.

The adapter deliberately does NOT touch intent labels (bundle_intent.csv): the
non-LLM baselines are structural/collaborative and consume interaction data only.
"""

from __future__ import annotations

import html
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels up (Code/5_Non-LLM-Baselines -> Code -> repo root),
# where "4_Bundle Generation" lives.
REPO_ROOT = os.path.join(ROOT, "..", "..")
LLM4BEAR_BASE = os.path.join(REPO_ROOT, "4_Bundle Generation", "data", "bundlerec")
CSV_BASE = os.path.join(ROOT, "BundleRec_dataset")
DOMAINS = ["clothing", "electronic", "food"]


def _load_npy_dict(path):
    return np.load(path, allow_pickle=True).tolist()


def leaf_category(raw: str) -> str:
    """Extract a single leaf category token from the raw bracketed CSV field.

    Raw form (not clean JSON, commas appear inside nodes):
        '[[Clothing, Shoes & Jewelry,Novelty, Costumes & More,Costumes & Accessories],[...]]'
    We take the FIRST category path and its LAST comma-separated segment as the
    leaf (deterministic, one leaf per item -- matching the original tune.py which
    used a single leaf category per item).
    """
    s = html.unescape(str(raw)).strip()
    s = s.strip("[]")
    if not s:
        return "UNKNOWN"
    first_path = s.split("],[")[0]
    leaf = first_path.split(",")[-1].strip()
    return leaf or "UNKNOWN"


@dataclass
class Session:
    sid: int
    items: list          # ordered list of global item indices (pN order)
    asins: list          # ordered list of ASINs (parallel to items)
    positives: list = field(default_factory=list)  # list[list[item_idx]] positive bundles


@dataclass
class Domain:
    name: str
    item2idx: dict
    idx2asin: list
    idx2title: list
    idx2leaf: list                 # leaf category per item index ("UNKNOWN" if absent)
    train_sessions: list           # list[Session]
    test_sessions: list            # list[Session]  (positives from test_p_data)
    pretrain_sequences: list       # list[list[item_idx]] user interaction sequences (time-ordered)
    cat_corpus_freq: Counter       # leaf-category -> count over training sessions

    @property
    def n_item(self):
        return len(self.idx2asin)

    def train_category_sessions(self):
        """Training sessions rendered as lists of leaf-category tokens (for Apriori)."""
        return [[self.idx2leaf[i] for i in s.items] for s in self.train_sessions]


def _parse_pN(bundle_pN, n_items):
    """['p2','p3'] -> [1,2] (0-based positions), dropping any out-of-range."""
    out = []
    for tok in bundle_pN:
        m = re.search(r"(\d+)", str(tok))
        if not m:
            continue
        pos = int(m.group(1)) - 1
        if 0 <= pos < n_items:
            out.append(pos)
    return out


def load_domain(domain: str) -> Domain:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    lb = os.path.join(LLM4BEAR_BASE, domain)
    cb = os.path.join(CSV_BASE, domain)

    session_items = _load_npy_dict(os.path.join(lb, "session_items.npy"))   # {sid: "asin,asin"}
    item_titles = _load_npy_dict(os.path.join(lb, "item_titles.npy"))        # {asin: title}
    train_keys = list(_load_npy_dict(os.path.join(lb, "training_set.npy")).keys())
    test_keys = list(_load_npy_dict(os.path.join(lb, "test_set.npy")).keys())
    train_p = _load_npy_dict(os.path.join(lb, "training_p_data.npy"))        # {sid: [[p1,p2],..]}
    test_p = _load_npy_dict(os.path.join(lb, "test_p_data.npy"))

    # ---- item -> leaf category (raw CSV join via ASIN) ---------------------- #
    idx_map = pd.read_csv(os.path.join(cb, "item_idx_mapping.csv"))          # item ID, source ID
    cats = pd.read_csv(os.path.join(cb, "item_categories.csv"))              # item ID, categories
    id2asin = dict(zip(idx_map["item ID"], idx_map["source ID"]))
    asin2leaf = {}
    for iid, craw in zip(cats["item ID"], cats["categories"]):
        a = id2asin.get(iid)
        if a is not None:
            asin2leaf[a] = leaf_category(craw)

    # ---- pretrain interaction sequences (item ID -> ASIN) ------------------ #
    pre = pd.read_csv(os.path.join(cb, "user_item_pretrain.csv"))            # user ID,item ID,timestamp
    pre["asin"] = pre["item ID"].map(id2asin)
    pre = pre.dropna(subset=["asin"])
    pre = pre.sort_values(["user ID", "timestamp"], kind="stable")

    # ---- build global item vocab (ASIN -> idx) ----------------------------- #
    # Cover every item that appears in a session (all splits) plus the pretrain
    # corpus, so embeddings exist for all test items we may score.
    item2idx = {}

    def _intern(asin):
        j = item2idx.get(asin)
        if j is None:
            j = len(item2idx)
            item2idx[asin] = j
        return j

    for sid, s in session_items.items():
        for a in str(s).split(","):
            _intern(a)
    for a in pre["asin"].unique():
        _intern(a)

    n_item = len(item2idx)
    idx2asin = [None] * n_item
    for a, j in item2idx.items():
        idx2asin[j] = a
    idx2title = [item_titles.get(a, a) for a in idx2asin]
    idx2leaf = [asin2leaf.get(a, "UNKNOWN") for a in idx2asin]

    # ---- sessions ---------------------------------------------------------- #
    def _make_session(sid, positives_pN):
        asins = str(session_items[sid]).split(",")
        items = [item2idx[a] for a in asins]
        pos = []
        for b in (positives_pN or []):
            idxs = [items[p] for p in _parse_pN(b, len(items))]
            if idxs:
                pos.append(idxs)
        return Session(sid=sid, items=items, asins=asins, positives=pos)

    train_sessions = [_make_session(k, train_p.get(k)) for k in train_keys]
    test_sessions = [_make_session(k, test_p.get(k)) for k in test_keys]

    # ---- pretrain sequences (as item indices) ------------------------------ #
    pretrain_sequences = []
    for _uid, grp in pre.groupby("user ID", sort=False):
        seq = [item2idx[a] for a in grp["asin"].tolist()]
        if len(seq) >= 2:
            pretrain_sequences.append(seq)

    # ---- category corpus frequency (training sessions) --------------------- #
    cat_freq = Counter()
    for s in train_sessions:
        for i in s.items:
            cat_freq[idx2leaf[i]] += 1

    return Domain(
        name=domain,
        item2idx=item2idx,
        idx2asin=idx2asin,
        idx2title=idx2title,
        idx2leaf=idx2leaf,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        pretrain_sequences=pretrain_sequences,
        cat_corpus_freq=cat_freq,
    )


def positions_to_products(positions) -> list:
    """0-based session positions -> ['productN', ...] (1-based, judge convention)."""
    return [f"product{p + 1}" for p in positions]


if __name__ == "__main__":
    # quick smoke test / sanity report
    for dom in DOMAINS:
        d = load_domain(dom)
        n_pos_tr = sum(len(s.positives) for s in d.train_sessions)
        example = d.test_sessions[0]
        print(f"\n=== {dom} ===")
        print(f"n_item(vocab)={d.n_item} train_sessions={len(d.train_sessions)} "
              f"test_sessions={len(d.test_sessions)} pretrain_seqs={len(d.pretrain_sequences)}")
        print(f"train positive bundles={n_pos_tr} | distinct leaf cats={len(d.cat_corpus_freq)}")
        print(f"example test sid={example.sid} len={len(example.items)} "
              f"leafcats={[d.idx2leaf[i] for i in example.items][:4]}")
        print(f"example positives(idx)={example.positives}")

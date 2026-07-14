"""Structural metrics (Precision / Recall / Jaccard) for the non-LLM baselines.

Faithful port of Comparing_Baselines.ipynb `compute_comprehensively` (cell 20) and
the bundle-stats logic (cell 16), scored against **BundleRec ground truth**
(`test_ground_indices.pkl`).  Runs entirely offline -- no API.

Predictions come from our Phase-1 output (`output/<baseline>/<domain>.npy`), which
is already the `{test_id: {bundleN: [productN]}}` dict the original function
consumes.  Ground truth is the 0-based session-relative index sets in
`test_ground_indices.pkl`.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from llm_as_a_judge import DATA_DIR, OUR_OUTPUT, convert_products_to_indices

BASELINES = ["freq", "bbpr", "byob"]
DOMAINS = ["clothing", "electronic", "food"]


def compute_comprehensively(ground_indices, predictions):
    """Verbatim port of the notebook's cell-20 logic."""
    session_precision = 0
    session_recall = 0
    session_jaccard = 0
    invalid_id = []
    valid_session_count = 0

    for test_id, pred in predictions.items():
        all_bundle = ground_indices[test_id]
        hit_bundle = 0

        if pred is None or isinstance(pred, int) or len(pred) == 0:
            invalid_id.append(test_id)
            continue

        valid_session_count += 1

        pred_sets = []
        for _bundle_key, product_nums in pred.items():
            indices = convert_products_to_indices(product_nums)
            if indices and len(indices) >= 2:
                pred_sets.append(set(indices))

        gt_sets = [set(gt) for gt in all_bundle]

        all_matches = []
        for g_idx, g_set in enumerate(gt_sets):
            for p_idx, p_set in enumerate(pred_sets):
                intersection = len(g_set & p_set)
                union = len(g_set | p_set)
                score = intersection / union if union > 0 else 0
                if score > 0:
                    all_matches.append((score, g_idx, p_idx))

        all_matches.sort(key=lambda x: x[0], reverse=True)

        assigned_gts, assigned_ps = set(), set()
        reward_sum = 0
        for score, g_idx, p_idx in all_matches:
            if g_idx not in assigned_gts and p_idx not in assigned_ps:
                assigned_gts.add(g_idx)
                assigned_ps.add(p_idx)
                reward_sum += score

        session_jaccard += reward_sum / max(len(gt_sets), 1 * len(pred_sets))

        for truth_bundle in gt_sets:
            for pred_bundle in pred_sets:
                if set(pred_bundle) == truth_bundle:
                    hit_bundle += 1

        session_precision += hit_bundle / len(pred_sets) if len(pred_sets) > 0 else 0
        session_recall += hit_bundle / len(gt_sets) if len(gt_sets) > 0 else 0

    valid_session_count = max(valid_session_count, 1)
    return (session_precision / valid_session_count,
            session_recall / valid_session_count,
            session_jaccard / valid_session_count,
            list(set(invalid_id)))


def bundle_stats(predictions):
    """sessions with >=1 bundle, total bundles, avg bundles/session, avg bundle size."""
    num_sessions = total_bundles = total_items = 0
    for _key, pred in predictions.items():
        if not isinstance(pred, dict) or len(pred) == 0:
            continue
        num_sessions += 1
        total_bundles += len(pred)
        for items in pred.values():
            total_items += len(items)
    return {
        "sessions": num_sessions,
        "total_bundles": total_bundles,
        "avg_bundles_per_sess": (total_bundles / num_sessions) if num_sessions else 0.0,
        "avg_bundle_size": (total_items / total_bundles) if total_bundles else 0.0,
    }


def _load_ground(domain):
    with open(os.path.join(DATA_DIR, domain, "test_ground_indices.pkl"), "rb") as f:
        return pickle.load(f)


def _load_preds(baseline, domain):
    path = os.path.join(OUR_OUTPUT, baseline, f"{domain}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).tolist()


def compute_all(baselines=BASELINES, domains=DOMAINS):
    """{(baseline, domain): {precision, recall, jaccard, sessions, ...}}"""
    out = {}
    for domain in domains:
        ground = _load_ground(domain)
        for baseline in baselines:
            preds = _load_preds(baseline, domain)
            if preds is None:
                continue
            p, r, j, invalids = compute_comprehensively(ground, preds)
            stats = bundle_stats(preds)
            out[(baseline, domain)] = {
                "precision": p, "recall": r, "jaccard": j,
                "n_invalid": len(invalids), **stats,
            }
    return out


if __name__ == "__main__":
    res = compute_all()
    print(f"{'baseline/domain':22s} {'sess':>4s} {'bund':>5s} {'b/s':>5s} {'sz':>5s} "
          f"{'Prec':>6s} {'Rec':>6s} {'Jacc':>6s}")
    for (b, d), m in res.items():
        print(f"{b+'/'+d:22s} {m['sessions']:>4d} {m['total_bundles']:>5d} "
              f"{m['avg_bundles_per_sess']:>5.2f} {m['avg_bundle_size']:>5.2f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['jaccard']:>6.3f}")

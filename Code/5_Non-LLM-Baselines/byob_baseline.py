"""BYOB baseline for BundleRec -- lightweight reimplementation.

The original BYOB (``old_BYOB/main_sim.py``) is a Ray/RLlib PPO agent over a
``BundleEnv`` fed by a pretrained SkipGram (item2vec) embedding
(``old_BYOB/main_vec.py``).  The ``byob`` package (env, models, config) is not
present locally, so we reimplement BYOB's essential loop without Ray, keeping the
two defining pieces:

  1. SkipGram / item2vec embedding pretrained on the ``user_item_pretrain``
     interaction corpus (this is exactly the embedding input the original agent
     consumes).
  2. Greedy bundle construction to a fixed size of 3 over the session candidate
     pool, scoring incremental additions by embedding compatibility -- a
     deterministic stand-in for the RL policy that maximises bundle reward.

Generation uses the session-context proxy (see NOTES.md): pool = the whole test
session, seed = the item most compatible with the session context, then greedily
add the item most compatible with the current partial bundle until size 3.
  * |S| < 2  -> []          (does not occur)
  * |S| == 2 -> both items

Hyperparameters: embed = 20, negative samples = 2, bundle size = 3.  The inherited
batch size = 64 / lr = 0.001 pertain to the original RL stage; the SkipGram
pretraining uses main_vec.py's own regime (window 5, batch 256) -- see NOTES.md.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import positions_to_products  # noqa: E402

EMBED_DIM = 20
NEG_SAMPLES = 2
WINDOW = 5
LR = 0.001
BATCH_SIZE = 4096        # SkipGram pretraining on the large interaction corpus,
                         # batched large for CPU feasibility (main_vec.py used 256;
                         # the inherited batch-64 pertained to the RL stage we replace).
EPOCHS = 6
BUNDLE_SIZE = 3
SEED = 42
NUM_THREADS = 8


class SkipGram(nn.Module):
    def __init__(self, n_item, dim):
        super().__init__()
        self.center = nn.Embedding(n_item, dim)
        self.context = nn.Embedding(n_item, dim)
        nn.init.xavier_normal_(self.center.weight)
        nn.init.xavier_normal_(self.context.weight)

    def forward(self, c, o, neg):
        c = self.center(c)                       # (B, E)
        o = self.context(o)                      # (B, E)
        neg = self.context(neg)                  # (B, K, E)
        pos = F.logsigmoid((c * o).sum(-1))                       # (B,)
        negl = F.logsigmoid(-(neg * c.unsqueeze(1)).sum(-1)).sum(-1)  # (B,)
        return -(pos + negl).mean()


def _build_pairs(seqs, window, rng):
    centers, contexts = [], []
    for seq in seqs:
        L = len(seq)
        for a in range(L):
            lo, hi = max(0, a - window), min(L, a + window + 1)
            for b in range(lo, hi):
                if b == a:
                    continue
                centers.append(seq[a])
                contexts.append(seq[b])
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


def train_skipgram(domain, device, epochs=EPOCHS):
    rng = np.random.default_rng(SEED)
    n_item = domain.n_item
    centers, contexts = _build_pairs(domain.pretrain_sequences, WINDOW, rng)
    n = len(centers)
    model = SkipGram(n_item, EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            c = torch.as_tensor(centers[b], device=device)
            o = torch.as_tensor(contexts[b], device=device)
            neg = torch.as_tensor(rng.integers(0, n_item, size=(len(b), NEG_SAMPLES)),
                                  device=device)
            loss = model(c, o, neg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"    [skipgram] epoch {ep+1}/{epochs} loss={total/(n//BATCH_SIZE+1):.4f}")
    return model.center.weight.detach().clone()


def greedy_bundle(items, W, size=BUNDLE_SIZE):
    n = len(items)
    emb = F.normalize(W[torch.as_tensor(items, dtype=torch.int64)], dim=-1)  # (n,E)
    ctx = emb.mean(dim=0)
    sims_ctx = emb @ ctx
    selected = [int(torch.argmax(sims_ctx))]
    while len(selected) < size and len(selected) < n:
        centroid = emb[selected].mean(dim=0)
        scores = emb @ centroid
        for s in selected:
            scores[s] = -1e9
        selected.append(int(torch.argmax(scores)))
    return selected


def generate(domain, item_weight):
    out = {}
    for s in domain.test_sessions:
        items = s.items
        n = len(items)
        if n < 2:
            out[s.sid] = []
        elif n == 2:
            out[s.sid] = {"bundle1": positions_to_products([0, 1])}
        else:
            pos = greedy_bundle(items, item_weight)
            out[s.sid] = {"bundle1": positions_to_products(sorted(pos))}
    return out


def run_byob(domain, device=None):
    device = device or torch.device("cpu")
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    W = train_skipgram(domain, device)
    res = generate(domain, W)
    return res, {"stage": "skipgram+greedy"}


if __name__ == "__main__":
    import time
    from adapter import load_domain
    dom = sys.argv[1] if len(sys.argv) > 1 else "food"
    t = time.time()
    d = load_domain(dom)
    print(f"loaded {dom} in {time.time()-t:.1f}s")
    res, _ = run_byob(d)
    nonempty = sum(1 for v in res.values() if v)
    print(f"{dom}: keys={len(res)} non_empty={nonempty} ({nonempty/len(res)*100:.1f}%) "
          f"| total {time.time()-t:.1f}s")
    print("   example:", next(iter(res.items())))

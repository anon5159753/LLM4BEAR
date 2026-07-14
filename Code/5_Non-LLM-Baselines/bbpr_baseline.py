"""BBPR baseline for BundleRec (two-stage Item-BPR -> Bundle-BPR).

Vendors ``ItemBPRModel`` and ``BundleBPRModel`` unchanged (mean-pooled bundle
embeddings) from the BYOB repo (https://github.com/fuxiAIlab/BYOB), originally in
``old_BBPR/baseline_bpr.py`` -- see the attribution header below.

Pipeline:
  1. Item-BPR: train item (and auxiliary user) embeddings on the large
     ``user_item_pretrain`` interaction corpus via BPR triplets
     (user, positive item, negative item).  This is what gives test items
     non-random embeddings (item cold-start otherwise ~90% in clothing).
  2. Bundle-BPR: carry the item embeddings forward and fine-tune them on the
     TRAINING bundle-sessions (positive bundle vs. random negative bundle,
     mean-pooled), so items that co-occur in real bundles become compatible.

Generation (session-context proxy -- chosen because ~99% of test-session users
are cold-start, so a user-ID embedding is untrained at test time):
  * pool = the whole test session's items (<=10 here; the inherited candidate
    set = 20 is vacuous).
  * |S| < 2  -> []          (does not occur)
  * |S| == 2 -> both items
  * |S| >= 3 -> context = mean of the session's item embeddings; score each item
    by sigmoid(context . item_embed); take the top-3.

Hyperparameters (inherited): embed = 20, negative samples = 2, lr = 0.01,
batch size = 64, bundle size = 3.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter import positions_to_products


# --------------------------------------------------------------------------- #
# ItemBPRModel and BundleBPRModel are vendored verbatim from
# old_BBPR/baseline_bpr.py -- originally from the BYOB repo
# (https://github.com/fuxiAIlab/BYOB).  Vendored (unmodified) so old_BBPR/ can be
# deleted.  BundleBPRModel scores a (user, bundle) via the mean-pooled item
# embeddings of the bundle; both models are trained from scratch on BPR triplets.
# --------------------------------------------------------------------------- #
class ItemBPRModel(nn.Module):

    def __init__(self, conf):
        super(ItemBPRModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

    def _forward(self, inputs):
        u, i, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)
        i = self.item_embed(i.long()).squeeze(dim=1)
        logits = torch.mul(u, i).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def forward(self, inputs):
        u, i, j, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)
        i = self.item_embed(i.long()).squeeze(dim=1)
        j = self.item_embed(j.long()).squeeze(dim=1)
        pos_logits = torch.mul(u, i).sum(dim=-1, keepdim=True)
        neg_logits = torch.mul(u, j).sum(dim=-1, keepdim=True)
        return pos_logits, neg_logits

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self._forward(inputs)
        return pred.detach().cpu().numpy()


class BundleBPRModel(nn.Module):

    def __init__(self, conf):
        super(BundleBPRModel, self).__init__()
        self.n_user = conf['n_user']
        self.n_item = conf['n_item']
        self.embed_dim = conf.get('embed_dim', 32)
        self._build()

    def _build(self):
        self.user_embed = nn.Embedding(self.n_user, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_item, self.embed_dim)
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)

    def _forward(self, inputs):
        u, b, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)
        b = self.item_embed(b.long()).squeeze(dim=1)
        b = torch.mean(b, dim=1)
        logits = torch.mul(u, b).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def forward(self, inputs):
        u, bi, bj, _ = inputs
        u = self.user_embed(u.long()).squeeze(dim=1)
        bi = self.item_embed(bi.long()).squeeze(dim=1)
        bj = self.item_embed(bj.long()).squeeze(dim=1)
        bi = torch.mean(bi, dim=1)
        bj = torch.mean(bj, dim=1)
        pos_logits = torch.mul(u, bi).sum(dim=-1, keepdim=True)
        neg_logits = torch.mul(u, bj).sum(dim=-1, keepdim=True)
        return pos_logits, neg_logits

    def predict(self, inputs):
        inputs = [torch.as_tensor(v, dtype=torch.int64) for v in inputs]
        pred = self._forward(inputs)
        return pred.detach().cpu().numpy()

EMBED_DIM = 20
NEG_SAMPLES = 2
LR = 0.01
BATCH_SIZE = 64          # Bundle-BPR model (inherited directive); small/fast data
ITEM_BATCH = 8192        # item-embedding PRETRAINING stage on the large interaction
                         # corpus -- a separate item2vec-style step, batched large for
                         # CPU feasibility (batch-64 there = ~100x more tiny batches).
BUNDLE_SIZE = 3
ITEM_EPOCHS = 6
BUNDLE_EPOCHS = 6
SEED = 42
NUM_THREADS = 8


def _rng():
    return np.random.default_rng(SEED)


def train_item_bpr(domain, device, epochs=ITEM_EPOCHS):
    """BPR over pretrain interactions; returns trained item-embedding weights."""
    rng = _rng()
    n_item = domain.n_item
    seqs = domain.pretrain_sequences
    n_user = len(seqs)
    # flatten (user, pos_item) pairs
    users, pos = [], []
    for u, seq in enumerate(seqs):
        for it in seq:
            users.append(u)
            pos.append(it)
    users = np.asarray(users, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.int64)
    n = len(users)

    conf = {"n_user": n_user, "n_item": n_item, "embed_dim": EMBED_DIM}
    model = ItemBPRModel(conf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, ITEM_BATCH):
            b = perm[start:start + ITEM_BATCH]
            u = torch.as_tensor(users[b], device=device).view(-1, 1)
            i = torch.as_tensor(pos[b], device=device).view(-1, 1)
            # NEG_SAMPLES negatives per positive, averaged
            loss = 0.0
            for _ in range(NEG_SAMPLES):
                j = torch.as_tensor(rng.integers(0, n_item, size=len(b)),
                                    device=device).view(-1, 1)
                seq = torch.zeros_like(i)
                p_log, n_log = model((u, i, j, seq))
                loss = loss - F.logsigmoid(p_log - n_log).mean()
            loss = loss / NEG_SAMPLES
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"    [item-bpr] epoch {ep+1}/{epochs} loss={total/(n//ITEM_BATCH+1):.4f}")
    return model.item_embed.weight.detach().clone()


def train_bundle_bpr(domain, device, init_item_weight, epochs=BUNDLE_EPOCHS):
    """Fine-tune item embeddings on bundle co-occurrence; returns final weights."""
    rng = _rng()
    n_item = domain.n_item
    # collect positive bundles (as fixed size-3 item lists) with a session-user id
    u_list, bundles = [], []
    for u, s in enumerate(domain.train_sessions):
        for pb in s.positives:
            items = list(pb)
            if len(items) >= 2:
                u_list.append(u)
                bundles.append(items)
    n = len(bundles)
    n_user = len(domain.train_sessions)

    conf = {"n_user": n_user, "n_item": n_item, "embed_dim": EMBED_DIM}
    model = BundleBPRModel(conf).to(device)
    with torch.no_grad():
        model.item_embed.weight.copy_(init_item_weight.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def _fix3(items):
        if len(items) >= BUNDLE_SIZE:
            return list(rng.choice(items, BUNDLE_SIZE, replace=False))
        return list(items) + list(rng.choice(items, BUNDLE_SIZE - len(items), replace=True))

    for ep in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, BATCH_SIZE):
            b = perm[start:start + BATCH_SIZE]
            u = torch.as_tensor([u_list[k] for k in b], device=device).view(-1, 1)
            bi = torch.as_tensor([_fix3(bundles[k]) for k in b], device=device)
            loss = 0.0
            for _ in range(NEG_SAMPLES):
                bj = torch.as_tensor(rng.integers(0, n_item, size=(len(b), BUNDLE_SIZE)),
                                     device=device)
                seq = torch.zeros((len(b), 1), dtype=torch.int64, device=device)
                p_log, n_log = model((u, bi, bj, seq))
                loss = loss - F.logsigmoid(p_log - n_log).mean()
            loss = loss / NEG_SAMPLES
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"    [bundle-bpr] epoch {ep+1}/{epochs} loss={total/(n//BATCH_SIZE+1):.4f}")
    return model.item_embed.weight.detach().clone()


def generate(domain, item_weight):
    """Session-context top-K generation over each test session."""
    W = item_weight  # (n_item, E)
    out = {}
    for s in domain.test_sessions:
        items = s.items
        n = len(items)
        if n < 2:
            out[s.sid] = []
            continue
        if n == 2:
            out[s.sid] = {"bundle1": positions_to_products([0, 1])}
            continue
        emb = W[torch.as_tensor(items, dtype=torch.int64)]      # (n, E)
        ctx = emb.mean(dim=0, keepdim=True)                     # (1, E)
        scores = torch.sigmoid((ctx * emb).sum(dim=-1))         # (n,)
        top = torch.topk(scores, BUNDLE_SIZE).indices.tolist()
        out[s.sid] = {"bundle1": positions_to_products(sorted(top))}
    return out


def run_bbpr(domain, device=None):
    device = device or torch.device("cpu")
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    iw = train_item_bpr(domain, device)
    fw = train_bundle_bpr(domain, device, iw)
    res = generate(domain, fw)
    return res, {"stage": "item-bpr+bundle-bpr"}


if __name__ == "__main__":
    import time
    from adapter import load_domain
    dom = sys.argv[1] if len(sys.argv) > 1 else "food"
    t = time.time()
    d = load_domain(dom)
    print(f"loaded {dom} in {time.time()-t:.1f}s")
    res, _ = run_bbpr(d)
    nonempty = sum(1 for v in res.values() if v)
    print(f"{dom}: keys={len(res)} non_empty={nonempty} ({nonempty/len(res)*100:.1f}%) "
          f"| total {time.time()-t:.1f}s")
    print("   example:", next(iter(res.items())))

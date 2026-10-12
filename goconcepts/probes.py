"""Stage one and stage three: layerwise linear probes trained on natural
positions, evaluated on held-out natural positions (split by game) and on the
cyclic test suite.

Every number here is agreement with an exact label. No representational
similarity score is computed anywhere.

Writes results.json (accuracy tables) and preds.npz (per-sample predictions,
kept for the stage-two cross-engine error-correlation analysis).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

LIB_BUCKETS = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 means 8+


def lib_class(libs):
    return np.clip(libs, 1, 8) - 1


def balanced_acc(y, p):
    cls = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cls]))


class LinearProbe:
    def __init__(self, n_classes, wd=1e-3, epochs=300, lr=3e-3):
        self.k, self.wd, self.epochs, self.lr = n_classes, wd, epochs, lr

    def fit(self, X, y, balance=False):
        X = torch.tensor(X, dtype=torch.float32)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-3
        X = (X - self.mu) / self.sd
        y = torch.tensor(y, dtype=torch.long)
        w = None
        if balance:
            cnt = torch.bincount(y, minlength=self.k).float().clamp(min=1)
            w = (cnt.sum() / (self.k * cnt))
        self.lin = torch.nn.Linear(X.shape[1], self.k)
        opt = torch.optim.Adam(self.lin.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        n = X.shape[0]
        bs = 4096
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad()
                loss = lossf(self.lin(X[idx]), y[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        X = (torch.tensor(X, dtype=torch.float32) - self.mu) / self.sd
        return self.lin(X).argmax(1).numpy()


def pair_features(acts_layer, pos, stone, chain, rng, max_pairs_per_pos=40, only_idx=None, far=0):
    """Same-chain pairs vs same-colour different-chain pairs within a position.
    Feature: elementwise product of the two points' activations."""
    I, J, Y = [], [], []
    idx_all = np.arange(len(pos)) if only_idx is None else only_idx
    for p in np.unique(pos[idx_all]):
        ids = idx_all[pos[idx_all] == p]
        for col in (1, 2):
            c_ids = ids[stone[ids] == col]
            if len(c_ids) < 2:
                continue
            a = rng.choice(c_ids, size=min(len(c_ids) * 3, 400))
            b = rng.choice(c_ids, size=len(a))
            keep = a != b
            a, b = a[keep], b[keep]
            same = chain[a] == chain[b]
            pos_pairs = np.nonzero(same)[0][: max_pairs_per_pos // 2]
            neg_pairs = np.nonzero(~same)[0][: len(pos_pairs)]
            for q in np.concatenate([pos_pairs, neg_pairs]):
                I.append(a[q]); J.append(b[q]); Y.append(int(same[q]))
    I, J, Y = np.array(I, int), np.array(J, int), np.array(Y, int)
    return I, J, Y


def run(run_dir, epochs=200, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nat = np.load(os.path.join(run_dir, "natural.npz"))
    cyc = np.load(os.path.join(run_dir, "cyclic.npz"))
    layers = list(nat["layer_names"])
    A, Ac = nat["acts"], cyc["acts"]
    games = nat["game"]
    ug = np.unique(games)
    rng.shuffle(ug)
    test_games = set(ug[: len(ug) // 5])
    te = np.array([g in test_games for g in games])
    tr = ~te

    concepts = {
        "libs": (lib_class(nat["libs"]), lib_class(cyc["libs"]), 8, None),
        "pass_alive": (nat["pass_alive"].astype(int), cyc["pass_alive"].astype(int), 2, None),
        "eyespace": (nat["eyespace"].astype(int), cyc["eyespace"].astype(int), 3, None),
        "ladder": (nat["ladder"].astype(int), cyc["ladder"].astype(int), 2, (nat["ladder"] >= 0, cyc["ladder"] >= 0)),
    }
    results = {"layers": layers, "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_cyclic": int(len(cyc["libs"])),
               "concepts": {}}
    preds = {}
    cyc_mask = cyc["cyclic"] == 1
    # resume: concepts finished by an interrupted run are kept, not redone
    part_json, part_npz = os.path.join(run_dir, "results_partial.json"), os.path.join(run_dir, "preds_partial.npz")
    if os.path.exists(part_json) and os.path.exists(part_npz):
        results["concepts"] = json.load(open(part_json))["concepts"]
        preds = dict(np.load(part_npz))
        print("resuming; done:", list(results["concepts"]), flush=True)

    def checkpoint():
        json.dump(results, open(part_json, "w"))
        np.savez(part_npz, **preds)

    for cname, (y, yc, k, masks) in concepts.items():
        if cname in results["concepts"]:
            continue
        m_nat = masks[0] if masks else np.ones(len(y), bool)
        m_cyc = masks[1] if masks else np.ones(len(yc), bool)
        trm, tem = tr & m_nat, te & m_nat
        if len(np.unique(y[trm])) < 2:
            print(f"skip {cname}: one class in training data"); continue
        rows = []
        for li, lname in enumerate(layers):
            probe = LinearProbe(k, epochs=epochs).fit(A[trm, li].astype(np.float32), y[trm], balance=(cname != "libs"))
            p_te = probe.predict(A[tem, li].astype(np.float32))
            p_cy = probe.predict(Ac[m_cyc, li].astype(np.float32))
            yt, ycy = y[tem], yc[m_cyc]
            cm = cyc_mask[m_cyc]
            row = {"layer": lname,
                   "test_acc": float((p_te == yt).mean()), "test_bal": balanced_acc(yt, p_te),
                   "cyc_acc": float((p_cy == ycy).mean()),
                   "cyc_acc_on_cyclic_chains": float((p_cy[cm] == ycy[cm]).mean()) if cm.any() else None,
                   "cyc_acc_off_cyclic_chains": float((p_cy[~cm] == ycy[~cm]).mean()) if (~cm).any() else None}
            if cname == "libs":
                hi, hic = yt >= 3, ycy >= 3  # 4+ liberties: not marked by input planes
                row["test_acc_4plus"] = float((p_te[hi] == yt[hi]).mean())
                row["test_within1"] = float((np.abs(p_te - yt) <= 1).mean())
                row["cyc_acc_4plus_cyclic"] = float((p_cy[hic & cm] == ycy[hic & cm]).mean()) if (hic & cm).any() else None
                row["cyc_within1_cyclic"] = float((np.abs(p_cy[cm] - ycy[cm]) <= 1).mean()) if cm.any() else None
            rows.append(row)
            preds[f"{cname}/{lname}/test"] = p_te
            preds[f"{cname}/{lname}/cyc"] = p_cy
            print(cname, lname, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items() if k2 != "layer"}, flush=True)
        results["concepts"][cname] = rows
        checkpoint()

    # group membership (pairwise)
    rows = []
    Itr, Jtr, Ytr = pair_features(None, nat["pos"], nat["stone"], nat["chain"], rng, only_idx=np.nonzero(tr)[0])
    Ite, Jte, Yte = pair_features(None, nat["pos"], nat["stone"], nat["chain"], rng, only_idx=np.nonzero(te)[0])
    Ic, Jc, Yc = pair_features(None, cyc["pos"], cyc["stone"], cyc["chain"], rng, max_pairs_per_pos=200)
    pair_cyc = (cyc["cyclic"][Ic] == 1) | (cyc["cyclic"][Jc] == 1)
    for li, lname in enumerate(layers):
        f = lambda X, I, J: (X[I, li].astype(np.float32) * X[J, li].astype(np.float32))
        probe = LinearProbe(2, epochs=epochs).fit(f(A, Itr, Jtr), Ytr, balance=True)
        p_te = probe.predict(f(A, Ite, Jte)); p_c = probe.predict(f(Ac, Ic, Jc))
        row = {"layer": lname, "test_bal": balanced_acc(Yte, p_te), "cyc_bal": balanced_acc(Yc, p_c),
               "cyc_bal_cyclic_pairs": balanced_acc(Yc[pair_cyc], p_c[pair_cyc]) if pair_cyc.any() and len(np.unique(Yc[pair_cyc])) > 1 else None}
        rows.append(row)
        print("same_chain", lname, {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in row.items() if k2 != "layer"}, flush=True)
    results["concepts"]["same_chain"] = rows
    results["n_pairs"] = {"train": int(len(Ytr)), "test": int(len(Yte)), "cyclic": int(len(Yc)), "cyclic_involving_cyclic_chain": int(pair_cyc.sum())}

    # policy on cyclic cases: probability mass on CORRECT moves
    pol = np.exp(cyc["policy"]); corr = cyc["correct_moves"]
    has = corr.sum(1) > 0
    results["policy_correct_mass"] = {str(n): float((pol[i] * corr[i]).sum()) for i, n in enumerate(cyc["case_names"]) if has[i]}

    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    np.savez(os.path.join(run_dir, "preds.npz"), **preds)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/humansl")
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    run(a.run, a.epochs)

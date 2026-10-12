"""Layer-averaged version of the shared-failure test (supersedes the best-layer
version in pairs_shared.py).

Why: pairs_shared.py picked each engine's layer by probe-D accuracy on the
training pairs, which probe D was fit on. That is in-sample and rewards
overfitting (tf3, with 60 training pairs and 768 channels, scored 0.996 at its
last block). Averaging each pair's gap over every trunk block needs no
selection at all and is also less noisy.

  python -m goconcepts.pairs_layeravg --tag m60 runs/lz runs/g170 runs/humansl runs/tf3

--tag picks prediction files pairs_preds_D_<tag>.npz where they exist (probe D
refit on a matched, reduced training set), falling back to pairs_preds[_D].npz.
Reports, per engine, the mean gap over early / mid / late thirds of the trunk
and by liberty count, and between engines the Spearman correlation of
layer-averaged per-pair gaps, with permutation p-values within liberty-count
strata and partial correlations controlling for the HumanSL gap.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata

from .pairs_shared import _chain_stats


def _preds(run, tag):
    for name in ([f"pairs_preds_D_{tag}.npz"] if tag else []) + ["pairs_preds_D.npz", "pairs_preds.npz"]:
        p = os.path.join(run, name)
        if os.path.exists(p):
            return np.load(p), name
    raise FileNotFoundError(run)


def main(runs, tag=None, control="runs/humansl", seed=0):
    rng = np.random.default_rng(seed)
    meta = json.load(open(os.path.join(runs[0], "pairs_meta.json")))
    lib_of = {m["pid"]: m["libs_cyclic"] for m in meta}
    matched = {m["pid"] for m in meta if m["libs_cyclic"] == m["libs_cut"]}
    all_p = np.array(sorted(lib_of)); np.random.default_rng(0).shuffle(all_p)
    train = set(all_p[: len(all_p) // 2].tolist())
    M, P, src = {}, {}, {}
    for r in runs:
        pr = np.load(os.path.join(r, "pairs.npz"), mmap_mode="r")
        M[r] = (np.asarray(pr["pid"]), np.asarray(pr["variant"]), np.clip(np.asarray(pr["libs"]), 1, 8) - 1)
        P[r], src[r] = _preds(r, tag)
    present = set.intersection(*[set(M[r][0].tolist()) for r in runs])
    test = sorted(p for p in present if p not in train and p in matched)
    st = np.array([lib_of[p] for p in test])

    def ci(v):
        bs = rng.choice(v, size=(4000, len(v))).mean(1)
        return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

    rep = {"n_test_pairs": len(test), "tag": tag, "engines": {}, "correlations": {}}
    avg = {}
    for r in runs:
        pid, var, y = M[r]
        keys = sorted((k for k in P[r].files if k.startswith("D/block")), key=lambda k: int(k.split("block")[1]))
        G = []
        for k in keys:
            acc, _ = _chain_stats(P[r][k], y, pid, var, rng)
            G.append([acc[(p, 1)] - acc[(p, 0)] for p in test])
        G = np.array(G)
        avg[r] = G.mean(0)
        n = len(keys)
        thirds = [G[: n // 3].mean(0), G[n // 3: 2 * n // 3].mean(0), G[2 * n // 3:].mean(0)]
        rep["engines"][os.path.basename(r)] = {
            "preds": src[r], "n_layers": n,
            "gap_all_layers": ci(avg[r]), "gap_early": ci(thirds[0]), "gap_mid": ci(thirds[1]), "gap_late": ci(thirds[2]),
            "gap_mid_by_libs": {str(L): round(float(thirds[1][st == L].mean()), 3) for L in range(2, 9)},
        }

    def resid(v, covs):
        X = np.column_stack([np.ones(len(v))] + [(st == s).astype(float) for s in np.unique(st)[1:]] + [rankdata(c) for c in covs])
        rv = rankdata(v); b, *_ = np.linalg.lstsq(X, rv, rcond=None)
        return rv - X @ b

    def rtest(a, b, covs=()):
        x, z = resid(a, covs), resid(b, covs)
        r = np.corrcoef(x, z)[0, 1]
        null = np.array([np.corrcoef(x, rng.permutation(z))[0, 1] for _ in range(4000)])
        return {"r": round(float(r), 3), "p": round(float((null >= r).mean()), 4)}

    names = [os.path.basename(r) for r in runs]
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            rep["correlations"][f"{names[i]}~{names[j]}"] = rtest(avg[runs[i]], avg[runs[j]])
            if control in runs and control not in (runs[i], runs[j]):
                rep["correlations"][f"{names[i]}~{names[j]} | {os.path.basename(control)}"] = rtest(avg[runs[i]], avg[runs[j]], [avg[control]])
    rep["n_by_libs"] = {str(L): int((st == L).sum()) for L in range(2, 9)}
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    print(json.dumps(main(a.runs, a.tag), indent=1))

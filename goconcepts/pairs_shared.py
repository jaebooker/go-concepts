"""Stage-two test on the minimal pairs: do engines fail on the SAME cyclic chains?

  python -m goconcepts.pairs_shared runs/lz runs/g170 runs/humansl

For each engine, probe D predictions at one layer, chosen on TRAIN pairs only
(highest mean chain accuracy), so layer choice never sees the test pairs.

Per held-out matched pair:  gap = acc(cut) - acc(cyclic), the cyclic-specific
failure of that chain. Between two engines we report
  r       Spearman correlation of per-pair gaps
  r_rel   each engine's split-half reliability of its own gaps (stones of
          each chain split in two, Spearman-Brown corrected); r can't exceed
          sqrt(r_rel_a * r_rel_b) by much, so it is the ceiling to compare to
  p       permutation p-value, gaps shuffled within liberty-count strata, so
          shared difficulty by liberty count can't produce the correlation

The proposal's reading: "shared wrong concept" predicts a positive r near
the ceiling; "coverage gap" predicts gaps in both engines but on different
chains (r near 0).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr


def _load(run):
    pr = np.load(os.path.join(run, "pairs.npz"), mmap_mode="r")
    meta = {"pid": np.asarray(pr["pid"]), "var": np.asarray(pr["variant"]), "libs": np.asarray(pr["libs"])}
    for name in ("pairs_preds.npz", "pairs_preds_D.npz"):
        p = os.path.join(run, name)
        if os.path.exists(p):
            P = np.load(p)
            break
    res = json.load(open(os.path.join(run, "pairs_results.json" if os.path.exists(os.path.join(run, "pairs_preds.npz")) else "pairs_results_D.json")))
    return meta, P, set(res["train_pids"])


def _chain_stats(pred, y, pid, var, rng):
    acc, half = {}, {}
    for key in set(zip(pid.tolist(), var.tolist())):
        idx = np.nonzero((pid == key[0]) & (var == key[1]))[0]
        ok = pred[idx] == y[idx]
        acc[key] = ok.mean()
        perm = rng.permutation(len(idx))
        a, b = perm[: len(idx) // 2], perm[len(idx) // 2:]
        half[key] = (ok[a].mean() if len(a) else np.nan, ok[b].mean() if len(b) else np.nan)
    return acc, half


def main(runs, seed=0):
    rng = np.random.default_rng(seed)
    metas, preds, trains = {}, {}, None
    for r in runs:
        m, P, tr = _load(r)
        metas[r], preds[r] = m, P
        trains = tr if trains is None else trains
        assert tr == trains, "train/test pair split differs between runs"
    meta = json.load(open(os.path.join(runs[0], "pairs_meta.json")))
    lib_of = {m["pid"]: m["libs_cyclic"] for m in meta}
    matched = {m["pid"] for m in meta if m["libs_cyclic"] == m["libs_cut"]}
    # runs may hold different subsets of pairs (slow engines extract fewer
    # training pairs); test = held-out matched pairs present in every run
    present = set.intersection(*[set(metas[r]["pid"].tolist()) for r in runs])
    test = sorted(p for p in present if p not in trains and p in matched)

    report = {"n_test_pairs": len(test), "engines": {}, "pairs": {}}
    gaps = {}
    for r in runs:
        P = preds[r]
        pid, var = metas[r]["pid"], metas[r]["var"]
        y = np.clip(metas[r]["libs"], 1, 8) - 1
        trainm = sorted(p for p in set(pid.tolist()) if p in trains and p in matched)
        keys = [k for k in P.files if k.startswith("D/")]
        best, best_acc = None, -1
        for k in keys:
            acc, _ = _chain_stats(P[k], y, pid, var, rng)
            a = np.mean([acc[(p, v)] for p in trainm for v in (0, 1)])
            if a > best_acc:
                best, best_acc = k, a
        acc, half = _chain_stats(P[best], y, pid, var, np.random.default_rng(1))
        g = np.array([acc[(p, 1)] - acc[(p, 0)] for p in test])
        h1 = np.array([half[(p, 1)][0] - half[(p, 0)][0] for p in test])
        h2 = np.array([half[(p, 1)][1] - half[(p, 0)][1] for p in test])
        ok = np.isfinite(h1) & np.isfinite(h2)
        rh = spearmanr(h1[ok], h2[ok]).correlation
        rel = 2 * rh / (1 + rh) if rh > 0 else rh
        gaps[r] = g
        bs = rng.choice(g, size=(4000, len(g))).mean(1)
        report["engines"][os.path.basename(r)] = {"layer": best.split("/")[1], "train_acc": round(float(best_acc), 3),
                                                  "gap_mean": round(float(g.mean()), 3),
                                                  "gap_ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                                  "gap_reliability": round(float(rel), 3)}
    strata = np.array([lib_of[p] for p in test])
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = gaps[runs[i]], gaps[runs[j]]
            r_obs = spearmanr(a, b).correlation
            null = []
            for _ in range(2000):
                bb = b.copy()
                for s in np.unique(strata):
                    m = strata == s
                    bb[m] = rng.permutation(bb[m])
                null.append(spearmanr(a, bb).correlation)
            null = np.array(null)
            ra = report["engines"][os.path.basename(runs[i])]["gap_reliability"]
            rb = report["engines"][os.path.basename(runs[j])]["gap_reliability"]
            ceiling = float(np.sqrt(max(ra, 0) * max(rb, 0)))
            report["pairs"][f"{os.path.basename(runs[i])}~{os.path.basename(runs[j])}"] = {
                "r": round(float(r_obs), 3), "p_within_liberty_strata": round(float((null >= r_obs).mean()), 4),
                "ceiling": round(ceiling, 3), "r_over_ceiling": round(float(r_obs / ceiling), 3) if ceiling > 0 else None}
    # by liberty count, per engine
    report["gap_by_libs"] = {os.path.basename(r): {str(L): round(float(gaps[r][strata == L].mean()), 3) for L in range(2, 9) if (strata == L).any()} for r in runs}
    report["n_by_libs"] = {str(L): int((strata == L).sum()) for L in range(2, 9)}
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1:])
    print(json.dumps(rep, indent=1))

"""Cross-engine analysis over finished runs.

1. Depth curves per concept (relative depth so 18 and 40 block nets line up).
2. Size-matched cyclic test: per-chain liberty accuracy on cyclic chains vs
   non-cyclic chains of 10+ stones in the same cases, bucketed by liberties.
3. Stage two: error correlation between engines on the identical held-out
   natural points, at each engine's best layer, against the independence
   baseline.
4. Policy mass on CORRECT moves in the cyclic answer cases.

Usage: python -m goconcepts.compare runs/humansl runs/lz
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def load(run):
    r = json.load(open(os.path.join(run, "results.json")))
    P = np.load(os.path.join(run, "preds.npz"))
    nat = np.load(os.path.join(run, "natural.npz"), mmap_mode="r")
    cyc = np.load(os.path.join(run, "cyclic.npz"), mmap_mode="r")
    return r, P, nat, cyc


def test_mask(nat):
    rng = np.random.default_rng(0)
    g = nat["game"]; ug = np.unique(g); rng.shuffle(ug)
    return np.isin(g, ug[: len(ug) // 5])


def best_layer(r, concept, key="test_acc"):
    rows = [x for x in r["concepts"][concept] if x["layer"] not in ("input",)]
    return max(rows, key=lambda x: x[key])["layer"]


def chain_sizes(d):
    if "chain_size" in d.files:
        return np.asarray(d["chain_size"])
    key = d["pos"].astype(np.int64) * 1000 + d["chain"]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv]


def size_matched(r, P, cyc, layer):
    yc = np.clip(np.asarray(cyc["libs"]), 1, 8) - 1
    cm = np.asarray(cyc["cyclic"]) == 1
    key = cyc["pos"].astype(np.int64) * 1000 + np.asarray(cyc["chain"])
    size = chain_sizes(cyc)
    pc = P[f"libs/{layer}/cyc"]
    out = {}
    for b in range(1, 8):  # 2..8+ liberties
        row = {}
        for name, m in (("cyclic", cm), ("large_noncyclic", (~cm) & (size >= 10))):
            mm = m & (yc == b)
            chains = np.unique(key[mm])
            if len(chains):
                row[name] = (float(np.mean([(pc[key == c] == b).mean() for c in chains])), int(len(chains)))
        out[f"{b + 1}{'+' if b == 7 else ''}"] = row
    return out


def error_corr(runs, concept="libs"):
    """Phi coefficient between engines' error indicators on identical points."""
    ref = runs[0]
    te = test_mask(ref[2])
    y = np.clip(np.asarray(ref[2]["libs"])[te], 1, 8) - 1
    errs = {}
    for name, (r, P, nat, _) in runs.items() if isinstance(runs, dict) else []:
        pass
    return y, te


def main(paths):
    runs = {os.path.basename(p.rstrip("/")): load(p) for p in paths}
    names = list(runs)
    # identical points check
    base = runs[names[0]][2]
    for n in names[1:]:
        o = runs[n][2]
        assert np.array_equal(base["pos"], o["pos"]) and np.array_equal(base["yx"], o["yx"]), f"{n}: sampled points differ"
    report = {}

    # 1. depth curves
    for n, (r, P, nat, cyc) in runs.items():
        layers = r["layers"]
        deep = [l for l in layers if l.startswith("block")]
        rep = {"n_blocks": len(deep)}
        for c in r["concepts"]:
            key = "test_acc" if c == "libs" else "test_bal"
            rows = {x["layer"]: x for x in r["concepts"][c]}
            rep[c] = {"input": rows["input"][key],
                      "best": max(x[key] for x in rows.values()),
                      "best_layer": max(rows.values(), key=lambda x: x[key])["layer"],
                      "curve": [(round((i + 1) / len(deep), 3), round(rows[l][key], 3)) for i, l in enumerate(deep)]}
        rep["libs_4plus_best"] = max(x["test_acc_4plus"] for x in r["concepts"]["libs"])
        pm = r.get("policy_correct_mass", {})
        rep["policy_correct_mass_mean"] = float(np.mean(list(pm.values()))) if pm else None
        report[n] = rep

    # 2. size matched, at best liberty layer and at the last layer
    for n, (r, P, nat, cyc) in runs.items():
        bl = best_layer(r, "libs")
        last = [l for l in r["layers"] if l.startswith("block")][-1]
        report[n]["size_matched"] = {bl: size_matched(r, P, cyc, bl), last: size_matched(r, P, cyc, last)}

    # 3. error correlation on liberties, natural held-out, best layer each
    te = test_mask(base)
    y = np.clip(np.asarray(base["libs"])[te], 1, 8) - 1
    err = {}
    for n, (r, P, nat, cyc) in runs.items():
        bl = best_layer(r, "libs")
        err[n] = P[f"libs/{bl}/test"] != y
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = err[names[i]], err[names[j]]
            both = float((a & b).mean()); indep = float(a.mean() * b.mean())
            phi = float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])
            pairs[f"{names[i]}~{names[j]}"] = {"err_rate": [float(a.mean()), float(b.mean())], "both_wrong": both,
                                               "both_wrong_if_independent": indep, "phi": phi}
    report["stage2_libs_error_correlation"] = pairs

    # cyclic chains: per-chain error correlation at best layers
    yc = np.clip(np.asarray(base and runs[names[0]][3]["libs"]), 1, 8) - 1
    cm = np.asarray(runs[names[0]][3]["cyclic"]) == 1
    cerr = {n: (P[f"libs/{best_layer(r, 'libs')}/cyc"] != yc) for n, (r, P, nat, cyc) in runs.items()}
    cp = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = cerr[names[i]][cm], cerr[names[j]][cm]
            cp[f"{names[i]}~{names[j]}"] = {"err_rate": [float(a.mean()), float(b.mean())],
                                            "both_wrong": float((a & b).mean()), "both_wrong_if_independent": float(a.mean() * b.mean()),
                                            "phi": float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])}
    report["stage2_libs_error_correlation_cyclic_stones"] = cp
    return report


if __name__ == "__main__":
    rep = main(sys.argv[1:])
    print(json.dumps(rep, indent=1))

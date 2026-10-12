"""Where does "loop" become "alive"? Per-block writes along the safety verdict.

  python -m goconcepts.verdict_path            # observational decomposition, g170 / b40 / tf3

For each net, from the mean trunk state on the D chain's stones after a D pass
(the state the ownership verdict is read from):
- verdict axis at layer k: unit vector from doomed edge groups to safe edge groups
  (edge positions only, so it carries no ring/edge geometry);
- position of doomed rings on that axis, scaled so doomed edge = 0 and safe edge = 1:
  s_k = (p(doomed ring) - p(doomed edge)) / (p(safe edge) - p(doomed edge)).
  s near 0: the net places doomed rings with doomed edge groups (correct);
  s near 1: with safe groups (the loop reads as alive).
- write decomposition: with w = the final layer's verdict axis, each block's update
  (difference of consecutive trunk states) projected on w, doomed rings minus
  doomed edges, in the same units. These terms sum exactly to s_final - s_0 along w.
Directions are fit on one half of the positions and measured on the other
(tf3: its after-pass states exist for one half only, split again 50/50).
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from .mech_cross import trunk_means

NETS = ("g170", "b40", "tf3")


def analyse(net, cases, geo, st, seed=0):
    T = trunk_means(net, cases, "pass")
    ok = ~np.isnan(T[:, 0, 0])
    rng = np.random.default_rng(seed)
    split = np.full(len(cases), -1)
    ids = np.nonzero(ok)[0]
    split[ids] = rng.permutation(len(ids)) % 2
    fit, ev = split == 0, split == 1
    nL = T.shape[1]
    groups = {"de": (geo == 1) & (st == 0), "se": (geo == 1) & (st == 1),
              "dr": (geo == 0) & (st == 0), "sr": (geo == 0) & (st == 1)}
    s = {"doomed_ring": [], "safe_ring": []}
    axes = []
    for k in range(nL):
        v = T[fit & groups["se"], k].mean(0) - T[fit & groups["de"], k].mean(0)
        v /= np.linalg.norm(v); axes.append(v)
        p = {g: float((T[ev & m, k] @ v).mean()) for g, m in groups.items()}
        gap = p["se"] - p["de"]
        s["doomed_ring"].append(round((p["dr"] - p["de"]) / gap, 3))
        s["safe_ring"].append(round((p["sr"] - p["de"]) / gap, 3))
    w = axes[-1]
    pe = {g: float((T[ev & m, -1] @ w).mean()) for g, m in groups.items()}
    gap = pe["se"] - pe["de"]
    writes = []
    for k in range(1, nL):
        d = T[:, k] - T[:, k - 1]
        writes.append(round(float(((d[ev & groups["dr"]] @ w).mean() - (d[ev & groups["de"]] @ w).mean()) / gap), 3))
    start = round(float(((T[ev & groups["dr"], 0] @ w).mean() - (T[ev & groups["de"], 0] @ w).mean()) / gap), 3)
    # the same writes for safe rings (to separate "loop -> alive" from "race -> alive")
    writes_safe = []
    for k in range(1, nL):
        d = T[:, k] - T[:, k - 1]
        writes_safe.append(round(float(((d[ev & groups["sr"]] @ w).mean() - (d[ev & groups["se"]] @ w).mean()) / gap), 3))
    return {"n_eval": int(ev.sum()), "position_on_verdict_axis": s,
            "final_axis_start": start, "writes_doomed_ring_minus_doomed_edge": writes,
            "writes_safe_ring_minus_safe_edge": writes_safe,
            "final_position_doomed_ring": round(start + sum(writes), 3)}


def patch(net, window, eval_limit=0, seed=0):
    """Causal version. At each block k in `window`, on the D chain's stones, replace
    the component of the block's update along that layer's verdict axis with the
    mean update edge groups of the same status get along it. This removes whatever
    the block writes onto the verdict axis because the chain is a ring, and keeps
    what it writes because of the race outcome. Readouts as in mech_cross."""
    import torch
    from .mech_cross import Runner, summarize
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(seed).permutation(len(cases)) % 2     # same split as mech_cross
    r = Runner(net)
    need = np.nonzero(half == 0)[0]
    U, ref = {}, {}
    for state in ("move", "pass"):
        T = trunk_means(net, cases, state, r, need)
        fit = half == 0
        U[state], ref[state] = {}, {}
        for k in window:
            v = T[fit & (geo == 1) & (st == 1), k].mean(0) - T[fit & (geo == 1) & (st == 0), k].mean(0)
            v /= np.linalg.norm(v)
            d = T[:, k] - T[:, k - 1]
            U[state][k] = v
            ref[state][k] = {s_: float((d[fit & (geo == 1) & (st == s_)] @ v).mean()) for s_ in (0, 1)}

    def hold(state, idx_b):
        stb = st[list(idx_b)]
        tg = {k: torch.tensor([ref[state][k][int(x)] for x in stb], dtype=torch.float32) for k in window}
        return (U[state], tg, "update")
    idx = []
    for g in (0, 1):
        for s_ in (0, 1):
            ii = [i for i in range(len(cases)) if half[i] == 1 and geo[i] == g and st[i] == s_]
            idx += ii[:eval_limit] if eval_limit else ii
    d0 = np.load(f"runs/{net}/status_bal.npz")
    base = summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], geo[idx], st[idx])
    cap, own = r.evaluate(cases, idx, hold)
    res = summarize(cap, own, geo[idx], st[idx])
    return base, res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", help="net:lo-hi[,lo-hi...] to run the causal patch instead")
    ap.add_argument("--eval-limit", type=int, default=0)
    a = ap.parse_args()
    if a.patch:
        net, spec = a.patch.split(":")
        out = {}
        for w in spec.split(","):
            lo, hi = (int(x) for x in w.split("-"))
            base, res = patch(net, list(range(lo, hi + 1)), a.eval_limit)
            out[w] = res
            print(net, w, "baseline", {g: (v["own_after_pass"], v["capture"]) for g, v in base.items()}, flush=True)
            print(net, w, "patched ", {g: (v["own_after_pass"], v["capture"]) for g, v in res.items()}, flush=True)
        out["baseline"] = base
        json.dump(out, open(f"runs/verdict_patch_{net}_{spec.replace(',', '_')}.json", "w"), indent=1)
        return
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    rep = {}
    for net in NETS:
        r = rep[net] = analyse(net, cases, geo, st)
        print(f"== {net} (n eval {r['n_eval']})")
        print("  doomed rings on the verdict axis (0 = doomed edge, 1 = safe edge), by layer:")
        print("  ", r["position_on_verdict_axis"]["doomed_ring"])
        print("  safe rings:", r["position_on_verdict_axis"]["safe_ring"])
        print("  writes along the final verdict axis, doomed ring - doomed edge, per block:")
        print("  ", r["writes_doomed_ring_minus_doomed_edge"], " start", r["final_axis_start"], " end", r["final_position_doomed_ring"])
    json.dump(rep, open("runs/verdict_path.json", "w"), indent=1)


if __name__ == "__main__":
    main()

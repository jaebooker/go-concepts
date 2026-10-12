"""Swap hold of the closure direction in the convolutional nets, to compare with tf3.

  python -m goconcepts.closure_swap --net g170 --blocks 4-14
  python -m goconcepts.closure_swap --net b40 --blocks 7-29

On the D stones of the eye-balanced set (eval half, 30 per cell), set the closure
component to the fit-half mean of the other geometry with the same status: rings get
an edge group's closure level, edge groups get a ring's. Closure directions from the
minimal pairs (closure_hold.closure_dirs). Control: random directions, same targets.
Same design as tf3_closure.hold, so the three nets compare directly.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from .closure_hold import closure_dirs
from .mech_cross import Runner, trunk_means, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="g170")
    ap.add_argument("--blocks", default="4-14")
    ap.add_argument("--eval-limit", type=int, default=30)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(0).permutation(len(cases)) % 2; fit = half == 0
    r = Runner(a.net)
    U = closure_dirs(r, blocks)
    T = {s: trunk_means(a.net, cases, s, r) for s in ("move", "pass")}
    idx = []
    for g in (0, 1):
        for s in (0, 1):
            ii = [i for i in range(len(cases)) if half[i] == 1 and geo[i] == g and st[i] == s]
            idx += ii[:a.eval_limit]

    def make(dirs):
        tg = {s: {k: np.zeros(len(cases), np.float32) for k in blocks} for s in dirs}
        for s in dirs:
            for k in blocks:
                for g in (0, 1):
                    for stt in (0, 1):
                        tg[s][k][(geo == g) & (st == stt)] = float(T[s][fit & (geo == 1 - g) & (st == stt), k].mean(0) @ dirs[s][k])

        def h(state, ids):
            return ({k: dirs[state][k] for k in blocks}, {k: torch.tensor(tg[state][k][ids]) for k in blocks})
        return h

    d0 = np.load(f"runs/{a.net}/status_bal.npz")
    rep = {"net": a.net, "blocks": a.blocks, "n_eval": len(idx),
           "baseline": summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], geo[idx], st[idx])}
    cap, own = r.evaluate(cases, idx, make(U))
    rep["swap_closure"] = summarize(cap, own, geo[idx], st[idx])
    rng = np.random.default_rng(3)
    Ur = {s: {k: (lambda v: v / np.linalg.norm(v))(rng.normal(size=len(U[s][k]))) for k in blocks} for s in U}
    cap, own = r.evaluate(cases, idx, make(Ur))
    rep["swap_random"] = summarize(cap, own, geo[idx], st[idx])
    for k in ("baseline", "swap_closure", "swap_random"):
        print(k, {g: (v["own_after_pass"], v["capture"]) for g, v in rep[k].items()}, flush=True)
    json.dump(rep, open(f"runs/closure_swap_{a.net}_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

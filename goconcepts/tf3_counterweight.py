"""Where is tf3's death evidence on doomed rings?

  python -m goconcepts.tf3_counterweight hold --blocks 3-9 --eval-limit 30

The loop direction (safe ring - safe edge, fit half) leans toward death in tf3 while its
closure part (minimal pairs) leans away (seventeenth run). Split the loop direction per
layer into its closure component and the rest, r = u_loop - (u_loop . c) c, normalised.
Hold r on the D stones at the doomed-edge level. If the counterweight lives in r, doomed
rings lose it but keep their closure, so they should read alive. Safe rings should move
little. Edge groups are already at edge level along r.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from .mech_cross import Runner, trunk_means, summarize

torch.set_num_threads(2)


def residual_dirs(cases, geo, st, fit, state, blocks):
    T = trunk_means("tf3", cases, state)
    d = np.load("runs/tf3/closure_dirs.npz")
    C = d["move"] if state == "move" else d["passed"]
    U, tgt, info = {}, {}, {}
    de = fit & (geo == 1) & (st == 0)
    for k in blocks:
        u = T[fit & (geo == 0) & (st == 1), k].mean(0) - T[fit & (geo == 1) & (st == 1), k].mean(0); u /= np.linalg.norm(u)
        c = C[k] / np.linalg.norm(C[k])
        r = u - (u @ c) * c; r /= np.linalg.norm(r)
        death = T[fit & (geo == 1) & (st == 0), k].mean(0) - T[fit & (geo == 1) & (st == 1), k].mean(0); death /= np.linalg.norm(death)
        U[k] = r; tgt[k] = float(T[de, k].mean(0) @ r)
        p = T[fit, k] @ r; sd = p.std()
        pr = lambda m: float((T[fit & m, k] @ r).mean())
        info[k] = {"cos_loop_closure": round(float(u @ c), 2), "cos_residual_death": round(float(r @ death), 2),
                   "doomed_ring_minus_doomed_edge_sd": round((pr((geo == 0) & (st == 0)) - pr((geo == 1) & (st == 0))) / sd, 2),
                   "safe_ring_minus_safe_edge_sd": round((pr((geo == 0) & (st == 1)) - pr((geo == 1) & (st == 1))) / sd, 2)}
    return U, tgt, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["info", "hold"])
    ap.add_argument("--blocks", default="3-9")
    ap.add_argument("--eval-limit", type=int, default=30)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(0).permutation(len(cases)) % 2; fit = half == 0
    U, tgt, info = {}, {}, {}
    for s in ("move", "pass"):
        U[s], tgt[s], info[s] = residual_dirs(cases, geo, st, fit, s, blocks)
    for s in info:
        print(s)
        for k, v in info[s].items():
            print("  block", k, v)
    if a.cmd == "info":
        return
    idx = []
    for g in (0, 1):
        for s in (0, 1):
            ii = [i for i in range(len(cases)) if half[i] == 1 and geo[i] == g and st[i] == s]
            idx += ii[:a.eval_limit]
    d0 = np.load("runs/tf3/status_bal.npz")
    rep = {"blocks": a.blocks, "info": {s: {str(k): v for k, v in info[s].items()} for s in info},
           "baseline": summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], geo[idx], st[idx])}
    print("baseline", rep["baseline"], flush=True)
    r = Runner("tf3")
    cap, own = r.evaluate(cases, idx, (U, tgt), batch=8)
    rep["hold_residual"] = summarize(cap, own, geo[idx], st[idx]); print("hold residual", rep["hold_residual"], flush=True)
    json.dump(rep, open(f"runs/tf3_counterweight_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

"""Does the pure closure signal drive the verdict? Hold it, instead of the ring - edge
loop direction, on the eye-balanced set.

  python -m goconcepts.closure_hold --net b40 --blocks 12-24

The ring - edge direction used in the eleventh run mixes loop closure with other
geometry (ring stones sit mid-board, edge-pocket stones near the edge). The minimal
pairs isolate closure: cyclic and cut members differ by one stone. Here the direction
per layer is the mean (cyclic - cut) trunk state on ring stones at chain distance >= 5
from the cut, computed separately for pairs where the ring's owner is to move (used on
the to-move forward) and where the opponent is (used on the after-pass forward), so each
matches KataGo's side-relative inputs. The hold then sets the D stones' component along
it to the doomed-edge level, at the chosen blocks, exactly as mech_cross does. Control:
random directions.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from .cyclegen import generate
from .katago_adapter import gamestate_from_game, board_array
from .loopcircuit import geometry, trunk_states
from .mech_cross import Runner, trunk_means, summarize


def closure_dirs(runner, blocks):
    model = runner.model; net = model.net
    pairs = [p for p in generate(400, seed=12345) if p.libs_cyclic == p.libs_cut]
    acc = {"move": np.zeros((net.n_blocks + 1, net.C)), "pass": np.zeros((net.n_blocks + 1, net.C))}
    for s in range(0, len(pairs), 8):
        ps = pairs[s:s + 8]
        e1 = [model.encode(gamestate_from_game(p.cyclic)) for p in ps]
        e2 = [model.encode(gamestate_from_game(p.cut)) for p in ps]
        g = torch.tensor(np.stack([e[1] for e in e1]))
        S1 = trunk_states(net, torch.tensor(np.stack([e[0] for e in e1])), g)
        S2 = trunk_states(net, torch.tensor(np.stack([e[0] for e in e2])), g)
        for j, p in enumerate(ps):
            gm = geometry(p)
            m = gm["ring"] >= 5
            if not m.any():
                continue
            gs = gamestate_from_game(p.cyclic)
            owner = int(board_array(gs)[p.ring_point])
            state = "move" if owner == int(gs.board.pla) else "pass"
            for k in blocks:
                acc[state][k] += (S1[k][j] - S2[k][j]).numpy()[:, m].mean(1)
    return {s: {k: acc[s][k] / np.linalg.norm(acc[s][k]) for k in blocks} for s in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="b40")
    ap.add_argument("--blocks", default="12-24")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(0).permutation(len(cases)) % 2
    r = Runner(a.net)
    U = closure_dirs(r, blocks)
    de = (half == 0) & (geo == 1) & (st == 0)
    tgt, cos = {}, {}
    for state in ("move", "pass"):
        T = trunk_means(a.net, cases, state, r)
        tgt[state] = {k: float(T[de, k].mean(0) @ U[state][k]) for k in blocks}
        # how far rings sit from edges along the closure direction, in pooled SDs (safe positions)
        cos[state] = {}
        for k in blocks:
            p = T[:, k] @ U[state][k]
            sr, se = (geo == 0) & (st == 1), (geo == 1) & (st == 1)
            cos[state][f"block{k}"] = round(float((p[sr].mean() - p[se].mean()) / p[sr | se].std()), 2)
    idx = [i for i in range(len(cases)) if half[i] == 1]
    d0 = np.load(f"runs/{a.net}/status_bal.npz")
    rep = {"net": a.net, "blocks": a.blocks,
           "rings_minus_edges_along_closure_dir_sd": cos,
           "baseline": summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], geo[idx], st[idx])}
    cap, own = r.evaluate(cases, idx, (U, tgt))
    rep["hold_closure_direction"] = summarize(cap, own, geo[idx], st[idx])
    rng = np.random.default_rng(2)
    Ur = {s: {} for s in U}; tr = {s: {} for s in U}
    for state in U:
        T = trunk_means(a.net, cases, state, r)
        for k in blocks:
            v = rng.normal(size=len(U[state][k])); v /= np.linalg.norm(v)
            Ur[state][k] = v; tr[state][k] = float(T[de, k].mean(0) @ v)
    cap, own = r.evaluate(cases, idx, (Ur, tr))
    rep["hold_random"] = summarize(cap, own, geo[idx], st[idx])
    for k in ("baseline", "hold_closure_direction", "hold_random"):
        print(k, {g: (v["own_after_pass"], v["capture"]) for g, v in rep[k].items()}, flush=True)
    print("rings - edges along the closure direction (SD), pass state:", cos["pass"])
    json.dump(rep, open(f"runs/closure_hold_{a.net}_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

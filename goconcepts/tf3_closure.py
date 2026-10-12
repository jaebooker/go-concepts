"""Does tf3 read closure as danger?

  python -m goconcepts.tf3_closure dirs            # closure directions from the minimal pairs (slow: ~260 forwards)
  python -m goconcepts.tf3_closure cos             # cosines with the loop direction and the death axis
  python -m goconcepts.tf3_closure hold --blocks 2-8 --eval-limit 40

1. Closure direction per layer and state: mean (cyclic - cut) trunk on ring stones at
   chain distance >= 5, on the 131 filtered minimal pairs (equal liberties, cut member
   given the cyclic member's global inputs, no input difference beyond 2 points of the
   cut). "move" from pairs where the ring's owner is to move, "pass" from the others, to
   match KataGo's side-relative inputs.
2. Cosines, per layer, with the loop direction (safe ring - safe edge) and the death
   axis (doomed edge - safe edge) on the eye-balanced set (fit half), and the doomed -
   safe separation of edge groups along the closure direction, in pooled SDs.
3. Swap hold on the eye-balanced set, eval half: on the D stones, set the closure
   component to the fit-half mean of the *other* geometry with the same status. Rings
   lose their closure (edge level), edge groups get a ring's closure. If tf3 reads closure
   as danger, rings should read more alive and edge groups more dead. Control: random
   directions with the same swap targets.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from .cyclegen import generate
from .katago_adapter import gamestate_from_game, board_array
from .loopcircuit import geometry
from .mech_cross import Runner, trunk_means, summarize

torch.set_num_threads(2)
CASES = "data/status_bal.pkl"


def closure_dirs():
    r = Runner("tf3"); model = r.model
    names = [n for n in model.layer_names if n != "input"]
    pairs = [p for p in generate(400, seed=12345) if p.libs_cyclic == p.libs_cut]
    keep = []
    for p in pairs:
        gm = geometry(p)
        e1 = model.encode(gamestate_from_game(p.cyclic)); e2 = model.encode(gamestate_from_game(p.cut))
        cy, cx = gm["cut"]
        dif = np.argwhere((e1[0] != e2[0]).any(0))
        if all(max(abs(y - cy), abs(x - cx)) <= 2 for y, x in dif) and (gm["ring"] >= 5).any():
            gs = gamestate_from_game(p.cyclic)
            state = "move" if int(board_array(gs)[p.ring_point]) == int(gs.board.pla) else "pass"
            keep.append((e1, (e2[0],) + tuple(e1[1:]), gm["ring"] >= 5, state))
    print(len(keep), "pairs;", sum(k[3] == "move" for k in keep), "move /", sum(k[3] == "pass" for k in keep), "pass", flush=True)
    acc = {s: np.zeros((len(names), 768)) for s in ("move", "pass")}; cnt = {"move": 0, "pass": 0}
    for s in range(0, len(keep), 6):
        ch = keep[s:s + 6]; B = len(ch)
        acts = model.activations([c[0] for c in ch] + [c[1] for c in ch], layers=names)
        for j, c in enumerate(ch):
            for k, nm in enumerate(names):
                d = acts[nm][j].astype(np.float32) - acts[nm][B + j].astype(np.float32)
                acc[c[3]][k] += d[:, c[2]].mean(1)
            cnt[c[3]] += 1
        print(f"  {min(s + 6, len(keep))}/{len(keep)}", flush=True)
    U = {s: acc[s] / np.linalg.norm(acc[s], axis=1, keepdims=True) for s in acc}
    np.savez("runs/tf3/closure_dirs.npz", move=U["move"], passed=U["pass"], n_move=cnt["move"], n_pass=cnt["pass"])


def load_dirs():
    d = np.load("runs/tf3/closure_dirs.npz")
    return {"move": d["move"], "pass": d["passed"]}


def cells():
    cases = pickle.load(open(CASES, "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(0).permutation(len(cases)) % 2
    return cases, geo, st, half


def cos():
    U = load_dirs(); cases, geo, st, half = cells(); fit = half == 0
    c = {"dr": (geo == 0) & (st == 0), "de": (geo == 1) & (st == 0), "sr": (geo == 0) & (st == 1), "se": (geo == 1) & (st == 1)}
    rep = {}
    for state in ("move", "pass"):
        T = trunk_means("tf3", cases, state)
        rows = []
        for k in range(T.shape[1]):
            u = U[state][k]
            loop = T[fit & c["sr"], k].mean(0) - T[fit & c["se"], k].mean(0); loop /= np.linalg.norm(loop)
            death = T[fit & c["de"], k].mean(0) - T[fit & c["se"], k].mean(0); death /= np.linalg.norm(death)
            p = T[fit, k] @ u; sd = p.std()
            pr = lambda m: float((T[fit & m, k] @ u).mean())
            rows.append({"layer": k, "cos_loop": round(float(u @ loop), 2), "cos_death": round(float(u @ death), 2),
                         "edge_doomed_minus_safe_sd": round((pr(c["de"]) - pr(c["se"])) / sd, 2),
                         "ring_doomed_minus_safe_sd": round((pr(c["dr"]) - pr(c["sr"])) / sd, 2),
                         "safe_ring_minus_safe_edge_sd": round((pr(c["sr"]) - pr(c["se"])) / sd, 2)})
        rep[state] = rows
        print(state)
        for r_ in rows:
            print("  ", r_)
    other = {}
    for state in ("move", "pass"):
        other[state] = [round(float(U["move"][k] @ U["pass"][k]), 2) for k in range(U["move"].shape[0])]
    rep["cos_move_vs_pass_closure"] = other["move"]
    json.dump(rep, open("runs/tf3_closure_cos.json", "w"), indent=1)


def hold(blocks, limit, only_random=False):
    U = load_dirs(); cases, geo, st, half = cells(); fit = half == 0
    r = Runner("tf3")
    idx = []
    for g in (0, 1):
        for s in (0, 1):
            ii = [i for i in range(len(cases)) if half[i] == 1 and geo[i] == g and st[i] == s]
            idx += ii[:limit]
    idx = np.array(idx)
    T = {s: trunk_means("tf3", cases, s) for s in ("move", "pass")}

    def make(dirs):
        # per case target: fit-half mean of the other geometry, same status
        tg = {s: {k: np.zeros(len(cases), np.float32) for k in blocks} for s in dirs}
        for s in dirs:
            for k in blocks:
                for g in (0, 1):
                    for stt in (0, 1):
                        lvl = float(T[s][fit & (geo == 1 - g) & (st == stt), k].mean(0) @ dirs[s][k])
                        tg[s][k][(geo == g) & (st == stt)] = lvl

        def h(state, ids):
            return ({k: dirs[state][k] for k in blocks}, {k: torch.tensor(tg[state][k][ids]) for k in blocks})
        return h

    d0 = np.load("runs/tf3/status_bal.npz")
    rep = {"blocks": blocks, "n_eval": len(idx),
           "baseline": summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], geo[idx], st[idx])}
    print("baseline", rep["baseline"], flush=True)
    Ud = {s: {k: U[s][k] for k in blocks} for s in U}
    if only_random:
        import ast, re
        line = [l for l in open("runs/tf3_closure_hold_part1.log") if l.startswith("swap closure")][0]
        rep["swap_closure"] = ast.literal_eval(line[len("swap closure "):])
    else:
        cap, own = r.evaluate(cases, list(idx), make(Ud), batch=8)
        rep["swap_closure"] = summarize(cap, own, geo[idx], st[idx]); print("swap closure", rep["swap_closure"], flush=True)
    rng = np.random.default_rng(3)
    Ur = {s: {} for s in U}
    for s in U:
        for k in blocks:
            v = rng.normal(size=U[s].shape[1]); Ur[s][k] = v / np.linalg.norm(v)
    cap, own = r.evaluate(cases, list(idx), make(Ur), batch=8)
    rep["swap_random"] = summarize(cap, own, geo[idx], st[idx]); print("swap random", rep["swap_random"], flush=True)
    json.dump(rep, open(f"runs/tf3_closure_hold_{blocks[0]}-{blocks[-1]}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dirs", "cos", "hold"])
    ap.add_argument("--blocks", default="2-8")
    ap.add_argument("--eval-limit", type=int, default=40)
    ap.add_argument("--only-random", action="store_true", help="reuse the swap-closure result from the log")
    a = ap.parse_args()
    if a.cmd == "dirs":
        closure_dirs()
    elif a.cmd == "cos":
        cos()
    else:
        lo, hi = (int(v) for v in a.blocks.split("-"))
        hold(list(range(lo, hi + 1)), a.eval_limit, a.only_random)


if __name__ == "__main__":
    main()

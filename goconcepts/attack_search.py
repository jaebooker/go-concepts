"""Search for doomed loops that tf3 reads as alive.

  python -m goconcepts.attack_search run --gens 50
  python -m goconcepts.attack_search report

Start from every doomed loop we have (eye-balanced set and shape set). Mutate the
board a few points at a time near the race (shape of the ring, its liberties, the
surrounding stones) and occasionally anywhere. Keep a mutant only if it is still
an exactly solved, closed race that D loses after a pass. Score = tf3's ownership of
the D chain after D passes (+1 = alive). Elitist search: population 12, 8 mutants per
parent per generation, at most 3 survivors per seed so it doesn't collapse onto one.

Closed race (so the exact label is right): every empty point in the region (ring and
inner liberties) has all its empty neighbours in the region; no D stone outside the
ring touches the region; no A stone outside the inner group touches a region point
next to the inner group; other A chains touching the region keep 2+ liberties outside
it; no chain other than ring and inner group is in atari.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time

import numpy as np
import torch

from . import labels as L
from .cyclegen import N, NB4, _cyclic, _to_game
from .katago_adapter import KataGoProbeModel, gamestate_from_game, board_array
from .racegen import solve, _Unresolved
from .status_eval import _d_color
from .paths import NETS_DIR

TF3 = NETS_DIR + "/kata1-tf3-b11c768-s11001M-d5973M.ckpt"
STATE = "runs/attack_search_state.pkl"
SEEDS = "runs/attack_search_seeds.pkl"


WALL_MARGIN = 2      # every wall chain needs ring liberties + 2 liberties (race_ext.fortify)


def seeds():
    from .race_ext import fortify
    out = []
    for path, tag in (("data/status_bal_fortified.pkl", "bal"), ("data/status_shapes.pkl", "shape")):
        for c in pickle.load(open(path, "rb")):
            if c.status != 0 or c.variant != 0:
                continue
            gs = gamestate_from_game(c.game)
            b = board_array(gs).astype(np.int8)
            if tag == "shape":
                b = fortify(b, _d_color(c), tuple(c.ring_point), tuple(c.inner_point))
                if b is None:
                    continue
            recent = [tuple(p) for _, p in c.game.moves]
            out.append({"b": b, "D": _d_color(c), "rp": tuple(c.ring_point), "ip": tuple(c.inner_point),
                        "recent": recent, "root": f"{tag}{c.pid}", "shape": getattr(c, "shape", "std"), "edits": 0})
    return out


class _Slow(Exception):
    pass


def validate(b, D, rp, ip, budget=1):
    """validate_ with a time budget in seconds for the exact solver (slow positions are rejected)."""
    import signal

    def h(*_):
        raise _Slow()
    old = signal.signal(signal.SIGALRM, h); signal.alarm(budget)
    try:
        return validate_(b, D, rp, ip)
    except _Slow:
        return None
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)


def validate_(b, D, rp, ip):
    """Returns info dict if b is a closed race that D loses after a pass, else None."""
    A = 3 - D
    if b[rp] != D or b[ip] != A:
        return None
    ch = L.find_chains(b)
    if any(len(ch.libs[k]) == 0 for k in range(len(ch.color))):
        return None
    kr = ch.chain_id[rp[0] * N + rp[1]]; ki = ch.chain_id[ip[0] * N + ip[1]]
    if kr not in _cyclic(b, ch):
        return None
    ring = {divmod(q, N) for q in ch.stones[kr]}; inner = {divmod(q, N) for q in ch.stones[ki]}
    rl = {divmod(q, N) for q in ch.libs[kr]}; il = {divmod(q, N) for q in ch.libs[ki]}
    region = rl | il
    for k in range(len(ch.color)):
        if k not in (kr, ki) and len(ch.libs[k]) == 1:
            return None
    from .race_ext import wall_chains
    _, _, _, ws = wall_chains(b, D, rp, ip)
    if any(len(ch.libs[k]) < len(rl) + WALL_MARGIN for k in ws):
        return None
    for p in region:
        for dy, dx in NB4:
            q = (p[0] + dy, p[1] + dx)
            if not (0 <= q[0] < N and 0 <= q[1] < N):
                continue
            v = b[q]
            if v == 0 and q not in region:
                return None
            if v == D and q not in ring:
                return None
            if v == A and q not in inner:
                if p in il:
                    return None
                k = ch.chain_id[q[0] * N + q[1]]
                if len({divmod(x, N) for x in ch.libs[k]} - region) < 2:
                    return None
    try:
        win, pass_value, per_move = solve(b, D, rp, ip, {q[0] * N + q[1] for q in region}, max_depth=40)
    except _Unresolved:
        return None
    if pass_value != 0:
        return None
    return {"ring_size": len(ring), "ring_libs": len(rl), "inner_size": len(inner), "inner_libs": len(il),
            "shared": len(rl & il), "savable": bool(win), "correct": [int(m) for m in win],
            "struct": ring | inner | region}


def mutate(rng, s, info):
    b = s["b"].copy()
    protect = set(s["recent"]) | {s["rp"], s["ip"]}
    struct = info["struct"]
    near = sorted({(y + dy, x + dx) for (y, x) in struct for dy in range(-3, 4) for dx in range(-3, 4)
                   if 0 <= y + dy < N and 0 <= x + dx < N} - protect)
    n_edit = int(rng.choice([1, 1, 2, 3]))
    for _ in range(n_edit):
        if rng.random() < 0.1:
            p = (int(rng.integers(N)), int(rng.integers(N)))
            if p in protect:
                continue
        else:
            p = near[int(rng.integers(len(near)))]
        choices = [v for v in (0, 1, 2) if v != b[p]]
        b[p] = choices[int(rng.integers(len(choices)))]
    return b, n_edit


def game_of(s):
    return _to_game(s["b"], s["D"], s["recent"])


def score(model, states):
    enc = []
    for s in states:
        g = game_of(s); g.moves = list(g.moves) + [(s["D"], None)]
        enc.append(model.encode(gamestate_from_game(g)))
    out = np.zeros(len(states))
    for i in range(0, len(enc), 16):
        a = model.activations(enc[i:i + 16], layers=[])
        for j, s in enumerate(states[i:i + 16]):
            ch = L.find_chains(s["b"]); cid = ch.chain_id.reshape(N, N)
            m = cid == cid[s["rp"]]
            out[i + j] = -float(a["ownership"][j][m].mean())      # A to move after the pass; D's side
    return out


def run(a):
    rng = np.random.default_rng(a.seed)
    model = KataGoProbeModel(TF3)
    if os.path.exists(STATE):
        st = pickle.load(open(STATE, "rb"))
        pop, log, gen0, n_eval = st["pop"], st["log"], st["gen"] + 1, st["n_eval"]
        print(f"resuming at generation {gen0}", flush=True)
    else:
        S = seeds()
        for s in S:
            s["info"] = validate(s["b"], s["D"], s["rp"], s["ip"])
        S = [s for s in S if s["info"] is not None]
        sc = score(model, S)
        for s, v in zip(S, sc):
            s["score"] = float(v); s["seed_score"] = float(v)
        n_eval = len(S)
        print(f"{len(S)} valid seeds; seed scores: max {sc.max():+.2f}, mean {sc.mean():+.2f}, "
              f"{(sc > 0).mean():.0%} read alive", flush=True)
        pickle.dump({"seeds": [{k: v for k, v in s.items() if k != "info"} for s in S]}, open("runs/attack_search_seeds.pkl", "wb"))
        pop = select(S, a.pop)
        log, gen0 = [], 0
    for gen in range(gen0, a.gens):
        t0 = time.time(); kids = []
        for s in pop:
            got, tries = 0, 0
            while got < a.kids and tries < 80:
                tries += 1
                b, ne = mutate(rng, s, s["info"])
                info = validate(b, s["D"], s["rp"], s["ip"])
                if info is None:
                    continue
                k = {**{x: s[x] for x in ("D", "rp", "ip", "recent", "root", "shape", "seed_score")},
                     "b": b, "info": info, "edits": s["edits"] + ne, "parent_score": s["score"]}
                kids.append(k); got += 1
        if kids:
            sc = score(model, kids)
            for k, v in zip(kids, sc):
                k["score"] = float(v)
            n_eval += len(kids)
        pop = select(pop + kids, a.pop)
        best = pop[0]
        log.append({"gen": gen, "best": best["score"], "mean_pop": float(np.mean([p["score"] for p in pop])),
                    "n_alive_pop": int(sum(p["score"] > 0 for p in pop)), "n_kids": len(kids), "n_eval": n_eval,
                    "secs": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        pickle.dump({"pop": pop, "log": log, "gen": gen, "n_eval": n_eval}, open(STATE, "wb"))


def select(cands, k, per_root=3):
    cands = sorted(cands, key=lambda s: -s["score"])
    out, cnt = [], {}
    for s in cands:
        if cnt.get(s["root"], 0) >= per_root:
            continue
        # drop exact duplicates
        if any(np.array_equal(s["b"], o["b"]) for o in out):
            continue
        out.append(s); cnt[s["root"]] = cnt.get(s["root"], 0) + 1
        if len(out) == k:
            break
    return out


def show(s):
    D = s["D"]; rows = []
    ch = L.find_chains(s["b"]); cid = ch.chain_id.reshape(N, N); ring = cid == cid[s["rp"]]
    for y in range(N):
        rows.append("".join("X" if ring[y, x] else ("x" if s["b"][y, x] == D else ("o" if s["b"][y, x] else "."))
                            for x in range(N)))
    return rows


def report(a):
    st = pickle.load(open(STATE, "rb"))
    pop = st["pop"]
    rep = {"log": st["log"], "n_eval": st["n_eval"], "found": []}
    for s in pop:
        seed = next((x for x in pickle.load(open("runs/attack_search_seeds.pkl", "rb"))["seeds"] if x["root"] == s["root"]), None)
        diff = [tuple(map(int, p)) for p in np.argwhere(seed["b"] != s["b"])] if seed else None
        info = {k: v for k, v in s["info"].items() if k != "struct"}
        rep["found"].append({"root": s["root"], "shape": s["shape"], "tf3_after_pass": round(s["score"], 3),
                             "seed_tf3_after_pass": round(s["seed_score"], 3), "edits_applied": s["edits"],
                             "points_changed_from_seed": diff, **info, "board": show(s)})
    json.dump(rep, open("runs/attack_search.json", "w"), indent=1)
    for f in rep["found"]:
        print(f["root"], f["shape"], "tf3", f["tf3_after_pass"], "seed", f["seed_tf3_after_pass"],
              "changed", len(f["points_changed_from_seed"] or []), "ring", f["ring_size"], "libs", f["ring_libs"],
              "savable", f["savable"])


def flip(a):
    """Per seed that tf3 reads confidently dead: beam search (width 3, 12 mutants per
    parent per step) for the fewest board edits that make tf3 read the loop alive.
    Records the tf3 score after each step and the step and edit count at which it
    first crosses 0 and 0.5."""
    rng = np.random.default_rng(a.seed)
    model = KataGoProbeModel(TF3)
    S = pickle.load(open("runs/attack_search_seeds.pkl", "rb"))["seeds"]
    S = sorted([s for s in S if s["score"] < -0.9], key=lambda s: s["root"])[:a.n_seeds]
    out_path = "runs/attack_flip.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for s0 in S:
        if s0["root"] in res:
            continue
        s0 = dict(s0); s0["info"] = validate(s0["b"], s0["D"], s0["rp"], s0["ip"], budget=20)
        if s0["info"] is None:
            continue
        beam, traj, first0, first5, best = [s0], [s0["score"]], None, None, s0
        t0 = time.time()
        for step in range(1, a.steps + 1):
            kids = []
            for s in beam:
                got, tries = 0, 0
                while got < a.kids and tries < 60:
                    tries += 1
                    b, ne = mutate(rng, s, s["info"])
                    info = validate(b, s["D"], s["rp"], s["ip"])
                    if info is None:
                        continue
                    kids.append({**{x: s[x] for x in ("D", "rp", "ip", "recent", "root", "shape", "seed_score")},
                                 "b": b, "info": info, "edits": s["edits"] + ne}); got += 1
            if not kids:
                break
            for k, v in zip(kids, score(model, kids)):
                k["score"] = float(v)
            beam = sorted(beam + kids, key=lambda x: -x["score"])[:a.beam]
            best = beam[0]; traj.append(round(best["score"], 3))
            if first0 is None and best["score"] > 0:
                first0 = (step, best["edits"])
            if first5 is None and best["score"] > 0.5:
                first5 = (step, best["edits"]); break
        diff = [tuple(map(int, p)) for p in np.argwhere(s0["b"] != best["b"])]
        res[s0["root"]] = {"shape": s0["shape"], "seed_score": round(s0["score"], 3), "trajectory": traj,
                           "first_alive": first0, "first_above_0.5": first5, "best": round(best["score"], 3),
                           "points_changed": diff, "savable": best["info"]["savable"],
                           "ring_size": best["info"]["ring_size"], "ring_libs": best["info"]["ring_libs"],
                           "board": show(best), "seed_board": show(s0), "secs": round(time.time() - t0)}
        pickle.dump(best, open(f"runs/attack_flip_{s0['root']}.pkl", "wb"))
        json.dump(res, open(out_path, "w"), indent=1)
        print(s0["root"], s0["shape"], "seed", round(s0["score"], 2), "best", round(best["score"], 2),
              "first alive", first0, "first >0.5", first5, "changed", len(diff), f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "flip"])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--gens", type=int, default=50)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--kids", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"run": run, "report": report, "flip": flip}[a.cmd](a)


if __name__ == "__main__":
    main()

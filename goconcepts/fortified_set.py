"""The eye-balanced set with its walls fortified, relabelled, and read by all three nets.

  python -m goconcepts.fortified_set build     # data/status_bal_fortified.pkl
  python -m goconcepts.fortified_set eval --net tf3

The opponent walls round the ring's outside liberties are often weak (2 or 3 liberties),
so D could sometimes save the ring by capturing a wall stone, which the race solver does
not model. Fortify: add opponent stones on wall liberties outside the race until every
wall chain has at least ring liberties + 2. Then re-solve the race and keep the case only
if its status is unchanged (doomed: lost after a pass but saved by the right move; safe:
not lost after a pass).
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys

import numpy as np

from .cyclegen import N, _to_game
from .katago_adapter import gamestate_from_game, board_array
from .mech_cross import Runner, summarize
from .race_ext import fortify
from .racegen import solve, _Unresolved
from .status_eval import _d_color
from . import labels as L

sys.setrecursionlimit(10000)


def build():
    cases = pickle.load(open("data/status_bal.pkl", "rb"))
    out, dropped, added = [], 0, []
    for c in cases:
        D = _d_color(c); rp, ip = tuple(c.ring_point), tuple(c.inner_point)
        b = board_array(gamestate_from_game(c.game)).astype(np.int8)
        fb = fortify(b, D, rp, ip)
        if fb is None:
            dropped += 1; continue
        ch = L.find_chains(fb)
        kr = ch.chain_id[rp[0] * N + rp[1]]; ki = ch.chain_id[ip[0] * N + ip[1]]
        region = set(ch.libs[kr]) | set(ch.libs[ki])
        try:
            win, pv, per = solve(fb, D, rp, ip, region)
        except _Unresolved:
            dropped += 1; continue
        status = 0 if (win and pv == 0) else (1 if pv == 2 else None)
        if status != c.status:
            dropped += 1; continue
        c2 = copy.deepcopy(c)
        c2.game = _to_game(fb, D, [tuple(p) for _, p in c.game.moves])
        c2.correct = [int(m) for m in win]
        c2.fortified_stones = int((fb != b).sum())
        added.append(c2.fortified_stones)
        c2.pid = len(out); c2.orig_pid = c.pid
        out.append(c2)
    pickle.dump(out, open("data/status_bal_fortified.pkl", "wb"))
    print(f"{len(out)} kept, {dropped} dropped; stones added median {np.median(added)}, max {max(added)}")


def evaluate(net):
    cases = pickle.load(open("data/status_bal_fortified.pkl", "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    r = Runner(net)
    cap, own = r.evaluate(cases, list(range(len(cases))), None, batch=16)
    np.savez(f"runs/{net}/status_bal_fortified.npz", cap=cap, own=own)
    s = summarize(cap, own, geo, st)
    for k, v in s.items():
        v["frac_read_alive"] = round(float((own[(geo == (0 if "ring" in k else 1)) & (st == (0 if "doomed" in k else 1))] > 0).mean()), 3)
    print(net, json.dumps(s))
    json.dump(s, open(f"runs/status_bal_fortified_{net}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--net", default="tf3")
    a = ap.parse_args()
    build() if a.cmd == "build" else evaluate(a.net)

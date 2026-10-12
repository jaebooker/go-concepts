"""Doomed vs safe groups at matched liberty counts.

Label: what happens to the D chain if D plays elsewhere (exact search, racegen.solve).
  doomed  the ring is captured after a D pass; some D move still wins
          (these are racegen's "urgent" races)
  safe    after a D pass, D still captures the inner group
Seki outcomes are excluded.

Cells: geometry (ring / edge pocket) x status (doomed / safe) x D-chain
liberties (2..5). Status depends on the whole race (the inner group's
liberties and eyes), not the D chain's liberty count alone, so within a cell
liberties can't give the status away.
"""
from __future__ import annotations

import time

import numpy as np

from .racegen import make_race


def _combos(L):
    return [(s, e, L - s) for s in range(1, L + 1) for e in range(0, 3) if L - s >= 0 and s + e + (L - s) <= 9]


def generate(n_per_cell, seed=31, libs=(2, 3, 4, 5), time_cap=240):
    rng = np.random.default_rng(seed)
    out, report = [], {}
    for L in libs:
        for edge in (False, True):
            for want in ("urgent", "safe"):
                got, t0 = 0, time.time()
                combos = _combos(L)
                while got < n_per_cell and time.time() - t0 < time_cap:
                    combo = combos[int(rng.integers(len(combos)))]
                    c = make_race(rng, len(out), edge=edge, max_tries=20, combo=combo, want=want, target_libs=L)
                    if c is not None:
                        c.status = 0 if want == "urgent" else 1   # 0 doomed, 1 safe
                        out.append(c); got += 1
                report[(L, int(edge), want)] = got
                print(f"  libs {L} {'edge' if edge else 'ring'} {want}: {got}", flush=True)
    return out, report


# Eye-balanced set.
#
# In data/status.pkl status tracks the inner group's eyes: doomed cells mostly
# have an inner eye, safe cells mostly none. A first fix that matched the
# nominal race parameters (shared, eyes, outside) is kept as
# data/status_bal_nominal.pkl, but it did not remove the confound: within a
# matched cell the status flipped on whether the "eye" point touched the shared
# liberty, i.e. whether the eye was real. Here every inner group has exactly
# one real eye (its point is surrounded by inner stones only) and the D chain
# has the same liberty count in both classes. What flips the status is how D's
# liberties split between shared and outside ones. With an eye against no eye,
# shared liberties count for the eyed side: D (to move) wins iff
# outside >= shared + 1, and survives a pass iff outside >= shared + 2. So the
# "doomed" class (lost after a pass, saved by the right move, as in the main
# set) needs outside = shared + 1 exactly, which fixes odd liberty counts:
#   libs 5: doomed (2,1,3)   safe (1,1,4)
#   libs 7: doomed (3,1,4)   safe (2,1,5),(1,1,6)
# (shared, eyes, outside). Inner-group liberties are shared + 1, so they differ
# by class; the D chain's own liberties and the eye structure do not.
# Exact search checks every case.
BALANCED_CELLS = [(5, [(2, 1, 3)], [(1, 1, 4)]),
                  (7, [(3, 1, 4)], [(2, 1, 5), (1, 1, 6)])]


def generate_balanced(n_per_cell, seed=77, time_cap=900):
    rng = np.random.default_rng(seed)
    out = []
    for L, doomed_combos, safe_combos in BALANCED_CELLS:
        for edge in (False, True):
            got = {}
            for want, combos in (("urgent", doomed_combos), ("safe", safe_combos)):
                cell, t0 = [], time.time()
                while len(cell) < n_per_cell and time.time() - t0 < time_cap:
                    combo = combos[int(rng.integers(len(combos)))]
                    c = make_race(rng, 0, edge=edge, max_tries=20, combo=combo, want=want,
                                  target_libs=L, real_eyes=True)
                    if c is not None:
                        c.status = 0 if want == "urgent" else 1
                        cell.append(c)
                got[want] = cell
            k = min(len(got["urgent"]), len(got["safe"]))   # equal counts per cell
            for c in got["urgent"][:k] + got["safe"][:k]:
                c.pid = len(out); out.append(c)
            print(f"  libs {L} {'edge' if edge else 'ring'}: {k} doomed + {k} safe "
                  f"(found {len(got['urgent'])}/{len(got['safe'])})", flush=True)
    return out


# Hopeless loops: the inner group has two real eyes and shares no liberty with
# the D chain, so it lives unconditionally; the D chain's liberties are all
# outside (3 to 5), so it dies whatever D does. Nothing about the race is close.
# A net that reads these rings as alive isn't misreading a race: it is treating
# the enclosed group as dead because it is enclosed.
def generate_hopeless(n_per_cell, seed=5, outside=(3, 4, 5), time_cap=600):
    rng = np.random.default_rng(seed)
    out = []
    for r in outside:
        for edge in (False, True):
            cell, t0 = [], time.time()
            while len(cell) < n_per_cell and time.time() - t0 < time_cap:
                c = make_race(rng, 0, edge=edge, max_tries=20, combo=(0, 2, r), want="dead", real_eyes=True)
                if c is not None:
                    c.status = 0
                    cell.append(c)
            for c in cell:
                c.pid = len(out); out.append(c)
            print(f"  outside libs {r} {'edge' if edge else 'ring'}: {len(cell)}", flush=True)
    return out


# Size-varied eye-balanced set (libs 5 cells only), to test whether the depth at
# which g170 flips a doomed ring to "safe" grows with the ring's size, as it
# would if the net had to propagate "this chain closes a loop" across the board.
SIZE_BINS = [(4, 10), (10, 20), (20, 35), (35, 60)]


def generate_sized(n_per_cell, seed=91, time_cap=400):
    rng = np.random.default_rng(seed)
    out = []
    for sr in SIZE_BINS:
        for edge in (False, True):
            got = {}
            for want, combo in (("urgent", (2, 1, 3)), ("safe", (1, 1, 4))):
                cell, t0 = [], time.time()
                while len(cell) < n_per_cell and time.time() - t0 < time_cap:
                    c = make_race(rng, 0, edge=edge, max_tries=20, combo=combo, want=want, target_libs=5,
                                  real_eyes=True, size_range=sr)
                    if c is not None:
                        c.status = 0 if want == "urgent" else 1
                        c.size_bin = sr
                        cell.append(c)
                got[want] = cell
            k = min(len(got["urgent"]), len(got["safe"]))
            for c in got["urgent"][:k] + got["safe"][:k]:
                c.pid = len(out); out.append(c)
            print(f"  region {sr} {'edge' if edge else 'ring'}: {k} + {k}", flush=True)
    return out

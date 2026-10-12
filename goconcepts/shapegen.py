"""Doomed loops with unusual shapes, to test whether tf3's fix generalises.

  python -m goconcepts.shapegen            # writes data/status_shapes.pkl

Same race construction and exact labels as the eye-balanced set (libs 5: doomed
combo (2,1,3), safe (1,1,4), one real eye), with the D ring changed:
  std    the standard thin ring (fresh seed, a within-set control)
  thick  a ring two stones wide
  tail   a ring with an arm of 4-6 stones sticking out
  edge1  a ring that runs along the first line
  big    a standard ring round a larger region (ring of about 30-45 stones)
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from .racegen import make_race

SHAPES = {"std": {}, "thick": {"shape": "thick"}, "tail": {"shape": "tail"},
          "edge1": {"shape": "edge1"}, "big": {"size_range": (18, 32)}}


def generate(n_doomed=20, n_safe=10, seed=2026, time_cap=600):
    rng = np.random.default_rng(seed)
    out = []
    for name, kw in SHAPES.items():
        for want, combo, n in (("urgent", (2, 1, 3), n_doomed), ("safe", (1, 1, 4), n_safe)):
            cell, t0 = [], time.time()
            while len(cell) < n and time.time() - t0 < time_cap:
                c = make_race(rng, 0, edge=False, max_tries=20, combo=combo, want=want, target_libs=5,
                              real_eyes=True, **kw)
                if c is not None:
                    c.status = 0 if want == "urgent" else 1
                    c.shape = name
                    c.pid = len(out)
                    cell.append(c); out.append(c)
            print(f"  {name:6s} {want:6s}: {len(cell)} (ring sizes {sorted(c.ring_size for c in cell)[:3]}..{max([c.ring_size for c in cell] or [0])})", flush=True)
    return out


if __name__ == "__main__":
    cases = generate(*(int(v) for v in sys.argv[1:3]))
    pickle.dump(cases, open("data/status_shapes.pkl", "wb"))
    print(len(cases), "cases")

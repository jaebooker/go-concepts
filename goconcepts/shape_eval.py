"""Readouts on the shape-variant loops (data/status_shapes.pkl).

  python -m goconcepts.shape_eval --net tf3
Per shape and status: ownership of the D chain after D passes (+1 = alive) and the
policy mass on the correct (capturing) moves when D is to move.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch

from .mech_cross import Runner

torch.set_num_threads(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="tf3")
    a = ap.parse_args()
    cases = pickle.load(open("data/status_shapes.pkl", "rb"))
    r = Runner(a.net)
    import os
    part = f"runs/{a.net}/shapes_partial.npz"
    cap = np.full(len(cases), np.nan); own = np.full(len(cases), np.nan)
    if os.path.exists(part):
        d = np.load(part); cap, own = d["cap"], d["own"]
    for s in range(0, len(cases), 10):
        idx = list(range(s, min(s + 10, len(cases))))
        if not np.isnan(own[idx]).any():
            continue
        c_, o_ = r.evaluate(cases, idx, None, batch=10)
        cap[idx] = c_; own[idx] = o_
        np.savez(part, cap=cap, own=own)
        print(f"  {idx[-1] + 1}/{len(cases)}", flush=True)
    np.savez(f"runs/{a.net}/shapes.npz", cap=cap, own=own)
    rep = {}
    for sh in dict.fromkeys(c.shape for c in cases):
        for s, sn in ((0, "doomed"), (1, "safe")):
            m = np.array([c.shape == sh and c.status == s for c in cases])
            rep[f"{sh}_{sn}"] = {"n": int(m.sum()), "own_after_pass": round(float(own[m].mean()), 3),
                                 "frac_read_alive": round(float((own[m] > 0).mean()), 3),
                                 "capture": round(float(cap[m].mean()), 3)}
            print(a.net, sh, sn, rep[f"{sh}_{sn}"], flush=True)
    json.dump(rep, open(f"runs/shapes_{a.net}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

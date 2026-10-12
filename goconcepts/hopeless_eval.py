"""Ownership of the ring and of the enclosed group on hopeless loops
(data/status_hopeless.pkl): inner group has two real eyes, ring has only
outside liberties. Truth: ring dead (-1), inner group alive (from D's side -1).

  python -m goconcepts.hopeless_eval --engine kgbin --ckpt ... --out runs/g170/hopeless.json
"""
import argparse, copy, json, pickle
import numpy as np
from . import labels as L
from .katago_adapter import gamestate_from_game, board_array
from .race_eval import _model
from .status_eval import _d_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine"); ap.add_argument("--ckpt"); ap.add_argument("--out")
    ap.add_argument("--cases", default="data/status_hopeless.pkl")
    a = ap.parse_args()
    cases = pickle.load(open(a.cases, "rb")); model = _model(a.engine, a.ckpt)
    import os
    part = a.out + ".partial"
    rows = json.load(open(part)) if os.path.exists(part) else []
    for s in range(0, len(cases), 12):
        if len(rows) >= min(s + 12, len(cases)):
            continue
        rows = rows[:s]
        chunk = cases[s:s + 12]
        for passed in (False, True):
            gss = []
            for c in chunk:
                g = copy.deepcopy(c.game)
                if passed:
                    g.moves = list(g.moves) + [(_d_color(c), None)]
                gss.append(gamestate_from_game(g))
            acts = model.activations([model.encode(gs) for gs in gss], layers=[])
            own = acts["ownership"]; sign = -1 if passed else 1
            for j, (c, gs) in enumerate(zip(chunk, gss)):
                cid = L.find_chains(board_array(gs)).chain_id.reshape(19, 19)
                ring = cid == cid[c.ring_point]; inner = cid == cid[c.inner_point]
                if not passed:
                    rows.append({"geo": c.variant, "libs": c.ring_libs})
                r = rows[s + j]
                key = "pass" if passed else "move"
                r[f"ring_{key}"] = float(sign * own[j][ring].mean())
                r[f"inner_{key}"] = float(sign * own[j][inner].mean())   # D's side: + = inner dead
        json.dump(rows, open(part, "w"))
        print(f"  {min(s + 12, len(cases))}/{len(cases)}", flush=True)
    out = {}
    for g, gn in ((0, "ring"), (1, "edge")):
        m = [r for r in rows if r["geo"] == g]
        out[gn] = {k: round(float(np.mean([r[k] for r in m])), 3) for k in ("ring_pass", "inner_pass", "ring_move", "inner_move")}
        out[gn]["ring_read_alive_after_pass"] = round(float(np.mean([r["ring_pass"] > 0 for r in m])), 3)
        out[gn]["by_libs_ring_pass"] = {L_: round(float(np.mean([r["ring_pass"] for r in m if r["libs"] == L_])), 3) for L_ in sorted({r["libs"] for r in m})}
    json.dump({"summary": out, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

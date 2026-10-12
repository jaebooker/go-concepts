"""Group-safety readouts on doomed vs safe positions (data/status.pkl).

  python -m goconcepts.status_eval extract --engine kgbin --ckpt ... --run runs/g170
  python -m goconcepts.status_eval score runs/humansl runs/lz runs/g170 runs/tf3

Label per position: status 0 doomed (the D chain is captured if D plays
elsewhere), 1 safe. D is to move in every position.

Readouts, cheapest first:
  ownership after a pass   the counterfactual that defines the label: D passes,
             A is to move, and the net predicts end-of-game ownership of the D
             chain (from D's side: +1 D keeps it, -1 it is captured). A net that
             knows the chain is doomed says -1 for doomed, +1 for safe. KataGo
             nets only; Leela Zero has no ownership head.
  value after a pass       D's win probability in the same counterfactual. All
             engines. Noisy (the rest of the board is random), so it is compared
             doomed vs safe within a geometry.
  (ownership and value with D to move are also stored. They reflect the
  outcome the net expects from its own play, not danger, so they are not the
  safety readout.)
  probe      a linear doomed/safe probe on the D chain's stones, per block.
             Trained on edge positions, tested on ring positions ("does a ring
             look like an edge group with the same fate?"), and cross-validated
             within rings ("is ring status decodable at all?").
Behaviour: policy mass on the capturing moves in doomed positions.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle

import numpy as np
from scipy.stats import rankdata

from .extract import _labels_for
from .katago_adapter import gamestate_from_game
from .probes import LinearProbe
from .race_eval import _model

STATUS = os.path.join(os.path.dirname(__file__), "..", "data", "status.pkl")


def extract(model, cases, batch=12):
    per = {k: [] for k in ("own_d", "own_inner", "value", "mass_capture", "own_d_after_pass", "value_after_pass")}
    rows_case, acts_out, pol = [], [], []
    names = None
    for s in range(0, len(cases), batch):
        chunk = cases[s:s + batch]
        gss = [gamestate_from_game(c.game) for c in chunk]
        acts = model.activations([model.encode(gs) for gs in gss])
        names = [n for n in model.layer_names if n in acts and n != "input"]
        C = max(acts[n].shape[1] for n in names)
        for j, (c, gs) in enumerate(zip(chunk, gss)):
            lab = _labels_for(gs)
            cid = lab["chain_id"]
            dmask = cid == cid[c.ring_point]
            imask = cid == cid[c.inner_point]
            own = acts.get("ownership")
            per["own_d"].append(float(own[j][dmask].mean()) if own is not None else np.nan)
            per["own_inner"].append(float(own[j][imask].mean()) if own is not None else np.nan)
            v = acts["value"][j]
            per["value"].append(float(v[0]) if np.ndim(v) else float((v + 1) / 2))  # LZ: tanh -> [0,1]
            p = np.exp(acts["policy"][j])
            per["mass_capture"].append(float(p[c.correct].sum()))
            pol.append(acts["policy"][j])
            ys, xs = np.nonzero(dmask)
            for y, x in zip(ys, xs):
                vec = np.zeros((len(names), C), np.float16)
                for li, nm in enumerate(names):
                    a = acts[nm][j, :, y, x]
                    vec[li, :a.shape[0]] = a
                acts_out.append(vec)
                rows_case.append(s + j)
        # counterfactual: D plays elsewhere (passes), A to move. Ownership and
        # value are then from A's side; flip them back to D's.
        from copy import deepcopy
        passed = []
        for c in chunk:
            g = deepcopy(c.game); g.moves = list(g.moves) + [(_d_color(c), None)]
            passed.append(gamestate_from_game(g))
        ap = model.activations([model.encode(gs) for gs in passed], layers=[])
        for j, (c, gs) in enumerate(zip(chunk, passed)):
            cid = _labels_for(gs)["chain_id"]
            dmask = cid == cid[c.ring_point]
            own = ap.get("ownership")
            per["own_d_after_pass"].append(float(-own[j][dmask].mean()) if own is not None else np.nan)
            v = ap["value"][j]
            per["value_after_pass"].append(1 - (float(v[0]) if np.ndim(v) else float((v + 1) / 2)))
        del acts, ap
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        print(f"  {min(s + batch, len(cases))}/{len(cases)} positions", flush=True)
    out = {k: np.asarray(v, np.float32) for k, v in per.items()}
    out["case"] = np.asarray(rows_case)
    out["acts"] = np.stack(acts_out)
    out["policy"] = np.asarray(pol, np.float32)
    out["layer_names"] = np.array(names)
    return out


def _d_color(c):
    """D (the side to move) is the colour of the chain at ring_point."""
    b = np.zeros((19, 19), np.int8)
    for p in c.game.setup_black: b[p] = 1
    for p in c.game.setup_white: b[p] = 2
    for col, p in c.game.moves: b[p] = col
    return int(b[c.ring_point])


def _auc(score, y):
    """P(score of a positive > score of a negative); y=1 positive."""
    r = rankdata(score)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _boot_diff(a, b, rng, n=4000):
    d = a.mean() - b.mean()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [round(float(d), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def probe_status(d, geo, status, libs, epochs=200, seed=0):
    """Per block: train on edge, test on ring; and 5-fold CV within ring.
    Scores are per case: mean predicted P(safe) over the D chain's stones."""
    rng = np.random.default_rng(seed)
    case = d["case"]; names = list(d["layer_names"])
    blocks = [n for n in names if n.startswith("block")]
    if len(blocks) > 22:
        blocks = [n for n in blocks if int(n[5:]) % 2 == 0]
    ys = status[case]
    out = []
    ncase = len(status)
    for n in blocks:
        li = names.index(n)
        X = d["acts"][:, li].astype(np.float32)
        tr = geo[case] == 1
        pr = LinearProbe(2, epochs=epochs).fit(X[tr], ys[tr], balance=True)
        p_safe = _proba(pr, X)
        score_x = np.array([p_safe[case == c].mean() for c in range(ncase)])
        ring = geo == 0
        # within-ring CV (folds by position)
        folds = rng.permutation(np.nonzero(ring)[0]) % 5
        fold_of = np.full(ncase, -1); fold_of[np.nonzero(ring)[0]] = rng.permutation(np.arange(ring.sum()) % 5)
        score_cv = np.full(ncase, np.nan)
        for f in range(5):
            te_c = fold_of == f
            tr_mask = ring[case] & ~te_c[case]
            pc = LinearProbe(2, epochs=epochs).fit(X[tr_mask], ys[tr_mask], balance=True)
            ps = _proba(pc, X)
            for c in np.nonzero(te_c)[0]:
                score_cv[c] = ps[case == c].mean()
        row = {"layer": n,
               "edge_to_ring_auc": _auc(score_x[ring], status[ring]),
               "edge_to_ring_doomed_called_safe": float((score_x[ring & (status == 0)] > 0.5).mean()),
               "edge_to_ring_safe_called_doomed": float((score_x[ring & (status == 1)] < 0.5).mean()),
               "edge_heldout_note": "edge scores are in-sample",
               "ring_cv_auc": _auc(score_cv[ring], status[ring]),
               "ring_auc_within_libs": {str(L): _auc(score_cv[ring & (libs == L)], status[ring & (libs == L)]) for L in sorted(set(libs))}}
        out.append(row)
        print("   ", n, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ("layer", "edge_heldout_note", "ring_auc_within_libs")}, flush=True)
    return out


def _proba(probe, X):
    import torch
    with torch.no_grad():
        Z = (torch.tensor(X) - probe.mu) / probe.sd
        return torch.softmax(probe.lin(Z), -1)[:, 1].numpy()


def score(runs, out_path, cases_path=STATUS, npz="status.npz"):
    rng = np.random.default_rng(0)
    cases = pickle.load(open(cases_path, "rb"))
    geo = np.array([c.variant for c in cases]); status = np.array([c.status for c in cases])
    libs = np.array([c.ring_libs for c in cases])
    rep = {"n_cases": len(cases), "engines": {}}
    for run in runs:
        name = os.path.basename(run)
        d = np.load(os.path.join(run, npz))
        e = {}
        for key in ("own_d_after_pass", "value_after_pass", "own_d", "value"):
            v = d[key]
            if np.all(np.isnan(v)):
                continue
            cell = {}
            for g, gname in ((0, "ring"), (1, "edge")):
                m = geo == g
                cell[gname] = {"doomed": round(float(v[m & (status == 0)].mean()), 3),
                               "safe": round(float(v[m & (status == 1)].mean()), 3),
                               "safe_minus_doomed": _boot_diff(v[m & (status == 1)], v[m & (status == 0)], rng),
                               "auc": round(_auc(v[m], status[m]), 3),
                               "auc_by_libs": {str(L): round(_auc(v[m & (libs == L)], status[m & (libs == L)]), 3) for L in sorted(set(libs))}}
            e[key] = cell
        mc = d["mass_capture"]
        e["capture_in_doomed"] = {"ring": round(float(mc[(geo == 0) & (status == 0)].mean()), 3),
                                  "edge": round(float(mc[(geo == 1) & (status == 0)].mean()), 3)}
        # does the ownership readout predict capture, within doomed ring positions?
        if not np.all(np.isnan(d["own_d_after_pass"])):
            m = (geo == 0) & (status == 0)
            x, y = d["own_d_after_pass"][m], mc[m]
            rx, ry = rankdata(x), rankdata(y)
            for L in sorted(set(libs)):  # remove liberty-level means
                k = libs[m] == L
                rx[k] -= rx[k].mean(); ry[k] -= ry[k].mean()
            r = float(np.corrcoef(rx, ry)[0, 1])
            null = [np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(4000)]
            e["ownership_predicts_capture_doomed_rings"] = {"r": round(r, 3), "p_one_sided_negative": round(float(np.mean(np.array(null) <= r)), 4)}
        e["probe"] = probe_status(d, geo, status, libs)
        rep["engines"][name] = e
    json.dump(rep, open(out_path, "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--engine", default="katago")
    ap.add_argument("--ckpt")
    ap.add_argument("--run")
    ap.add_argument("--out", default="runs/status_report.json")
    ap.add_argument("--cases", default=STATUS, help="data/status.pkl or data/status_bal.pkl (eye-balanced)")
    ap.add_argument("--npz", default="status.npz")
    a = ap.parse_args()
    if a.cmd == "extract":
        cases = pickle.load(open(a.cases, "rb"))
        np.savez(os.path.join(a.run, a.npz), **extract(_model(a.engine, a.ckpt), cases))
    else:
        print(json.dumps(score(a.runs, a.out, a.cases, a.npz), indent=1))


if __name__ == "__main__":
    main()

"""The loop direction across nets: g170 (2020 self-play), b40 (the attacked
net, FAR's "Latest") and tf3 (post-2023, adversarially trained).

  python -m goconcepts.mech_cross --net b40 --blocks 1-24 --fit-on safe
  python -m goconcepts.mech_cross --net tf3 --blocks 1-7  --fit-on safe --eval-limit 80

Same recipe as mech.direction_cmd, made uniform across architectures:
- Direction per block k: unit vector from the mean trunk state on D stones of
  edge groups to that of rings, from the stored to-move activations
  (runs/<net>/status_bal.npz), fit on one half of the positions. fit-on "safe"
  uses safe positions only (same verdict in both geometries: a pure shape
  direction); "doomed" uses doomed positions.
- Hold: on the other half, set the D stones' component along u_k to the
  doomed-edge mean, at every block in the window, in both the to-move and the
  after-pass forward passes. Readouts: ownership of the D chain after a pass
  (D's side) and policy mass on the capture.
- Control: random unit directions, same blocks.
Also reported without any intervention: how far rings sit from edge groups
along u_k, in units of the doomed-vs-safe separation along the same axis, so
nets can be compared on whether the loop feature exists at all.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle

import numpy as np
import torch

from . import labels as L
from .katago_adapter import gamestate_from_game, board_array
from .status_eval import _d_color
from .paths import NETS_DIR

NETS = {"g170": ("kgbin", NETS_DIR + "/g170e-b20.bin.gz"),
        "b40": ("kgbin", NETS_DIR + "/b40-s11840935168.bin.gz"),
        "tf3": ("katago", NETS_DIR + "/kata1-tf3-b11c768-s11001M-d5973M.ckpt")}
CASES = "data/status_bal.pkl"


def trunk_means(net, cases, state, runner=None, need=None):
    """(n_cases, n_layers, C): mean trunk state over the D chain's stones.
    to-move: from the stored activations; after-pass: computed once and cached.
    The two differ a lot: KataGo inputs are relative to the side to move, so
    after D's pass the D stones are the opponent's."""
    if state == "move":
        d = np.load(f"runs/{net}/status_bal.npz")
        case = d["case"]; A = d["acts"]
        T = np.zeros((len(cases), A.shape[1], A.shape[2]), np.float32)
        for c in range(len(cases)):
            T[c] = A[case == c].astype(np.float32).mean(0)
        return T
    path = f"runs/{net}/status_bal_pass_trunk.npy"
    try:
        return np.load(path)
    except FileNotFoundError:
        pass
    runner = runner or Runner(net)
    model = runner.model
    need = np.arange(len(cases)) if need is None else np.asarray(need)
    T = None
    for s in range(0, len(need), 12):
        ids = need[s:s + 12]
        chunk = [cases[i] for i in ids]
        gss, masks = [], []
        for c in chunk:
            g2 = copy.deepcopy(c.game); g2.moves = list(g2.moves) + [(_d_color(c), None)]
            gs = gamestate_from_game(g2); gss.append(gs)
            cid = L.find_chains(board_array(gs)).chain_id.reshape(19, 19)
            masks.append((cid == cid[c.ring_point]).astype(np.float32))
        acts = model.activations([model.encode(gs) for gs in gss])
        names = [n for n in model.layer_names if n != "input"]
        for j, i in enumerate(ids):
            mm = masks[j]
            v = np.stack([(acts[nm][j].astype(np.float32) * mm[None]).sum((1, 2)) / mm.sum() for nm in names])
            if T is None:
                T = np.full((len(cases),) + v.shape, np.nan, np.float32)
            T[i] = v
        print(f"  pass trunk {min(s + 12, len(need))}/{len(need)}", flush=True)
    np.save(path, T)   # rows outside `need` stay NaN
    return T


def directions(T, blocks, fit_on, half, geo, st):
    fit = half == 0
    fs = 0 if fit_on == "doomed" else 1
    fr, fe = fit & (geo == 0) & (st == fs), fit & (geo == 1) & (st == fs)
    de = fit & (geo == 1) & (st == 0)
    U, tgt, stats = {}, {}, {}
    for k in blocks:          # layer index k = trunk after block k (index 0 is trunk0)
        v = T[fr, k].mean(0) - T[fe, k].mean(0)
        u = v / np.linalg.norm(v)
        U[k] = u; tgt[k] = float(T[de, k].mean(0) @ u)
        p = T[:, k] @ u
        ev = (half == 1) & ~np.isnan(p)
        sep = abs(p[ev & (geo == 1) & (st == 1)].mean() - p[ev & (geo == 1) & (st == 0)].mean())
        stats[f"block{k}"] = {
            "ring_minus_edge_doomed": round(float(p[ev & (geo == 0) & (st == 0)].mean() - p[ev & (geo == 1) & (st == 0)].mean()), 3),
            "ring_minus_edge_safe": round(float(p[ev & (geo == 0) & (st == 1)].mean() - p[ev & (geo == 1) & (st == 1)].mean()), 3),
            "edge_safe_minus_doomed": round(float(sep), 3),
            "pooled_sd": round(float(p[ev].std()), 3)}
    return U, tgt, stats


class Runner:
    def __init__(self, net):
        kind, ckpt = NETS[net]
        self.kind = kind
        if kind == "kgbin":
            from .kgbin_adapter import KataGoBinProbeModel
            self.model = KataGoBinProbeModel(ckpt)
        else:
            from .katago_adapter import KataGoProbeModel
            self.model = KataGoProbeModel(ckpt)
            self._hooks = []

    def evaluate(self, cases, idx, hold=None, batch=16):
        """hold: (U, tgt) or None. Returns capture mass and after-pass ownership per idx."""
        cap, own = [], []
        for s in range(0, len(idx), batch):
            chunk = [cases[i] for i in idx[s:s + batch]]
            masks = []
            states = {"move": [], "pass": []}
            for c in chunk:
                gs = gamestate_from_game(c.game)
                g2 = copy.deepcopy(c.game); g2.moves = list(g2.moves) + [(_d_color(c), None)]
                states["move"].append(gs); states["pass"].append(gamestate_from_game(g2))
                cid = L.find_chains(board_array(gs)).chain_id.reshape(19, 19)
                masks.append((cid == cid[c.ring_point]).astype(np.float32))
            m = torch.tensor(np.stack(masks))
            for state in ("move", "pass"):
                if callable(hold):
                    h_ = hold(state, idx[s:s + batch])
                else:
                    h_ = None if hold is None else (hold[0][state], hold[1][state])
                pol, ow = self._forward(states[state], m, h_)
                if state == "move":
                    p = np.exp(pol)
                    cap += [float(p[j, c.correct].sum()) for j, c in enumerate(chunk)]
                else:
                    mm = np.stack(masks)
                    own += list(-(ow * mm).sum((1, 2)) / mm.sum((1, 2)))
        return np.array(cap), np.array(own)

    def _forward(self, gss, m, hold):
        enc = [self.model.encode(gs) for gs in gss]
        if self.kind == "kgbin":
            from .mech import forward
            x = torch.tensor(np.stack([e[0] for e in enc])); g = torch.tensor(np.stack([e[1] for e in enc]))
            ed = None
            if hold is not None:
                U, tgt = hold[0], hold[1]
                on_update = len(hold) > 2 and hold[2] == "update"
                Ut = {k: torch.tensor(v, dtype=torch.float32)[None, :, None, None] for k, v in U.items()}

                def ed(k, h, upd):
                    if k not in Ut:
                        return upd
                    base = upd if on_update else h + upd
                    proj = (base * Ut[k]).sum(1, keepdim=True)
                    t = tgt[k]
                    if torch.is_tensor(t):
                        t = t[:, None, None, None]
                    return upd - m[:, None] * (proj - t) * Ut[k]
            return forward(self.model.net, x, g, ed)
        # KataGo pytorch model: forward hooks on the blocks (each returns its residual)
        hooks = []
        if hold is not None:
            U, tgt = hold[0], hold[1]
            on_update = len(hold) > 2 and hold[2] == "update"
            for k, u in U.items():
                blk = self.model.model.blocks[k - 1]
                ut = torch.tensor(u, dtype=torch.float32)

                def hook(_mod, args, out, ut=ut, t=tgt[k]):
                    h = args[0]
                    base = out if on_update else h + out
                    if out.dim() == 3:        # (B, S, C) sequence layout
                        tt = t[:, None, None] if torch.is_tensor(t) else t
                        mm = m.reshape(m.shape[0], -1)[:, :, None]
                        proj = (base * ut[None, None, :]).sum(-1, keepdim=True)
                        return out - mm * (proj - tt) * ut[None, None, :]
                    tt = t[:, None, None, None] if torch.is_tensor(t) else t
                    mm = m[:, None]
                    proj = (base * ut[None, :, None, None]).sum(1, keepdim=True)
                    return out - mm * (proj - tt) * ut[None, :, None, None]
                hooks.append(blk.register_forward_hook(hook))
        try:
            acts = self.model.activations(enc, layers=[])
        finally:
            for h in hooks:
                h.remove()
        return acts["policy"], acts["ownership"]


def summarize(cap, own, geo, st):
    out = {}
    for g, gn in ((0, "ring"), (1, "edge")):
        for s, sn in ((0, "doomed"), (1, "safe")):
            m = (geo == g) & (st == s)
            if m.sum():
                out[f"{sn}_{gn}"] = {"own_after_pass": round(float(own[m].mean()), 3), "capture": round(float(cap[m].mean()), 3), "n": int(m.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, choices=list(NETS))
    ap.add_argument("--blocks", default="1-12")
    ap.add_argument("--fit-on", default="safe", choices=["safe", "doomed"])
    ap.add_argument("--eval-limit", type=int, default=0, help="per geometry x status cell, on the eval half (0 = all)")
    ap.add_argument("--random", type=int, default=1)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    cases = pickle.load(open(CASES, "rb"))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    half = np.random.default_rng(0).permutation(len(cases)) % 2
    lo, hi = (int(v) for v in a.blocks.split("-"))
    blocks = list(range(lo, hi + 1))
    r = Runner(a.net)
    U, tgt, stats = {}, {}, {}
    need = np.nonzero(half == 0)[0]           # directions and targets use the fit half only
    for state in ("move", "pass"):
        T = trunk_means(a.net, cases, state, r, need)
        U[state], tgt[state], stats[state] = directions(T, blocks, a.fit_on, half, geo, st)
    idx = []
    for g in (0, 1):
        for s in (0, 1):
            ii = [i for i in range(len(cases)) if half[i] == 1 and geo[i] == g and st[i] == s]
            idx += ii[:a.eval_limit] if a.eval_limit else ii
    ev_geo, ev_st = geo[idx], st[idx]
    rep = {"net": a.net, "blocks": a.blocks, "fit_on": a.fit_on, "n_eval": len(idx), "geometry_along_u": stats}
    if a.skip_baseline:   # the unmodified readouts are already stored
        d0 = np.load(f"runs/{a.net}/status_bal.npz")
        rep["baseline"] = summarize(d0["mass_capture"][idx], d0["own_d_after_pass"][idx], ev_geo, ev_st)
        print("baseline (stored)", rep["baseline"], flush=True)
    else:
        cap, own = r.evaluate(cases, idx)
        rep["baseline"] = summarize(cap, own, ev_geo, ev_st); print("baseline", rep["baseline"], flush=True)
    cap, own = r.evaluate(cases, idx, (U, tgt))
    rep["hold_loop_direction"] = summarize(cap, own, ev_geo, ev_st); print("hold", rep["hold_loop_direction"], flush=True)
    rng = np.random.default_rng(1)
    rep["hold_random"] = []
    de = (half == 0) & (geo == 1) & (st == 0)
    Ts = {s_: trunk_means(a.net, cases, s_, r, need) for s_ in ("move", "pass")}
    for _ in range(a.random):
        Ur, tr = {"move": {}, "pass": {}}, {"move": {}, "pass": {}}
        for s_ in ("move", "pass"):
            for k in blocks:
                v = rng.normal(size=Ts[s_].shape[2]); v /= np.linalg.norm(v)
                Ur[s_][k] = v; tr[s_][k] = float(Ts[s_][de, k].mean(0) @ v)
        cap, own = r.evaluate(cases, idx, (Ur, tr))
        rep["hold_random"].append(summarize(cap, own, ev_geo, ev_st)); print("random", rep["hold_random"][-1], flush=True)
    json.dump(rep, open(a.out or f"runs/mech_cross_{a.net}_{a.fit_on}_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

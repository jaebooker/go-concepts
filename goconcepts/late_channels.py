"""Which channels in the last blocks write "alive" for doomed loops, and do they
read the loop feature?  (kgbin nets: b40 blocks 38-40, g170 blocks 19-20)

  python -m goconcepts.late_channels --net b40 --blocks 38-40
  python -m goconcepts.late_channels --net g170 --blocks 19-20

Each residual block computes upd = conv3x3(t, fin), t = relu(bn(conv3x3(relu(bn(h)), reg))).
fin is linear, so the block's write onto the final verdict axis w splits exactly
over t's channels: contribution_j = conv3x3(t_j, sum_c w_c fin[c, j]).

1. Rank: per mid channel, mean contribution on the D chain's stones, doomed rings
   minus doomed edge groups, in units of the final verdict gap (safe edge - doomed
   edge on w). Fit half of the positions. Report how concentrated it is.
2. Ablate: zero the top-k channels' activations t_j on the D chain and its
   neighbours, at the chosen blocks, in both forward passes (to-move and after-pass).
   Eval half. Controls: k random channels; the k channels ranked highest for SAFE
   rings (ring minus edge) instead.
3. Reading the loop: hold the loop direction (safe-fit ring - edge axis, as in
   mech_cross) at the edge level on the D stones at the input of the first chosen
   block only, and measure how much each top channel's activation on the D stones
   drops, against random channels. Also the channels' mean activation by cell.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from . import labels as L
from .katago_adapter import gamestate_from_game, board_array
from .kgbin_adapter import KataGoBinProbeModel, _apply_bn, _cv, _gpool
from .mech_cross import NETS, trunk_means
from .status_eval import _d_color

CASES = "data/status_bal.pkl"


def forward(net, x, g, blocks, capture=None, zero=None, hold=None):
    """capture: dict filled with {k: t} for k in blocks. zero: {k: (mask (B,1,19,19), channel idx tensor)}.
    hold: (k, u (C,), target, mask) applied to h at the input of block k.
    Returns policy log-probs, ownership."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        for i, b in enumerate(net.blocks):
            k = i + 1
            if hold is not None and hold[0] == k:
                _, u, tg, m = hold
                ut = u[None, :, None, None]
                proj = (h * ut).sum(1, keepdim=True)
                h = h - m[:, None] * (proj - tg) * ut
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            if zero is not None and k in zero:
                zm, ch = zero[k]
                t = t.clone()
                t[:, ch] = t[:, ch] * (1 - zm)
            if capture is not None and k in blocks:
                capture[k] = t
            h = h + _cv(t, b["fin"])
        tr = F.relu(_apply_bn(h, net.tip))
        p = _cv(tr, net.p1)
        gp = F.relu(_apply_bn(_cv(tr, net.g1), net.g1bn))
        pooled = _gpool(gp)
        p = F.relu(_apply_bn(p + (pooled @ net.g2b)[:, :, None, None], net.p1bn))
        pol = torch.log_softmax(torch.cat([_cv(p, net.p2)[:, 0].flatten(1), (pooled @ net.g2pass)[:, :1]], 1), -1)
        v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
        own = torch.tanh(_cv(v1, net.own)[:, 0])
        return pol.numpy(), own.numpy()


def load(net_name):
    cases = pickle.load(open(CASES, "rb"))
    model = KataGoBinProbeModel(NETS[net_name][1])
    X = {"move": [], "pass": []}
    masks = []
    for c in cases:
        gs = gamestate_from_game(c.game)
        g2 = copy.deepcopy(c.game); g2.moves = list(g2.moves) + [(_d_color(c), None)]
        X["move"].append(model.encode(gs)); X["pass"].append(model.encode(gamestate_from_game(g2)))
        cid = L.find_chains(board_array(gs)).chain_id.reshape(19, 19)
        masks.append((cid == cid[c.ring_point]).astype(np.float32))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    return cases, model, X, np.stack(masks), geo, st


def batches(n, size=40):
    for s in range(0, n, size):
        yield list(range(s, min(n, s + size)))


def dilate(m):
    t = torch.tensor(m)[:, None]
    return (F.max_pool2d(t, 3, 1, 1) > 0).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="b40", choices=["b40", "g170"])
    ap.add_argument("--blocks", default="38-40")
    ap.add_argument("--ks", default="4,8,16,32")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
    cases, model, X, masks, geo, st = load(a.net)
    net = model.net; n = len(cases); C = net.C
    half = np.random.default_rng(0).permutation(n) % 2
    fit, ev = half == 0, half == 1
    # final verdict axis and loop direction from the stored after-pass trunk means
    T = trunk_means(a.net, cases, "pass")
    nL = T.shape[1] - 1
    w = T[fit & (geo == 1) & (st == 1), nL].mean(0) - T[fit & (geo == 1) & (st == 0), nL].mean(0)
    w /= np.linalg.norm(w)
    pw = T[:, nL] @ w
    gap = pw[fit & (geo == 1) & (st == 1)].mean() - pw[fit & (geo == 1) & (st == 0)].mean()
    kin = lo - 1                                   # trunk index at the input of the first block
    u = T[fit & (geo == 0) & (st == 1), kin].mean(0) - T[fit & (geo == 1) & (st == 1), kin].mean(0)
    u /= np.linalg.norm(u)
    u_tgt = float(T[fit & (geo == 1) & (st == 0), kin].mean(0) @ u)
    wt = torch.tensor(w, dtype=torch.float32)
    K = {k: torch.einsum("c,cjyx->jyx", wt, net.blocks[k - 1]["fin"]["w"])[:, None] for k in blocks}   # (C,1,3,3)

    # 1. per-channel contributions onto w, per case, on D stones (after-pass state)
    contrib = {k: np.zeros((n, C)) for k in blocks}
    act = {k: np.zeros((n, C)) for k in blocks}
    for idx in batches(n):
        x = torch.tensor(np.stack([X["pass"][i][0] for i in idx])); g = torch.tensor(np.stack([X["pass"][i][1] for i in idx]))
        cap = {}
        forward(net, x, g, blocks, capture=cap)
        mm = masks[idx]; den = mm.sum((1, 2))[:, None]
        for k in blocks:
            cm = F.conv2d(cap[k], K[k], padding=1, groups=C).numpy()           # (B,C,19,19)
            contrib[k][idx] = (cm * mm[:, None]).sum((2, 3)) / den
            act[k][idx] = (cap[k].numpy() * mm[:, None]).sum((2, 3)) / den
    cells = {"dr": (geo == 0) & (st == 0), "de": (geo == 1) & (st == 0), "sr": (geo == 0) & (st == 1), "se": (geo == 1) & (st == 1)}
    rank_d = {k: (contrib[k][fit & cells["dr"]].mean(0) - contrib[k][fit & cells["de"]].mean(0)) / gap for k in blocks}
    rank_s = {k: (contrib[k][fit & cells["sr"]].mean(0) - contrib[k][fit & cells["se"]].mean(0)) / gap for k in blocks}
    allc = [(k, j, rank_d[k][j]) for k in blocks for j in range(C)]
    allc.sort(key=lambda r: -r[2])
    total = sum(r[2] for r in allc)
    pos_total = sum(r[2] for r in allc if r[2] > 0)
    cum = np.cumsum([r[2] for r in allc])
    rep = {"net": a.net, "blocks": blocks, "total_ring_minus_edge_doomed": round(float(total), 3),
           "positive_total": round(float(pos_total), 3),
           "share_of_total_top": {str(q): round(float(cum[q - 1] / total), 3) for q in (1, 2, 4, 8, 16, 32, 64)},
           "top_channels": [{"block": k, "ch": j, "doomed_ring_minus_edge": round(float(v), 4),
                             "safe_ring_minus_edge": round(float(rank_s[k][j]), 4),
                             "act_by_cell": {c: round(float(act[k][ev & m, j].mean()), 3) for c, m in cells.items()}}
                            for k, j, v in allc[:16]]}
    print(json.dumps({x: rep[x] for x in ("total_ring_minus_edge_doomed", "positive_total", "share_of_total_top")}))
    for r in rep["top_channels"][:12]:
        print("  ", r)

    # 2. ablation
    def run(zero_sets):
        cap_m, own_p = np.zeros(n), np.zeros(n)
        for idx in batches(n):
            dm = dilate(masks[idx])
            for state in ("move", "pass"):
                x = torch.tensor(np.stack([X[state][i][0] for i in idx])); g = torch.tensor(np.stack([X[state][i][1] for i in idx]))
                zero = {k: (dm, torch.tensor(chs, dtype=torch.long)) for k, chs in zero_sets.items() if len(chs)} if zero_sets else None
                pol, ow = forward(net, x, g, blocks, zero=zero)
                if state == "move":
                    p = np.exp(pol); cap_m[idx] = [p[j, cases[i].correct].sum() for j, i in enumerate(idx)]
                else:
                    mm = masks[idx]; own_p[idx] = -(ow * mm).sum((1, 2)) / mm.sum((1, 2))
        return {c: {"own": round(float(own_p[ev & m].mean()), 3), "capture": round(float(cap_m[ev & m].mean()), 3)} for c, m in cells.items()}

    def as_sets(lst):
        d = {k: [] for k in blocks}
        for k, j in lst:
            d[k].append(j)
        return d
    rep["ablation"] = {"none": run(None)}
    print("none", rep["ablation"]["none"], flush=True)
    rng = np.random.default_rng(3)
    ranked_s = sorted([(k, j, rank_s[k][j]) for k in blocks for j in range(C)], key=lambda r: -r[2])
    for q in [int(v) for v in a.ks.split(",")]:
        top = [(k, j) for k, j, _ in allc[:q]]
        rep["ablation"][f"top{q}"] = run(as_sets(top))
        rnd = [(blocks[int(rng.integers(len(blocks)))], int(rng.integers(C))) for _ in range(q)]
        rep["ablation"][f"random{q}"] = run(as_sets(rnd))
        rep["ablation"][f"topsafe{q}"] = run(as_sets([(k, j) for k, j, _ in ranked_s[:q]]))
        for key in (f"top{q}", f"random{q}", f"topsafe{q}"):
            print(key, rep["ablation"][key], flush=True)

    # 3. do the top channels read the loop direction?
    top16 = [(k, j) for k, j, _ in allc[:16]]
    rnd16 = [(blocks[int(rng.integers(len(blocks)))], int(j)) for j in rng.choice(C, 16, replace=False)]
    base_act = {k: np.zeros((n, C)) for k in blocks}; held_act = {k: np.zeros((n, C)) for k in blocks}
    for idx in batches(n):
        x = torch.tensor(np.stack([X["pass"][i][0] for i in idx])); g = torch.tensor(np.stack([X["pass"][i][1] for i in idx]))
        mm = masks[idx]; den = mm.sum((1, 2))[:, None]
        for store, hold in ((base_act, None), (held_act, (lo, torch.tensor(u, dtype=torch.float32), u_tgt, torch.tensor(mm)))):
            cap = {}
            forward(net, x, g, blocks, capture=cap, hold=hold)
            for k in blocks:
                store[k][idx] = (cap[k].numpy() * mm[:, None]).sum((2, 3)) / den

    def drop(chs, cell):
        m = ev & cells[cell]
        b = np.array([base_act[k][m, j].mean() for k, j in chs]); h = np.array([held_act[k][m, j].mean() for k, j in chs])
        return round(float((b - h).sum() / max(b.sum(), 1e-6)), 3)
    rep["loop_reading"] = {"hold_at_input_of_block": lo,
                           "activation_drop_top16_doomed_ring": drop(top16, "dr"),
                           "activation_drop_random16_doomed_ring": drop(rnd16, "dr"),
                           "activation_drop_top16_safe_ring": drop(top16, "sr"),
                           "activation_drop_random16_safe_ring": drop(rnd16, "sr"),
                           "activation_drop_top16_doomed_edge": drop(top16, "de")}
    print("loop reading", rep["loop_reading"])
    json.dump(rep, open(f"runs/late_channels_{a.net}_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

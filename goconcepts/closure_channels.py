"""Which channels carry the closure wave?  (g170, b40)

  python -m goconcepts.closure_channels --net g170 --upto 6 --late 12
  python -m goconcepts.closure_channels --net b40  --upto 12 --late 24

Same minimal pairs and input hygiene as loopcircuit.wave (equal liberties, cut member
gets the cyclic member's global inputs, no input difference beyond 2 points of the cut).

1. Attribution. Each block writes fin(t) into the trunk, and fin is a linear conv, so the
   block's write along the closure direction splits exactly over its hidden channels t_j.
   Contribution of (block k, channel j) = mean over far ring stones (chain and board
   distance >= 5) of U_k . fin_j(t_j cyclic - t_j cut). Fit half. Also where each top
   channel's difference sits: interior, ring, exterior band, near the cut.
2. Causal test, eval half. In the cyclic member's forward, set the top-n channels (by
   attribution, blocks 1..upto) to the cut member's values at every point. Report the
   closure signal left at far ring stones at layer `upto` and at layer `late`, and the
   ownership difference (cyclic - cut) on far ring stones, from the ring owner's side.
   Control: n random channels spread over the same blocks.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from .cyclegen import generate
from .katago_adapter import gamestate_from_game, board_array
from .kgbin_adapter import KataGoBinProbeModel, _apply_bn, _cv, _gpool
from .loopcircuit import geometry, dist_to_ring
from .mech_cross import NETS

N = 19


def load_pairs(model, n=400):
    pairs = [p for p in generate(n, seed=12345) if p.libs_cyclic == p.libs_cut]
    geo = [geometry(p) for p in pairs]
    enc = [(model.encode(gamestate_from_game(p.cyclic)), model.encode(gamestate_from_game(p.cut))) for p in pairs]
    keep = []
    for i in range(len(pairs)):
        cy, cx = geo[i]["cut"]
        dif = np.argwhere((enc[i][0][0] != enc[i][1][0]).any(0))
        if all(max(abs(y - cy), abs(x - cx)) <= 2 for y, x in dif):
            keep.append(i)
    pairs = [pairs[i] for i in keep]; geo = [geo[i] for i in keep]
    enc = [(enc[i][0], (enc[i][1][0], enc[i][0][1])) for i in keep]
    sign = []
    for p in pairs:
        gs = gamestate_from_game(p.cyclic)
        sign.append(1.0 if int(board_array(gs)[p.ring_point]) == int(gs.board.pla) else -1.0)
    return pairs, geo, enc, np.array(sign)


def forward(net, x, g, capture_t=False, set_t=None):
    """Returns trunk states per layer, hidden t per block (if capture_t), ownership.
    set_t: {k: (chan LongTensor, values (B, n, 19, 19))} replaces t[:, chan] in block k."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        S, T = [h], {}
        for i, b in enumerate(net.blocks):
            k = i + 1
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            if set_t is not None and k in set_t:
                ch, vals = set_t[k]
                t = t.clone(); t[:, ch] = vals
            if capture_t:
                T[k] = t
            h = h + _cv(t, b["fin"])
            S.append(h)
        tr = F.relu(_apply_bn(h, net.tip))
        v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
        own = torch.tanh(_cv(v1, net.own)[:, 0])
        return S, T, own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="g170")
    ap.add_argument("--upto", type=int, default=6)
    ap.add_argument("--late", type=int, default=12)
    ap.add_argument("--ns", default="8,16,32,64,128")
    a = ap.parse_args()
    model = KataGoBinProbeModel(NETS[a.net][1]); net = model.net
    K, Lt = a.upto, a.late
    pairs, geo, enc, sign = load_pairs(model)
    n = len(pairs)
    half = np.random.default_rng(0).permutation(n) % 2
    fit = [i for i in range(n) if half[i] == 0]; ev = [i for i in range(n) if half[i] == 1]
    far = [(gm["ring"] >= 5) & (gm["board_d"] >= 5) for gm in geo]
    fit = [i for i in fit if far[i].any()]; ev = [i for i in ev if far[i].any()]
    print(f"{n} pairs; {len(fit)} fit / {len(ev)} eval with far ring stones", flush=True)

    def xs(idx, member):
        x = torch.tensor(np.stack([enc[i][member][0] for i in idx]))
        g = torch.tensor(np.stack([enc[i][0][1] for i in idx]))     # cyclic globals for both
        return x, g

    # closure directions per layer (fit half, far stones)
    layers = sorted(set(range(1, K + 1)) | {Lt})
    acc = {k: np.zeros(net.C) for k in layers}
    for s in range(0, len(fit), 8):
        idx = fit[s:s + 8]
        S1, _, _ = forward(net, *xs(idx, 0)); S2, _, _ = forward(net, *xs(idx, 1))
        for j, i in enumerate(idx):
            for k in layers:
                acc[k] += (S1[k][j] - S2[k][j]).numpy()[:, far[i]].mean(1)
    U = {k: acc[k] / np.linalg.norm(acc[k]) for k in layers}
    Ut = {k: torch.tensor(U[k], dtype=torch.float32) for k in layers}
    # projected fin kernels: write along U_k per unit of channel j
    Kp = {k: torch.einsum("c,cjyx->jyx", Ut[k], net.blocks[k - 1]["fin"]["w"])[:, None] for k in range(1, K + 1)}
    Cmk = {k: Kp[k].shape[0] for k in Kp}

    # 1. attribution
    contrib = {k: np.zeros(Cmk[k]) for k in Kp}; total = np.zeros(K + 1)
    regions = ("interior", "ring_far", "ring_near_cut", "exterior_band", "elsewhere")
    mass = {k: np.zeros((len(regions), Cmk[k])) for k in range(1, K + 1)}
    for s in range(0, len(fit), 8):
        idx = fit[s:s + 8]
        _, T1, _ = forward(net, *xs(idx, 0), capture_t=True); _, T2, _ = forward(net, *xs(idx, 1), capture_t=True)
        fm = torch.tensor(np.stack([far[i] for i in idx]).astype(np.float32))
        reg = []
        for i in idx:
            gm = geo[i]; dr = dist_to_ring(gm)
            r_in = gm["inside"]; r_far = gm["ring"] >= 5; r_near = (gm["ring"] >= 0) & (gm["ring"] < 5) | (gm["board_d"] == 0)
            r_ext = gm["outside"] & (dr <= 2)
            r_else = ~(r_in | r_far | r_near | r_ext)
            reg.append(np.stack([r_in, r_far, r_near, r_ext, r_else]).astype(np.float32))
        reg = torch.tensor(np.stack(reg))                                  # (B, 5, 19, 19)
        for k in range(1, K + 1):
            dT = T1[k] - T2[k]
            w = F.conv2d(dT, Kp[k], padding=1, groups=Cmk[k])                  # (B, Cm, 19, 19)
            c = (w * fm[:, None]).sum((2, 3)) / fm.sum((1, 2))[:, None]
            contrib[k] += c.sum(0).numpy(); total[k] += c.sum().item()
            mass[k] += torch.einsum("bjyx,bryx->rj", dT.abs(), reg).numpy()
    contrib = {k: v / len(fit) for k, v in contrib.items()}; total /= len(fit)
    flat = [(k, j, contrib[k][j]) for k in range(1, K + 1) for j in range(Cmk[k])]
    flat.sort(key=lambda r: -r[2])
    pos = sum(v for _, _, v in flat if v > 0)
    cum = np.cumsum([v for _, _, v in flat])
    rep = {"net": a.net, "upto": K, "late": Lt, "n_pairs": n, "n_fit": len(fit), "n_eval": len(ev),
           "write_along_closure_by_block": [round(float(v), 3) for v in total[1:]],
           "sum_of_positive_contributions": round(float(pos), 3),
           "channels_for_50pct_of_positive": int(np.argmax(cum >= 0.5 * pos)) + 1,
           "channels_for_80pct_of_positive": int(np.argmax(cum >= 0.8 * pos)) + 1,
           "n_channels_total": sum(Cmk.values())}
    top = []
    for k, j, v in flat[:20]:
        m = mass[k][:, j]; m = m / m.sum()
        top.append({"block": k, "ch": j, "contrib": round(float(v), 3), **{r: round(float(x), 2) for r, x in zip(regions, m)}})
    rep["top_channels"] = top
    # area share of each region, for comparison
    area = np.zeros(len(regions))
    for i in fit:
        gm = geo[i]; dr = dist_to_ring(gm)
        r_in = gm["inside"]; r_far = gm["ring"] >= 5; r_near = (gm["ring"] >= 0) & (gm["ring"] < 5) | (gm["board_d"] == 0)
        r_ext = gm["outside"] & (dr <= 2)
        area += [r_in.sum(), r_far.sum(), r_near.sum(), r_ext.sum(), (~(r_in | r_far | r_near | r_ext)).sum()]
    rep["area_share"] = {r: round(float(x), 2) for r, x in zip(regions, area / area.sum())}
    # difference mass by region over all top-64 channels vs all channels
    def share(sel):
        m = sum(mass[k][:, j] for k, j in sel); return {r: round(float(x), 2) for r, x in zip(regions, m / m.sum())}
    rep["diff_mass_share_top64"] = share([(k, j) for k, j, _ in flat[:64]])
    rep["diff_mass_share_all"] = share([(k, j) for k, j, _ in flat])
    print(json.dumps({x: rep[x] for x in rep if x != "top_channels"}), flush=True)
    for t in top[:10]:
        print(t)

    # 2. causal
    def run(sel):
        byk = {}
        for k, j in sel:
            byk.setdefault(k, []).append(j)
        sig_K, sig_L, own_d = [], [], []
        for s in range(0, len(ev), 8):
            idx = ev[s:s + 8]
            x1, g = xs(idx, 0); x2, _ = xs(idx, 1)
            S2, T2, o2 = forward(net, x2, g, capture_t=True)
            set_t = {k: (torch.tensor(js), T2[k][:, js]) for k, js in byk.items()} if sel else None
            S1, _, o1 = forward(net, x1, g, set_t=set_t)
            for j, i in enumerate(idx):
                fmk = far[i]
                sig_K.append(float(np.einsum("c,cyx->yx", U[K], (S1[K][j] - S2[K][j]).numpy())[fmk].mean()))
                sig_L.append(float(np.einsum("c,cyx->yx", U[Lt], (S1[Lt][j] - S2[Lt][j]).numpy())[fmk].mean()))
                own_d.append(float(sign[i] * (o1[j] - o2[j]).numpy()[fmk].mean()))
        return np.array(sig_K), np.array(sig_L), np.array(own_d)

    b = run([])
    rep["baseline"] = {"signal_layer_upto": round(float(b[0].mean()), 3), "signal_layer_late": round(float(b[1].mean()), 3),
                       "own_far_ring_cyclic_minus_cut": round(float(b[2].mean()), 3)}
    print("baseline", rep["baseline"], flush=True)
    rng = np.random.default_rng(7)
    rep["patch"] = {}
    for q in [int(v) for v in a.ns.split(",")]:
        sel = [(k, j) for k, j, _ in flat[:q]]
        rnd = []
        for _ in range(q):
            kk = int(rng.integers(1, K + 1)); rnd.append((kk, int(rng.integers(Cmk[kk]))))
        out = {}
        for name, s_ in (("top", sel), ("random", rnd)):
            r = run(s_)
            out[name] = {"signal_left_upto": round(float(r[0].mean() / b[0].mean()), 3),
                         "signal_left_late": round(float(r[1].mean() / b[1].mean()), 3),
                         "own_diff_left": round(float(r[2].mean() / b[2].mean()), 3) if abs(b[2].mean()) > 1e-3 else None,
                         "own_diff": round(float(r[2].mean()), 3)}
        rep["patch"][q] = out
        print(q, out, flush=True)
    json.dump(rep, open(f"runs/closure_channels_{a.net}_{K}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

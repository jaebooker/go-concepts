"""How is the "closed loop" signal built? Wavefront analysis on the minimal pairs.

  python -m goconcepts.loopcircuit wave --net g170
  python -m goconcepts.loopcircuit wave --net b40

Each minimal pair is a ring enclosing an opponent group (cyclic) and the same
position with one ring stone replaced by an opponent stone (cut): the two differ
at one point only, the cut point. Whatever the net computes that depends on the
ring being closed must start at the cut point and spread from there.

Closure direction at layer k: the mean difference (cyclic - cut) of the trunk
state on ring stones far from the cut (chain distance >= 5), fit on half the pairs.
Closure signal of a point on the other half: (h_cyc - h_cut) projected on that
direction, divided by the spread of the same projection over ring stones of cut
members (so 1 = one standard deviation).

Reported per layer, as a function of distance from the cut point:
  ring stones by chain distance (steps along the ring) and by board distance,
  interior points (the enclosed region) and exterior points by board distance.
If closure is computed by local propagation, the signal at chain distance d should
appear only after enough blocks for the receptive field to reach d. Global-pooling
blocks (g170: 7, 12, 17; b40: 6, 11, ..., 36) could spread it at once.
"""
from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from . import labels as L
from .cyclegen import generate
from .katago_adapter import gamestate_from_game, board_array
from .kgbin_adapter import KataGoBinProbeModel, _apply_bn, _cv, _gpool
from .mech_cross import NETS

N = 19


def trunk_states(net, x, g):
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        out = [h]
        for b in net.blocks:
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            out.append(h)
        return out


def trunk_states_patched(net, x, g, patch_states, mask, upto):
    """Cyclic-member forward in which, after every block k <= upto, the trunk state at
    `mask` points is overwritten with the cut member's state at that layer."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        m = mask[:, None]
        h = h * (1 - m) + m * patch_states[0]
        out = [h]
        for i, b in enumerate(net.blocks):
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            if i + 1 <= upto:
                h = h * (1 - m) + m * patch_states[i + 1]
            out.append(h)
        return out


def geometry(p):
    """Masks and distances for one pair, from the cyclic member's board."""
    bc = board_array(gamestate_from_game(p.cyclic))
    bx = board_array(gamestate_from_game(p.cut))
    diff = np.argwhere(bc != bx)
    cut = tuple(diff[0])
    ch = L.find_chains(bc)
    k = ch.chain_id[p.ring_point[0] * N + p.ring_point[1]]
    ring = {divmod(s, N) for s in ch.stones[k]}
    # enclosed region: points not on the ring, not connected to the board edge without crossing the ring
    seen = np.zeros((N, N), bool)
    q = deque()
    for y in range(N):
        for x in range(N):
            if (y in (0, N - 1) or x in (0, N - 1)) and (y, x) not in ring:
                seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (y + dy, x + dx)
            if 0 <= v[0] < N and 0 <= v[1] < N and not seen[v] and v not in ring:
                seen[v] = True; q.append(v)
    inside = {(y, x) for y in range(N) for x in range(N) if not seen[y, x] and (y, x) not in ring}
    # chain distance along the ring from the cut point
    cd = {cut: 0}
    q = deque([cut])
    while q:
        u = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + dy, u[1] + dx)
            if v in ring and v not in cd:
                cd[v] = cd[u] + 1; q.append(v)
    ring_map = np.full((N, N), -1); in_map = np.zeros((N, N), bool); out_map = np.zeros((N, N), bool)
    for v, d in cd.items():
        if v != cut:
            ring_map[v] = d
    for v in inside:
        in_map[v] = True
    yy, xx = np.mgrid[0:N, 0:N]
    board_d = np.abs(yy - cut[0]) + np.abs(xx - cut[1])
    near = board_d <= 12
    out_map = (~in_map) & (ring_map < 0) & near
    out_map[cut] = False
    return {"cut": cut, "ring": ring_map, "inside": in_map, "outside": out_map, "board_d": board_d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wave", "route"])
    ap.add_argument("--upto", type=int, default=6)
    ap.add_argument("--net", default="g170", choices=["g170", "b40", "tf3"])
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--all-pairs", action="store_true",
                    help="keep pairs whose members differ in liberty count (by default they are dropped: "
                         "KataGo's input planes mark 1-3 liberty chains, which would hand chain-wide "
                         "information to the stem)")
    a = ap.parse_args()
    pairs = generate(a.pairs, seed=12345)
    if not a.all_pairs:
        pairs = [p for p in pairs if p.libs_cyclic == p.libs_cut]
    if a.net == "tf3":
        from .katago_adapter import KataGoProbeModel
        model = KataGoProbeModel(NETS[a.net][1]); net = None
        tf_names = [n for n in model.layer_names if n != "input"]
        nL = len(tf_names); C = model.model.blocks[0].__dict__.get("c_main", None) or 768
    else:
        model = KataGoBinProbeModel(NETS[a.net][1])
        net = model.net; nL = net.n_blocks + 1; C = net.C
    geo = [geometry(p) for p in pairs]
    half = np.random.default_rng(0).permutation(len(pairs)) % 2
    enc = [(model.encode(gamestate_from_game(p.cyclic)), model.encode(gamestate_from_game(p.cut))) for p in pairs]
    # Input hygiene. (1) KataGo's features under territory scoring shift the komi input by
    # the difference in stones placed, so the two members differ in a global input: give
    # the cut member the cyclic member's global inputs. (2) Chain-wide input planes (the
    # ladder planes) can differ far from the cut: drop pairs whose spatial inputs differ
    # anywhere beyond the stem's 5x5 reach of the cut point.
    keep = []
    for i, p in enumerate(pairs):
        cy, cx = geo[i]["cut"]
        dif = np.argwhere((enc[i][0][0] != enc[i][1][0]).any(0))
        if all(max(abs(y - cy), abs(x - cx)) <= 2 for y, x in dif):
            keep.append(i)
    dropped = len(pairs) - len(keep)
    pairs = [pairs[i] for i in keep]; geo = [geo[i] for i in keep]
    enc = [(enc[i][0], (enc[i][1][0],) + tuple(enc[i][0][1:])) for i in keep]
    half = np.random.default_rng(0).permutation(len(pairs)) % 2
    print(f"{len(pairs)} pairs kept, {dropped} dropped for far input differences", flush=True)

    def states(idx):
        if a.net == "tf3":
            acts = model.activations([enc[i][0] for i in idx] + [enc[i][1] for i in idx], layers=tf_names)
            B = len(idx)
            S = [acts[n].astype(np.float32) for n in tf_names]
            return [s[:B] - s[B:] for s in S], [s[B:] for s in S]
        x = torch.tensor(np.stack([enc[i][0][0] for i in idx] + [enc[i][1][0] for i in idx]))
        g = torch.tensor(np.stack([enc[i][0][1] for i in idx] + [enc[i][1][1] for i in idx]))
        S = trunk_states(net, x, g)
        B = len(idx)
        return [(s[:B] - s[B:]).numpy() for s in S], [s[B:].numpy() for s in S]

    if a.cmd == "route":
        return route(a, net, pairs, geo, half, enc, nL, C)
    # pass 1: closure directions from far ring stones (fit half)
    acc = np.zeros((nL, C)); cnt = 0
    far_idx = [i for i in range(len(pairs)) if half[i] == 0]
    for s in range(0, len(far_idx), 8):
        idx = far_idx[s:s + 8]
        D, _ = states(idx)
        for j, i in enumerate(idx):
            m = geo[i]["ring"] >= 5
            if m.sum() == 0:
                continue
            for k in range(nL):
                acc[k] += D[k][j][:, m].mean(1)
            cnt += 1
    U = acc / np.linalg.norm(acc, axis=1, keepdims=True)
    # spread of the projection over cut-member ring stones (fit half)
    sd = np.zeros(nL); vals = [[] for _ in range(nL)]
    for s in range(0, min(len(far_idx), 80), 8):
        idx = far_idx[s:s + 8]
        _, Hc = states(idx)
        for j, i in enumerate(idx):
            m = geo[i]["ring"] >= 0
            for k in range(nL):
                vals[k] += list(U[k] @ Hc[k][j][:, m])
    sd = np.array([np.std(v) for v in vals])

    # pass 2: signal by distance on the eval half
    maxd = 16
    sums = {r: np.zeros((nL, maxd + 1)) for r in ("ring_chain", "ring_board", "inside", "outside")}
    ns = {r: np.zeros(maxd + 1) for r in sums}
    ev_idx = [i for i in range(len(pairs)) if half[i] == 1]
    recs = []
    for s in range(0, len(ev_idx), 8):
        idx = ev_idx[s:s + 8]
        D, _ = states(idx)
        for j, i in enumerate(idx):
            gm = geo[i]
            proj = np.stack([np.einsum("c,cyx->yx", U[k], D[k][j]) / sd[k] for k in range(nL)])   # (nL,19,19)
            for (yy, xx) in zip(*np.nonzero(gm["ring"] >= 0)):
                recs.append((int(gm["ring"][yy, xx]), int(gm["board_d"][yy, xx]), proj[:, yy, xx].copy()))
                d = min(int(gm["ring"][yy, xx]), maxd)
                sums["ring_chain"][:, d] += proj[:, yy, xx]; ns["ring_chain"][d] += 1
                db = min(int(gm["board_d"][yy, xx]), maxd)
                sums["ring_board"][:, db] += proj[:, yy, xx]; ns["ring_board"][db] += 1
            for r in ("inside", "outside"):
                for (yy, xx) in zip(*np.nonzero(gm[r])):
                    db = min(int(gm["board_d"][yy, xx]), maxd)
                    sums[r][:, db] += proj[:, yy, xx]; ns[r][db] += 1
    rep = {"net": a.net, "gpool_blocks": ([i + 1 for i, b in enumerate(net.blocks) if b["kind"] == "gpool_block"] if net is not None else "attention"),
           "n_fit_pairs": cnt, "proj_sd": [round(float(v), 3) for v in sd]}
    for r in sums:
        mean = sums[r] / np.maximum(ns[r], 1)
        rep[r] = {"n_by_distance": ns[r].astype(int).tolist(),
                  "signal_by_layer_distance": [[round(float(v), 2) for v in row] for row in mean]}
    # wavefront: for ring chain distance, first layer where the signal passes half of its final value
    fin = sums["ring_chain"][-1] / np.maximum(ns["ring_chain"], 1)
    front = {}
    for d in range(1, maxd + 1):
        if ns["ring_chain"][d] < 15:
            continue
        m = sums["ring_chain"][:, d] / ns["ring_chain"][d]
        thr = 0.5 * max(fin[d], m.max())
        front[d] = int(np.argmax(m >= thr))
    rep["first_layer_half_signal_by_chain_distance"] = front
    # along the chain or through the board? On ring stones whose chain distance exceeds
    # their board distance (the ring folds back), regress the signal on both distances.
    cdist = np.array([r[0] for r in recs]); bdist = np.array([r[1] for r in recs]); P = np.stack([r[2] for r in recs])
    rep["chain_vs_board"] = {}
    for k in range(1, min(nL, 13)):
        X = np.stack([np.ones(len(cdist)), cdist, bdist], 1)
        beta = np.linalg.lstsq(X, P[:, k], rcond=None)[0]
        folded = cdist >= bdist + 4
        same = cdist == bdist
        rep["chain_vs_board"][f"layer{k}"] = {"slope_chain": round(float(beta[1]), 3), "slope_board": round(float(beta[2]), 3),
                                              "folded_mean": round(float(P[folded, k].mean()), 2) if folded.any() else None,
                                              "n_folded": int(folded.sum())}
    print("chain vs board distance:", rep["chain_vs_board"])
    print(a.net, "gpool blocks", rep["gpool_blocks"])
    print("first layer (0 = stem) where the closure signal reaches half its peak, by chain distance from the cut:")
    print(front)
    for r in ("ring_chain", "inside", "outside"):
        print(r, "n by distance", rep[r]["n_by_distance"][:13])
        sig = np.array(rep[r]["signal_by_layer_distance"])
        for k in range(0, nL, max(1, nL // 10)):
            print(f"  layer {k:2d}", np.round(sig[k, :13], 2).tolist())
    rep["matched_liberties_only"] = not a.all_pairs; rep["n_pairs"] = len(pairs); rep["dropped_far_input_diff"] = dropped
    json.dump(rep, open(f"runs/loopcircuit_wave_{a.net}{'' if not a.all_pairs else '_allpairs'}.json", "w"), indent=1)


def route(a, net, pairs, geo, half, enc, nL, C):
    """Which region carries the closure signal to far ring stones? In the cyclic member,
    overwrite a region with the cut member's states at every layer up to --upto, and
    measure how much of the closure difference survives at far ring stones (chain
    distance >= 5 and board distance >= 5 from the cut) at that layer, along the
    closure direction fit on the other half."""
    K = a.upto
    fit = [i for i in range(len(pairs)) if half[i] == 0]
    ev = [i for i in range(len(pairs)) if half[i] == 1]
    def batch_states(idx):
        x1 = torch.tensor(np.stack([enc[i][0][0] for i in idx])); x2 = torch.tensor(np.stack([enc[i][1][0] for i in idx]))
        g = torch.tensor(np.stack([enc[i][0][1] for i in idx]))
        return x1, x2, g
    acc = np.zeros(C)
    for s in range(0, len(fit), 8):
        idx = fit[s:s + 8]; x1, x2, g = batch_states(idx)
        S1 = trunk_states(net, x1, g); S2 = trunk_states(net, x2, g)
        D = (S1[K] - S2[K]).numpy()
        for j, i in enumerate(idx):
            m = (geo[i]["ring"] >= 5) & (geo[i]["board_d"] >= 5)
            if m.any():
                acc += D[j][:, m].mean(1)
    u = acc / np.linalg.norm(acc)
    regions = {
        "interior": lambda gm: gm["inside"],
        "exterior_band": lambda gm: gm["outside"] & (dist_to_ring(gm) <= 2),
        "ring_near_cut": lambda gm: (gm["ring"] >= 0) & (gm["ring"] <= 3),
        "interior_and_exterior": lambda gm: gm["inside"] | (gm["outside"] & (dist_to_ring(gm) <= 2)),
    }
    res = {r: [] for r in regions}; base = []
    for s in range(0, len(ev), 8):
        idx = ev[s:s + 8]; x1, x2, g = batch_states(idx)
        S1 = trunk_states(net, x1, g); S2 = trunk_states(net, x2, g)
        far = [(geo[i]["ring"] >= 5) & (geo[i]["board_d"] >= 5) for i in idx]
        d0 = np.einsum("c,bcyx->byx", u, (S1[K] - S2[K]).numpy())
        base += [d0[j][far[j]].mean() for j in range(len(idx)) if far[j].any()]
        for r, fn in regions.items():
            mask = torch.tensor(np.stack([fn(geo[i]).astype(np.float32) for i in idx]))
            P = trunk_states_patched(net, x1, g, S2, mask, K)
            d1 = np.einsum("c,bcyx->byx", u, (P[K] - S2[K]).numpy())
            res[r] += [d1[j][far[j]].mean() for j in range(len(idx)) if far[j].any()]
    b = np.mean(base)
    out = {"net": a.net, "layer": K, "n_eval_pairs_with_far_stones": len(base),
           "closure_signal_left_at_far_ring_stones": {r: round(float(np.mean(v) / b), 3) for r, v in res.items()}}
    print(json.dumps(out))
    json.dump(out, open(f"runs/loopcircuit_route_{a.net}_L{K}.json", "w"), indent=1)


def dist_to_ring(gm):
    ring = gm["ring"] >= 0
    ys, xs = np.nonzero(ring)
    yy, xx = np.mgrid[0:N, 0:N]
    d = np.full((N, N), 99)
    for y, x in zip(ys, xs):
        d = np.minimum(d, np.abs(yy - y) + np.abs(xx - x))
    return d


if __name__ == "__main__":
    main()

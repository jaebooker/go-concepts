"""Mechanism of the loop-safety flip in g170 (blocks 3 to 7).

  python -m goconcepts.mech blocks     # which block's update, at the D chain, makes rings read safe
  python -m goconcepts.mech channels   # which channels carry it, and what they respond to

Data: the eye-balanced status set (data/status_bal.pkl). Every position is run twice:
as is (D to move; policy -> capture mass) and after a D pass (ownership of the D
chain from D's side, the safety readout).

Intervention: at block k, the block's residual update on the D chain's stones is
replaced by the mean update that block writes on D stones of doomed EDGE groups
(computed separately for the to-move and after-pass states). If block k is where a
ring gets written as "safe", this should make doomed rings read doomed and get
captured, while leaving doomed edge groups (patched with their own mean) and safe
rings (whose safety should come from the race, not the loop) less affected.
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
from .kgbin_adapter import KataGoBinProbeModel, _apply_bn, _cv, _gpool, _vpool
from .status_eval import _d_color
from .paths import NETS_DIR

CKPT = NETS_DIR + "/g170e-b20.bin.gz"
CASES = "data/status_bal.pkl"


def forward(net, x, g, edit=None, keep=None):
    """g170 forward pass. edit(k, h_prev, upd) -> new upd, for blocks k = 1..n.
    keep: dict to store per-block updates. Returns policy log-probs, ownership."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        for i, b in enumerate(net.blocks):
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            upd = _cv(t, b["fin"])
            if keep is not None:
                keep[i + 1] = upd
            if edit is not None:
                upd = edit(i + 1, h, upd)
            h = h + upd
        tr = F.relu(_apply_bn(h, net.tip))
        p = _cv(tr, net.p1)
        gp = F.relu(_apply_bn(_cv(tr, net.g1), net.g1bn))
        pooled = _gpool(gp)
        p = F.relu(_apply_bn(p + (pooled @ net.g2b)[:, :, None, None], net.p1bn))
        pol = torch.log_softmax(torch.cat([_cv(p, net.p2)[:, 0].flatten(1), (pooled @ net.g2pass)[:, :1]], 1), -1)
        v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
        own = torch.tanh(_cv(v1, net.own)[:, 0])
        return pol.numpy(), own.numpy()


def load(cases_path=CASES):
    cases = pickle.load(open(cases_path, "rb"))
    model = KataGoBinProbeModel(CKPT)
    X = {"move": [], "pass": []}
    masks = []
    for c in cases:
        gs = gamestate_from_game(c.game)
        g2 = copy.deepcopy(c.game); g2.moves = list(g2.moves) + [(_d_color(c), None)]
        gp = gamestate_from_game(g2)
        X["move"].append(model.encode(gs)); X["pass"].append(model.encode(gp))
        arr = board_array(gs)
        cid = L.find_chains(arr).chain_id.reshape(19, 19)
        masks.append((cid == cid[c.ring_point]).astype(np.float32))
    geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
    return cases, model, X, np.stack(masks), geo, st


def run(model, X, masks, cases, edit_for=None, batch=40, keep_blocks=()):
    """Returns capture mass (to-move), ownership of D chain after pass (D's side),
    and, if keep_blocks, the mean update over D stones per case per kept block."""
    n = len(cases)
    cap, own = np.zeros(n), np.zeros(n)
    kept = {k: {"move": np.zeros((n, 256)), "pass": np.zeros((n, 256))} for k in keep_blocks}
    for s in range(0, n, batch):
        idx = list(range(s, min(n, s + batch)))
        m = torch.tensor(masks[idx])
        for state in ("move", "pass"):
            x = torch.tensor(np.stack([X[state][i][0] for i in idx]))
            g = torch.tensor(np.stack([X[state][i][1] for i in idx]))
            keep = {} if keep_blocks else None
            ed = edit_for(state, m, idx) if edit_for else None
            pol, ow = forward(model.net, x, g, ed, keep)
            if state == "move":
                p = np.exp(pol)
                cap[idx] = [p[j, cases[i].correct].sum() for j, i in enumerate(idx)]
            else:
                mm = masks[idx]
                own[idx] = -(ow * mm).sum((1, 2)) / mm.sum((1, 2))   # after D's pass the net speaks for A
            for k in keep_blocks:
                u = keep[k].numpy()                       # (B, C, 19, 19)
                kept[k][state][idx] = (u * masks[idx][:, None]).sum((2, 3)) / masks[idx].sum((1, 2))[:, None]
    return cap, own, kept


def summarize(cap, own, geo, st):
    out = {}
    for g, gn in ((0, "ring"), (1, "edge")):
        for s, sn in ((0, "doomed"), (1, "safe")):
            m = (geo == g) & (st == s)
            out[f"{sn}_{gn}"] = {"own_after_pass": round(float(own[m].mean()), 3),
                                 "read_alive": round(float((own[m] > 0).mean()), 3),
                                 "capture": round(float(cap[m].mean()), 3)}
    return out


def ranges_cmd(out_path, ranges):
    """Mean-edge patch of the D chain's updates over ranges of blocks."""
    cases, model, X, masks, geo, st = load()
    _, _, kept = run(model, X, masks, cases, keep_blocks=range(1, 21))
    de = (geo == 1) & (st == 0)
    mean_upd = {k: {s: torch.tensor(kept[k][s][de].mean(0), dtype=torch.float32) for s in ("move", "pass")} for k in kept}
    rep = {}
    for a, b in ranges:
        blocks = set(range(a, b + 1))

        def edit_for(state, m, idx, blocks=blocks):
            def ed(k, h, upd):
                if k not in blocks:
                    return upd
                mm = m[:, None]
                return upd * (1 - mm) + mm * mean_upd[k][state][None, :, None, None]
            return ed
        cap, own, _ = run(model, X, masks, cases, edit_for)
        r = rep[f"blocks{a}-{b}"] = summarize(cap, own, geo, st)
        print(f"blocks {a}-{b}: doomed ring own {r['doomed_ring']['own_after_pass']:+.2f} alive {r['doomed_ring']['read_alive']:.2f} cap {r['doomed_ring']['capture']:.2f} | "
              f"safe ring own {r['safe_ring']['own_after_pass']:+.2f} cap {r['safe_ring']['capture']:.2f} | doomed edge own {r['doomed_edge']['own_after_pass']:+.2f} cap {r['doomed_edge']['capture']:.2f} | safe edge own {r['safe_edge']['own_after_pass']:+.2f}", flush=True)
    json.dump(rep, open(out_path, "w"), indent=1)


def direction_cmd(out_path, blocks=range(1, 13), seed=0, fit_on="doomed"):
    """Is the flip carried by one direction? Per block k, u_k = unit vector along
    (mean trunk state on D stones: doomed rings) - (same: doomed edges), fit on half
    the positions. On the other half, set the D stones' component along u_k to the
    doomed-edge mean at every block in `blocks` (a 1-dim intervention per block).
    Control: random unit directions, same blocks."""
    cases, model, X, masks, geo, st = load()
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(cases)) % 2
    trunk = {}
    def keep_trunk(state):
        def ed(k, h, upd):
            hh = (h + upd).numpy()
            trunk.setdefault(state, {}).setdefault(k, []).append(hh)
            return upd
        return ed
    # trunk states on D stones, per case, per block (mean over stones)
    T = {s: np.zeros((len(cases), 21, 256), np.float32) for s in ("move", "pass")}
    for s0 in range(0, len(cases), 40):
        idx = list(range(s0, min(len(cases), s0 + 40)))
        for state in ("move", "pass"):
            x = torch.tensor(np.stack([X[state][i][0] for i in idx])); g = torch.tensor(np.stack([X[state][i][1] for i in idx]))
            store = {}
            def ed(k, h, upd, store=store):
                store[k] = (h + upd).numpy(); return upd
            forward(model.net, x, g, ed)
            mm = masks[idx]
            for k in range(1, 21):
                T[state][idx, k] = (store[k] * mm[:, None]).sum((2, 3)) / mm.sum((1, 2))[:, None]
    fit = half == 0
    fs = 0 if fit_on == "doomed" else 1
    dr, de = fit & (geo == 0) & (st == 0), fit & (geo == 1) & (st == 0)
    fr, fe = fit & (geo == 0) & (st == fs), fit & (geo == 1) & (st == fs)
    U, tgt = {}, {}
    for state in ("move", "pass"):
        for k in blocks:
            v = T[state][fr, k].mean(0) - T[state][fe, k].mean(0)
            u = v / np.linalg.norm(v)
            U[(state, k)] = torch.tensor(u, dtype=torch.float32)
            tgt[(state, k)] = float(T[state][de, k].mean(0) @ u)
    rep = {"blocks": list(blocks)}
    ks = list(blocks)
    cos = [[round(float(U[("pass", a)] @ U[("pass", b)]), 2) for b in ks] for a in ks]
    rep["cosine_between_blocks_pass"] = cos
    np.save(out_path.replace(".json", "_U.npy"), np.stack([U[("pass", k)].numpy() for k in ks]))
    print("cosine u_k (pass):"); [print("  ", r) for r in cos]
    evalm = half == 1

    def make(Ud, tg):
        def edit_for(state, m, idx):
            def ed(k, h, upd):
                if k not in blocks:
                    return upd
                u = Ud[(state, k)][None, :, None, None]
                proj = ((h + upd) * u).sum(1, keepdim=True)
                return upd - m[:, None] * (proj - tg[(state, k)]) * u
            return ed
        return edit_for

    base_cap, base_own, _ = run(model, X, masks, cases)
    rep["baseline_eval_half"] = summarize(base_cap[evalm], base_own[evalm], geo[evalm], st[evalm])
    cap, own, _ = run(model, X, masks, cases, make(U, tgt))
    rep["loop_direction"] = summarize(cap[evalm], own[evalm], geo[evalm], st[evalm])
    rand = []
    for r in range(3):
        Ur, tr_ = {}, {}
        for key in U:
            v = rng.normal(size=256); v /= np.linalg.norm(v)
            Ur[key] = torch.tensor(v, dtype=torch.float32)
            state, k = key
            tr_[key] = float(T[state][de, k].mean(0) @ v)
        c2, o2, _ = run(model, X, masks, cases, make(Ur, tr_))
        rand.append(summarize(c2[evalm], o2[evalm], geo[evalm], st[evalm]))
    rep["random_directions"] = rand
    # how much of the ring-edge gap is along u_k, per block
    rep["share_of_gap_along_u"] = {}
    for k in blocks:
        v = T["pass"][dr, k].mean(0) - T["pass"][de, k].mean(0)
        sd = T["pass"][de, k].std(0).mean()
        rep["share_of_gap_along_u"][f"block{k}"] = {"gap_norm_over_edge_sd": round(float(np.linalg.norm(v) / sd), 2)}
    for key in ("baseline_eval_half", "loop_direction"):
        r = rep[key]
        print(key, {g: (r[g]["own_after_pass"], r[g]["capture"]) for g in r})
    for r in rand:
        print("random", {g: (r[g]["own_after_pass"], r[g]["capture"]) for g in r})
    json.dump(rep, open(out_path, "w"), indent=1)


def blocks_cmd(out_path):
    cases, model, X, masks, geo, st = load()
    base_cap, base_own, kept = run(model, X, masks, cases, keep_blocks=range(1, 21))
    rep = {"baseline": summarize(base_cap, base_own, geo, st), "patch_one_block": {}, "patch_range": {}}
    de = (geo == 1) & (st == 0)
    mean_upd = {k: {s: torch.tensor(kept[k][s][de].mean(0), dtype=torch.float32) for s in ("move", "pass")} for k in kept}

    def make(blocks, rings_only=True):
        def edit_for(state, m, idx):
            def ed(k, h, upd):
                if k not in blocks:
                    return upd
                mm = m[:, None]
                return upd * (1 - mm) + mm * mean_upd[k][state][None, :, None, None]
            return ed
        return edit_for

    for k in range(1, 13):
        cap, own, _ = run(model, X, masks, cases, make({k}))
        rep["patch_one_block"][f"block{k}"] = summarize(cap, own, geo, st)
        r = rep["patch_one_block"][f"block{k}"]
        print(f"block {k}: doomed ring own {r['doomed_ring']['own_after_pass']:+.2f} cap {r['doomed_ring']['capture']:.2f} | "
              f"safe ring own {r['safe_ring']['own_after_pass']:+.2f} | doomed edge own {r['doomed_edge']['own_after_pass']:+.2f} cap {r['doomed_edge']['capture']:.2f}", flush=True)
    for a, b in ((3, 5), (4, 6), (4, 7), (3, 7), (5, 7), (6, 8), (1, 7), (1, 12)):
        cap, own, _ = run(model, X, masks, cases, make(set(range(a, b + 1))))
        rep["patch_range"][f"blocks{a}-{b}"] = summarize(cap, own, geo, st)
        r = rep["patch_range"][f"blocks{a}-{b}"]
        print(f"blocks {a}-{b}: doomed ring own {r['doomed_ring']['own_after_pass']:+.2f} cap {r['doomed_ring']['capture']:.2f} | "
              f"safe ring own {r['safe_ring']['own_after_pass']:+.2f} cap {r['safe_ring']['capture']:.2f} | doomed edge own {r['doomed_edge']['own_after_pass']:+.2f} cap {r['doomed_edge']['capture']:.2f}", flush=True)
    b = rep["baseline"]
    print("baseline", json.dumps(b))
    json.dump(rep, open(out_path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blocks", "ranges", "attribution", "direction", "channels"])
    ap.add_argument("--dir-blocks", default="1-12")
    ap.add_argument("--fit-on", default="doomed", choices=["doomed", "safe"],
                    help="safe: the ring-edge direction from SAFE positions only, where both geometries have the same verdict")
    ap.add_argument("--ranges", default="8-12,7-12,9-12,10-12,8-10,1-20,13-20,8-20,1-12")
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--out", default="runs/mech_blocks.json")
    a = ap.parse_args()
    if a.cmd == "blocks":
        blocks_cmd(a.out)
    elif a.cmd == "ranges":
        ranges_cmd(a.out, [tuple(int(v) for v in r.split("-")) for r in a.ranges.split(",")])
    elif a.cmd == "direction":
        lo, hi = (int(v) for v in a.dir_blocks.split("-"))
        direction_cmd(a.out, range(lo, hi + 1), fit_on=a.fit_on)
    elif a.cmd == "attribution":
        attribution_cmd(a.out, a.cases)



REGIONS = ("ring", "inner", "eye", "shared", "outside", "wall", "rest")


def region_masks(c):
    """Integer map of region ids (index into REGIONS) for one case."""
    arr = np.zeros((19, 19), np.int8)
    for q in c.game.setup_black: arr[q] = 1
    for q in c.game.setup_white: arr[q] = 2
    for col, q in c.game.moves: arr[q] = col
    ch = L.find_chains(arr); n = 19
    kr = ch.chain_id[c.ring_point[0] * n + c.ring_point[1]]; ki = ch.chain_id[c.inner_point[0] * n + c.inner_point[1]]
    ring_l, inner_l = set(ch.libs[kr]), set(ch.libs[ki])
    reg = np.full(n * n, REGIONS.index("rest"), np.int8)
    D = ch.color[kr]; f = arr.reshape(-1); nb = L._nb(n)
    for s in ch.stones[kr]: reg[s] = 0
    for s in ch.stones[ki]: reg[s] = 1
    for s in inner_l - ring_l: reg[s] = 2
    for s in inner_l & ring_l: reg[s] = 3
    for s in ring_l - inner_l: reg[s] = 4
    for s in list(ch.stones[kr]) + list(ring_l - inner_l):
        for v in nb[s]:
            if f[v] == 3 - D and ch.chain_id[v] != ki:
                reg[v] = 5
    return reg.reshape(n, n)


def forward_grad(net, x, g, target_mask):
    """Gradient x update attribution: for each block k and board point, the
    first-order contribution of block k's update at that point to the ring's
    summed ownership logit (sign flipped to D's side after a pass)."""
    x = x.clone().requires_grad_(True)
    h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
    upds = []
    for b in net.blocks:
        a = F.relu(_apply_bn(h, b["pre"]))
        t = _cv(a, b["reg"])
        if b["kind"] == "gpool_block":
            gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
            t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
        t = F.relu(_apply_bn(t, b["mid"]))
        upd = _cv(t, b["fin"])
        upd.retain_grad(); upds.append(upd)
        h = h + upd
    tr = F.relu(_apply_bn(h, net.tip))
    v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
    logit = _cv(v1, net.own)[:, 0]
    obj = -(logit * target_mask).sum()          # D's side after D's pass
    obj.backward()
    return [(u.grad * u).sum(1).detach().numpy() for u in upds]   # per block: (B,19,19)


def attribution_cmd(out_path, cases_path=CASES):
    cases, model, X, masks, geo, st = load(cases_path)
    regs = np.stack([region_masks(c) for c in cases])
    nbk = model.net.n_blocks
    A = np.zeros((len(cases), nbk, len(REGIONS)))
    for s in range(0, len(cases), 20):
        idx = list(range(s, min(len(cases), s + 20)))
        x = torch.tensor(np.stack([X["pass"][i][0] for i in idx])); g = torch.tensor(np.stack([X["pass"][i][1] for i in idx]))
        tm = torch.tensor(masks[idx])
        att = forward_grad(model.net, x, g, tm)
        for k in range(nbk):
            for j, i in enumerate(idx):
                for r in range(len(REGIONS)):
                    A[i, k, r] = att[k][j][regs[i] == r].sum()
    rep = {}
    for gg, gn in ((0, "ring"), (1, "edge")):
        for ss, sn in ((0, "doomed"), (1, "safe")):
            m = (geo == gg) & (st == ss)
            rep[f"{sn}_{gn}"] = {f"block{k + 1}": {REGIONS[r]: round(float(A[m, k, r].mean()), 3) for r in range(len(REGIONS))} for k in range(nbk)}
    np.save(out_path.replace(".json", ".npy"), A)
    json.dump(rep, open(out_path, "w"), indent=1)
    d = A[(geo == 0) & (st == 0)].mean(0) - A[(geo == 1) & (st == 0)].mean(0)
    print("doomed ring minus doomed edge, contribution to ring's 'alive' logit, by block x region")
    print("block " + " ".join(f"{r:>8s}" for r in REGIONS))
    for k in range(nbk):
        print(f"{k + 1:5d} " + " ".join(f"{v:8.2f}" for v in d[k]))
    print("total by region", np.round(d.sum(0), 2), "grand", round(float(d.sum()), 2))


if __name__ == "__main__":
    main()

"""Follow-up to late_channels: the late channels that separate doomed loops from
doomed edge groups are mostly "this group is dying" channels. They fire on doomed
edge groups and stay silent on doomed loops (and on safe groups). So the question
becomes whether the loop feature is what keeps them silent.

  python -m goconcepts.late_channels2 --net b40 --blocks 38-40 --mid 12-24
  python -m goconcepts.late_channels2 --net g170 --blocks 19-20 --mid 6-12

1. Restore: on the D chain's stones, set the top-k death channels' activations to
   the mean that same-status edge groups get (status-matched, fit half). If silence
   on these few channels is what lets doomed loops read alive, doomed loops should
   now read dead, while safe loops stay alive. Control: k random channels.
2. Loop reading: hold the loop direction (safe-fit ring - edge axis) at the
   doomed-edge level on the D stones, (a) only at the input of the first late block,
   (b) across the mid-trunk window that the tenth/eleventh runs found. Report the
   death channels' activation on doomed loops as a fraction of their activation on
   doomed edge groups, and the ownership readout.
All readouts on the eval half, after-pass state (restore also runs the to-move state
for the capture rate).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from .kgbin_adapter import _apply_bn, _cv, _gpool
from .late_channels import load, batches
from .mech_cross import trunk_means


def forward(net, x, g, late, capture=None, setval=None, holds=None):
    """setval: {k: (mask (B,19,19), ch LongTensor, values (B, len(ch)))} sets t[:, ch] on mask.
    holds: {k: (u, tgt, mask)} applied to h after block k (k = 0: before block 1)."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        for i, b in enumerate(net.blocks):
            k = i + 1
            if holds is not None and (k - 1) in holds:
                u, tg, m = holds[k - 1]
                ut = u[None, :, None, None]
                h = h - m[:, None] * ((h * ut).sum(1, keepdim=True) - tg) * ut
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            if setval is not None and k in setval:
                m, ch, vals = setval[k]
                t = t.clone()
                mm = m[:, None]
                t[:, ch] = t[:, ch] * (1 - mm) + mm * vals[:, :, None, None]
            if capture is not None and k in late:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="b40")
    ap.add_argument("--blocks", default="38-40")
    ap.add_argument("--mid", default="12-24")
    ap.add_argument("--ks", default="8,16,32")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.blocks.split("-")); late = list(range(lo, hi + 1))
    mlo, mhi = (int(v) for v in a.mid.split("-"))
    rep_prev = json.load(open(f"runs/late_channels_{a.net}_{a.blocks}.json"))
    cases, model, X, masks, geo, st = load(a.net)
    net = model.net; n = len(cases); C = net.C
    half = np.random.default_rng(0).permutation(n) % 2
    fit, ev = half == 0, half == 1
    cells = {"dr": (geo == 0) & (st == 0), "de": (geo == 1) & (st == 0), "sr": (geo == 0) & (st == 1), "se": (geo == 1) & (st == 1)}

    # channel activations on D stones, both states, all late blocks
    act = {s: {k: np.zeros((n, C), np.float32) for k in late} for s in ("move", "pass")}
    for idx in batches(n):
        mm = masks[idx]; den = mm.sum((1, 2))[:, None]
        for s in ("move", "pass"):
            x = torch.tensor(np.stack([X[s][i][0] for i in idx])); g = torch.tensor(np.stack([X[s][i][1] for i in idx]))
            cap = {}
            forward(net, x, g, late, capture=cap)
            for k in late:
                act[s][k][idx] = (cap[k].numpy() * mm[:, None]).sum((2, 3)) / den
    # rank again with the full contribution ranking from late_channels, keep "death" type:
    # more active on doomed edge groups than on doomed loops (fit half, after-pass)
    ranked = []
    # recompute ranking over all channels from the saved top list is not enough; use activation gap x write
    from .late_channels import CASES  # noqa: F401
    T = trunk_means(a.net, cases, "pass")
    nL = T.shape[1] - 1
    w = T[fit & cells["se"], nL].mean(0) - T[fit & cells["de"], nL].mean(0); w /= np.linalg.norm(w)
    wt = torch.tensor(w, dtype=torch.float32)
    for k in late:
        Kk = torch.einsum("c,cjyx->j", wt, net.blocks[k - 1]["fin"]["w"]).numpy()   # summed kernel: write per unit activation
        gapact = act["pass"][k][fit & cells["dr"]].mean(0) - act["pass"][k][fit & cells["de"]].mean(0)
        for j in range(C):
            ranked.append((k, j, float(gapact[j] * Kk[j]), float(gapact[j])))
    ranked.sort(key=lambda r: -r[2])
    death = [(k, j) for k, j, sc, ga in ranked if ga < 0][:max(int(v) for v in a.ks.split(","))]
    loopy = [(k, j) for k, j, sc, ga in ranked if ga > 0][:16]
    rep = {"net": a.net, "late_blocks": late,
           "n_death_in_top32_by_write": sum(1 for k, j, sc, ga in ranked[:32] if ga < 0),
           "death_channels": [{"block": k, "ch": j, **{c: round(float(act["pass"][k][ev & m, j].mean()), 2) for c, m in cells.items()}} for k, j in death[:12]],
           "loop_positive_channels": [{"block": k, "ch": j, **{c: round(float(act["pass"][k][ev & m, j].mean()), 2) for c, m in cells.items()}} for k, j in loopy[:6]]}
    print("death-type among top 32 writers:", rep["n_death_in_top32_by_write"])

    def refs(chs, s):
        """per case, status-matched edge mean activation for each channel (fit half)."""
        out = np.zeros((n, len(chs)), np.float32)
        for ci, (k, j) in enumerate(chs):
            for status in (0, 1):
                m = fit & (geo == 1) & (st == status)
                out[st == status, ci] = act[s][k][m, j].mean()
        return out

    def run(chs=None, holds=None, states=("move", "pass"), capture_death=None):
        capm, own = np.full(n, np.nan), np.zeros(n)
        dact = np.zeros((n, len(capture_death))) if capture_death else None
        R = {s: refs(chs, s) for s in states} if chs else None
        for idx in batches(n):
            mm = masks[idx]; mt = torch.tensor(mm)
            for s in states:
                x = torch.tensor(np.stack([X[s][i][0] for i in idx])); g = torch.tensor(np.stack([X[s][i][1] for i in idx]))
                setval = None
                if chs:
                    setval = {}
                    for k in late:
                        sel = [ci for ci, (kk, j) in enumerate(chs) if kk == k]
                        if sel:
                            setval[k] = (mt, torch.tensor([chs[ci][1] for ci in sel]), torch.tensor(R[s][idx][:, sel]))
                hh = None
                if holds is not None and s == "pass":
                    hh = {k: (u, tg, mt) for k, (u, tg) in holds.items()}
                cap = {}
                pol, ow = forward(net, x, g, late, capture=cap if capture_death else None, setval=setval, holds=hh)
                if s == "move":
                    p = np.exp(pol); capm[idx] = [p[jj, cases[i].correct].sum() for jj, i in enumerate(idx)]
                else:
                    own[idx] = -(ow * mm).sum((1, 2)) / mm.sum((1, 2))
                    if capture_death:
                        for ci, (k, j) in enumerate(capture_death):
                            dact[idx, ci] = (cap[k][:, j].numpy() * mm).sum((1, 2)) / mm.sum((1, 2))
        res = {c: {"own": round(float(own[ev & m].mean()), 3),
                   "capture": (round(float(np.nanmean(capm[ev & m])), 3) if "move" in states else None)} for c, m in cells.items()}
        if capture_death:
            de_level = dact[ev & cells["de"]].mean(0)
            res["death_act_doomed_loop_over_doomed_edge"] = round(float(dact[ev & cells["dr"]].mean(0).sum() / de_level.sum()), 3)
            res["death_act_safe_loop_over_doomed_edge"] = round(float(dact[ev & cells["sr"]].mean(0).sum() / de_level.sum()), 3)
        return res

    rng = np.random.default_rng(5)
    rep["restore"] = {}
    for q in [int(v) for v in a.ks.split(",")]:
        rep["restore"][f"death{q}"] = run(death[:q])
        rnd = [(late[int(rng.integers(len(late)))], int(rng.integers(C))) for _ in range(q)]
        rep["restore"][f"random{q}"] = run(rnd)
        print(q, "death", rep["restore"][f"death{q}"], "\n   random", rep["restore"][f"random{q}"], flush=True)

    # loop reading
    Tm = T
    def loop_holds(ks):
        H = {}
        for k in ks:
            u = Tm[fit & cells["sr"], k].mean(0) - Tm[fit & cells["se"], k].mean(0); u /= np.linalg.norm(u)
            H[k] = (torch.tensor(u, dtype=torch.float32), float(Tm[fit & cells["de"], k].mean(0) @ u))
        return H
    d16 = death[:16]
    rep["loop_reading"] = {
        "baseline": run(capture_death=d16, states=("pass",)),
        f"hold_at_input_of_block{lo}": run(holds=loop_holds([lo - 1]), capture_death=d16, states=("pass",)),
        f"hold_blocks{mlo}-{mhi}": run(holds=loop_holds(range(mlo, mhi + 1)), capture_death=d16, states=("pass",)),
    }
    for kk, v in rep["loop_reading"].items():
        print("loop reading", kk, v, flush=True)
    json.dump(rep, open(f"runs/late_channels2_{a.net}_{a.blocks}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

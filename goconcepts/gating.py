"""Where in mid-trunk does the loop feature block the death evidence?  (b40, g170)

  python -m goconcepts.gating --net b40 --mid 12-24
  python -m goconcepts.gating --net g170 --mid 6-12

After-pass state, eye-balanced set, directions fit on one half, readouts on the other.

Death axis at layer k: from safe edge groups to doomed edge groups on the D chain's
stones (edge positions only). "Death evidence" on a chain at layer k = its position
on that axis, scaled so safe edge = 0 and doomed edge = 1.

1. Baseline trajectory: death evidence on doomed loops and doomed edges per layer,
   and each block's write along its own layer's death axis (loop minus edge).
2. Holds of the loop direction (safe-fit, doomed-edge level, D stones) over single
   blocks, prefixes and suffixes of the mid window. For each: ownership of doomed and
   safe loops, and where the released death evidence is written: per-block writes
   along the death axis under the hold minus baseline, summed over blocks inside the
   window, after it, and in the last blocks.
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


def forward(net, x, g, holds, mask):
    """Returns ownership map and the trunk state (mean over mask) after each block."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        den = mask.sum((1, 2))[:, None]
        states = [(h * mask[:, None]).sum((2, 3)) / den]
        for i, b in enumerate(net.blocks):
            k = i + 1
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            if k in holds:
                u, tg = holds[k]
                ut = u[None, :, None, None]
                h = h - mask[:, None] * ((h * ut).sum(1, keepdim=True) - tg) * ut
            states.append((h * mask[:, None]).sum((2, 3)) / den)
        tr = F.relu(_apply_bn(h, net.tip))
        v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
        own = torch.tanh(_cv(v1, net.own)[:, 0])
        return own.numpy(), torch.stack(states, 1).numpy()      # (B, nL+1, C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="b40")
    ap.add_argument("--mid", default="12-24")
    a = ap.parse_args()
    mlo, mhi = (int(v) for v in a.mid.split("-"))
    cases, model, X, masks, geo, st = load(a.net)
    net = model.net; n = len(cases); nb = net.n_blocks
    half = np.random.default_rng(0).permutation(n) % 2
    fit, ev = half == 0, half == 1
    cells = {"dr": (geo == 0) & (st == 0), "de": (geo == 1) & (st == 0), "sr": (geo == 0) & (st == 1), "se": (geo == 1) & (st == 1)}
    T = trunk_means(a.net, cases, "pass")
    # death axes and scale per layer
    D, lo_, hi_ = [], [], []
    for k in range(nb + 1):
        v = T[fit & cells["de"], k].mean(0) - T[fit & cells["se"], k].mean(0); v /= np.linalg.norm(v)
        D.append(v); lo_.append(float(T[fit & cells["se"], k].mean(0) @ v)); hi_.append(float(T[fit & cells["de"], k].mean(0) @ v))
    D = np.stack(D); lo_ = np.array(lo_); gap = np.array(hi_) - lo_
    U = {}
    for k in range(1, nb + 1):
        u = T[fit & cells["sr"], k].mean(0) - T[fit & cells["se"], k].mean(0); u /= np.linalg.norm(u)
        U[k] = (torch.tensor(u, dtype=torch.float32), float(T[fit & cells["de"], k].mean(0) @ u))

    def run(hold_blocks):
        holds = {k: U[k] for k in hold_blocks}
        own = np.zeros(n); S = np.zeros((n, nb + 1, net.C), np.float32)
        for idx in batches(n):
            x = torch.tensor(np.stack([X["pass"][i][0] for i in idx])); g = torch.tensor(np.stack([X["pass"][i][1] for i in idx]))
            m = torch.tensor(masks[idx])
            ow, states = forward(net, x, g, holds, m)
            own[idx] = -(ow * masks[idx]).sum((1, 2)) / masks[idx].sum((1, 2)); S[idx] = states
        evid = (np.einsum("nkc,kc->nk", S, D) - lo_) / gap                  # death evidence per layer
        writes = np.einsum("nkc,kc->nk", S[:, 1:] - S[:, :-1], D[1:]) / gap[1:]   # block k's write on its layer's axis
        return own, evid, writes

    own0, ev0, wr0 = run([])
    rep = {"net": a.net, "mid": [mlo, mhi]}
    rep["baseline"] = {"own": {c: round(float(own0[ev & m].mean()), 3) for c, m in cells.items()},
                       "death_evidence_doomed_loop": [round(float(v), 2) for v in ev0[ev & cells["dr"]].mean(0)],
                       "death_evidence_safe_loop": [round(float(v), 2) for v in ev0[ev & cells["sr"]].mean(0)],
                       "writes_loop_minus_edge_doomed": [round(float(v), 3) for v in (wr0[ev & cells["dr"]].mean(0) - wr0[ev & cells["de"]].mean(0))]}
    print("baseline death evidence on doomed loops:", rep["baseline"]["death_evidence_doomed_loop"])
    print("writes (doomed loop - doomed edge) per block:", rep["baseline"]["writes_loop_minus_edge_doomed"], flush=True)
    W = mhi - mlo + 1
    thirds = [(mlo + i * W // 3, mlo + (i + 1) * W // 3 - 1) for i in range(3)]
    thirds[-1] = (thirds[-1][0], mhi)
    conds = [(f"{k}", [k]) for k in range(mlo, mhi + 1)]
    conds += [(f"{a_}-{b_}", list(range(a_, b_ + 1))) for a_, b_ in thirds]
    conds += [(f"{mlo}-{b_}", list(range(mlo, b_ + 1))) for _, b_ in thirds[:-1]]
    conds += [(f"{a_}-{mhi}", list(range(a_, mhi + 1))) for a_, _ in thirds[1:]]
    conds += [(f"{mlo}-{mhi}", list(range(mlo, mhi + 1)))]
    rep["holds"] = {}
    for name, blocks in conds:
        own, evd, wr = run(blocks)
        dw = wr[ev & cells["dr"]].mean(0) - wr0[ev & cells["dr"]].mean(0)      # extra death writes on doomed loops
        dws = wr[ev & cells["sr"]].mean(0) - wr0[ev & cells["sr"]].mean(0)
        blk = np.arange(1, nb + 1)
        r = {"own": {c: round(float(own[ev & m].mean()), 3) for c, m in cells.items()},
             "final_death_evidence_doomed_loop": round(float(evd[ev & cells["dr"], -1].mean()), 3),
             "extra_death_writes_doomed_loop": {"inside_hold": round(float(dw[(blk >= blocks[0]) & (blk <= blocks[-1])].sum()), 3),
                                                "between_hold_and_last4": round(float(dw[(blk > blocks[-1]) & (blk <= nb - 4)].sum()), 3),
                                                "last4": round(float(dw[blk > nb - 4].sum()), 3)},
             "extra_death_writes_safe_loop_total": round(float(dws.sum()), 3),
             "extra_death_writes_doomed_loop_by_block": [round(float(v), 3) for v in dw]}
        rep["holds"][name] = r
        print(f"hold {name:>7}: doomed loop own {r['own']['dr']:+.2f}  safe loop {r['own']['sr']:+.2f}  "
              f"final death evidence {r['final_death_evidence_doomed_loop']:+.2f}  extra writes {r['extra_death_writes_doomed_loop']}", flush=True)
    json.dump(rep, open(f"runs/gating_{a.net}_{a.mid}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

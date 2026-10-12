"""Causal test on g170: push the D chain's activations along the edge-trained
"doomed" direction and see whether the policy starts capturing in doomed rings.

  python -m goconcepts.steer --layers 6,10,14 --alphas 0,2,4,8

Direction: a linear doomed/safe probe trained on the D-chain stones of EDGE
positions at one block (from runs/g170/status.npz). In raw activation space
the doomed direction is (w_doomed - w_safe) / sd, normalised to unit length,
then scaled by the spread of edge activations along it. At test time it is
added to the trunk after that block, on the D chain's stones only.

Controls: a random direction with the same norm; the same steering on doomed
edge positions (already handled correctly) and on safe rings (where D should
not suddenly panic, though capturing there is also sound).
Metric: policy mass on the capturing moves.
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from .extract import _labels_for
from .katago_adapter import gamestate_from_game
from .kgbin_adapter import KataGoBinProbeModel, _apply_bn, _cv, _gpool
from .probes import LinearProbe
from .status_eval import STATUS
from .paths import NETS_DIR


def forward_steered(net, x, g, steer):
    """Copy of KataGoBinNet.forward's trunk with an additive intervention.
    steer: {block_index (1-based): (mask (B,19,19) float tensor, vec (C,) tensor)}"""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        for i, b in enumerate(net.blocks):
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            if (i + 1) in steer:
                mask, vec = steer[i + 1]
                h = h + mask[:, None] * vec[None, :, None, None]
        tr = F.relu(_apply_bn(h, net.tip))
        p = _cv(tr, net.p1)
        gp = F.relu(_apply_bn(_cv(tr, net.g1), net.g1bn))
        pooled = _gpool(gp)
        p = F.relu(_apply_bn(p + (pooled @ net.g2b)[:, :, None, None], net.p1bn))
        pol = _cv(p, net.p2)[:, 0].flatten(1)
        pas = (pooled @ net.g2pass)[:, :1]
        return torch.log_softmax(torch.cat([pol, pas], 1), -1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=NETS_DIR + "/g170e-b20.bin.gz")
    ap.add_argument("--run", default="runs/g170")
    ap.add_argument("--layers", default="6,10,14")
    ap.add_argument("--alphas", default="0,2,4,8")
    ap.add_argument("--out", default="runs/steer_g170.json")
    a = ap.parse_args()
    cases = pickle.load(open(STATUS, "rb"))
    geo = np.array([c.variant for c in cases]); status = np.array([c.status for c in cases])
    d = np.load(f"{a.run}/status.npz")
    names = list(d["layer_names"]); case = d["case"]
    model = KataGoBinProbeModel(a.ckpt)
    gss = [gamestate_from_game(c.game) for c in cases]
    enc = [model.encode(gs) for gs in gss]
    masks = []
    for c, gs in zip(cases, gss):
        cid = _labels_for(gs)["chain_id"]
        masks.append((cid == cid[c.ring_point]).astype(np.float32))
    groups = {"doomed_ring": (geo == 0) & (status == 0), "doomed_edge": (geo == 1) & (status == 0),
              "safe_ring": (geo == 0) & (status == 1)}
    rng = np.random.default_rng(0)
    report = {"layers": {}}
    for L in [int(x) for x in a.layers.split(",")]:
        li = names.index(f"block{L}")
        X = d["acts"][:, li].astype(np.float32)
        tr = geo[case] == 1
        probe = LinearProbe(2, epochs=200).fit(X[tr], status[case][tr], balance=True)
        W = probe.lin.weight.detach().numpy()
        dvec = (W[0] - W[1]) / probe.sd.numpy()           # towards "doomed" (class 0)
        u = dvec / np.linalg.norm(dvec)
        scale = float((X[tr] @ u).std())
        r = rng.normal(size=u.shape); r /= np.linalg.norm(r)
        rows = {}
        for dname, dirv in (("doomed_dir", u), ("random_dir", r)):
            for alpha in [float(x) for x in a.alphas.split(",")]:
                vec = torch.tensor(dirv * alpha * scale, dtype=torch.float32)
                res = {}
                for gname, gm in groups.items():
                    idx = np.nonzero(gm)[0]
                    mass = []
                    for s in range(0, len(idx), 25):
                        bi = idx[s:s + 25]
                        x = torch.tensor(np.stack([enc[i][0] for i in bi]))
                        g = torch.tensor(np.stack([enc[i][1] for i in bi]))
                        m = torch.tensor(np.stack([masks[i] for i in bi]))
                        lp = forward_steered(model.net, x, g, {L: (m, vec)} if alpha else {})
                        p = np.exp(lp)
                        mass += [float(p[k, cases[i].correct].sum()) for k, i in enumerate(bi)]
                    res[gname] = round(float(np.mean(mass)), 3)
                rows[f"{dname} alpha={alpha:g}"] = res
                print(f"block{L}", dname, alpha, res, flush=True)
        report["layers"][f"block{L}"] = {"scale": scale, "results": rows}
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Behaviour on solved capturing races, linked to representation.

  python -m goconcepts.race_eval extract --engine lz --ckpt ... --run runs/lz
  python -m goconcepts.race_eval score runs/humansl runs/lz runs/g170 runs/tf3

extract: for every race position (data/races.pkl, from racegen) record the raw
policy and the activations of every stone of the D chain (the ring, or the
edge-pocket chain) at every layer.

score:
  behaviour   policy mass on the capturing moves, and whether the top move is one
  representation  probe D (natural positions + the training half of the cyclic/
              cut pairs, using the same reduced training set tf3 had, so all
              engines are on one footing) reads the liberty count of each stone
              of the D chain; per position, the fraction read correctly,
              averaged over all trunk blocks
  links       within each engine, rank correlation between representation and
              behaviour across positions, controlling for liberty structure,
              geometry and chain size; across engines, whether the same
              positions are failed
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
from .probes import LinearProbe, lib_class

RACES = os.path.join(os.path.dirname(__file__), "..", "data", "races.pkl")


def _model(engine, ckpt):
    if engine == "lz":
        from .lz_adapter import LeelaZeroProbeModel
        return LeelaZeroProbeModel(ckpt)
    if engine == "kgbin":
        from .kgbin_adapter import KataGoBinProbeModel
        return KataGoBinProbeModel(ckpt)
    from .katago_adapter import KataGoProbeModel
    return KataGoProbeModel(ckpt)


def extract(model, cases, batch=12):
    rows = {k: [] for k in ("case", "libs")}
    acts_out, pol = [], []
    names = None
    for s in range(0, len(cases), batch):
        chunk = cases[s:s + batch]
        gss = [gamestate_from_game(c.game) for c in chunk]
        acts = model.activations([model.encode(gs) for gs in gss])
        names = [n for n in model.layer_names if n in acts and n != "input"]
        C = max(acts[n].shape[1] for n in names)
        for j, (c, gs) in enumerate(zip(chunk, gss)):
            pol.append(acts["policy"][j])
            lab = _labels_for(gs)
            cid = lab["chain_id"]
            ys, xs = np.nonzero(cid == cid[c.ring_point])
            for y, x in zip(ys, xs):
                v = np.zeros((len(names), C), np.float16)
                for li, nm in enumerate(names):
                    a = acts[nm][j, :, y, x]
                    v[li, :a.shape[0]] = a
                acts_out.append(v)
                rows["case"].append(s + j); rows["libs"].append(int(lab["libs"][y, x]))
        del acts
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        print(f"  {min(s + batch, len(cases))}/{len(cases)} positions", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["acts"] = np.stack(acts_out)
    out["policy"] = np.asarray(pol, np.float32)
    out["layer_names"] = np.array(names)
    return out


def _probe_training(run, restrict_to):
    """Probe-D training data, reduced to what the restrict_to run had."""
    nat = np.load(os.path.join(run, "natural.npz"), mmap_mode="r")
    pr = np.load(os.path.join(run, "pairs.npz"), mmap_mode="r")
    meta = json.load(open(os.path.join(run, "pairs_meta.json")))
    r0 = np.random.default_rng(0); ug = np.unique(nat["game"]); r0.shuffle(ug)
    nat_tr = ~np.isin(nat["game"], ug[: len(ug) // 5])
    all_p = np.array(sorted(m["pid"] for m in meta)); np.random.default_rng(0).shuffle(all_p)
    train_p = set(all_p[: len(all_p) // 2].tolist())
    if restrict_to:
        other = set(np.unique(np.load(os.path.join(restrict_to, "pairs.npz"))["pid"]).tolist())
        train_p &= other
        n_nat = int(np.load(os.path.join(restrict_to, "natural.npz"))["pos"].max()) + 1
        nat_tr = nat_tr & (np.asarray(nat["pos"]) < n_nat)
    ptr = np.isin(np.asarray(pr["pid"]), list(train_p))
    return nat, np.nonzero(nat_tr)[0], pr, np.nonzero(ptr)[0]


def representation(run, restrict_to="runs/tf3", epochs=150):
    """Per case: fraction of D-chain stones whose liberty bucket probe D reads
    correctly, averaged over trunk blocks. Returns (per-case array, per-layer)."""
    re = np.load(os.path.join(run, "races.npz"))
    nat, ni, pr, pi = _probe_training(run, restrict_to)
    nat_names = list(nat["layer_names"])
    names = [n for n in re["layer_names"] if n.startswith("block") and n in nat_names]
    if len(names) > 22:  # Leela Zero: every second block, as elsewhere
        names = [n for n in names if int(n[5:]) % 2 == 0]
    y = lib_class(re["libs"]); case = re["case"]
    ncase = int(case.max()) + 1
    per_layer = []
    for n in names:
        li_n, li_r = nat_names.index(n), list(re["layer_names"]).index(n)
        X = np.concatenate([nat["acts"][ni, li_n].astype(np.float32), pr["acts"][pi, li_n].astype(np.float32)])
        Y = np.concatenate([lib_class(np.asarray(nat["libs"])[ni]), lib_class(np.asarray(pr["libs"])[pi])])
        probe = LinearProbe(8, epochs=epochs).fit(X, Y)
        ok = probe.predict(re["acts"][:, li_r].astype(np.float32)) == y
        per_layer.append([ok[case == c].mean() for c in range(ncase)])
        print(" ", os.path.basename(run), n, round(float(ok.mean()), 3), flush=True)
    return np.array(per_layer).mean(0), names


def _behaviour(run, cases):
    re = np.load(os.path.join(run, "races.npz"))
    p = np.exp(re["policy"])
    mass = np.array([p[i, c.correct].sum() for i, c in enumerate(cases)])
    top = np.array([int(np.argmax(p[i])) in c.correct for i, c in enumerate(cases)])
    save = np.array([p[i, c.saving].sum() for i, c in enumerate(cases)])
    return mass, top, save


def _ci(v, rng):
    bs = rng.choice(v, size=(4000, len(v))).mean(1)
    return [round(float(v.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def score(runs, out_path, restrict_to="runs/tf3"):
    rng = np.random.default_rng(0)
    cases = pickle.load(open(RACES, "rb"))
    var = np.array([c.variant for c in cases])
    combo = np.array([f"{c.shared}{c.eyes}{c.outside}" for c in cases])
    libs = np.array([c.ring_libs for c in cases])
    size = np.array([c.ring_size for c in cases])
    rep = {"n_cases": len(cases), "engines": {}, "links": {}, "shared_behaviour": {}}
    B, Rp = {}, {}
    for run in runs:
        name = os.path.basename(run)
        mass, top, save = _behaviour(run, cases)
        rep_case, layers = representation(run, restrict_to)
        B[name], Rp[name] = mass, rep_case
        e = {"mass_correct": {"cyclic": _ci(mass[var == 0], rng), "edge": _ci(mass[var == 1], rng)},
             "top1_correct": {"cyclic": round(float(top[var == 0].mean()), 3), "edge": round(float(top[var == 1].mean()), 3)},
             "mass_saving": {"cyclic": round(float(save[var == 0].mean()), 3), "edge": round(float(save[var == 1].mean()), 3)},
             "mass_by_ring_libs": {str(L): {"cyclic": round(float(mass[(libs == L) & (var == 0)].mean()), 3),
                                            "edge": round(float(mass[(libs == L) & (var == 1)].mean()), 3)}
                                   for L in sorted(set(libs))},
             "representation": {"cyclic": _ci(rep_case[var == 0], rng), "edge": _ci(rep_case[var == 1], rng)},
             "probe_layers": layers}
        # cyclic minus edge, within liberty structure (combo), regressing out ring size
        X = np.column_stack([np.ones(len(cases)), var == 0, rankdata(size)] + [(combo == k) for k in sorted(set(combo))[1:]]).astype(float)
        for label, v in (("behaviour", mass), ("representation", rep_case)):
            beta, *_ = np.linalg.lstsq(X, v, rcond=None)
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, len(v), len(v))
                bb, *_ = np.linalg.lstsq(X[idx], v[idx], rcond=None)
                bs.append(bb[1])
            e[f"cyclic_effect_on_{label}"] = [round(float(beta[1]), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        rep["engines"][name] = e

    def resid(v):
        Xc = np.column_stack([np.ones(len(v)), var == 0, rankdata(size)] + [(combo == k) for k in sorted(set(combo))[1:]]).astype(float)
        rv = rankdata(v); b, *_ = np.linalg.lstsq(Xc, rv, rcond=None)
        return rv - Xc @ b

    def rtest(a, b):
        x, z = resid(a), resid(b)
        r = np.corrcoef(x, z)[0, 1]
        null = np.array([np.corrcoef(x, rng.permutation(z))[0, 1] for _ in range(4000)])
        return {"r": round(float(r), 3), "p": round(float((null >= r).mean()), 4)}

    for name in B:
        rep["links"][name] = {"all": rtest(Rp[name], B[name])}
    names = list(B)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rep["shared_behaviour"][f"{names[i]}~{names[j]}"] = rtest(B[names[i]], B[names[j]])
    json.dump(rep, open(out_path, "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--engine", default="katago")
    ap.add_argument("--ckpt")
    ap.add_argument("--run")
    ap.add_argument("--out", default="runs/races_report.json")
    a = ap.parse_args()
    if a.cmd == "extract":
        cases = pickle.load(open(RACES, "rb"))
        d = extract(_model(a.engine, a.ckpt), cases)
        np.savez(os.path.join(a.run, "races.npz"), **d)
    else:
        print(json.dumps(score(a.runs, a.out), indent=1))


if __name__ == "__main__":
    main()

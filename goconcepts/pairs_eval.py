"""Cyclic vs cut minimal pairs: extraction and paired evaluation.

  python -m goconcepts.pairs_eval extract --engine lz --ckpt nets/best-network --run runs/lz
  python -m goconcepts.pairs_eval eval --run runs/lz

Four probes per layer, all trained on liberty buckets:
  A  natural positions only (the stage-one probe). Tests transfer.
  B  natural + cut chains from the TRAIN half of the pairs (favours cut)
  C  natural + cyclic chains from the TRAIN half (favours cyclic)
  D  natural + both variants from the TRAIN half. The primary probe: it has
     seen both shapes equally, so a remaining gap on TEST pairs is not a
     training-distribution artefact. B and C bracket how much the choice of
     training chains alone can move the gap.
Metric: per chain, the fraction of its stones whose liberty bucket is read
correctly. Main statistic: paired difference cut minus cyclic over test
pairs with equal liberties in both members, bootstrap 95% CI over pairs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os

import numpy as np

from .cyclegen import generate
from .extract import _labels_for
from .katago_adapter import gamestate_from_game
from .probes import LinearProbe, lib_class
from .paths import NETS_DIR


def extract(model, pairs, batch=16):
    items = []
    for p in pairs:
        items.append((p, 0, p.cyclic))
        items.append((p, 1, p.cut))
    rows = {k: [] for k in ("pid", "variant", "libs", "chain_size", "is_ring")}
    acts_out = []
    names = None
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        gss = [gamestate_from_game(g) for _, _, g in chunk]
        acts = model.activations([model.encode(gs) for gs in gss])
        names = [n for n in model.layer_names if n in acts]
        C = max(acts[n].shape[1] for n in names)
        for j, ((p, var, _), gs) in enumerate(zip(chunk, gss)):
            lab = _labels_for(gs)
            cid = lab["chain_id"]
            k = cid[p.ring_point]
            ys, xs = np.nonzero(cid == k)
            for y, x in zip(ys, xs):
                v = np.zeros((len(names), C), np.float16)
                for li, nm in enumerate(names):
                    a = acts[nm][j, :, y, x]
                    v[li, :a.shape[0]] = a
                acts_out.append(v)
                rows["pid"].append(p.pid); rows["variant"].append(var)
                rows["libs"].append(int(lab["libs"][y, x])); rows["chain_size"].append(len(ys))
                rows["is_ring"].append(1)
        del acts
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        print(f"  {min(s + batch, len(items))}/{len(items)} positions, {len(acts_out)} points", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["acts"] = np.stack(acts_out)
    out["layer_names"] = np.array(names)
    return out


def _chain_acc(pred, y, pid, var):
    """Per (pid, variant): fraction of that chain's stones read correctly."""
    res = {}
    for key in set(zip(pid.tolist(), var.tolist())):
        m = (pid == key[0]) & (var == key[1])
        res[key] = float((pred[m] == y[m]).mean())
    return res


def _boot(d, rng, n=4000):
    d = np.asarray(d)
    if len(d) == 0:
        return (float("nan"),) * 3
    bs = rng.choice(d, size=(n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def evaluate(run, epochs=150, layers=None, seed=0, probes="ABCD", restrict_train_to=None, tag=None):
    rng = np.random.default_rng(seed)
    nat = np.load(os.path.join(run, "natural.npz"))
    pr = np.load(os.path.join(run, "pairs.npz"))
    meta = json.load(open(os.path.join(run, "pairs_meta.json")))
    names = list(pr["layer_names"])
    layers = layers or names
    # same game split as probes.py
    r0 = np.random.default_rng(0); ug = np.unique(nat["game"]); r0.shuffle(ug)
    nat_tr = ~np.isin(nat["game"], ug[: len(ug) // 5])
    yn = lib_class(nat["libs"])
    yp = lib_class(pr["libs"]); pid = pr["pid"]; var = pr["variant"]
    # split over ALL generated pairs (from meta), so a run that extracted only a
    # subset of pairs still gets the same train/test assignment as full runs
    all_p = np.array(sorted(m["pid"] for m in meta)); rng.shuffle(all_p)
    train_p = set(all_p[: len(all_p) // 2].tolist())
    split_train = set(train_p)            # defines the test set; never changes
    if restrict_train_to is not None:
        # match another run's smaller training data, for a like-for-like comparison:
        # same training pairs and the same number of natural positions
        other = set(np.unique(np.load(os.path.join(restrict_train_to, "pairs.npz"))["pid"]).tolist())
        train_p = train_p & other
        n_nat = int(np.load(os.path.join(restrict_train_to, "natural.npz"))["pos"].max()) + 1
        nat_tr = nat_tr & (nat["pos"] < n_nat)
    pids = np.unique(pid)
    ptr = np.isin(pid, list(train_p))
    matched = {m["pid"] for m in meta if m["libs_cyclic"] == m["libs_cut"]}
    lib_of = {m["pid"]: m["libs_cyclic"] for m in meta}
    out = {"layers": [], "n_pairs": len(pids), "n_train_pairs_used": len(train_p & set(pids.tolist())),
           "n_natural_train_points": int(nat_tr.sum()),
           "n_test_matched": int(len([p for p in pids if p not in split_train and p in matched]))}
    all_preds = {}
    for lname in layers:
        li = names.index(lname)
        Xn = nat["acts"][nat_tr, li].astype(np.float32)
        Xp = pr["acts"][:, li].astype(np.float32)
        row = {"layer": lname}
        for probe_name, add_vars in (("A", ()), ("B", (1,)), ("C", (0,)), ("D", (0, 1))):
            if probe_name not in probes:
                continue
            if not add_vars:
                X, y = Xn, yn[nat_tr]
            else:
                addm = ptr & np.isin(var, add_vars)
                X = np.concatenate([Xn, Xp[addm]]); y = np.concatenate([yn[nat_tr], yp[addm]])
            probe = LinearProbe(8, epochs=epochs).fit(X, y)
            pred = probe.predict(Xp)
            all_preds[f"{probe_name}/{lname}"] = pred.astype(np.int8)
            acc = _chain_acc(pred, yp, pid, var)
            test = [p for p in pids if p not in split_train and p in matched]
            d = [acc[(p, 1)] - acc[(p, 0)] for p in test]
            cyc = [acc[(p, 0)] for p in test]; cut = [acc[(p, 1)] for p in test]
            row[probe_name] = {"cut": float(np.mean(cut)), "cyclic": float(np.mean(cyc)),
                               "diff_cut_minus_cyclic": _boot(d, rng),
                               "by_libs": {str(L): {"n": int(sum(1 for p in test if lib_of[p] == L)),
                                                    "cut": float(np.mean([acc[(p, 1)] for p in test if lib_of[p] == L])) if any(lib_of[p] == L for p in test) else None,
                                                    "cyclic": float(np.mean([acc[(p, 0)] for p in test if lib_of[p] == L])) if any(lib_of[p] == L for p in test) else None}
                                           for L in range(2, 9)}}
        print(lname, " | ".join("%s cut/cyc %.2f/%.2f d=%.3f [%.3f,%.3f]" % (k, row[k]["cut"], row[k]["cyclic"], *row[k]["diff_cut_minus_cyclic"]) for k in probes), flush=True)
        out["layers"].append(row)
    out["train_pids"] = sorted(int(p) for p in split_train)
    suffix = ("" if probes == "ABCD" else "_" + probes) + (f"_{tag}" if tag else "")
    json.dump(out, open(os.path.join(run, f"pairs_results{suffix}.json"), "w"), indent=1)
    np.savez(os.path.join(run, f"pairs_preds{suffix}.npz"), **all_preds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval"])
    ap.add_argument("--engine", default="katago", choices=["katago", "lz", "kgbin"])
    ap.add_argument("--ckpt", default=NETS_DIR + "/humanv0.ckpt")
    ap.add_argument("--run", required=True)
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--layers", default=None, help="comma-separated subset")
    ap.add_argument("--probes", default="ABCD")
    ap.add_argument("--restrict-train-to", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-train-pairs", type=int, default=None,
                    help="extract all test pairs but only this many training pairs (slow engines)")
    a = ap.parse_args()
    if a.cmd == "extract":
        pairs = generate(a.pairs, seed=12345)
        keep = pairs
        if a.max_train_pairs is not None:
            r = np.random.default_rng(0); all_p = np.array(sorted(p.pid for p in pairs)); r.shuffle(all_p)
            tr = set(all_p[: len(all_p) // 2].tolist())
            tr_keep = set(sorted(tr)[: a.max_train_pairs])
            keep = [p for p in pairs if p.pid not in tr or p.pid in tr_keep]
        json.dump([{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items() if k not in ("cyclic", "cut")} for p in pairs],
                  open(os.path.join(a.run, "pairs_meta.json"), "w"))
        if a.engine == "lz":
            from .lz_adapter import LeelaZeroProbeModel
            model = LeelaZeroProbeModel(a.ckpt)
        elif a.engine == "kgbin":
            from .kgbin_adapter import KataGoBinProbeModel
            model = KataGoBinProbeModel(a.ckpt)
        else:
            from .katago_adapter import KataGoProbeModel
            model = KataGoProbeModel(a.ckpt)
        d = extract(model, keep)
        np.savez(os.path.join(a.run, "pairs.npz"), **d)
    else:
        evaluate(a.run, a.epochs, a.layers.split(",") if a.layers else None, probes=a.probes,
                 restrict_train_to=a.restrict_train_to, tag=a.tag)


if __name__ == "__main__":
    main()

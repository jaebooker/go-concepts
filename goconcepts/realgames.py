"""Ownership-after-pass on chains that were really captured in real games.

  python -m goconcepts.realgames build                 # -> data/real_positions.pkl
  python -m goconcepts.realgames extract --engine kgbin --ckpt ... --run runs/g170
  python -m goconcepts.realgames score runs/g170 runs/humansl runs/tf3

Sources
  far  FAR's published cyclic-attack games (goattack.far.ai): the adversary vs
       KataGo at 1 visit to 10M visits, humans running the attack against
       KataGo and Leela Zero, transfer games vs Leela Zero and ELF.
  pro  Fox professional games (data/sgf), as a control for ordinary captures.

Events: every chain of >= 6 stones that is captured. For each event and lead
L in LEADS we take the position L moves before the capturing move (L=1: the
V moves once more, then the capture lands) with the
captured side (V) to move, find V's chain there (the chain holding most of the
stones that are later captured), and read the counterfactual that defines
"doomed": V passes, the capturer is to move, and the net predicts end-of-game
ownership of the chain from V's side (+1 V keeps it, -1 it is lost).
Each chain is also labelled cyclic (it encloses an opponent group on its own)
at that position.

The comparison that matters: cyclic vs non-cyclic chains that were in fact
captured, at the same lead and liberty count. "Loops look alive" predicts
positive ownership for cyclic chains and negative for ordinary ones.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import glob
import gzip
import json
import os
import pickle
import random

import numpy as np

from . import labels as L
from .katago_adapter import Board, gamestate_from_game, board_array
from .sgf import Game, read as read_sgf
from .paths import SGF_DIR

ROOT = os.path.join(os.path.dirname(__file__), "..")
FAR = os.path.join(ROOT, "data", "far", "far_adversarial_games.txt.gz")
PRO = SGF_DIR
OUT = os.path.join(ROOT, "data", "real_positions.pkl")
LEADS = (1, 3, 5, 9, 17, 33)
# FAR sets whose victim is kata1-b40c256-s11840935168 ("Latest"/cp505; the
# "hardened" games use the same weights with a search-side pass fix)
B40_SETS = ["no_search_hardened", "4096_visits_hardened", "10mil_visits", "win-rate",
            "position-analysis", "position-analysis-1b", "qualitative", "no_search", "8_visits"]
MIN_CAPTURE = 6


def loop_chains(board: np.ndarray, ch: L.Chains, min_inner=4) -> set[int]:
    """Chains that form a closed loop around an opponent group: removing the
    chain leaves a region that does not reach the board edge and holds an
    opponent chain of >= min_inner stones. Unlike extract.cyclic_chains, the
    region may also hold other stones of the loop's own colour (in real
    attacks the victim usually has stones inside the loop too)."""
    n = board.shape[0]
    b = board.reshape(-1)
    nb = L._nb(n)
    out = set()
    for k, st in enumerate(ch.stones):
        if len(st) < 8:
            continue
        c = ch.color[k]
        mine = set(st)
        seen = set(mine)
        for s0 in range(n * n):
            if s0 in seen:
                continue
            comp, q, edge = [], [s0], False
            seen.add(s0)
            while q:
                u = q.pop(); comp.append(u)
                y, x = divmod(u, n)
                if y in (0, n - 1) or x in (0, n - 1):
                    edge = True
                for v in nb[u]:
                    if v not in seen:
                        seen.add(v); q.append(v)
            if edge:
                continue
            inner = {ch.chain_id[u] for u in comp if b[u] == L.opp(c)}
            if any(len(ch.stones[j]) >= min_inner for j in inner):
                out.add(k)
                break
    return out


def load_far():
    games, seen = [], set()
    with gzip.open(FAR, "rt") as fh:
        for line in fh:
            key, hdr, ab, aw, col, mv = line.rstrip("\n").split("\t")
            if mv in seen:          # position-analysis repeats some games
                continue
            seen.add(mv)
            hdr = json.loads(hdr)
            g = Game(size=int(hdr.get("SZ", 19)))
            g.setup_black = [(ord(ab[i + 1]) - 97, ord(ab[i]) - 97) for i in range(0, len(ab), 2)]
            g.setup_white = [(ord(aw[i + 1]) - 97, ord(aw[i]) - 97) for i in range(0, len(aw), 2)]
            for k, c in enumerate(col):
                s = mv[2 * k:2 * k + 2]
                g.moves.append((1 if c == "B" else 2, None if s == "tt" else (ord(s[1]) - 97, ord(s[0]) - 97)))
            victim = 1 if hdr.get("PB") == "Victim" else 2
            games.append({"src": "far", "key": key, "set": key.split("/")[0], "victim": victim, "game": g})
    return games


def load_pro(n, seed=0):
    files = sorted(glob.glob(os.path.join(PRO, "**", "*.sgf"), recursive=True))
    random.Random(seed).shuffle(files)
    out = []
    for f in files:
        if len(out) >= n:
            break
        try:
            g = read_sgf(f)
        except Exception:
            continue
        if g.size != 19 or len(g.moves) < 80 or g.setup_black or g.setup_white:
            continue
        out.append({"src": "pro", "key": os.path.basename(f), "set": "pro", "victim": 0, "game": g})
    return out


def replay(g: Game):
    """Board arrays after each prefix of the move list (boards[t] = after t moves)."""
    gs = gamestate_from_game(Game(size=g.size, setup_black=g.setup_black, setup_white=g.setup_white,
                                  first_player=g.first_player, moves=[]))
    boards = [board_array(gs)]
    for col, mv in g.moves:
        loc = Board.PASS_LOC if mv is None else gs.board.loc(mv[1], mv[0])
        try:
            gs.play(col, loc)
        except Exception:
            break
        boards.append(board_array(gs))
    return boards


def events(item):
    g = item["game"]
    boards = replay(g)
    n = g.size
    ev = []
    for i in range(len(boards) - 1):
        before, after = boards[i], boards[i + 1]
        col = g.moves[i][0]
        v = 3 - col
        gone = (before == v) & (after == 0)
        if gone.sum() < MIN_CAPTURE:
            continue
        ch = L.find_chains(before)
        for k in set(ch.chain_id[gone.reshape(-1)].tolist()):
            st = ch.stones[k]
            if len(st) < MIN_CAPTURE:
                continue
            ev.append({"i": i, "v": v, "stones": set(st), "size_at_capture": len(st),
                       "cyclic_at_capture": k in loop_chains(before, ch)})
    out = []
    for e in ev:
        i, v = e["i"], e["v"]
        for lead in LEADS:
            s = i - lead            # position after s moves; move s is the next one
            if s < 1 or g.moves[s][0] != v:
                continue
            b = boards[s]
            ch = L.find_chains(b)
            ids = [ch.chain_id[p] for p in e["stones"] if b.reshape(-1)[p] == v]
            if not ids:
                continue
            k = max(set(ids), key=ids.count)
            st = ch.stones[k]
            if len(st) < 4:
                continue
            out.append({"src": item["src"], "set": item["set"], "key": item["key"],
                        "victim_side": (v == item["victim"]) if item["src"] == "far" else None,
                        "capture_move": i, "lead": lead, "s": s, "v": v,
                        "point": divmod(st[0], n), "size": len(st), "libs": len(ch.libs[k]),
                        "overlap": ids.count(k) / len(st),
                        "cyclic": k in loop_chains(b, ch),
                        "cyclic_at_capture": e["cyclic_at_capture"],
                        "size_at_capture": e["size_at_capture"],
                        "game": item["game"]})
    return out


def build(n_pro=1500):
    far, pro = load_far(), load_pro(n_pro)
    pos = []
    for it in far + pro:
        pos += events(it)
    # the Game objects are shared per game; positions store the prefix length s
    pickle.dump(pos, open(OUT, "wb"))
    tab = {}
    for p in pos:
        k = (p["src"], p["cyclic"], p["lead"])
        tab[k] = tab.get(k, 0) + 1
    for k in sorted(tab):
        print(k, tab[k])
    return pos


def _state(p, passed):
    g = copy.copy(p["game"])
    g.moves = list(p["game"].moves[:p["s"]]) + ([(p["v"], None)] if passed else [])
    return gamestate_from_game(g)


def extract(model, pos, batch=16, only_pass=False):
    keys = ["own_after_pass", "own", "value_after_pass"]
    res = {k: [] for k in keys}
    for s in range(0, len(pos), batch):
        chunk = pos[s:s + batch]
        for passed in ((True,) if only_pass else (True, False)):
            gss = [_state(p, passed) for p in chunk]
            a = model.activations([model.encode(gs) for gs in gss], layers=[])
            for j, (p, gs) in enumerate(zip(chunk, gss)):
                arr = board_array(gs)
                cid = L.find_chains(arr).chain_id.reshape(arr.shape)
                m = cid == cid[p["point"]]
                own = a.get("ownership")
                sign = -1.0 if passed else 1.0   # after a pass the net speaks for the capturer
                o = float(sign * own[j][m].mean()) if own is not None else np.nan
                v = a["value"][j]
                v = float(v[0]) if np.ndim(v) else float((v + 1) / 2)
                if passed:
                    res["own_after_pass"].append(o); res["value_after_pass"].append(1 - v)
                else:
                    res["own"].append(o)
            del a
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        print(f"  {min(s + batch, len(pos))}/{len(pos)}", flush=True)
    if only_pass:
        res["own"] = [np.nan] * len(pos)
    return {k: np.asarray(v, np.float32) for k, v in res.items()}


def subset(pos, cap_noncyc=120, leads=(3, 9, 17), seed=0):
    """For slow nets (tf3, ~15 s a position on CPU): every cyclic position at
    three leads plus a fixed sample of ordinary ones per lead (same sample for
    every engine)."""
    rng = random.Random(seed)
    idx = [i for i, p in enumerate(pos) if p["cyclic"] and p["lead"] in leads]
    by = {}
    for i, p in enumerate(pos):
        if not p["cyclic"] and p["lead"] in leads:
            by.setdefault(p["lead"], []).append(i)
    for lead, ii in sorted(by.items()):
        rng.shuffle(ii)
        idx += ii[:cap_noncyc // len(by) + 1]
    return sorted(idx)


def _ci(x, rng, n=4000):
    if len(x) == 0:
        return [None, None, None]
    bs = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def score(runs, out):
    pos = pickle.load(open(OUT, "rb"))
    rng = np.random.default_rng(0)
    rep = {"n_positions": len(pos), "engines": {}}
    cyc = np.array([p["cyclic"] for p in pos]); lead = np.array([p["lead"] for p in pos])
    libs = np.array([p["libs"] for p in pos]); size = np.array([p["size"] for p in pos])
    sets = np.array([p["set"] for p in pos])
    src = np.array([p["src"] for p in pos]); vic = np.array([bool(p["victim_side"]) for p in pos])
    for run in runs:
        d = np.load(os.path.join(run, "real.npz"))
        idx = d["idx"]; o = np.full(len(pos), np.nan); o[idx] = d["own_after_pass"]
        o2 = np.full(len(pos), np.nan); o2[idx] = d["own"]
        have = ~np.isnan(o)
        e = {}
        groups = {"cyclic_far_victim": cyc & (src == "far") & vic,
                  "noncyclic_far": ~cyc & (src == "far"),
                  "noncyclic_pro": ~cyc & (src == "pro"),
                  "cyclic_pro": cyc & (src == "pro"),
                  "noncyclic_large_20plus": ~cyc & (size >= 20),
                  "cyclic_b40_victim_games": cyc & vic & np.isin(sets, B40_SETS)}
        for gname, gm in groups.items():
            e[gname] = {}
            for Ld in sorted(set(lead)):
                m = gm & (lead == Ld) & have
                e[gname][f"lead{Ld}"] = {"n": int(m.sum()), "own_after_pass": _ci(o[m], rng),
                                         "frac_read_alive": round(float((o[m] > 0).mean()), 3) if m.sum() else None,
                                         "own_to_move": round(float(np.nanmean(o2[m])), 3) if m.sum() else None}
        # liberty-matched contrast: cyclic (far victim) minus non-cyclic (all), per liberty count, leads >= 3
        rows = []
        for lb in range(2, 9):
            a = o[groups["cyclic_far_victim"] & (libs == lb) & (lead >= 3) & have]
            b = o[~cyc & (libs == lb) & (lead >= 3) & have]
            if len(a) >= 3 and len(b) >= 3:
                rows.append({"libs": lb, "n_cyc": len(a), "n_non": len(b), "cyc": round(float(a.mean()), 3),
                             "non": round(float(b.mean()), 3), "diff": round(float(a.mean() - b.mean()), 3)})
        e["by_libs_lead3plus"] = rows
        # regression: own_after_pass ~ cyclic + libs + log(size) + lead dummies
        m = have & (vic | (src == "pro") | ~cyc)
        X = [np.ones(m.sum()), cyc[m].astype(float), np.minimum(libs[m], 10), np.log(size[m])]
        for Ld in sorted(set(lead))[1:]:
            X.append((lead[m] == Ld).astype(float))
        X = np.stack(X, 1); y = o[m]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        boots = []
        ii = np.arange(len(y))
        for _ in range(2000):
            b = rng.choice(ii, len(ii))
            boots.append(np.linalg.lstsq(X[b], y[b], rcond=None)[0][1])
        e["cyclic_coef_adjusted"] = [round(float(beta[1]), 3), round(float(np.percentile(boots, 2.5)), 3),
                                     round(float(np.percentile(boots, 97.5)), 3)]
        rep["engines"][os.path.basename(run)] = e
    json.dump(rep, open(out, "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "extract", "score"])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--engine", default="katago")
    ap.add_argument("--ckpt")
    ap.add_argument("--run")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--only-pass", action="store_true")
    ap.add_argument("--n-pro", type=int, default=1500)
    ap.add_argument("--out", default="runs/real_report.json")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.n_pro)
    elif a.cmd == "extract":
        from .race_eval import _model
        pos = pickle.load(open(OUT, "rb"))
        idx = subset(pos) if a.subset else list(range(len(pos)))
        r = extract(_model(a.engine, a.ckpt), [pos[i] for i in idx], only_pass=a.only_pass)
        np.savez(os.path.join(a.run, "real.npz"), idx=np.asarray(idx), **r)
    else:
        print(json.dumps(score(a.runs, a.out), indent=1))


if __name__ == "__main__":
    main()

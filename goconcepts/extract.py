"""Build probe datasets: per-point activations at every layer + exact labels.

Natural set: positions sampled from human pro games (Fox 'Pro' collection),
split by game. Cyclic set: every stone in every cyclic test case.

Output: one .npz per set with
  acts      (P, L, C) float16   activations at P sampled points, L layers
  libs, pass_alive, eyespace, ladder, stone, chain, chain_size, cyclic  (P,) labels
  pos       (P,) position index;  game (P,) game index;  yx (P, 2)
  layer_names
Engine-specific code lives in the adapter; this file only needs
model.encode(gamestate) and model.activations(list).
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import random
from collections import deque

import numpy as np

from . import labels as L
from . import sgf
from .katago_adapter import KataGoProbeModel, gamestate_from_game, board_array
from .paths import NETS_DIR, SGF_DIR


def cyclic_chains(board: np.ndarray, ch: L.Chains) -> set[int]:
    """Chains that on their own enclose a region containing an opponent stone,
    i.e. removing the chain leaves a component that touches neither the edge
    nor any other stone of the chain's colour and contains an opponent stone."""
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
        for s in range(n * n):
            if s in seen or (b[s] == c):
                continue
            comp, q, touches_edge, has_opp, other_own = [], deque([s]), False, False, False
            seen.add(s)
            while q:
                u = q.popleft(); comp.append(u)
                y, x = divmod(u, n)
                if y in (0, n - 1) or x in (0, n - 1):
                    touches_edge = True
                if b[u] == L.opp(c):
                    has_opp = True
                for v in nb[u]:
                    if v in mine:
                        continue
                    if b[v] == c:
                        other_own = True
                        continue
                    if v not in seen:
                        seen.add(v); q.append(v)
            if has_opp and not touches_edge and not other_own:
                out.add(k)
                break
    return out


def _labels_for(gs):
    arr = board_array(gs)
    to_move = int(gs.board.pla)
    lab = L.all_labels(arr, to_move)
    ch = L.find_chains(arr)
    cyc = cyclic_chains(arr, ch)
    cyc_map = np.zeros(arr.size, np.int8)
    for k in cyc:
        cyc_map[ch.stones[k]] = 1
    lab["cyclic"] = cyc_map.reshape(arr.shape)
    return lab


def _collect(model, items, points_fn, batch=16, layers=None):
    """items: list of (game_idx, make_gamestate) where make_gamestate() builds the
    GameState on demand (a GameState keeps a Board per move, so holding 1000+
    of them at once exhausts memory). points_fn(lab, rng) -> list of (y,x)."""
    rng = np.random.default_rng(0)
    rows = {k: [] for k in ("chain_size", "libs", "pass_alive", "eyespace", "ladder", "stone", "chain", "cyclic", "pos", "game", "yx")}
    acts_out, policy_out = [], []
    for s in range(0, len(items), batch):
        chunk = [(gi, mk()) for gi, mk in items[s:s + batch]]
        enc = [model.encode(gs) for _, gs in chunk]
        acts = model.activations(enc, layers=layers)
        names = [n for n in model.layer_names if n in acts]
        for j, (gi, gs) in enumerate(chunk):
            lab = _labels_for(gs)
            pts = points_fn(lab, rng)
            pidx = s + j
            cid = lab["chain_id"]
            csz = np.bincount(cid[cid >= 0].ravel(), minlength=int(cid.max()) + 1) if (cid >= 0).any() else np.zeros(1, int)
            policy_out.append(acts["policy"][j])
            for (y, x) in pts:
                C = max(acts[nm].shape[1] for nm in names)
                vec = np.zeros((len(names), C), np.float16)
                for li, nm in enumerate(names):
                    v = acts[nm][j, :, y, x]
                    vec[li, :v.shape[0]] = v
                acts_out.append(vec)
                rows["libs"].append(lab["libs"][y, x]); rows["pass_alive"].append(lab["pass_alive"][y, x])
                rows["eyespace"].append(lab["eyespace"][y, x]); rows["ladder"].append(lab["ladder"][y, x])
                rows["stone"].append(lab["stone"][y, x]); rows["chain"].append(lab["chain_id"][y, x])
                rows["cyclic"].append(lab["cyclic"][y, x]); rows["pos"].append(pidx); rows["game"].append(gi)
                rows["yx"].append((y, x))
                rows["chain_size"].append(int(csz[cid[y, x]]))
        del acts
        try:  # glibc fragmentation from the per-batch activation buffers
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass
        import psutil
        print(f"  {min(s + batch, len(items))}/{len(items)} positions, {len(acts_out)} points, rss {psutil.Process().memory_info().rss // 2**20} MB", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    # layers differ in channel count (input has 22): zero-padded to the max
    out["acts"] = np.stack(acts_out)
    out["policy"] = np.asarray(policy_out, np.float32)
    out["layer_names"] = np.array(names)
    return out


def natural_points(k):
    def fn(lab, rng):
        ys, xs = np.nonzero(lab["stone"] > 0)
        if len(ys) == 0:
            return []
        # oversample high-liberty stones, which the input planes do not mark
        w = np.where(lab["libs"][ys, xs] >= 4, 2.0, 1.0)
        idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False, p=w / w.sum())
        return list(zip(ys[idx], xs[idx]))
    return fn


def all_stone_points(lab, rng):
    ys, xs = np.nonzero(lab["stone"] > 0)
    return list(zip(ys, xs))


def natural_items(sgf_dir, n_games, per_game, seed=0):
    files = sorted(glob.glob(os.path.join(sgf_dir, "**", "*.sgf"), recursive=True))
    rnd = random.Random(seed)
    rnd.shuffle(files)
    items, gi = [], 0
    for f in files:
        if gi >= n_games:
            break
        g = sgf.read(f)
        if g.size != 19 or len(g.moves) < 80 or g.setup_black or g.setup_white:
            continue
        ts = sorted(rnd.sample(range(30, len(g.moves)), per_game))
        try:
            gamestate_from_game(g)  # reject games with illegal moves up front
        except Exception:
            continue
        for t in ts:
            items.append((gi, (lambda g=g, t=t: gamestate_from_game(g, upto=t))))
        gi += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=NETS_DIR + "/humanv0.ckpt")
    ap.add_argument("--sgf-dir", default=SGF_DIR)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--out", default="runs/humansl")
    ap.add_argument("--which", default="natural,cyclic")
    ap.add_argument("--engine", default="katago", choices=["katago", "lz", "kgbin"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.engine == "lz":
        from .lz_adapter import LeelaZeroProbeModel
        model = LeelaZeroProbeModel(args.ckpt)
    elif args.engine == "kgbin":
        from .kgbin_adapter import KataGoBinProbeModel
        model = KataGoBinProbeModel(args.ckpt)
    else:
        model = KataGoProbeModel(args.ckpt)
    if "natural" in args.which:
        items = natural_items(args.sgf_dir, args.games, args.per_game)
        print(f"natural: {len(items)} positions from {args.games} games", flush=True)
        d = _collect(model, items, natural_points(args.points))
        np.savez(os.path.join(args.out, "natural.npz"), **d)
    if "cyclic" in args.which:
        from .cyclic import load_cases
        cases = load_cases()
        items = [(i, (lambda c=c: gamestate_from_game(c.game))) for i, c in enumerate(cases)]
        print(f"cyclic: {len(items)} cases", flush=True)
        d = _collect(model, items, all_stone_points)
        d["case_names"] = np.array([c.name for c in cases])
        corr = np.zeros((len(cases), 362), np.float32)
        for i, c in enumerate(cases):
            for col, mv in c.correct:
                corr[i, 361 if mv is None else mv[0] * 19 + mv[1]] = 1
        d["correct_moves"] = corr
        np.savez(os.path.join(args.out, "cyclic.npz"), **d)


if __name__ == "__main__":
    main()

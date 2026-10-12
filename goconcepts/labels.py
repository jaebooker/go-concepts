"""Exact concept labels computed from a Go board.

Engine-independent. Every label here is a deterministic function of the board
(plus, for ladders, a bounded exact search). No model is consulted.

Board convention: np.int8 array (N, N), 0 = empty, 1 = black, 2 = white.
Labels are returned in absolute colour; callers flip to side-to-move frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2


def opp(c: int) -> int:
    return 3 - c


def neighbours(n: int):
    """Precomputed orthogonal neighbour lists, indexed by flat point."""
    nb = []
    for i in range(n * n):
        y, x = divmod(i, n)
        lst = []
        if y > 0: lst.append(i - n)
        if y < n - 1: lst.append(i + n)
        if x > 0: lst.append(i - 1)
        if x < n - 1: lst.append(i + 1)
        nb.append(tuple(lst))
    return nb


_NB_CACHE: dict[int, list] = {}


def _nb(n: int):
    if n not in _NB_CACHE:
        _NB_CACHE[n] = neighbours(n)
    return _NB_CACHE[n]


@dataclass
class Chains:
    chain_id: np.ndarray          # (N*N,) int32, -1 on empty points
    color: list[int]              # colour of each chain
    stones: list[list[int]]       # flat points of each chain
    libs: list[set[int]]          # liberty points of each chain


def find_chains(board: np.ndarray) -> Chains:
    n = board.shape[0]
    b = board.reshape(-1)
    nb = _nb(n)
    cid = np.full(n * n, -1, dtype=np.int32)
    colors, stones, libs = [], [], []
    for p in range(n * n):
        if b[p] == EMPTY or cid[p] >= 0:
            continue
        c = b[p]
        k = len(colors)
        q = deque([p]); cid[p] = k
        st, lb = [], set()
        while q:
            u = q.popleft(); st.append(u)
            for v in nb[u]:
                if b[v] == EMPTY:
                    lb.add(v)
                elif b[v] == c and cid[v] < 0:
                    cid[v] = k; q.append(v)
        colors.append(int(c)); stones.append(st); libs.append(lb)
    return Chains(cid, colors, stones, libs)


def liberty_map(board: np.ndarray, ch: Chains | None = None) -> np.ndarray:
    """Per point: liberty count of the chain occupying it, 0 on empty."""
    ch = ch or find_chains(board)
    n = board.shape[0]
    out = np.zeros(n * n, dtype=np.int16)
    for k, st in enumerate(ch.stones):
        out[st] = len(ch.libs[k])
    return out.reshape(n, n)


# ---------------------------------------------------------------- Benson
def _regions(board: np.ndarray, color: int):
    """Maximal connected sets of points NOT of `color` (empty or opponent)."""
    n = board.shape[0]
    b = board.reshape(-1)
    nb = _nb(n)
    rid = np.full(n * n, -1, dtype=np.int32)
    regions = []
    for p in range(n * n):
        if b[p] == color or rid[p] >= 0:
            continue
        k = len(regions)
        q = deque([p]); rid[p] = k; pts = []
        while q:
            u = q.popleft(); pts.append(u)
            for v in nb[u]:
                if b[v] != color and rid[v] < 0:
                    rid[v] = k; q.append(v)
        regions.append(pts)
    return rid, regions


def benson(board: np.ndarray, color: int, ch: Chains | None = None):
    """Benson's algorithm for unconditional life.

    Returns (alive_chain_ids, vital_count) where vital_count maps every chain of
    `color` to the number of regions vital to it in the FIRST pass (a
    structural eye-space count), and alive_chain_ids is the set of chains that
    are pass-alive (cannot be captured even if the owner always passes).
    """
    ch = ch or find_chains(board)
    n = board.shape[0]
    b = board.reshape(-1)
    nb = _nb(n)
    rid, regions = _regions(board, color)

    X = {k for k, c in enumerate(ch.color) if c == color}
    # chains bordering each region; empty points of each region
    border = []
    empties = []
    for pts in regions:
        bd = set()
        for u in pts:
            for v in nb[u]:
                if b[v] == color:
                    bd.add(int(ch.chain_id[v]))
        border.append(bd)
        empties.append([u for u in pts if b[u] == EMPTY])
    R = set(range(len(regions)))

    def vital_to(r, x):
        libs = ch.libs[x]
        return all(e in libs for e in empties[r])

    first_counts = None
    while True:
        counts = {x: 0 for x in X}
        for r in R:
            for x in border[r]:
                if x in X and vital_to(r, x):
                    counts[x] += 1
        if first_counts is None:
            first_counts = dict(counts)
        drop = {x for x in X if counts[x] < 2}
        if not drop:
            break
        X -= drop
        R = {r for r in R if border[r] <= X}
    return X, first_counts


def eye_regions(board: np.ndarray, ch: Chains | None = None) -> dict[int, int]:
    """Per chain: number of enclosed regions vital to it that contain no
    opponent stones and have at most 7 points. A conservative, exact
    'eye-shaped space' count; it is not a life-and-death judgement."""
    ch = ch or find_chains(board)
    b = board.reshape(-1)
    nb = _nb(board.shape[0])
    out = {k: 0 for k in range(len(ch.color))}
    for color in (BLACK, WHITE):
        rid, regions = _regions(board, color)
        for pts in regions:
            if len(pts) > 7 or any(b[u] != EMPTY for u in pts):
                continue
            bd = {int(ch.chain_id[v]) for u in pts for v in nb[u] if b[v] == color}
            for x in bd:
                if all(u in ch.libs[x] for u in pts):
                    out[x] += 1
    return out


# ---------------------------------------------------------------- ladders
def _play(b: np.ndarray, n: int, p: int, c: int):
    """Play on a flat board copy in place. Returns captured count or -1 if illegal."""
    if b[p] != EMPTY:
        return -1
    nb = _nb(n)
    b[p] = c
    captured = 0
    for v in nb[p]:
        if b[v] == opp(c):
            grp, lib = _group(b, n, v)
            if not lib:
                b[list(grp)] = EMPTY
                captured += len(grp)
    grp, lib = _group(b, n, p)
    if not lib:
        b[p] = EMPTY
        return -1
    return captured


def _group(b: np.ndarray, n: int, p: int):
    nb = _nb(n)
    c = b[p]
    seen = {p}; q = [p]; libs = set()
    while q:
        u = q.pop()
        for v in nb[u]:
            if b[v] == EMPTY:
                libs.add(v)
            elif b[v] == c and v not in seen:
                seen.add(v); q.append(v)
    return seen, libs


class LadderBudget(Exception):
    pass


def _ladder_captured(b: np.ndarray, n: int, p: int, defender_to_move: bool, depth: int = 0, budget=None) -> bool:
    """Exact ladder reading: attacker may only play atari moves on the target,
    defender may extend or capture an adjacent attacker chain in atari.
    Raises LadderBudget if the search exceeds its node budget."""
    if budget is None:
        budget = [5000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 400:
        raise LadderBudget()
    grp, libs = _group(b, n, p)
    d = b[p]; a = opp(d)
    if defender_to_move:
        if len(libs) != 1:
            return len(libs) == 0
        # defender options: capture an adjacent attacker chain in atari, or extend
        options = set(libs)
        for u in grp:
            for v in _nb(n)[u]:
                if b[v] == a:
                    ag, al = _group(b, n, v)
                    if len(al) == 1:
                        options |= al
        for m in options:
            bb = b.copy()
            if _play(bb, n, m, d) < 0:
                continue
            g2, l2 = _group(bb, n, p)
            if len(l2) >= 3:
                return False
            if len(l2) == 2 and not _ladder_captured(bb, n, p, False, depth + 1, budget):
                return False
        return True
    else:
        if len(libs) == 1:
            return True
        if len(libs) != 2:
            return False
        for m in libs:
            bb = b.copy()
            if _play(bb, n, m, a) < 0:
                continue
            if bb[p] == EMPTY:
                return True
            g2, l2 = _group(bb, n, p)
            if len(l2) == 1 and _ladder_captured(bb, n, p, True, depth + 1, budget):
                return True
        return False


def ladder_map(board: np.ndarray, to_move: int, ch: Chains | None = None) -> np.ndarray:
    """Per stone: 1 if its chain is captured by a ladder with `to_move` to play,
    0 if not, -1 where undefined (empty, chain with >= 3 liberties, a
    2-liberty chain whose owner is to move, or a search over budget)."""
    ch = ch or find_chains(board)
    n = board.shape[0]
    b = board.reshape(-1)
    out = np.full(n * n, -1, dtype=np.int8)
    for k, st in enumerate(ch.stones):
        L = len(ch.libs[k]); c = ch.color[k]
        try:
            if L == 1:
                val = _ladder_captured(b.copy(), n, st[0], defender_to_move=(c == to_move))
            elif L == 2 and c != to_move:
                val = _ladder_captured(b.copy(), n, st[0], defender_to_move=False)
            else:
                continue
        except LadderBudget:
            continue  # search too large to settle exactly: leave undefined
        out[st] = int(val)
    return out.reshape(n, n)


# ---------------------------------------------------------------- bundle
def all_labels(board: np.ndarray, to_move: int) -> dict[str, np.ndarray]:
    """All per-point labels, in the side-to-move frame ('own' = to_move)."""
    n = board.shape[0]
    ch = find_chains(board)
    libs = liberty_map(board, ch)
    alive = np.zeros(n * n, dtype=np.int8)
    eyespace = np.zeros(n * n, dtype=np.int8)
    for color in (BLACK, WHITE):
        X, _ = benson(board, color, ch)
        for x in X:
            alive[ch.stones[x]] = 1
    er = eye_regions(board, ch)
    for k, st in enumerate(ch.stones):
        eyespace[st] = min(er[k], 2)
    stone = board.reshape(-1)
    rel = np.where(stone == EMPTY, 0, np.where(stone == to_move, 1, 2)).astype(np.int8)
    return {
        "stone": rel.reshape(n, n),                 # 0 empty, 1 own, 2 opp
        "chain_id": ch.chain_id.reshape(n, n),
        "libs": libs,
        "pass_alive": alive.reshape(n, n),
        "eyespace": eyespace.reshape(n, n),
        "ladder": ladder_map(board, to_move, ch),
    }

"""Procedural cyclic-group minimal pairs.

Why not reflections or colour swaps of lightvector's cases: both engines were
trained with all 8 board symmetries and see the board relative to the side to
move, so those variants give near-identical activations. They inflate n
without adding independent chains.

Each pair here is two positions that differ by one stone:
  cyclic  a ring chain of colour R that on its own encloses a group of the
          other colour (the inner group), with a target liberty count
  cut     the same position with one ring stone replaced by an inner-colour
          stone, so the ring becomes an open C-shaped chain of nearly the same
          size and liberties that encloses nothing

Chain size, shape, surroundings and liberties are matched; the only thing that
differs is whether the chain closes on itself. Context stones are scattered
elsewhere so the board is not empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import labels as L
from .sgf import Game

N = 19
NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Pair:
    pid: int
    cyclic: Game
    cut: Game
    ring_point: tuple          # a stone on the ring / C chain, same point in both
    target_libs: int
    libs_cyclic: int
    libs_cut: int
    ring_size: int
    inner_size: int


def _grow_region(rng, center, size):
    """Random 4-connected blob, kept at least 3 points from the edge."""
    R = {center}
    frontier = [center]
    while len(R) < size:
        y, x = frontier[rng.integers(len(frontier))]
        dy, dx = NB4[rng.integers(4)]
        p = (y + dy, x + dx)
        if 3 <= p[0] <= N - 4 and 3 <= p[1] <= N - 4 and p not in R:
            R.add(p); frontier.append(p)
    return R


def _fill_holes(R):
    """Make R simply connected: add any point not reachable from the edge."""
    inside = np.zeros((N, N), bool)
    for p in R:
        inside[p] = True
    seen = np.zeros((N, N), bool)
    stack = [(y, x) for y in range(N) for x in (0, N - 1)] + [(y, x) for x in range(N) for y in (0, N - 1)]
    for p in stack:
        seen[p] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in NB4:
            q = (y + dy, x + dx)
            if 0 <= q[0] < N and 0 <= q[1] < N and not seen[q] and not inside[q]:
                seen[q] = True; stack.append(q)
    return {(y, x) for y in range(N) for x in range(N) if not seen[y, x]}


def _ring(R):
    """8-neighbour outer boundary of R: points outside R at Chebyshev distance 1."""
    out = set()
    for (y, x) in R:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y + dy, x + dx)
                if q not in R:
                    out.add(q)
    return out


def _chain_of(board, p):
    ch = L.find_chains(board)
    k = ch.chain_id[p[0] * N + p[1]]
    return ch, k


def _legalize(board, protect):
    """Remove zero-liberty chains not in `protect` until none remain.
    Returns False if a protected chain has no liberties."""
    while True:
        ch = L.find_chains(board)
        dead = [k for k in range(len(ch.color)) if len(ch.libs[k]) == 0]
        if not dead:
            return True
        for k in dead:
            pts = ch.stones[k]
            if any(divmod(p, N) in protect for p in pts):
                return False
            board.reshape(-1)[pts] = 0


def _to_game(board, to_move, recent):
    """Setup stones plus the `recent` context stones replayed as the last
    moves, so both engines see a full, ordinary move history (Leela Zero
    zero-fills missing history planes, which setup-only positions would hit)."""
    g = Game(size=N)
    rec = set(recent)
    g.setup_black = [tuple(p) for p in np.argwhere(board == 1) if tuple(p) not in rec]
    g.setup_white = [tuple(p) for p in np.argwhere(board == 2) if tuple(p) not in rec]
    g.moves = [(int(board[p]), p) for p in recent]
    g.first_player = int(board[recent[0]])
    return g


def _recent_moves(rng, b, near, to_move, k=8):
    """Pick k context stones, alternating colour, the last played by the
    opponent of `to_move`. They are far from the structure."""
    cols = [3 - to_move if i % 2 == 0 else to_move for i in range(k)][::-1]
    pools = {c: [tuple(p) for p in np.argwhere(b == c) if tuple(p) not in near] for c in (1, 2)}
    out = []
    for c in cols:
        if not pools[c]:
            return None
        out.append(pools[c].pop(int(rng.integers(len(pools[c])))))
    return out


def make_pair(rng, pid, max_tries=200) -> Pair | None:
    for _ in range(max_tries):
        size = int(rng.integers(3, 22))
        c = (int(rng.integers(6, 13)), int(rng.integers(6, 13)))
        R = _fill_holes(_grow_region(rng, c, size))
        ring = _ring(R)
        if any(not (1 <= y <= N - 2 and 1 <= x <= N - 2) for y, x in ring):
            continue
        rc = int(rng.integers(1, 3))            # ring colour
        ic = 3 - rc                              # inner / attacker colour
        target = int(rng.integers(2, 9))         # ring liberties, 2..8
        b = np.zeros((N, N), np.int8)
        for p in ring:
            b[p] = rc
        # inside: inner-colour stones with k empty points (inner liberties)
        Rl = sorted(R)
        k_in = int(rng.integers(1, min(len(Rl), max(1, target - 1)) + 1)) if len(Rl) > 1 else 1
        empties = set(map(tuple, (Rl[i] for i in rng.choice(len(Rl), size=min(k_in, len(Rl)), replace=False))))
        for p in Rl:
            if p not in empties:
                b[p] = ic
        if not any(b[p] == ic for p in Rl):
            continue
        # outside liberties of the ring
        outer = sorted({(y + dy, x + dx) for (y, x) in ring for dy, dx in NB4
                        if (y + dy, x + dx) not in ring and (y + dy, x + dx) not in R})
        inner_libs = sum(1 for p in empties if any((p[0] + dy, p[1] + dx) in ring for dy, dx in NB4))
        keep_out = target - inner_libs
        if keep_out < 0 or keep_out > len(outer):
            continue
        fill = rng.choice(len(outer), size=len(outer) - keep_out, replace=False)
        for i in fill:
            b[outer[i]] = ic
        # context stones away from the structure
        struct = R | ring | set(outer)
        near = {(y + dy, x + dx) for (y, x) in struct for dy in range(-2, 3) for dx in range(-2, 3)}
        for y in range(N):
            for x in range(N):
                if (y, x) not in near and rng.random() < 0.18:
                    b[y, x] = 1 if rng.random() < 0.5 else 2
        rp = sorted(ring)[0]
        if not _legalize(b, protect=ring | R | set(outer)):
            continue
        ch, kr = _chain_of(b, rp)
        if set(divmod(s, N) for s in ch.stones[kr]) != ring:
            continue  # ring is not one chain (or merged with something)
        libs_cyc = len(ch.libs[kr])
        if libs_cyc != target:
            continue
        if kr not in _cyclic(b, ch):
            continue
        # cut: replace one ring stone that touches both R and the outside
        cands = [p for p in sorted(ring)
                 if p != rp
                 and any((p[0] + dy, p[1] + dx) in R for dy, dx in NB4)
                 and any((p[0] + dy, p[1] + dx) not in R and (p[0] + dy, p[1] + dx) not in ring for dy, dx in NB4)]
        rng.shuffle(cands)
        for cp in cands:
            bc = b.copy()
            bc[cp] = ic
            if not _legalize(bc, protect=ring | R):
                continue
            chc, kc = _chain_of(bc, rp)
            if set(divmod(s, N) for s in chc.stones[kc]) != ring - {cp}:
                continue  # removing that stone split the ring
            if kc in _cyclic(bc, chc):
                continue
            to_move = int(rng.integers(1, 3))
            recent = _recent_moves(rng, b, near, to_move)
            if recent is None:
                break
            return Pair(pid, _to_game(b, to_move, recent), _to_game(bc, to_move, recent), rp, target,
                        libs_cyc, len(chc.libs[kc]), len(ring), int(sum(b[p] == ic for p in R)))
    return None


def _cyclic(board, ch):
    from .extract import cyclic_chains
    return cyclic_chains(board, ch)


def generate(n_pairs, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_pairs:
        p = make_pair(rng, len(out))
        if p is not None:
            out.append(p)
    return out

"""Capturing-race positions with exactly solved correct moves.

A race position: a ring chain of colour D encloses a single chain of colour A
(the inner group). Every empty point inside is a liberty of both (shared), so
the inner group has only shared liberties and no eyes; the ring also has r
outside liberties. D is to move. D wins by capturing the inner group, A wins by
capturing the ring.

`solve` plays the race out exactly over the only moves that matter (the
liberties of the chains in the structure, plus pass for "play elsewhere") with
memoised minimax. A position is kept only if it is URGENT: some race move wins
for D, and passing loses. The correct set is every winning race move. A net
that plays elsewhere, or fills the wrong liberty, loses a ring of 12 to 35
stones.

Construction makes the race closed: the ring's r outside liberties are dead
ends walled in by A stones (extending gains nothing), the inner group has s
liberties shared with the ring and e single-point eyes. `solve` is an exact
memoised minimax over the liberties in that closed region, with outcomes
capture (2) > seki (1) > ring captured (0). A position is kept only if some D
move captures the inner group AND passing loses the ring outright. Correct
moves: the capturing ones. So a net that plays elsewhere, or plays a liberty in
the wrong order, loses a 12 to 35 stone ring.

Control: the same race counting with no cyclic group. The inner group sits in a
pocket formed by a D chain and the board edge, so the D chain encloses nothing.
Matched on s, r and roughly on size; not a one-stone minimal pair.

The positions are generated fresh from a seed, so no engine can have trained
on them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import labels as L
from .cyclegen import (N, NB4, _chain_of, _cyclic, _fill_holes, _grow_region, _legalize, _recent_moves,
                       _ring, _to_game)
from .sgf import Game


@dataclass
class RaceCase:
    pid: int
    variant: int                 # 0 cyclic ring, 1 edge-pocket control
    game: Game
    ring_point: tuple            # a stone of the D chain that matters (ring, or C chain)
    inner_point: tuple
    correct: list = field(default_factory=list)     # flat indices y*19+x
    ring_libs: int = 0
    inner_libs: int = 0
    ring_size: int = 0
    shared: int = 0
    eyes: int = 0
    outside: int = 0
    saving: list = field(default_factory=list)   # moves that at least avoid losing the ring


class _Unresolved(Exception):
    pass


def solve(board, d_color, ring_point, inner_point, region, max_depth=40):
    """Exact minimax over the closed race region, D to move.
    Outcome values: 2 inner group captured, 1 seki (both pass, nothing
    captured), 0 ring captured. D maximises, A minimises.
    Returns (moves with value 2, value after a D pass, {move: value})."""
    n = board.shape[0]
    memo = {}
    ring_flat = ring_point[0] * n + ring_point[1]
    inner_flat = inner_point[0] * n + inner_point[1]

    def moves(b):
        ys = set()
        for p in (ring_flat, inner_flat):
            if b[p] != 0:
                ys |= L._group(b, n, p)[1]
        return sorted(q for q in ys if q in region)

    def value(b, side, passes, depth):
        if b[ring_flat] != d_color:
            return 0
        if b[inner_flat] != 3 - d_color:
            return 2
        if passes >= 2:
            return 1
        if depth > max_depth:
            raise _Unresolved()
        key = (b.tobytes(), side, passes)
        if key in memo:
            return memo[key]
        vals = []
        for m in moves(b):
            bb = b.copy()
            if L._play(bb, n, m, side) < 0:
                continue
            vals.append(value(bb, 3 - side, 0, depth + 1))
        vals.append(value(b, 3 - side, passes + 1, depth + 1))
        v = max(vals) if side == d_color else min(vals)
        memo[key] = v
        return v

    b0 = board.reshape(-1).copy()
    per_move = {}
    for m in moves(b0):
        bb = b0.copy()
        if L._play(bb, n, m, d_color) < 0:
            continue
        per_move[m] = value(bb, 3 - d_color, 0, 1)
    # the hypothetical D pass does not count toward a double-pass ending:
    # it asks "what if D plays elsewhere", after which play goes on normally
    pass_value = value(b0, 3 - d_color, 0, 1)
    return sorted(m for m, v in per_move.items() if v == 2), pass_value, per_move


def _no_other_ataris(b, keep):
    ch = L.find_chains(b)
    for k in range(len(ch.color)):
        if len(ch.libs[k]) == 1 and not any(divmod(s, N) in keep for s in ch.stones[k]):
            return False
    return True


def make_race(rng, pid, edge=False, max_tries=400, combo=None, want="urgent", target_libs=None, real_eyes=False,
              size_range=(4, 16), shape="std"):
    """One race where D, to move, must act: some move captures the inner group,
    and if D plays elsewhere the ring is captured (exact search).
    edge=False: the D chain is a closed ring (cyclic).
    edge=True: the inner group sits in a pocket of a D chain and the board
    edge, so no chain encloses anything (non-cyclic control)."""
    for _ in range(max_tries):
        size = int(rng.integers(*size_range))
        if edge:
            c = (0, int(rng.integers(5, 14)))
            R = _grow_region_edge(rng, c, size)
        elif shape == "edge1":
            c = (1, int(rng.integers(5, 14)))
            R = _fill_holes(_grow_region_rows(rng, c, size, ymin=1, ymax=6))
        else:
            c = (int(rng.integers(6, 13)), int(rng.integers(6, 13)))
            R = _fill_holes(_grow_region(rng, c, size))
        ring = {p for p in _ring(R) if 0 <= p[0] < N and 0 <= p[1] < N}
        if shape == "edge1":
            # the ring runs along the first line: its top side is the board edge row 0
            if not any(y == 0 for y, x in ring):
                continue
        if shape == "thick":
            ring = ring | {p for p in _ring(R | ring) if 0 <= p[0] < N and 0 <= p[1] < N}
        if shape == "tail":
            cy = sum(p[0] for p in R) / len(R); cx = sum(p[1] for p in R) / len(R)
            rp0 = sorted(ring)[int(rng.integers(len(ring)))]
            vy, vx = rp0[0] - cy, rp0[1] - cx
            d = (int(np.sign(vy)), 0) if abs(vy) >= abs(vx) else (0, int(np.sign(vx)))
            if d == (0, 0):
                continue
            k = int(rng.integers(4, 7))
            arm = [(rp0[0] + d[0] * i, rp0[1] + d[1] * i) for i in range(1, k + 1)]
            if any(not (1 <= y <= N - 2 and 1 <= x <= N - 2) or p in ring or p in R for p in arm for y, x in [p]):
                continue
            ring = ring | set(arm)
        lo = 0 if (edge or shape in ("edge1", "thick", "tail")) else 2
        hi_ = N - 1 if shape in ("thick", "tail") else N - 3
        xlo = 0 if shape in ("thick", "tail") else 2
        if any(not (lo <= y <= hi_ and xlo <= x <= hi_) for y, x in ring):
            continue
        D = int(rng.integers(1, 3)); A = 3 - D
        Rl = sorted(R)
        touching = [p for p in Rl if any((p[0] + dy, p[1] + dx) in ring for dy, dx in NB4)]
        interior = [p for p in Rl if p not in touching]
        if combo is not None:
            ns, ne, r = combo
        else:
            ns = int(rng.integers(1, 5)); ne = int(rng.integers(0, 3)); r = int(rng.integers(0, 5))
        if ns + ne + r > 9 or len(touching) < ns or len(interior) < ne or ns + ne >= len(Rl):
            continue
        shared = set(touching[i] for i in rng.choice(len(touching), size=ns, replace=False))
        eyes = set(interior[i] for i in rng.choice(len(interior), size=ne, replace=False)) if ne else set()
        # real_eyes: every eye point is surrounded by inner stones only, so it
        # cannot merge with a shared liberty into a larger, non-eye space
        if real_eyes and any((e[0] + dy, e[1] + dx) in shared or (e[0] + dy, e[1] + dx) in eyes
                             or (e[0] + dy, e[1] + dx) not in R for e in eyes for dy, dx in NB4):
            continue
        b = np.zeros((N, N), np.int8)
        for p in ring:
            b[p] = D
        for p in Rl:
            if p not in shared and p not in eyes:
                b[p] = A
        outer = sorted({(y + dy, x + dx) for (y, x) in ring for dy, dx in NB4
                        if 0 <= y + dy < N and 0 <= x + dx < N
                        and (y + dy, x + dx) not in ring and (y + dy, x + dx) not in R})
        if r > len(outer):
            continue
        keep_empty = set(outer[i] for i in rng.choice(len(outer), size=r, replace=False)) if r else set()
        for p in outer:
            if p not in keep_empty:
                b[p] = A
        wall = {(p[0] + dy, p[1] + dx) for p in keep_empty for dy, dx in NB4} - R - ring - set(outer)
        if any(not (0 <= y < N and 0 <= x < N) for y, x in wall):
            continue
        for p in wall:
            b[p] = A
        struct = R | ring | set(outer) | wall
        near = {(y + dy, x + dx) for (y, x) in struct for dy in range(-2, 3) for dx in range(-2, 3)}
        for y in range(N):
            for x in range(N):
                if (y, x) not in near and rng.random() < 0.18:
                    b[y, x] = 1 if rng.random() < 0.5 else 2
        if not _legalize(b, protect=struct):
            continue
        rp = sorted(ring)[0]
        ch, kr = _chain_of(b, rp)
        if set(divmod(q, N) for q in ch.stones[kr]) != ring:
            continue
        if (kr in _cyclic(b, ch)) == edge:
            continue
        inner_stones = [p for p in Rl if b[p] == A]
        ip = inner_stones[0]
        ci = ch.chain_id[ip[0] * N + ip[1]]
        if set(divmod(q, N) for q in ch.stones[ci]) != set(inner_stones):
            continue
        inner_libs = set(divmod(q, N) for q in ch.libs[ci])
        ring_libs = set(divmod(q, N) for q in ch.libs[kr])
        if inner_libs != shared | eyes or ring_libs != shared | keep_empty:
            continue
        # outside liberties are dead ends: extending gains nothing
        if any(b[(p[0] + dy, p[1] + dx)] == 0 for p in keep_empty for dy, dx in NB4
               if 0 <= p[0] + dy < N and 0 <= p[1] + dx < N):
            continue
        if not _no_other_ataris(b, keep=ring | set(inner_stones)):
            continue
        region = {q[0] * N + q[1] for q in ring_libs | inner_libs}
        try:
            win, pass_value, per_move = solve(b, D, rp, ip, region)
        except _Unresolved:
            continue
        if target_libs is not None and len(ring_libs) != target_libs:
            continue
        if want == "urgent" and (not win or pass_value != 0):
            continue
        if want == "safe" and pass_value != 2:
            continue
        # "dead": the ring is lost after a pass and no D move saves it either
        if want == "dead" and (win or pass_value != 0):
            continue
        recent = _recent_moves(rng, b, near, D)
        if recent is None:
            continue
        # A's last move is in the race: one of its stones next to the ring,
        # so the fight is where the opponent just played, as in a real game
        # (and as policy nets expect). The seven earlier moves are context.
        attackers = [p for p in outer if b[p] == A]
        if not attackers:
            continue
        recent[-1] = attackers[int(rng.integers(len(attackers)))]
        case = RaceCase(pid, int(edge), _to_game(b, D, recent), rp, ip, win,
                        len(ring_libs), len(inner_libs), len(ring))
        case.shared, case.eyes, case.outside = ns, ne, r
        case.saving = sorted(m for m, v in per_move.items() if v >= 1)
        return case
    return None


def _grow_region_rows(rng, center, size, ymin=1, ymax=6):
    """Random 4-connected blob in rows ymin..ymax, away from the side edges."""
    R = {center}
    frontier = [center]
    tries = 0
    while len(R) < size and tries < 10000:
        tries += 1
        y, x = frontier[rng.integers(len(frontier))]
        dy, dx = NB4[rng.integers(4)]
        p = (y + dy, x + dx)
        if ymin <= p[0] <= ymax and 3 <= p[1] <= N - 4 and p not in R:
            R.add(p); frontier.append(p)
    return R


def _grow_region_edge(rng, center, size):
    """Random 4-connected blob touching the top edge (row 0)."""
    R = {center}
    frontier = [center]
    while len(R) < size:
        y, x = frontier[rng.integers(len(frontier))]
        dy, dx = NB4[rng.integers(4)]
        p = (y + dy, x + dx)
        if 0 <= p[0] <= 5 and 3 <= p[1] <= N - 4 and p not in R:
            R.add(p); frontier.append(p)
    return R


URGENT_COMBOS = [(1, 0, 0), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 2, 3), (2, 1, 3)]
# (shared, eyes, outside). The only combinations (up to 9 liberties in all) for
# which random constructions came out urgent in a sweep; ring liberties are
# shared + outside: 1, 2, 3, 3, 4, 5.


def generate(n_per_cell, seed=777, combos=URGENT_COMBOS):
    """For each combo, n cyclic-ring races and n edge-pocket controls."""
    rng = np.random.default_rng(seed)
    out = []
    for combo in combos:
        for i in range(n_per_cell):
            for edge in (False, True):
                c = None
                while c is None:
                    c = make_race(rng, len(out), edge=edge, max_tries=200, combo=combo)
                out.append(c)
    return out

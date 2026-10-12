"""A stricter exact label for the race positions: the solver may also attack the walls.

The original solver (racegen.solve) plays only on the liberties of the ring and the inner
group. The opponent stones walling in the ring's outside liberties are assumed safe. In
the generated positions they are often weak (2 or 3 liberties), so D might save the ring
by capturing a wall stone. solve_ext adds, as moves for both sides, the liberties of every
wall chain (opponent chains touching the ring or the race region) that has 3 or fewer
liberties, and scores the capture of any original wall stone as an escape for D (value 2,
the same as capturing the inner group). "Doomed" under this solver means D loses the ring
after a pass even when it may attack the walls, a conservative label.
"""
from __future__ import annotations

import numpy as np

from . import labels as L
from .cyclegen import N, NB4


class Budget(Exception):
    pass


def solve_ext(board, D, rp, ip, max_nodes=400000, max_depth=40):
    A = 3 - D
    n = N
    b0 = board.reshape(-1).copy()
    ring_f = rp[0] * n + rp[1]; inner_f = ip[0] * n + ip[1]
    rg, rl = L._group(b0, n, ring_f); ig, il = L._group(b0, n, inner_f)
    region = rl | il
    walls0 = set()
    for p in list(rg) + list(region):
        for dy, dx in NB4:
            y, x = divmod(p, n); q = (y + dy) * n + (x + dx)
            if 0 <= y + dy < n and 0 <= x + dx < n and b0[q] == A and q not in ig:
                walls0 |= L._group(b0, n, q)[0]
    walls0 = frozenset(walls0)
    memo = {}
    nodes = [0]

    def moves(b):
        race, extra = set(), set()
        for p in (ring_f, inner_f):
            race |= L._group(b, n, p)[1]
        seen = set()
        for w in walls0:
            if b[w] != A or w in seen:
                continue
            g, lib = L._group(b, n, w)
            seen |= g
            if len(lib) <= 3:
                extra |= lib
        return sorted(race) + sorted(extra - race)

    def value(b, side, passes, depth):
        if b[ring_f] != D:
            return 0
        if b[inner_f] != A:
            return 2
        if any(b[w] != A for w in walls0):
            return 2
        if passes >= 2:
            return 1
        nodes[0] += 1
        if nodes[0] > max_nodes or depth > max_depth:
            raise Budget()
        key = (b.tobytes(), side, passes)
        if key in memo:
            return memo[key]
        best = 0 if side == D else 2
        target = 2 if side == D else 0
        for m in moves(b):
            bb = b.copy()
            if L._play(bb, n, m, side) < 0:
                continue
            v = value(bb, 3 - side, 0, depth + 1)
            best = max(best, v) if side == D else min(best, v)
            if best == target:
                break
        if best != target:
            v = value(b, 3 - side, passes + 1, depth + 1)
            best = max(best, v) if side == D else min(best, v)
        memo[key] = best
        return best

    return value(b0, A, 0, 1)        # D has passed: A to move


def doomed_ext(board, D, rp, ip, **kw):
    """True: ring lost after a D pass even with wall attacks. False: not. None: over budget."""
    try:
        return solve_ext(board, D, rp, ip, **kw) == 0
    except (Budget, RecursionError):
        return None


def wall_chains(b, D, rp, ip):
    """Opponent chains (other than the inner group) touching the ring or the race region."""
    A = 3 - D
    ch = L.find_chains(b)
    kr = ch.chain_id[rp[0] * N + rp[1]]; ki = ch.chain_id[ip[0] * N + ip[1]]
    pts = {divmod(q, N) for q in ch.stones[kr]} | {divmod(q, N) for q in ch.libs[kr]} | {divmod(q, N) for q in ch.libs[ki]}
    ws = set()
    for (y, x) in pts:
        for dy, dx in NB4:
            q = (y + dy, x + dx)
            if 0 <= q[0] < N and 0 <= q[1] < N and b[q] == A:
                k = ch.chain_id[q[0] * N + q[1]]
                if k != ki:
                    ws.add(k)
    return ch, kr, ki, ws


def fortify(board, D, rp, ip, margin=2, max_stones=40):
    """Add opponent stones on wall liberties outside the race region until every wall
    chain has at least ring liberties + margin liberties. Returns the new board or None."""
    b = board.copy(); A = 3 - D
    for _ in range(max_stones):
        ch, kr, ki, ws = wall_chains(b, D, rp, ip)
        need = len(ch.libs[kr]) + margin
        region = set(ch.libs[kr]) | set(ch.libs[ki])
        weak = [k for k in ws if len(ch.libs[k]) < need]
        if not weak:
            return b
        k = min(weak, key=lambda k: len(ch.libs[k]))
        best, bestv = None, -1
        for q in ch.libs[k]:
            if q in region:
                continue
            bb = b.reshape(-1).copy()
            if L._play(bb, N, q, A) < 0:
                continue
            g, lib = L._group(bb, N, q)
            if (bb.reshape(N, N) != b).sum() != 1:      # no captures
                continue
            if len(lib) > bestv:
                best, bestv = q, len(lib)
        if best is None:
            return None
        b.reshape(-1)[best] = A
    return None

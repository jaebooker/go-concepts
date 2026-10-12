"""lightvector's cyclic-group test suite (via FAR AI's go_attack repo,
scripts/runcycletestsandplot/sgfs). Each SGF holds one or more test cases:
a setup position, moves since setup, and either CORRECT/WRONG answer moves
(node commented START) or a target winner (BLACKWIN / WHITEWIN).

This is the stage-three position set: hand-built positions whose difficulty
is exactly cyclic-group liberty and status reading.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import sgfmill.sgf
import sgfmill.sgf_properties

from .sgf import Game

CYCLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cyclic_sgfs")


@dataclass
class CyclicCase:
    name: str
    game: Game                      # setup stones + moves since setup
    correct: list = field(default_factory=list)   # (color, (y, x) or None)
    wrong: list = field(default_factory=list)
    target_winner: int | None = None             # 1 black, 2 white


def _col(c):
    return 1 if c == "b" else 2


def _walk(name, node, setup, moves, out):
    ab, aw, ae = node.get_setup_stones()
    if ab or aw or ae:
        pts = dict(setup)
        for p in ae: pts.pop(p, None)
        for p in ab: pts[p] = 1
        for p in aw: pts[p] = 2
        setup = pts
        moves = []
    color, raw = node.get_raw_move()
    if color:
        mv = sgfmill.sgf_properties.interpret_go_point(raw, 19)
        moves = moves + [(_col(color), _flip(mv))]

    def mk():
        g = Game(size=19)
        g.setup_black = [_flip(p) for p, c in setup.items() if c == 1]
        g.setup_white = [_flip(p) for p, c in setup.items() if c == 2]
        g.moves = list(moves)
        return g

    if node.has_property("C"):
        c = node.get_raw("C").decode().strip()
        if c == "START":
            case = CyclicCase(f"{name}#{len(out)}", mk())
            for child in node:
                if child.has_property("C"):
                    cc = child.get_raw("C").decode().strip()
                    col, r = child.get_raw_move()
                    if col and cc in ("CORRECT", "WRONG"):
                        mv = (_col(col), _flip(sgfmill.sgf_properties.interpret_go_point(r, 19)))
                        (case.correct if cc == "CORRECT" else case.wrong).append(mv)
            out.append(case)
        elif c in ("BLACKWIN", "WHITEWIN"):
            # the commented node's move is part of the position
            out.append(CyclicCase(f"{name}#{len(out)}", mk(), target_winner=1 if c == "BLACKWIN" else 2))
    for child in node:
        _walk(name, child, setup, moves, out)


def _flip(p):
    """sgfmill uses (row from bottom, col); convert to (y from top, x)."""
    if p is None:
        return None
    r, c = p
    return (18 - r, c)


def load_cases(directory: str = CYCLIC_DIR) -> list[CyclicCase]:
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "*.sgf"))):
        with open(f, "rb") as fh:
            root = sgfmill.sgf.Sgf_game.from_bytes(fh.read()).get_root()
        _walk(os.path.basename(f)[:-4], root, {}, [], out)
    return out

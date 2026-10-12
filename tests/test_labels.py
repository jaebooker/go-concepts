"""Labels are checked against KataGo's independent Board implementation on
real pro positions, plus hand-built cases for Benson and eyes."""
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from goconcepts import labels as L, sgf  # noqa: E402
from goconcepts.katago_adapter import Board, gamestate_from_game, board_array  # noqa: E402

SGF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sgf")


def _positions(n_games=30, seed=0):
    files = sorted(glob.glob(os.path.join(SGF_DIR, "**", "*.sgf"), recursive=True))
    rnd = random.Random(seed)
    for f in rnd.sample(files, n_games):
        g = sgf.read(f)
        if g.size != 19 or len(g.moves) < 60:
            continue
        for t in rnd.sample(range(20, len(g.moves)), 4):
            try:
                gs = gamestate_from_game(g, upto=t)
            except Exception:
                break
            yield gs


def test_liberties_and_ladders_match_katago():
    checked_lad = 0
    for gs in _positions():
        b = gs.board
        arr = board_array(gs)
        lab = L.all_labels(arr, int(b.pla))
        for y in range(19):
            for x in range(19):
                loc = b.loc(x, y)
                if b.board[loc] in (Board.BLACK, Board.WHITE):
                    assert lab["libs"][y, x] == b.num_liberties(loc), (y, x)
        # ladders: compare with KataGo's search on 1- and 2-liberty chains
        for y in range(19):
            for x in range(19):
                v = lab["ladder"][y, x]
                if v < 0:
                    continue
                loc = b.loc(x, y)
                libs = b.num_liberties(loc)
                if libs == 1:
                    kg = b.searchIsLadderCaptured(loc, b.board[loc] == b.pla)
                else:
                    kg = b.searchIsLadderCapturedAttackerFirst2Libs(loc)
                    kg = bool(kg) if not isinstance(kg, list) else len(kg) > 0
                assert bool(v) == bool(kg), ((y, x), v, kg, libs)
                checked_lad += 1
    assert checked_lad > 20


def test_pass_alive_matches_katago():
    for gs in _positions(n_games=15, seed=1):
        b = gs.board
        arr = board_array(gs)
        lab = L.all_labels(arr, int(b.pla))
        area = [0] * b.arrsize
        b.calculateArea(area, False, False, False, False)
        for y in range(19):
            for x in range(19):
                loc = b.loc(x, y)
                if b.board[loc] in (Board.BLACK, Board.WHITE):
                    kg_alive = area[loc] == b.board[loc]
                    assert bool(lab["pass_alive"][y, x]) == kg_alive, (y, x)


def _board(rows):
    m = {".": 0, "X": 1, "O": 2}
    return np.array([[m[c] for c in r] for r in rows], dtype=np.int8)


def test_benson_two_eyes_alive_one_eye_not():
    rows = ["." * 9 for _ in range(9)]
    rows[0] = ".X.X....."
    rows[1] = "XXXX....."
    b = np.zeros((9, 9), np.int8); b[:] = _board(rows)
    X, counts = L.benson(b, 1)
    assert len(X) == 1
    rows[0] = ".XXX....."
    b[:] = _board(rows)
    X, _ = L.benson(b, 1)
    assert len(X) == 0


def test_eyespace_counts():
    rows = ["." * 9 for _ in range(9)]
    rows[0] = ".X.X....."
    rows[1] = "XXXX....."
    lab = L.all_labels(_board(rows), 1)
    assert lab["eyespace"][1, 0] == 2


def test_katago_trunk_hooks_are_cumulative():
    """blockK must be the trunk state: trunk(K) = trunk(K-1) + residual(K)."""
    import torch
    from goconcepts.katago_adapter import KataGoProbeModel
    ck = os.path.join(os.environ.get("GOCONCEPTS_NETS", "nets"), "humanv0.ckpt")
    if not os.path.exists(ck):
        return
    m = KataGoProbeModel(ck)
    gs = next(_positions(n_games=2, seed=3))
    res = {}
    hs = [b.register_forward_hook(lambda mod, a, o, i=i: res.__setitem__(i, o.detach())) for i, b in enumerate(m.model.blocks)]
    acts = m.activations([m.encode(gs)])
    for h in hs:
        h.remove()
    names = m.layer_names[1:]  # trunk0, block1..
    for k in range(1, len(names)):
        prev = torch.tensor(acts[names[k - 1]].astype(np.float32))
        cur = torch.tensor(acts[names[k]].astype(np.float32))
        assert torch.allclose(prev + res[k - 1], cur, atol=0.05), names[k]


def test_cyclegen_pairs_are_minimal_and_correctly_labelled():
    from goconcepts.cyclegen import generate
    from goconcepts.extract import cyclic_chains
    from goconcepts.sgf import Game

    def arr(g):
        b = np.zeros((19, 19), np.int8)
        for p in g.setup_black: b[p] = 1
        for p in g.setup_white: b[p] = 2
        for c, p in g.moves: b[p] = c
        return b
    for p in generate(25, seed=7):
        a, c = arr(p.cyclic), arr(p.cut)
        assert (a != c).sum() == 1
        for b, want in ((a, True), (c, False)):
            ch = L.find_chains(b)
            k = ch.chain_id[p.ring_point[0] * 19 + p.ring_point[1]]
            assert (k in cyclic_chains(b, ch)) == want
            assert all(len(l) > 0 for l in ch.libs)
        assert L.liberty_map(a)[p.ring_point] == p.libs_cyclic == p.target_libs
        assert len(p.cyclic.moves) == 8 and p.cyclic.moves == p.cut.moves
        cols = [c for c, _ in p.cyclic.moves]
        assert all(cols[i] != cols[i + 1] for i in range(7))


def test_race_positions_match_exact_search():
    """racegen's counting rule (urgent iff r = s - 1, correct = shared liberties)
    agrees with exhaustive search over the closed race region."""
    from goconcepts.racegen import generate, solve
    from goconcepts.katago_adapter import gamestate_from_game, board_array
    for c in generate(1, seed=11):
        gs = gamestate_from_game(c.game)
        b = board_array(gs)
        D = int(gs.board.pla)
        assert b[c.ring_point] == D and b[c.inner_point] == 3 - D
        ch = L.find_chains(b)
        kr = ch.chain_id[c.ring_point[0] * 19 + c.ring_point[1]]
        ki = ch.chain_id[c.inner_point[0] * 19 + c.inner_point[1]]
        region = set(ch.libs[kr]) | set(ch.libs[ki])
        win, pass_value, _ = solve(b, D, c.ring_point, c.inner_point, region)
        assert win == c.correct and pass_value == 0, (c.variant, c.ring_libs, win, c.correct, pass_value)
        assert len(gs.moves) == 8 and gs.moves[-1][0] == 3 - D  # full history, opponent moved last

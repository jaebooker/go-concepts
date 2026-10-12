"""A small PUCT search over tf3 (or any KataGoProbeModel), to test whether search fixes
the raw network's reading of a loop.

  python -m goconcepts.mcts --visits 800

For each position in runs/attack_flip_*.pkl (the strong-wall flips) and its seed, two
searches: D to move (does search find a saving capture, and what does it think of the
ring?) and after D passes (A to move: does A find the kill?). Reported: root value for D,
the search's ownership of the ring from D's side (visit-weighted average of the leaf
ownership, like KataGo's), the visit share on the race's capturing moves, the most
visited move. Japanese rules, as in every other run.

Simplifications against KataGo's search: no symmetry averaging, no dynamic cpuct or
variance-scaled FPU, all legal moves with prior >= 1e-5 plus pass at each node (--topk; the first run used
40, which cut the saving capture in one position), batches of 8 leaves
with virtual loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle

import numpy as np
import torch

from . import labels as L
from .cyclegen import N
from .katago_adapter import KataGoProbeModel, gamestate_from_game, Board
from .paths import NETS_DIR

TF3 = NETS_DIR + "/kata1-tf3-b11c768-s11001M-d5973M.ckpt"
torch.set_num_threads(2)


class Node:
    __slots__ = ("gs", "pla", "prior", "children", "N", "W", "O", "vloss", "expanded", "move")

    def __init__(self, gs, move=None, prior=1.0):
        self.gs = gs; self.move = move; self.prior = prior
        self.pla = gs.board.pla if gs is not None else None
        self.children = []; self.N = 0; self.W = 0.0; self.O = 0.0; self.vloss = 0; self.expanded = False


def loc_of(idx):
    return Board.PASS_LOC if idx == N * N else Board.loc_static(idx % N, idx // N, N)


class Search:
    def __init__(self, model, D, ring_mask, cpuct=1.1, topk=40, score_factor=0.3, score_scale=14.0):
        self.m = model; self.D = D; self.ring = ring_mask; self.cpuct = cpuct; self.topk = topk
        # KataGo-style utility: win/loss plus a dynamic score term centred on the root's
        # expected lead, so the search still prefers saving 20 stones in a lost game
        self.sf = score_factor; self.ss = score_scale; self.center_D = None

    hold = None      # {"U": {state: {k: vec}}, "tgt": {state: {k: float}}, "ring0": mask} for kgbin nets

    def _held(self, nodes):
        """Network outputs with a direction held on the D ring's stones still on the board.
        The direction and target depend on whether D is to move (inputs are side-relative)."""
        from .kgbin_adapter import forward_hold
        from .katago_adapter import board_array
        enc = [self.m.encode(n.gs) for n in nodes]
        x = torch.tensor(np.stack([e[0] for e in enc])); g = torch.tensor(np.stack([e[1] for e in enc]))
        masks, states = [], []
        for n in nodes:
            b = board_array(n.gs)
            masks.append(((b == self.D) & self.hold["ring0"]).astype(np.float32))
            states.append("move" if n.gs.board.pla == self.D else "pass")
        ks = sorted(self.hold["U"]["move"])
        U = {k: torch.tensor(np.stack([self.hold["U"][s][k] for s in states]), dtype=torch.float32) for k in ks}
        T = {k: torch.tensor([self.hold["tgt"][s][k] for s in states], dtype=torch.float32) for k in ks}
        pol, val, own, lead = forward_hold(self.m.net, x, g, torch.tensor(np.stack(masks)), U, T)
        return {"policy": torch.log_softmax(pol, -1).numpy(), "value": torch.softmax(val, -1).numpy(),
                "ownership": own.numpy(), "lead": lead.numpy()}

    def evaluate(self, nodes):
        if self.hold is not None:
            a = self._held(nodes)
        else:
            a = self.m.activations([self.m.encode(n.gs) for n in nodes], layers=[])
        out = []
        for j, n in enumerate(nodes):
            pol = np.exp(a["policy"][j]); v = a["value"][j]
            val_tm = float(v[0] - v[1])                      # side to move
            lead_tm = float(a["lead"][j])
            sgn_tm = 1.0 if n.gs.board.pla == self.D else -1.0
            if self.center_D is None:
                self.center_D = sgn_tm * lead_tm
            val_tm += self.sf * (2 / math.pi) * math.atan((lead_tm - sgn_tm * self.center_D) / self.ss)
            own_tm = a["ownership"][j]
            sgn = 1.0 if n.gs.board.pla == self.D else -1.0
            ring_own_D = sgn * float(own_tm[self.ring].mean())
            out.append((pol, val_tm, ring_own_D))
        return out

    def expand(self, n, pol):
        b = n.gs.board; pla = b.pla
        order = np.argsort(-pol[:N * N])
        kids = []
        for idx in order:
            if len(kids) >= self.topk or pol[idx] < 1e-5:
                break
            loc = loc_of(int(idx))
            if b.would_be_legal(pla, loc):
                kids.append((int(idx), float(pol[idx])))
        kids.append((N * N, float(pol[N * N])))
        s = sum(p for _, p in kids) or 1.0
        n.children = [Node(None, move=i, prior=p / s) for i, p in kids]
        n.expanded = True

    def child_state(self, parent, child):
        gs = copy.deepcopy(parent.gs)
        gs.play(parent.pla, loc_of(child.move))
        child.gs = gs; child.pla = gs.board.pla

    def select(self, n):
        sq = math.sqrt(max(1, n.N + n.vloss))
        parent_q = n.W / n.N if n.N else 0.0
        best, bs = None, -1e9
        for c in n.children:
            nn = c.N + c.vloss
            # value of the child from the parent's side to move
            q = (-(c.W - c.vloss) / nn) if nn else parent_q - 0.2
            u = self.cpuct * c.prior * sq / (1 + nn)
            if q + u > bs:
                best, bs = c, q + u
        return best

    def run(self, gs, visits, batch=8):
        root = Node(gs)
        (pol, v, o), = self.evaluate([root])
        self.expand(root, pol); root.N = 1; root.W = v; root.O = o
        while root.N < visits:
            paths, leaves = [], []
            for _ in range(batch):
                path = [root]; n = root
                while n.expanded and n.children:
                    c = self.select(n)
                    if c.gs is None:
                        self.child_state(n, c)
                    path.append(c); n = c
                for x in path:
                    x.vloss += 1
                paths.append(path); leaves.append(n)
            uniq = list({id(l): l for l in leaves}.values())
            res = dict(zip([id(l) for l in uniq], self.evaluate(uniq)))
            for path, leaf in zip(paths, leaves):
                pol, v, o = res[id(leaf)]
                terminal = len(leaf.gs.boards) >= 3 and False
                if not leaf.expanded:
                    self.expand(leaf, pol)
                # back up: v is for leaf's side to move; flip sign going up
                for x in reversed(path):
                    x.vloss -= 1
                    x.N += 1; x.W += v; x.O += o
                    v = -v
        return root


def summarize(root, D, correct):
    """root_value_for_D is the search utility (win/loss plus the score term)."""
    tot = sum(c.N for c in root.children) or 1
    vis = {c.move: c.N for c in root.children}
    top = max(root.children, key=lambda c: c.N)
    val_D = (root.W / root.N) * (1 if root.pla == D else -1)
    return {"visits": root.N, "root_value_for_D": round(val_D, 3), "ring_ownership_D": round(root.O / root.N, 3),
            "visit_share_on_capture": round(sum(vis.get(m, 0) for m in correct) / tot, 3),
            "top_move": (divmod(top.move, N) if top.move < N * N else "pass"), "top_move_share": round(top.N / tot, 3)}


def run_set(a):
    """Search on the fortified eye-balanced set with any net: doomed rings, doomed edge
    groups and safe rings, D to move and after a D pass."""
    from .mech_cross import NETS
    from .status_eval import _d_color
    from .katago_adapter import board_array
    kind, ckpt = NETS[a.net]
    if kind == "kgbin":
        from .kgbin_adapter import KataGoBinProbeModel
        model = KataGoBinProbeModel(ckpt)
    else:
        model = KataGoProbeModel(ckpt)
    cases = pickle.load(open("data/status_bal_fortified.pkl", "rb"))
    rng = np.random.default_rng(0)
    pick = []
    for (g, st, n) in ((0, 0, a.n_ring), (1, 0, a.n_edge), (0, 1, a.n_safe)):
        ii = [i for i, c in enumerate(cases) if c.variant == g and c.status == st]
        pick += [int(i) for i in rng.permutation(ii)[:n]]
    out_path = f"runs/mcts_set_{a.net}_{a.visits}{'' if not a.hold else '_hold_' + a.hold + '_' + a.blocks}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    hold = None
    if a.hold:
        # loop direction (safe ring - safe edge) and doomed-edge target per block and state,
        # fit on the fit half of the original eye-balanced set, as in mech_cross
        from .mech_cross import trunk_means, directions
        oc = pickle.load(open("data/status_bal.pkl", "rb"))
        ogeo = np.array([c.variant for c in oc]); ost = np.array([c.status for c in oc])
        half = np.random.default_rng(0).permutation(len(oc)) % 2
        lo, hi = (int(v) for v in a.blocks.split("-")); blocks = list(range(lo, hi + 1))
        hold = {"U": {}, "tgt": {}}
        hrng = np.random.default_rng(11)
        for s_ in ("move", "pass"):
            T = trunk_means(a.net, oc, s_)
            U, tgt, _ = directions(T, blocks, "safe", half, ogeo, ost)
            if a.hold == "random":
                de = (half == 0) & (ogeo == 1) & (ost == 0)
                for k in blocks:
                    v = hrng.normal(size=T.shape[2]); v /= np.linalg.norm(v)
                    U[k] = v; tgt[k] = float(T[de, k].mean(0) @ v)
            hold["U"][s_], hold["tgt"][s_] = U, tgt
        fit_pids = {int(c.pid) for c, h_ in zip(oc, half) if h_ == 0}
    for i in pick:
        c = cases[i]; D = _d_color(c)
        b = board_array(gamestate_from_game(c.game))
        ch = L.find_chains(b); cid = ch.chain_id.reshape(N, N); ring = cid == cid[tuple(c.ring_point)]
        for state in ("D_to_move", "after_D_pass"):
            key = f"{i}_{state}"
            if key in res:
                continue
            gg = copy.deepcopy(c.game)
            if state == "after_D_pass":
                gg.moves = list(gg.moves) + [(D, None)]
            gs = gamestate_from_game(gg)
            S = Search(model, D, ring, topk=a.topk)
            S0 = Search(model, D, ring)
            if hold is not None:
                S.hold = {**hold, "ring0": ring}; S0.hold = S.hold
            root = S.run(gs, a.visits)
            r = summarize(root, D, c.correct if state == "D_to_move" else [])
            r["raw_ring_ownership_D"] = round(S0.evaluate([Node(gs)])[0][2], 3)
            if hold is not None:
                r["in_fit_half"] = int(c.orig_pid) in fit_pids
            r.update({"geo": int(c.variant), "status": int(c.status), "ring_size": int(ring.sum())})
            res[key] = r
            json.dump(res, open(out_path, "w"), indent=1)
            print(key, r, flush=True)
    # summary
    for (g, st, nm) in ((0, 0, "doomed ring"), (1, 0, "doomed edge"), (0, 1, "safe ring")):
        for state in ("D_to_move", "after_D_pass"):
            rows = [v for k, v in res.items() if k.endswith(state) and v["geo"] == g and v["status"] == st]
            if rows:
                print(a.net, nm, state, "n", len(rows),
                      "raw own", round(float(np.mean([v["raw_ring_ownership_D"] for v in rows])), 2),
                      "search own", round(float(np.mean([v["ring_ownership_D"] for v in rows])), 2),
                      "search read alive", round(float(np.mean([v["ring_ownership_D"] > 0 for v in rows])), 2),
                      "capture visits", round(float(np.mean([v["visit_share_on_capture"] for v in rows])), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="store_true", help="run on the fortified eye-balanced set (run_set)")
    ap.add_argument("--net", default="tf3")
    ap.add_argument("--n-ring", type=int, default=12)
    ap.add_argument("--n-edge", type=int, default=4)
    ap.add_argument("--n-safe", type=int, default=4)
    ap.add_argument("--hold", default="", choices=["", "loop", "random"],
                    help="hold the loop direction (or a random one) on the ring during the search")
    ap.add_argument("--blocks", default="12-24")
    ap.add_argument("--visits", type=int, default=800)
    ap.add_argument("--roots", default="bal12,bal161,bal181")
    ap.add_argument("--topk", type=int, default=400, help="moves considered per node, by prior (400 = all legal)")
    ap.add_argument("--only", default="flipped,seed")
    a = ap.parse_args()
    if a.set:
        return run_set(a)
    from .attack_search import game_of, validate
    model = KataGoProbeModel(TF3)
    seeds = {s["root"]: s for s in pickle.load(open("runs/attack_search_seeds.pkl", "rb"))["seeds"]}
    out_path = f"runs/mcts_{a.visits}_top{a.topk}_score.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for r in a.roots.split(","):
        flipped = pickle.load(open(f"runs/attack_flip_{r}.pkl", "rb"))
        for tag, s in (("flipped", flipped), ("seed", seeds[r])):
            if tag not in a.only.split(","):
                continue
            info = validate(s["b"], s["D"], s["rp"], s["ip"], budget=30)
            ch = L.find_chains(s["b"]); cid = ch.chain_id.reshape(N, N); ring = cid == cid[s["rp"]]
            g = game_of(s)
            for state in ("D_to_move", "after_D_pass"):
                key = f"{r}_{tag}_{state}"
                if key in res:
                    continue
                gg = copy.deepcopy(g)
                if state == "after_D_pass":
                    gg.moves = list(gg.moves) + [(s["D"], None)]
                gs = gamestate_from_game(gg)
                root = Search(model, s["D"], ring, topk=a.topk).run(gs, a.visits)
                res[key] = summarize(root, s["D"], info["correct"] if (info and state == "D_to_move") else [])
                res[key]["raw_ring_ownership_D"] = round(Search(model, s["D"], ring).evaluate([Node(gs)])[0][2], 3)
                json.dump(res, open(out_path, "w"), indent=1)
                print(key, res[key], flush=True)


if __name__ == "__main__":
    main()

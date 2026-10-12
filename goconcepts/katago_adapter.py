"""KataGo side: replay games into KataGo's GameState, encode inputs, and pull
activations from every trunk block with forward hooks.

Rules default to Japanese with encorePhase 0. Under those rules KataGo's area
input planes (18, 19) are left blank, so pass-alive status is NOT handed to
the network as an input. Liberty (1/2/3) and ladder planes still are: see
INPUT_LEAKS below and report input-layer probe baselines for those concepts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Some tf3 weights or activations are denormal floats; without flushing them to zero,
# MKL's fp32 matmul slows to ~2% speed (17 s per position instead of 0.4 s). Outputs
# are bit-identical on the positions checked.
torch.set_flush_denormal(True)

KATAGO_PY = os.environ.get("KATAGO_PY", os.path.join(os.path.dirname(__file__), "..", "ext", "KataGo", "python"))
if KATAGO_PY not in sys.path:
    sys.path.insert(0, os.path.abspath(KATAGO_PY))

from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402

# Concepts KataGo receives (partially) as hand-built input planes.
INPUT_LEAKS = {
    "libs": "planes 3-5 mark stones whose chain has exactly 1, 2 or 3 liberties",
    "ladder": "planes 14-17 mark ladderable stones (current and two previous positions)",
    "pass_alive": "planes 18-19 under area/no-tax rules only; blank under the default Japanese rules here",
}


def gamestate_from_game(game, upto: int | None = None, rules=None) -> GameState:
    gs = GameState(game.size, rules or GameState.RULES_JAPANESE)
    b = gs.board
    for (y, x) in game.setup_black:
        b.set_stone(Board.BLACK, b.loc(x, y))
    for (y, x) in game.setup_white:
        b.set_stone(Board.WHITE, b.loc(x, y))
    if game.first_player:
        b.set_pla(game.first_player)
    elif game.moves:
        b.set_pla(game.moves[0][0])
    gs.boards = [b.copy()]
    moves = game.moves if upto is None else game.moves[:upto]
    for col, mv in moves:
        loc = Board.PASS_LOC if mv is None else b.loc(mv[1], mv[0])
        gs.play(col, loc)
        b = gs.board
    return gs


def board_array(gs: GameState) -> np.ndarray:
    b = gs.board
    n = b.x_size
    out = np.zeros((n, n), dtype=np.int8)
    for y in range(n):
        for x in range(n):
            out[y, x] = b.board[b.loc(x, y)]
    return out


class KataGoProbeModel:
    """Wraps a KataGo pytorch Model; returns per-layer spatial activations."""

    def __init__(self, ckpt_path: str, device: str = "cpu", human_rank: str | None = "9d"):
        from katago.train.load_model import load_model
        self.model, _, _ = load_model(ckpt_path, use_swa=False, device=device)
        self.model.eval()
        self.device = device
        self.features = Features(self.model.config, self.model.pos_len)
        self.has_meta = self.model.metadata_encoder is not None
        self.meta = None
        if self.has_meta:
            # Rank-conditioned HumanSL: condition on a strong human by default.
            inv = {"9d": 0, "5d": 4, "1d": 8, "5k": 13, "10k": 18, "20k": 28}[human_rank or "9d"]
            self.meta = SGFMetadata(inverseBRank=inv, inverseWRank=inv, bIsHuman=True, wIsHuman=True,
                                    gameIsUnrated=False, tcIsUnknown=True, source=1)
        # Trunk states, not block outputs: each KataGo block returns only its
        # residual, and the model adds it to the trunk. So "blockK" is the trunk
        # after block K = the input to block K+1 (pre-hook), and the last one is
        # its input plus its residual. "trunk0" is the trunk before block 1
        # (stem conv + global + metadata embeddings).
        nb = len(self.model.blocks)
        self.layer_names = ["input", "trunk0"] + [f"block{i+1}" for i in range(nb)]
        self._acts = {}
        self._hooks = []
        for i, blk in enumerate(self.model.blocks):
            name = "trunk0" if i == 0 else f"block{i}"
            self._hooks.append(blk.register_forward_pre_hook(self._save_in(name)))
        self._hooks.append(self.model.blocks[-1].register_forward_hook(self._save_last(f"block{nb}", f"block{nb - 1}" if nb > 1 else "trunk0")))

    def _save_in(self, name):
        def hook(_m, args):
            self._acts[name] = args[0].detach()
        return hook

    def _save_last(self, name, prev):
        def hook(_m, args, out):
            self._acts[name] = (args[0] + out).detach()
        return hook

    def encode(self, gs: GameState):
        bin_in, glob_in = gs.get_input_features(self.features)
        meta = None
        if self.has_meta:
            meta = self.meta.get_metadata_row(nextPlayer=gs.board.pla, boardArea=gs.board.x_size * gs.board.y_size)
        return bin_in[0], glob_in[0], meta

    @torch.no_grad()
    def activations(self, encoded: list, layers: list[str] | None = None):
        """encoded: list of (bin, glob, meta). Returns {layer: (B, C, 19, 19) float16 numpy},
        plus 'policy' (B, 362) log-probs from the main policy head."""
        bins = torch.tensor(np.stack([e[0] for e in encoded]), device=self.device)
        globs = torch.tensor(np.stack([e[1] for e in encoded]), device=self.device)
        meta = None
        if self.has_meta:
            meta = torch.tensor(np.stack([e[2] for e in encoded]), dtype=torch.float32, device=self.device)
        self._acts = {}
        out = self.model(bins, globs, input_meta=meta)
        policy_logits = out[0][0][:, 0, :]  # N, moves
        post = self.model.postprocess_output(out)[0]
        # ownership: +1 = owned by the side to move at the end of the game
        ownership = torch.tanh(post[4][:, 0].float()).cpu().numpy()
        value = torch.softmax(post[1].float(), dim=-1).cpu().numpy()  # win, loss, noresult for side to move
        acts = {"input": bins.cpu().numpy().astype(np.float16)}
        for k, v in self._acts.items():
            if v.dim() == 3:
                # transformer trunks run in sequence layout (B, H*W, C): back to NCHW
                B, S, C = v.shape
                side = int(round(S ** 0.5))
                v = v.transpose(1, 2).reshape(B, C, side, side)
            acts[k] = v.cpu().numpy().astype(np.float16)
        if layers is not None:
            acts = {k: v for k, v in acts.items() if k in layers}
        acts["policy"] = torch.log_softmax(policy_logits.float(), dim=-1).cpu().numpy()
        acts["ownership"] = ownership.reshape(ownership.shape[0], -1)[:, : 19 * 19].reshape(-1, 19, 19)
        acts["value"] = value
        acts["lead"] = post[10].float().cpu().numpy()     # expected score lead, side to move
        return acts

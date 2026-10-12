"""Leela Zero side: load the text weight file into a PyTorch residual tower and
expose the same interface as KataGoProbeModel (encode / activations /
layer_names).

Leela Zero sees stones only: 8 history planes for the side to move, 8 for the
opponent, and two colour planes. No liberties, no ladders, no area. So every
concept probed here is learned, not given.

Weight file layout (version 1), one tensor per line:
  input conv w, b, bn mean, bn var
  per block x B: conv1 w, b, mean, var, conv2 w, b, mean, var
  policy: conv w, b, mean, var, fc w, b
  value:  conv w, b, mean, var, fc1 w, b, fc2 w, b
Batchnorm has no learned scale or shift; conv biases are folded into the
mean, as Leela Zero's own loader does. eps = 1e-5.
"""
from __future__ import annotations

import gzip

import numpy as np
import torch
import torch.nn.functional as F

from .katago_adapter import Board, GameState

EPS = 1e-5
HISTORY = 8


def _read(path):
    op = gzip.open if path.endswith(".gz") or open(path, "rb").read(2) == b"\x1f\x8b" else open
    with op(path, "rt") as fh:
        lines = fh.read().strip().split("\n")
    version = int(lines[0])
    if version != 1:
        raise ValueError(f"unsupported Leela Zero weight version {version}")
    return [np.array(l.split(), dtype=np.float32) for l in lines[1:]]


class _BN:
    def __init__(self, bias, mean, var):
        self.shift = torch.tensor(mean - bias)
        self.inv = torch.tensor(1.0 / np.sqrt(var + EPS))

    def __call__(self, x):
        return (x - self.shift[None, :, None, None]) * self.inv[None, :, None, None]


class LeelaZeroNet:
    def __init__(self, path):
        t = _read(path)
        n_blocks = (len(t) - 4 - 14) // 8
        C = len(t[1])
        self.C, self.n_blocks = C, n_blocks
        k = 0

        def conv(n_in, n_out, ks):
            nonlocal k
            w = torch.tensor(t[k].reshape(n_out, n_in, ks, ks)); bn = _BN(t[k + 1], t[k + 2], t[k + 3])
            k += 4
            return w, bn

        self.inp = conv(18, C, 3)
        self.blocks = [(conv(C, C, 3), conv(C, C, 3)) for _ in range(n_blocks)]
        self.pol_conv = conv(C, 2, 1)
        self.pol_fc = (torch.tensor(t[k].reshape(362, 2 * 361)), torch.tensor(t[k + 1])); k += 2
        self.val_conv = conv(C, 1, 1)
        self.val_fc1 = (torch.tensor(t[k].reshape(256, 361)), torch.tensor(t[k + 1])); k += 2
        self.val_fc2 = (torch.tensor(t[k].reshape(1, 256)), torch.tensor(t[k + 1])); k += 2
        assert k == len(t), (k, len(t))

    @torch.no_grad()
    def forward(self, x, keep: dict):
        def cbr(h, cv, pad):
            w, bn = cv
            return bn(F.conv2d(h, w, padding=pad))
        h = F.relu(cbr(x, self.inp, 1))
        keep["stem"] = h
        for i, (c1, c2) in enumerate(self.blocks):
            r = F.relu(cbr(h, c1, 1))
            h = F.relu(cbr(r, c2, 1) + h)
            keep[f"block{i + 1}"] = h
        p = F.relu(cbr(h, self.pol_conv, 0)).flatten(1)
        policy = F.linear(p, *self.pol_fc)
        v = F.relu(cbr(h, self.val_conv, 0)).flatten(1)
        v = torch.tanh(F.linear(F.relu(F.linear(v, *self.val_fc1)), *self.val_fc2))
        return policy, v


class LeelaZeroProbeModel:
    def __init__(self, path, device="cpu", threads=None):
        if threads:
            torch.set_num_threads(threads)
        self.net = LeelaZeroNet(path)
        self.layer_names = ["input", "stem"] + [f"block{i + 1}" for i in range(self.net.n_blocks)]

    def encode(self, gs: GameState):
        pla = gs.board.pla
        opp = Board.get_opp(pla)
        x = np.zeros((18, 19, 19), np.float32)
        boards = gs.boards[::-1][:HISTORY]  # most recent first; missing history stays zero
        for h, b in enumerate(boards):
            arr = np.array([[b.board[b.loc(xx, yy)] for xx in range(19)] for yy in range(19)])
            x[h] = arr == pla
            x[HISTORY + h] = arr == opp
        x[16 if pla == Board.BLACK else 17] = 1.0
        return x

    def activations(self, encoded: list, layers=None):
        x = torch.tensor(np.stack(encoded))
        keep = {}
        policy, value = self.net.forward(x, keep)
        acts = {"input": x.numpy().astype(np.float16)}
        for k, v in keep.items():
            acts[k] = v.numpy().astype(np.float16)
        if layers is not None:
            acts = {k: v for k, v in acts.items() if k in layers}
        acts["policy"] = torch.log_softmax(policy, -1).numpy()
        acts["value"] = value.numpy()[:, 0]
        return acts

"""KataGo .bin.gz models (the format KataGo's C++ engine reads) loaded into
PyTorch, for model versions 8 to 10: plain residual towers with global-pooling
blocks, ReLU, batchnorm. That covers the 2020 g170 self-play nets on KataGo's
GitHub releases, which predate KataGo's anti-cyclic training.

Parser follows cpp/neuralnet/desc.cpp; the forward pass follows the Eigen
backend (cpp/neuralnet/eigenbackend.cpp):
  trunk  x = conv(in) + matmul(global); each block adds a residual; tip = relu(bn(x))
  gpool  [mean, mean * (sqrt(area) - 14) / 10, max]
  value  [mean, mean * (s - 14) / 10, mean * ((s - 14)^2 / 100 - 0.1)], s = sqrt(area)
Conv weights are stored y, x, ic, oc; matmul weights ic, oc.

Inputs are KataGo's V7 features, the same planes HumanSL gets, so the liberty
(1-3) and ladder input planes caveat applies here too.
"""
from __future__ import annotations

import gzip

import numpy as np
import torch
import torch.nn.functional as F

from .katago_adapter import Features, GameState


class _Reader:
    def __init__(self, path):
        with gzip.open(path, "rb") as fh:
            self.b = fh.read()
        self.i = 0

    def tok(self):
        b, i = self.b, self.i
        while b[i:i + 1].isspace():
            i += 1
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        self.i = j
        return b[i:j].decode()

    def int(self):
        return int(self.tok())

    def float(self):
        return float(self.tok())

    def floats(self, n):
        k = self.b.index(b"@BIN@", self.i)
        assert k - self.i < 100, "binary block not where expected"
        s = k + 5
        arr = np.frombuffer(self.b, dtype="<f4", count=n, offset=s).copy()
        self.i = s + 4 * n
        return arr


def _conv(r):
    name = r.tok(); ky, kx, ic, oc, dy, dx = (r.int() for _ in range(6))
    w = r.floats(ky * kx * ic * oc).reshape(ky, kx, ic, oc).transpose(3, 2, 0, 1)
    return {"name": name, "w": torch.tensor(np.ascontiguousarray(w)), "pad": ky // 2}


def _bn(r):
    name = r.tok(); c = r.int(); eps = r.float(); hs = r.int(); hb = r.int()
    mean, var = r.floats(c), r.floats(c)
    scale = r.floats(c) if hs else np.ones(c, np.float32)
    bias = r.floats(c) if hb else np.zeros(c, np.float32)
    mul = scale / np.sqrt(var + eps)
    return {"mul": torch.tensor(mul), "add": torch.tensor(bias - mean * mul)}


def _act(r, version):
    r.tok()
    if version >= 11:
        kind = r.tok()
        assert kind == "ACTIVATION_RELU", kind


def _matmul(r):
    r.tok(); ic, oc = r.int(), r.int()
    return torch.tensor(r.floats(ic * oc).reshape(ic, oc))


def _matbias(r):
    r.tok(); c = r.int()
    return torch.tensor(r.floats(c))


def _apply_bn(x, bn):
    return x * bn["mul"][None, :, None, None] + bn["add"][None, :, None, None]


def _cv(x, c):
    return F.conv2d(x, c["w"], padding=c["pad"])


def _gpool(x):
    mean = x.mean((2, 3))
    s = (x.shape[2] * x.shape[3]) ** 0.5
    return torch.cat([mean, mean * (s - 14) * 0.1, x.amax((2, 3))], 1)


def _vpool(x):
    mean = x.mean((2, 3))
    s = (x.shape[2] * x.shape[3]) ** 0.5
    return torch.cat([mean, mean * (s - 14) * 0.1, mean * ((s - 14) ** 2 * 0.01 - 0.1)], 1)


class KataGoBinNet:
    def __init__(self, path):
        r = _Reader(path)
        self.name = r.tok()
        self.version = r.int()
        if not 8 <= self.version <= 10:
            raise ValueError(f"model version {self.version} not supported by this loader")
        self.n_in, self.n_glob = r.int(), r.int()
        r.tok()  # trunk name
        nb = r.int(); self.C = r.int(); r.int(); r.int(); r.int(); r.int()
        self.init_conv = _conv(r)
        self.init_mm = _matmul(r)
        self.blocks = []
        for _ in range(nb):
            kind = r.tok(); r.tok()
            b = {"kind": kind, "pre": _bn(r)}; _act(r, self.version)
            b["reg"] = _conv(r)
            if kind == "gpool_block":
                b["gconv"] = _conv(r); b["gbn"] = _bn(r); _act(r, self.version); b["g2b"] = _matmul(r)
            elif kind != "ordinary_block":
                raise ValueError(kind)
            b["mid"] = _bn(r); _act(r, self.version)
            b["fin"] = _conv(r)
            self.blocks.append(b)
        self.tip = _bn(r); _act(r, self.version)
        r.tok()  # policy head
        self.p1 = _conv(r); self.g1 = _conv(r); self.g1bn = _bn(r); _act(r, self.version)
        self.g2b = _matmul(r); self.p1bn = _bn(r); _act(r, self.version)
        self.p2 = _conv(r); self.g2pass = _matmul(r)
        r.tok()  # value head
        self.v1 = _conv(r); self.v1bn = _bn(r); _act(r, self.version)
        self.v2 = _matmul(r); self.v2b = _matbias(r); _act(r, self.version)
        self.v3 = _matmul(r); self.v3b = _matbias(r)
        self.sv3 = _matmul(r); self.sv3b = _matbias(r); self.own = _conv(r)
        self.bytes_left = len(r.b[r.i:].strip())  # 0 when the whole file was parsed
        self.n_blocks = nb

    @torch.no_grad()
    def forward(self, x, g, keep: dict):
        h = _cv(x, self.init_conv) + (g @ self.init_mm)[:, :, None, None]
        keep["trunk0"] = h
        for i, b in enumerate(self.blocks):
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            keep[f"block{i + 1}"] = h
        tr = F.relu(_apply_bn(h, self.tip))
        p = _cv(tr, self.p1)
        gp = F.relu(_apply_bn(_cv(tr, self.g1), self.g1bn))
        pooled = _gpool(gp)
        p = F.relu(_apply_bn(p + (pooled @ self.g2b)[:, :, None, None], self.p1bn))
        pol = _cv(p, self.p2)[:, 0].flatten(1)
        pas = (pooled @ self.g2pass)[:, :1]
        policy = torch.cat([pol, pas], 1)
        v1 = F.relu(_apply_bn(_cv(tr, self.v1), self.v1bn))
        v = F.relu(_vpool(v1) @ self.v2 + self.v2b)
        value = v @ self.v3 + self.v3b
        # misc value outputs: scoreMean, scoreStdev, lead, ... (x20 as in KataGo's C++ code)
        self.last_lead = (v @ self.sv3 + self.sv3b)[:, 2] * 20.0
        self.last_ownership = torch.tanh(_cv(v1, self.own)[:, 0])  # +1 = side to move owns it
        return policy, value


def forward_hold(net, x, g, mask, U, T):
    """net.forward with a direction held on `mask` after chosen blocks.
    U: {block: (B, C) unit directions}, T: {block: (B,) target projections}. Returns
    policy logits, value logits, ownership (+1 side to move), lead."""
    with torch.no_grad():
        h = _cv(x, net.init_conv) + (g @ net.init_mm)[:, :, None, None]
        m = mask[:, None]
        for i, b in enumerate(net.blocks):
            a = F.relu(_apply_bn(h, b["pre"]))
            t = _cv(a, b["reg"])
            if b["kind"] == "gpool_block":
                gp = F.relu(_apply_bn(_cv(a, b["gconv"]), b["gbn"]))
                t = t + (_gpool(gp) @ b["g2b"])[:, :, None, None]
            t = F.relu(_apply_bn(t, b["mid"]))
            h = h + _cv(t, b["fin"])
            k = i + 1
            if k in U:
                u = U[k][:, :, None, None]
                h = h - m * ((h * u).sum(1, keepdim=True) - T[k][:, None, None, None]) * u
        tr = F.relu(_apply_bn(h, net.tip))
        p = _cv(tr, net.p1)
        gp = F.relu(_apply_bn(_cv(tr, net.g1), net.g1bn))
        pooled = _gpool(gp)
        p = F.relu(_apply_bn(p + (pooled @ net.g2b)[:, :, None, None], net.p1bn))
        policy = torch.cat([_cv(p, net.p2)[:, 0].flatten(1), (pooled @ net.g2pass)[:, :1]], 1)
        v1 = F.relu(_apply_bn(_cv(tr, net.v1), net.v1bn))
        v = F.relu(_vpool(v1) @ net.v2 + net.v2b)
        value = v @ net.v3 + net.v3b
        lead = (v @ net.sv3 + net.sv3b)[:, 2] * 20.0
        own = torch.tanh(_cv(v1, net.own)[:, 0])
        return policy, value, own, lead


class KataGoBinProbeModel:
    """Same interface as KataGoProbeModel / LeelaZeroProbeModel."""

    def __init__(self, path, device="cpu"):
        self.net = KataGoBinNet(path)
        self.features = Features({"version": 10}, 19)  # V7 input features, shared by model versions 8-17
        self.layer_names = ["input", "trunk0"] + [f"block{i + 1}" for i in range(self.net.n_blocks)]

    def encode(self, gs: GameState):
        b, g = gs.get_input_features(self.features)
        return b[0], g[0]

    def activations(self, encoded: list, layers=None):
        x = torch.tensor(np.stack([e[0] for e in encoded]))
        g = torch.tensor(np.stack([e[1] for e in encoded]))
        keep = {}
        policy, value = self.net.forward(x, g, keep)
        acts = {"input": x.numpy().astype(np.float16)}
        for k, v in keep.items():
            acts[k] = v.numpy().astype(np.float16)
        if layers is not None:
            acts = {k: v for k, v in acts.items() if k in layers}
        acts["policy"] = torch.log_softmax(policy, -1).numpy()
        acts["value"] = torch.softmax(value, -1).numpy()
        acts["ownership"] = self.net.last_ownership.numpy()
        acts["lead"] = self.net.last_lead.numpy()           # expected score lead, side to move
        return acts

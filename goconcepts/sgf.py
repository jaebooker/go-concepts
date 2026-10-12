"""Minimal SGF reader: main line only, setup stones, passes. Enough for Fox pro
games and lightvector's cyclic test positions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_PROP = re.compile(r"([A-Z]{1,2})((?:\s*\[(?:\\.|[^\]\\])*\])+)", re.S)
_VAL = re.compile(r"\[((?:\\.|[^\]\\])*)\]", re.S)


@dataclass
class Game:
    size: int = 19
    setup_black: list = field(default_factory=list)   # (y, x)
    setup_white: list = field(default_factory=list)
    first_player: int | None = None                  # from PL, 1 black 2 white
    moves: list = field(default_factory=list)          # (color, (y,x) or None)
    comments: dict = field(default_factory=dict)       # move index -> comment
    props: dict = field(default_factory=dict)          # root properties


def _coord(s: str, size: int):
    if s == "" or (s == "tt" and size <= 19):
        return None
    x = ord(s[0]) - 97
    y = ord(s[1]) - 97
    return (y, x)


def _main_line_nodes(text: str) -> list[str]:
    """Parse the game tree and return node strings along the first variation."""
    i, n = 0, len(text)

    def parse_tree():
        nonlocal i
        assert text[i] == "("; i += 1
        seq, children = [], []
        while i < n:
            ch = text[i]
            if ch == ";":
                i += 1; start = i
                while i < n and text[i] not in ";()":
                    if text[i] == "[":
                        i += 1
                        while i < n and text[i] != "]":
                            i += 2 if text[i] == "\\" else 1
                    i += 1
                seq.append(text[start:i])
            elif ch == "(":
                children.append(parse_tree())
            elif ch == ")":
                i += 1
                return (seq, children)
            else:
                i += 1
        return (seq, children)

    while i < n and text[i] != "(":
        i += 1
    seq, children = parse_tree()
    out = list(seq)
    while children:
        seq, children = children[0]
        out.extend(seq)
    return out


def parse(text: str) -> Game:
    g = Game()
    nodes = _main_line_nodes(text)
    for ni, node in enumerate(nodes):
        props = {}
        for m in _PROP.finditer(node):
            props[m.group(1)] = _VAL.findall(m.group(2))
        if ni == 0:
            g.props = {k: v[0] if v else "" for k, v in props.items()}
            if "SZ" in props:
                g.size = int(props["SZ"][0].split(":")[0])
        for key, lst in (("AB", g.setup_black), ("AW", g.setup_white)):
            for v in props.get(key, []):
                if ":" in v:  # compressed rectangle
                    a, b = _coord(v[:2], g.size), _coord(v[3:], g.size)
                    for y in range(a[0], b[0] + 1):
                        for x in range(a[1], b[1] + 1):
                            lst.append((y, x))
                else:
                    c = _coord(v, g.size)
                    if c: lst.append(c)
        if "PL" in props and not g.moves:
            g.first_player = 1 if props["PL"][0].upper() == "B" else 2
        for key, col in (("B", 1), ("W", 2)):
            if key in props:
                g.moves.append((col, _coord(props[key][0], g.size)))
        if "C" in props:
            g.comments[len(g.moves)] = props["C"][0]
    return g


def read(path: str) -> Game:
    with open(path, "rb") as f:
        raw = f.read()
    return parse(raw.decode("utf-8", errors="replace"))

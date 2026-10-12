"""Figures for the paper, from the result files in runs/. Run from the repository root:
  PYTHONPATH=. python3 paper/make_figures.py
"""
import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures") + "/"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"
INK, MUTED = "#1f2328", "#5b6270"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e6ea", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42})

# ---- Figure 1: capture mass by ring liberties, ring vs edge
rep = json.load(open(G + "runs/races_report.json"))["engines"]
cases = pickle.load(open(G + "data/races.pkl", "rb"))
pol = np.exp(np.load(G + "runs/b40/races_policy.npz")["policy"].astype(np.float64))
pol = pol / pol.sum(1, keepdims=True) if pol.max() > 1.0001 or pol.sum(1).mean() < 0.99 else pol
b40 = {}
for L in range(1, 6):
    b40[str(L)] = {}
    for v, nm in ((0, "cyclic"), (1, "edge")):
        m = [i for i, c in enumerate(cases) if c.ring_libs == L and c.variant == v]
        b40[str(L)][nm] = float(np.mean([pol[i, c].sum() for i in m for c in [cases[i].correct]])) if m else np.nan
nets = [("g170", rep["g170"]["mass_by_ring_libs"]), ("b40 (attacked)", b40),
        ("Leela Zero", rep["lz"]["mass_by_ring_libs"]), ("tf3 (fixed)", rep["tf3"]["mass_by_ring_libs"])]
fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.2), sharey=True)
x = np.arange(1, 6)
for ax, (name, d) in zip(axs, nets):
    ring = [d[str(L)]["cyclic"] for L in x]; edge = [d[str(L)]["edge"] for L in x]
    ax.plot(x, edge, color=ORANGE, lw=2, marker="o", ms=4, label="edge pocket")
    ax.plot(x, ring, color=BLUE, lw=2, marker="s", ms=4, label="closed ring")
    ax.set_title(name, fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_ylim(-0.03, 1.0); ax.set_xlabel("ring liberties")
axs[0].set_ylabel("policy mass on\nthe capture")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, handlelength=1.8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.4, w_pad=0.6, rect=(0, 0.09, 1, 1))
fig.savefig(OUT + "fig_races.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("b40 by libs", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in b40.items()})

# ---- Figure 2: b40 on its own lost loops vs ordinary chains, by lead time
r = json.load(open(G + "runs/real_report.json"))["engines"]["b40"]
leads = [1, 3, 5, 9, 17, 33]
fig, ax = plt.subplots(figsize=(3.3, 2.4))
for grp, col, mk, lab in (("cyclic_b40_victim_games", BLUE, "s", "b40's own lost loops (FAR games)"),
                          ("noncyclic_pro", ORANGE, "o", "ordinary captured chains (pro games)")):
    y = [r[grp][f"lead{L}"]["frac_read_alive"] for L in leads]
    ax.plot(range(len(leads)), y, color=col, lw=2, marker=mk, ms=4, label=lab)
ax.set_xticks(range(len(leads))); ax.set_xticklabels(leads)
ax.set_xlabel("moves before the capture"); ax.set_ylabel("fraction read alive\nafter a pass")
ax.set_ylim(-0.02, 1.0)
ax.legend(loc="lower center", fontsize=7, bbox_to_anchor=(0.45, 1.0), ncol=1)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_real.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---- Figure 3: doomed-ring ownership after a pass across interventions and search (fortified set)
def setmean(path, geo=0, st=0, key="ring_ownership_D"):
    d = json.load(open(G + path))
    rows = [v for k, v in d.items() if k.endswith("after_D_pass") and v["geo"] == geo and v["status"] == st]
    return float(np.mean([v[key] for v in rows])), len(rows)
vals = {
    "b40": [("raw", setmean("runs/mcts_set_b40_800.json", key="raw_ring_ownership_D")[0]),
            ("search", setmean("runs/mcts_set_b40_800.json")[0]),
            ("search +\nrandom hold", setmean("runs/mcts_set_b40_800_hold_random_12-24.json")[0]),
            ("search +\nloop hold", setmean("runs/mcts_set_b40_800_hold_loop_12-24.json")[0])],
    "g170": [("raw", setmean("runs/mcts_set_g170_800.json", key="raw_ring_ownership_D")[0]),
             ("search", setmean("runs/mcts_set_g170_800.json")[0])],
    "tf3": [("raw", setmean("runs/mcts_set_tf3_800.json", key="raw_ring_ownership_D")[0]),
            ("search", setmean("runs/mcts_set_tf3_800.json")[0])],
}
rows = []
for net, col in (("g170", GRAY), ("b40", BLUE), ("tf3", AQUA)):
    for lab, v in vals[net]:
        rows.append((f"{net}: " + lab.replace("\n", " "), v, col))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ys = np.arange(len(rows))[::-1]
ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.7, edgecolor="white", linewidth=1)
for y, (lab, v, c) in zip(ys, rows):
    ax.text(v + (0.04 if v >= 0 else -0.04), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-1.45, 1.45); ax.grid(axis="y", visible=False)
ax.set_xlabel("ownership after a pass")
fig.tight_layout(pad=0.4)
fig.savefig(OUT + "fig_search.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print({k: [(a.replace(chr(10), ' '), round(b, 2)) for a, b in v] for k, v in vals.items()})

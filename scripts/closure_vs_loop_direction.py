# Usage: PYTHONPATH=. python3 scripts/closure_vs_loop_direction.py g170
# Cosine, per layer, between the minimal-pair closure direction (cyclic - cut, far ring
# stones) and the safe-fit loop direction used in the holds (ring - edge, status_bal, to-move).
import sys, pickle, numpy as np, torch
from goconcepts.cyclegen import generate
from goconcepts.loopcircuit import geometry, trunk_states
from goconcepts.kgbin_adapter import KataGoBinProbeModel
from goconcepts.katago_adapter import gamestate_from_game
from goconcepts.mech_cross import NETS, trunk_means
net_name = sys.argv[1]
model = KataGoBinProbeModel(NETS[net_name][1]); net = model.net
pairs = [p for p in generate(400, seed=12345) if p.libs_cyclic == p.libs_cut]
acc = None
for s in range(0, len(pairs), 8):
    ps = pairs[s:s + 8]; geo = [geometry(p) for p in ps]
    e1 = [model.encode(gamestate_from_game(p.cyclic)) for p in ps]; e2 = [model.encode(gamestate_from_game(p.cut)) for p in ps]
    g = torch.tensor(np.stack([e[1] for e in e1]))
    S1 = trunk_states(net, torch.tensor(np.stack([e[0] for e in e1])), g); S2 = trunk_states(net, torch.tensor(np.stack([e[0] for e in e2])), g)
    if acc is None: acc = np.zeros((len(S1), net.C))
    for j, gm in enumerate(geo):
        m = gm["ring"] >= 5
        if m.any():
            for k in range(len(S1)): acc[k] += (S1[k][j] - S2[k][j]).numpy()[:, m].mean(1)
cases = pickle.load(open('data/status_bal.pkl', 'rb'))
geo_ = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
T = trunk_means(net_name, cases, 'move')
out = []
for k in range(acc.shape[0]):
    u = T[(geo_ == 0) & (st == 1), k].mean(0) - T[(geo_ == 1) & (st == 1), k].mean(0)
    out.append(round(float(acc[k] @ u / np.linalg.norm(acc[k]) / np.linalg.norm(u)), 2))
print(net_name, out)

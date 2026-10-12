# Usage: PYTHONPATH=. python3 scripts/closure_vs_death_cosine.py
# Closure direction (minimal pairs, closure_hold.closure_dirs) vs loop direction and death axis, g170 and b40.
import json
import pickle

import numpy as np

from goconcepts.closure_hold import closure_dirs
from goconcepts.mech_cross import Runner, trunk_means

cases = pickle.load(open('data/status_bal.pkl', 'rb'))
geo = np.array([c.variant for c in cases]); st = np.array([c.status for c in cases])
fit = np.random.default_rng(0).permutation(len(cases)) % 2 == 0
c = {"dr": (geo == 0) & (st == 0), "de": (geo == 1) & (st == 0), "sr": (geo == 0) & (st == 1), "se": (geo == 1) & (st == 1)}
out = {}
for net in ['g170', 'b40']:
    r = Runner(net); nb = r.model.net.n_blocks
    U = closure_dirs(r, list(range(1, nb + 1)))
    for state in ('move', 'pass'):
        T = trunk_means(net, cases, state)
        rows = []
        for k in range(1, nb + 1):
            u = U[state][k]
            loop = T[fit & c['sr'], k].mean(0) - T[fit & c['se'], k].mean(0); loop /= np.linalg.norm(loop)
            death = T[fit & c['de'], k].mean(0) - T[fit & c['se'], k].mean(0); death /= np.linalg.norm(death)
            p = T[fit, k] @ u; sd = p.std()
            pr = lambda m: float((T[fit & m, k] @ u).mean())
            rows.append({"layer": k, "cos_loop": round(float(u @ loop), 2), "cos_death": round(float(u @ death), 2),
                         "edge_doomed_minus_safe_sd": round((pr(c['de']) - pr(c['se'])) / sd, 2),
                         "ring_doomed_minus_safe_sd": round((pr(c['dr']) - pr(c['sr'])) / sd, 2)})
        out[f'{net}_{state}'] = rows
        print(net, state, [(r_['layer'], r_['cos_death'], r_['edge_doomed_minus_safe_sd']) for r_ in rows], flush=True)
json.dump(out, open('runs/closure_vs_death_cosine.json', 'w'), indent=1)

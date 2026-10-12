import os
import numpy as np, torch, random, glob, time
from goconcepts.katago_adapter import KataGoProbeModel, gamestate_from_game
from goconcepts import sgf
from goconcepts.cyclic import load_cases
m=KataGoProbeModel(os.path.join(os.environ.get('GOCONCEPTS_NETS', 'nets'), 'kata1-tf3-b11c768-s11001M-d5973M.ckpt'))
cases=[c for c in load_cases() if c.correct]
ps=[]
for s in range(0,len(cases),12): ps.append(m.activations([m.encode(gamestate_from_game(c.game)) for c in cases[s:s+12]],layers=[])['policy'])
p=np.exp(np.concatenate(ps)); corr=np.zeros_like(p)
for i,c in enumerate(cases):
    for col,mv in c.correct: corr[i,361 if mv is None else mv[0]*19+mv[1]]=1
print('tf3 mean policy mass on CORRECT', round(float((p*corr).sum(1).mean()),3),flush=True)

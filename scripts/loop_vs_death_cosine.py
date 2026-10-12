# Usage: PYTHONPATH=. python3 scripts/loop_vs_death_cosine.py   (cosine of loop direction and death axis per layer)
import numpy as np, pickle
from goconcepts.mech_cross import trunk_means
cases=pickle.load(open('data/status_bal.pkl','rb')); geo=np.array([c.variant for c in cases]); st=np.array([c.status for c in cases])
half=np.random.default_rng(0).permutation(len(cases))%2; fit=half==0
c={"dr":(geo==0)&(st==0),"de":(geo==1)&(st==0),"sr":(geo==0)&(st==1),"se":(geo==1)&(st==1)}
out={}
for net in ['g170','b40','tf3']:
  for state in ['pass','move']:
    T=trunk_means(net,cases,state)
    cs=[]
    for k in range(T.shape[1]):
        u=T[fit&c['sr'],k].mean(0)-T[fit&c['se'],k].mean(0); u/=np.linalg.norm(u)
        d=T[fit&c['de'],k].mean(0)-T[fit&c['se'],k].mean(0); d/=np.linalg.norm(d)
        cs.append(round(float(u@d),2))
    out[f'{net}_{state}']=cs
    print(net,state,cs)
import json; json.dump(out,open('runs/loop_vs_death_cosine.json','w'),indent=1)

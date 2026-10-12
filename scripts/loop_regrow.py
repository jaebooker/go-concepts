# Usage: PYTHONPATH=. python3 scripts/loop_regrow.py b40 12   (hold the loop direction after one block; track it downstream)
import numpy as np, torch, json, sys
from goconcepts.gating import forward
from goconcepts.late_channels import load, batches
from goconcepts.mech_cross import trunk_means
net_name, hk = sys.argv[1], int(sys.argv[2])
cases, model, X, masks, geo, st = load(net_name); net=model.net; n=len(cases); nb=net.n_blocks
half=np.random.default_rng(0).permutation(n)%2; fit,ev=half==0,half==1
cells={"dr":(geo==0)&(st==0),"de":(geo==1)&(st==0),"sr":(geo==0)&(st==1),"se":(geo==1)&(st==1)}
T=trunk_means(net_name,cases,"pass")
Uu=[];
for k in range(nb+1):
    u=T[fit&cells["sr"],k].mean(0)-T[fit&cells["se"],k].mean(0); u/=np.linalg.norm(u); Uu.append(u)
Uu=np.stack(Uu)
u=torch.tensor(Uu[hk],dtype=torch.float32); tg=float(T[fit&cells["de"],hk].mean(0)@Uu[hk])
out={}
for name,holds in (("base",{}),("hold",{hk:(u,tg)})):
    S=np.zeros((n,nb+1,net.C),np.float32)
    for idx in batches(n):
        x=torch.tensor(np.stack([X["pass"][i][0] for i in idx])); g=torch.tensor(np.stack([X["pass"][i][1] for i in idx]))
        _,s=forward(net,x,g,holds,torch.tensor(masks[idx])); S[idx]=s
    p=np.einsum("nkc,kc->nk",S,Uu)
    out[name]=p
edge=out["base"][ev&cells["de"]].mean(0); ring_b=out["base"][ev&cells["dr"]].mean(0); ring_h=out["hold"][ev&cells["dr"]].mean(0)
frac=(ring_h-edge)/(ring_b-edge)
print(net_name,'hold at',hk,'loop signal on doomed loops, share of baseline ring-edge gap, layers',hk,'..end:')
print(np.round(frac[hk:],2).tolist())

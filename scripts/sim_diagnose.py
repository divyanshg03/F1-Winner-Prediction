"""Calibration-set (2019-21) diagnostic: which simulator ingredient erodes the starting-grid advantage?"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from f1pred.sim import Params, simulate, laplace
from f1pred.simrun import Context
ctx = Context()
won = ctx.df[ctx.df.won.notna()]
keys = [k for k in ctx.lap_keys if "2019" <= k[:4] <= "2021" and k in set(won.race_key)]
setups = {k: ctx.setup(k) for k in keys}
act = np.mean([setups[k][1]["rows"].query("grid==grid.min()").won.sum() for k in keys])
print(f"races={len(keys)}  actual pole win rate={act:.3f}")
base = dict(pass_thr=0.035, start_sigma=0.6, pace_sigma_mult=0.45)
def run(label, prm_kw=None, mut=None):
    pk = {**base, **(prm_kw or {})}; prm = Params(s=600, **pk); pole, ll = [], []
    for k in keys:
        st, aux = setups[k]
        if mut: 
            import copy; st = copy.copy(st); mut(st)
        o = simulate(st, prm, seed=3); r = aux["rows"]
        pole.append(o["win"][int(np.argmin(st.grid))]); p = laplace(o["win"], prm.s)
        w = np.where(r.won.to_numpy() == 1)[0]; ll.append(-np.log(p[w[0]]))
    print(f"{label:34s} pole-win {np.mean(pole):.3f}   logloss {np.mean(ll):.3f}", flush=True)
run("v1 (independent strategies)")
run("shared strategy 0.7", dict(strategy_share=0.7))
run("shared strategy 0.9", dict(strategy_share=0.9))
run("no retirements", None, lambda s: setattr(s, "p_retire", s.p_retire * 0))
run("no safety cars", None, lambda s: setattr(s, "force_sc", False))
run("no start noise", dict(start_sigma=0.0))
run("no pace shocks", dict(pace_sigma_mult=0.0))
run("shared 0.9 + no pace shocks", dict(strategy_share=0.9, pace_sigma_mult=0.0))

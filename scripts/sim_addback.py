"""Add-back ablation on the 2019-21 calibration races: start from a noiseless race, switch ingredients on one by one."""
import sys, copy, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from f1pred.sim import Params, simulate
from f1pred.simrun import Context
ctx = Context()
keys = [k for k in ctx.lap_keys if "2019" <= k[:4] <= "2021" and k in set(ctx.df[ctx.df.won.notna()].race_key)]
setups = {k: ctx.setup(k) for k in keys}
print(f"races={len(keys)}  actual pole win=0.475")
def run(label, pits=False, sc=False, dnf=False, start=False, shocks=False, share=0.0, thr=0.035):
    prm = Params(s=500, pass_thr=thr, start_sigma=0.6 if start else 0.0, pace_sigma_mult=0.45 if shocks else 0.0, strategy_share=share)
    pole = []
    for k in keys:
        st, aux = setups[k]; st = copy.copy(st)
        if not pits: st.strategies = [[]]
        st.force_sc = None if sc else False
        if not dnf: st.p_retire = st.p_retire * 0
        o = simulate(st, prm, seed=5); pole.append(o["win"][int(np.argmin(st.grid))])
    print(f"{label:46s} pole-win {np.mean(pole):.3f}", flush=True)
run("nothing random, no stops")
run("+ pit stops (independent)", pits=True)
run("+ pit stops (shared 0.9)", pits=True, share=0.9)
run("+ safety cars", sc=True)
run("+ retirements", dnf=True)
run("+ start noise", start=True)
run("+ pace shocks", shocks=True)
run("everything (v1)", True, True, True, True, True)
run("nothing random, no stops, easy passing 0.006", thr=0.006)
run("no stops, only pace shocks, easy passing 0.006", shocks=True, thr=0.006)

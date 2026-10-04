"""Reactive-strategy simulator (v2). EXPLORATORY: the 2022+ test races were already scored once by sim v1.
Protocol: tune ONLY on 2019-21 (objective = winner log-loss; pole-win rate reported as a realism check),
freeze, then score 2022+ once. v1 numbers are printed alongside for context."""
import itertools, json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from f1pred import backtest as B
from f1pred.sim import Params, laplace, simulate
from f1pred.simrun import Context

REP = ROOT / "reports"; ctx = Context()
has = set(ctx.df[ctx.df.won.notna()].race_key)
calib = [k for k in ctx.lap_keys if "2019" <= k[:4] <= "2021" and k in has]
test = [k for k in ctx.lap_keys if k[:4] >= "2022" and k in has]
print(f"calibration races {len(calib)}, test races {len(test)}", flush=True)
BASE = dict(start_sigma=0.6, pace_sigma_mult=0.45, shared_shift=True, indiv_jitter=1)
cs = {k: ctx.setup(k) for k in calib}

def score(setups, prm, keys):
    ll, pole = [], []
    for k in keys:
        st, aux = setups[k]; o = simulate(st, prm, seed=1); p = laplace(o["win"], prm.s)
        w = np.where(aux["rows"].won.to_numpy() == 1)[0]
        ll.append(-np.log(p[w[0]]) if len(w) else np.nan); pole.append(o["win"][int(np.argmin(st.grid))])
    return float(np.nanmean(ll)), float(np.mean(pole))

rows = []
for share, pu, pc, thr in itertools.product([0.7, 0.9], [0.0, 0.35], [0.0, 0.6], [0.014, 0.035]):
    kw = dict(strategy_share=share, p_under=pu, p_cover=pc, pass_thr=thr)
    ll, pole = score(cs, Params(s=600, **BASE, **kw), calib); rows.append({**kw, "logloss": ll, "pole_win": pole})
    print(kw, round(ll, 4), "pole", round(pole, 3), flush=True)
cal = pd.DataFrame(rows).sort_values("logloss"); cal.to_csv(REP / "sim_v2_calibration.csv", index=False)
best = {k: float(cal.iloc[0][k]) for k in ["strategy_share", "p_under", "p_cover", "pass_thr"]}
print("FROZEN:", best, "calib pole-win", round(float(cal.iloc[0].pole_win), 3), "(actual 0.475)", flush=True)
(REP / "sim_v2_params.json").write_text(json.dumps({**BASE, **best}, indent=2))
prm = Params(s=4000, **BASE, **best)

out = []
for i, k in enumerate(test):
    st, aux = ctx.setup(k); o = simulate(st, prm, seed=7); r = aux["rows"]
    out.append(pd.DataFrame({"race_key": k, "driver_id": r.driver_id, "p_sim": laplace(o["win"], prm.s), "grid": r.grid.to_numpy()}))
    if i % 25 == 0: print("  scored", i, k, flush=True)
v2 = pd.concat(out); v2.to_csv(REP / "sim_v2_scores.csv", index=False)
bs = pd.read_csv(REP / "backtest_scores.csv"); v1 = pd.read_csv(REP / "sim_scores.csv")[["race_key", "driver_id", "p_sim"]].rename(columns={"p_sim": "p_v1"})
res = bs.merge(v2[["race_key", "driver_id", "p_sim"]], on=["race_key", "driver_id"]).merge(v1, on=["race_key", "driver_id"])
res = res[res.groupby("race_key").race_key.transform("size") >= 15]
res["s_sim2"] = np.log(res.p_sim); res["s_sim1"] = np.log(res.p_v1); res["s_blend2"] = 0.5 * res.s_ens_post + 0.5 * res.s_sim2
tab = B.summarize(res, ["grid_logit", "ens_post", "sim1", "sim2", "blend2"], n_boot=1000)
print(f"\nraces compared: {res.race_key.nunique()}"); print(tab[["model", "top1", "top3", "logloss", "ll_lo", "ll_hi"]].round(3).to_string(index=False))
hold = res.season >= 2024
for a, b in [("sim2", "sim1"), ("sim2", "grid_logit"), ("sim2", "ens_post"), ("blend2", "ens_post")]:
    for lab, m in [("2022+", None), ("2024+", hold)]:
        d = B.paired_diff(res, a, b, m); print(f"  {a} vs {b} [{lab}]: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] n={d['races']}")
pw = lambda c: res[res.grid == res.groupby("race_key").grid.transform("min")].groupby("race_key")[c].sum().mean()
print(f"\nDIAGNOSTIC pole win: sim1 {pw('p_v1'):.3f} | sim2 {pw('p_sim'):.3f} | actual {pw('won'):.3f}")
res.to_csv(REP / "sim_v2_vs_tabular.csv", index=False)

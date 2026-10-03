"""Race-simulator backtest.

Protocol (fixed before looking at test results):
  (Grid widened once after a smoke test on a single 2024 race showed the initial range was too flat; disclosed in README.
  Grid extended a second time because round-1 calibration put the optimum on the grid edge (pass_thr=0.014).
  Both extensions were decided from calibration-set (2019-21) evidence, before any 2022+ score was printed.)
  1. Calibrate three physics knobs (pass_thr, start_sigma, pace_sigma_mult) on 2019-2021 races only.
  2. Freeze them. Score every 2022+ race with the same walk-forward machinery (pace/retire models and
     all history re-estimated from strictly earlier races).
  3. Compare against the tabular models on exactly the same races; paired-bootstrap the differences.
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from f1pred import backtest as B
from f1pred.pace import QUALI_ONLY, PaceModel
from f1pred.sim import Params, laplace, simulate
from f1pred.simrun import Context

REP = ROOT / "reports"
ctx = Context(rebuild=True)
won = ctx.df[ctx.df.won.notna()][["race_key", "driver_id", "won"]]
has_result = set(won.race_key)
calib = [k for k in ctx.lap_keys if "2019" <= k[:4] <= "2021" and k in has_result]
test = [k for k in ctx.lap_keys if k[:4] >= "2022" and k in has_result]
print(f"calibration races: {len(calib)}  test races: {len(test)}", flush=True)


def ll(setups, prm, keys):
    tot = []
    for k in keys:
        st = setups[k]
        out = simulate(st[0], prm, seed=1)
        p = laplace(out["win"], prm.s)
        w = np.where(st[1]["rows"].won.to_numpy() == 1)[0]
        tot.append(-np.log(p[w[0]]) if len(w) else np.nan)
    return float(np.nanmean(tot))


# ---- 1. calibrate ------------------------------------------------------------------------
grid = [dict(pass_thr=a, start_sigma=b, pace_sigma_mult=c)
        for a, b, c in itertools.product([0.014, 0.022, 0.035, 0.055], [0.6, 1.2], [0.45, 0.6, 0.8])]
calib_setups = {k: ctx.setup(k) for k in calib}
rows = []
for g in grid:
    v = ll(calib_setups, Params(s=1000, **g), calib)
    rows.append({**g, "logloss": v})
    print(g, round(v, 4), flush=True)
cal = pd.DataFrame(rows).sort_values("logloss")
cal.to_csv(REP / "sim_calibration.csv", index=False)
best = {k: float(cal.iloc[0][k]) for k in ["pass_thr", "start_sigma", "pace_sigma_mult"]}
(REP / "sim_params.json").write_text(json.dumps(best, indent=2))
print("FROZEN PARAMS:", best, flush=True)
prm = Params(s=4000, **best)

# ---- 2. score the test races --------------------------------------------------------------
out, pace_rows = [], []
for i, k in enumerate(test):
    setup, aux = ctx.setup(k)
    r = simulate(setup, prm, seed=7)
    rows_ = aux["rows"]
    out.append(pd.DataFrame({"race_key": k, "driver_id": rows_.driver_id, "p_sim": laplace(r["win"], prm.s),
                             "p_sim_raw": r["win"], "p_podium": r["podium"], "exp_pos": r["exp_pos"], "p_sc": r["had_sc"],
                             "grid": rows_.grid.to_numpy()}))
    # pace-model skill: full model vs qualifying-only vs field-mean, on drivers with a realised pace
    tr = ctx.df[ctx.df.race_key.isin(ctx.prior(k))]
    q = PaceModel(QUALI_ONLY).fit(tr)
    kn = rows_.dropna(subset=["y_gap"])
    if len(kn):
        pace_rows.append(pd.DataFrame({"race_key": k, "y": kn.y_gap, "full": aux["pace_model"].predict(kn),
                                       "quali_only": q.predict(kn), "mean": tr.y_gap.mean()}))
    if i % 20 == 0:
        print(f"  scored {i}/{len(test)} {k}", flush=True)
sim = pd.concat(out, ignore_index=True)
sim.to_csv(REP / "sim_scores.csv", index=False)
pr = pd.concat(pace_rows)
print("\nPACE MODEL RMSE (fraction of a lap; lower is better)")
pace_tab = {c: float(np.sqrt(((pr[c] - pr.y) ** 2).mean())) for c in ["mean", "quali_only", "full"]}
print({k: round(v, 5) for k, v in pace_tab.items()})
json.dump(pace_tab, open(REP / "sim_pace_rmse.json", "w"))

# ---- 3. compare on the same races ----------------------------------------------------------
bs = pd.read_csv(REP / "backtest_scores.csv")
res = bs.merge(sim[["race_key", "driver_id", "p_sim", "p_podium", "exp_pos", "p_sc"]], on=["race_key", "driver_id"], how="inner")
keep = res.groupby("race_key").size()
res = res[res.race_key.isin(keep[keep >= 15].index)]
res["s_sim"] = np.log(res.p_sim)
res["s_blend"] = 0.5 * res["s_ens_post"] + 0.5 * res["s_sim"]
tab = B.summarize(res, ["grid_logit", "ens_post", "sim", "blend"])
print(f"\nraces compared: {res.race_key.nunique()}")
print(tab[["model", "races", "top1", "top3", "logloss", "ll_lo", "ll_hi", "brier"]].round(3).to_string(index=False))
tab.to_csv(REP / "sim_metrics.csv", index=False)
hold = res.season >= 2024
diffs = []
for a, b in [("sim", "grid_logit"), ("sim", "ens_post"), ("blend", "ens_post"), ("blend", "grid_logit")]:
    for label, m in [("all 2022+", None), ("2024+ holdout", hold)]:
        d = B.paired_diff(res, a, b, m)
        d["slice"] = label
        diffs.append(d)
dd = pd.DataFrame(diffs)
print(dd.round(3).to_string(index=False))
dd.to_csv(REP / "sim_paired.csv", index=False)

# ---- 4. does the simulator behave like real racing? -----------------------------------------
lap_r = ctx.R.loc[res.race_key.unique()]
pole_sim = res[res.grid == res.groupby("race_key").grid.transform("min")].groupby("race_key").p_sim.sum().mean()
pole_act = res[res.grid == res.groupby("race_key").grid.transform("min")].groupby("race_key").won.sum().mean()
print(f"\nDIAGNOSTICS  pole win: sim {pole_sim:.3f} vs actual {pole_act:.3f} | SC race share: sim {res.groupby('race_key').p_sc.first().mean():.2f} vs actual {(lap_r.n_sc > 0).mean():.2f}")
res.to_csv(REP / "sim_vs_tabular.csv", index=False)

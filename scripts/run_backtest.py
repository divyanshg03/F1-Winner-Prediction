"""Build features, run the walk-forward backtest, write reports/backtest.csv + metrics tables."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from f1pred import backtest as B
from f1pred import features as F

REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

df = F.make()
df.to_csv(ROOT / "data" / "processed" / "features.csv", index=False)
print("feature rows:", len(df), "races:", df.race_key.nunique())

res = B.walk_forward(df, first_test_season=2018, refit_every=1)
res.to_csv(REP / "backtest_scores.csv", index=False)

names = ["grid_logit", "logit_pre", "gbm_pre", "ens_pre", "ens_pre_cal",
         "logit_post", "gbm_post", "ens_post", "ens_post_cal"]
tab = B.summarize(res, names)

# baselines that need no model
pole = res.groupby("race_key").apply(lambda g: float(g.loc[g.grid.idxmin(), "won"]), include_groups=False)
uniform_ll = float(np.mean(np.log(res.groupby("race_key").size())))
print(f"\nraces scored: {res.race_key.nunique()}  uniform logloss={uniform_ll:.3f}  pole top1={pole.mean():.3f}")
print(tab.round(3).to_string(index=False))
tab.to_csv(REP / "metrics_overall.csv", index=False)

# by era: regulation resets are where models should struggle
res["era"] = pd.cut(res.season, [2017, 2021, 2025, 2030], labels=["2018-21", "2022-25 (ground effect)", "2026 (new regs)"])
rows = []
for era, g in res.groupby("era", observed=True):
    t = B.summarize(g, ["grid_logit", "ens_pre_cal", "ens_post_cal"], n_boot=500)
    t["era"] = era
    t["pole_top1"] = g.groupby("race_key").apply(lambda x: float(x.loc[x.grid.idxmin(), "won"]), include_groups=False).mean()
    rows.append(t)
era = pd.concat(rows)
print(era[["era", "model", "races", "top1", "logloss", "pole_top1"]].round(3).to_string(index=False))
era.to_csv(REP / "metrics_by_era.csv", index=False)

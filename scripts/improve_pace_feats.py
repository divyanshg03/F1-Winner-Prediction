"""Does adding lap-derived race-pace history to the winner model improve it?  (walk-forward, same 188 races)"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from f1pred import backtest as B, features as F, lapdata
from f1pred.models import BoostedRace, ConditionalLogit
from f1pred.pace import add_pace_history

df = F.make(); D, _ = lapdata.build_all()
df = add_pace_history(df, D)
for c in ["drv_pace_s", "con_pace_s", "drv_pace_l", "con_pace_l"]:
    df[c + "_rel"] = df[c] - df.groupby("race_key")[c].transform("min")
PACE = ["drv_pace_s", "drv_pace_l", "con_pace_s", "con_pace_l", "drv_pace_s_rel", "con_pace_s_rel", "drv_pace_l_rel", "con_pace_l_rel"]
cols = B.POST + PACE
B.MODELS["logit_pace"] = (lambda: ConditionalLogit(l2=3.0), cols)
B.MODELS["gbm_pace"] = (lambda: BoostedRace(), cols)
res = B.walk_forward(df, 2018, verbose=False)
res["s_ens_pace"] = 0.5 * res.s_logit_pace + 0.5 * res.s_gbm_pace
res.to_csv(ROOT / "reports" / "pace_feature_scores.csv", index=False)
names = ["grid_logit", "ens_post", "logit_pace", "gbm_pace", "ens_pace"]
for lab, m in [("DEV 2018-23", res.season <= 2023), ("HOLDOUT 2024-26", res.season >= 2024), ("ALL", res.season > 0)]:
    t = B.summarize(res[m], names, n_boot=300); print(lab); print(t[["model", "races", "top1", "top3", "logloss", "brier"]].round(3).to_string(index=False))
    d = B.paired_diff(res, "ens_pace", "ens_post", m); print(f"  ens_pace vs ens_post: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] nats/race")

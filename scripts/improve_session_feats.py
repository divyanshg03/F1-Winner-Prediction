"""Practice pace (FP2, else FP1) and race-day weather as features. Practice is known pre-race; weather here is the
ACTUAL race-day weather, i.e. optimistic vs a forecast, so any gain from it is an upper bound."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from f1pred import backtest as B, features as F
from f1pred.models import BoostedRace, ConditionalLogit

df = F.make(); rows = []
for f in (ROOT / "data/processed/sessions").glob("*.json"):
    d = json.load(open(f)); k = f.stem; w = d.get("weather") or {}
    for drv, v in (d.get("practice", {}).get("drivers") or {}).items():
        rows.append(dict(race_key=k, driver_id=drv, fp_best=v["best"], fp_long=v["longrun"]))
fp = pd.DataFrame(rows)
wx = pd.DataFrame([dict(race_key=f.stem, **(json.load(open(f)).get("weather") or {})) for f in (ROOT / "data/processed/sessions").glob("*.json")])
df = df.merge(fp, on=["race_key", "driver_id"], how="left").merge(wx, on="race_key", how="left")
print("practice coverage:", round(df[df.season >= 2018].fp_best.notna().mean(), 3), "weather coverage:", round(df[df.season >= 2018].rain.notna().mean(), 3))
for c in ["fp_best", "fp_long"]:
    df[c] = df[c].fillna(df.groupby("race_key")[c].transform(lambda s: s.quantile(0.9))).fillna(df[c].median())
    df[c + "_rel"] = df[c] - df.groupby("race_key")[c].transform("min")
df["fp_con_best"] = df.groupby(["race_key", "constructor_id"]).fp_best.transform("min")
for c in ["rain", "track", "air", "humidity", "wind"]:
    df[c] = df[c].fillna(df[c].median())
df["rain_x_grid"] = df.rain * df.grid_rank_sq
PRAC = ["fp_best", "fp_long", "fp_best_rel", "fp_long_rel", "fp_con_best"]
WX = ["rain", "track", "air", "humidity", "wind", "rain_x_grid"]
B.MODELS["logit_prac"] = (lambda: ConditionalLogit(l2=3.0), B.POST + PRAC)
B.MODELS["logit_wx"] = (lambda: ConditionalLogit(l2=3.0), B.POST + WX)
B.MODELS["logit_pw"] = (lambda: ConditionalLogit(l2=3.0), B.POST + PRAC + WX)
B.MODELS["gbm_pw"] = (lambda: BoostedRace(), B.POST + PRAC + WX)
res = B.walk_forward(df, 2018, verbose=False)
res["s_ens_pw"] = 0.5 * res.s_logit_pw + 0.5 * res.s_gbm_pw
res.to_csv(ROOT / "reports" / "session_feature_scores.csv", index=False)
for lab, m in [("DEV 2018-23", res.season <= 2023), ("HOLDOUT 2024-26", res.season >= 2024)]:
    t = B.summarize(res[m], ["ens_post", "logit_post", "logit_prac", "logit_wx", "logit_pw", "gbm_pw", "ens_pw"], n_boot=200)
    print(lab); print(t[["model", "races", "top1", "top3", "logloss"]].round(3).to_string(index=False))
    for n, ref in [("logit_prac", "logit_post"), ("logit_wx", "logit_post"), ("logit_pw", "logit_post"), ("ens_pw", "ens_post")]:
        d = B.paired_diff(res, n, ref, m); print(f"  {n} vs {ref}: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")

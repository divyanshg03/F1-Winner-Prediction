"""Predict the next un-run race and freeze the prediction to predictions/<season>-<round>.json.

Usage:  python scripts/predict_next.py [--refresh]   (--refresh re-pulls the current season first)
Trains on every completed race, scores the upcoming one with the post-qualifying
ensemble, and records a UTC timestamp + git commit so the call can be audited later.
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from f1pred import backtest as B
from f1pred import features as F
from f1pred import ingest
from f1pred.models import race_softmax, temperature_fit

if "--refresh" in sys.argv:
    ingest.build(2014, datetime.now().year)

res, qua, spr = F.load_raw()
df = F.build(res, qua, spr, include_future=True)
future = df[df["won"].isna()]
if future.empty:
    sys.exit("No upcoming race with qualifying data yet; run with --refresh after qualifying.")
key = future.race_key.iloc[0]
test = future[future.race_key == key].reset_index(drop=True)
train = df[df["won"].notna()].sort_values(["date", "round"]).reset_index(drop=True)
rid = train.race_key.to_numpy()

score = np.zeros(len(test))
parts = {}
for name in ("logit_post", "gbm_post"):
    mk, cols = B.MODELS[name]
    m = mk().fit(train[cols].to_numpy(float), train["won"].to_numpy(), rid)
    parts[name] = m.score(test[cols].to_numpy(float))
    score += 0.5 * parts[name]

# temperature learned from the walk-forward out-of-sample races (no extra leakage)
bt = pd.read_csv(ROOT / "reports" / "backtest_scores.csv")
hist = [(g["s_ens_post"].to_numpy(), int(np.argmax(g["won"].to_numpy()))) for _, g in bt.groupby("race_key")]
T = temperature_fit(hist)
p = race_softmax(score / T, np.zeros(len(test), dtype=int))

out = test[["driver", "constructor", "grid"]].copy()
out["win_prob"] = p
out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
rec = dict(
    race=str(test.race_name.iloc[0]), race_key=key, race_date=str(test.date.iloc[0].date()),
    generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit=commit,
    model="post-qualifying ensemble (conditional logit + LightGBM), trained on %d completed races" % train.race_key.nunique(),
    temperature=round(T, 3), note="grid = qualifying order (penalties not yet applied)",
    probabilities=[dict(driver=r.driver, constructor=r.constructor, grid=int(r.grid), p_win=round(float(r.win_prob), 4))
                   for r in out.itertuples()],
)
(ROOT / "predictions").mkdir(exist_ok=True)
path = ROOT / "predictions" / f"{key}.json"
path.write_text(json.dumps(rec, indent=2), encoding="utf8")
print(f"{rec['race']} ({rec['race_date']})  trained on {train.race_key.nunique()} races, T={T:.2f}")
print(out.head(8).assign(win_prob=lambda d: (100 * d.win_prob).round(1)).to_string(index=False))
print("saved", path)

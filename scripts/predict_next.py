"""Predict the next un-run race and freeze the prediction to predictions/<season>-<round>.json.

Usage:  python scripts/predict_next.py [--refresh] [--out PATH]
  --refresh   re-pull the current season first (do this after qualifying, before the race)
Decided model for race-week winner probabilities: the 'ens_post' ensemble (conditional logit + LightGBM + neural net,
post-qualifying features, temperature-calibrated), hyper-parameters from reports/final_hyperparams.json (chosen on
2022-24; scored 0.946 log-loss on the unseen 2025 season). It is refit on every completed race before the target race.
A frozen prediction file is never overwritten.
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
from f1pred.models import BoostedRace, ConditionalLogit, race_softmax
from f1pred.nn import NeuralRace

if "--refresh" in sys.argv:
    ingest.build(2014, datetime.now().year)

hp = json.load(open(ROOT / "reports" / "final_hyperparams.json"))
df = F.make(include_future=True)
future = df[df["won"].isna()]
if future.empty:
    sys.exit("No upcoming race with qualifying data yet; run with --refresh after qualifying.")
key = future.race_key.iloc[0]
test = future[future.race_key == key].reset_index(drop=True)
train = df[(df["won"].notna()) & (df.season >= 2016)].sort_values(["date", "round"]).reset_index(drop=True)
out_path = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "predictions" / f"{key}.json"
if out_path.exists():
    sys.exit(f"{out_path} already exists (frozen predictions are never overwritten). Use --out for a scratch copy.")

cols = B.POST
makers = {"logit": lambda: ConditionalLogit(**hp["chosen"]["logit_post"]), "gbm": lambda: BoostedRace(**hp["chosen"]["gbm_post"]),
          "nn": lambda: NeuralRace(**hp["chosen"]["nn_post"])}
Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
scores = [makers[n]().fit(Xtr, train.won.to_numpy(), train.race_key.to_numpy()).score(Xte) for n in makers]
p = race_softmax(np.mean(scores, axis=0) / hp["temps"]["post"], np.zeros(len(test), dtype=int))

out = test[["driver", "constructor", "grid"]].copy()
out["win_prob"] = p
out = out.sort_values("win_prob", ascending=False).reset_index(drop=True)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
rec = dict(
    race=str(test.race_name.iloc[0]), race_key=key, race_date=str(test.date.iloc[0].date()),
    generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), git_commit=commit,
    model=f"post-qualifying ensemble (logit + LightGBM + neural net), refit on {train.race_key.nunique()} completed races (2016-)",
    temperature=hp["temps"]["post"], note="grid = qualifying order (penalties not yet applied)",
    probabilities=[dict(driver=r.driver, constructor=r.constructor, grid=int(r.grid), p_win=round(float(r.win_prob), 4)) for r in out.itertuples()],
)
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(rec, indent=2), encoding="utf8")
print(f"{rec['race']} ({rec['race_date']})  refit on {train.race_key.nunique()} races")
print(out.head(8).assign(win_prob=lambda d: (100 * d.win_prob).round(1)).to_string(index=False))
print("saved", out_path)

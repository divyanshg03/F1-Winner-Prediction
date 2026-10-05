"""Demo server: browse backtested races, see the frozen next-race call, and play with the grid.

Run:  python -m uvicorn app.server:app --port 8000      (from the repo root)
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from f1pred import features as F
from f1pred.models import race_softmax
from f1pred.serve import ServeModel

app = FastAPI(title="F1 Winner Predictor")

DF = F.make(include_future=True).sort_values(["date", "round"]).reset_index(drop=True)
MODEL = ServeModel(ROOT / "deploy" / "serve")  # pre-trained weights: no training (and no PyTorch) at startup
BT = pd.read_csv(ROOT / "reports" / "backtest_scores.csv")
FROZEN = {p.stem: json.loads(p.read_text(encoding="utf8")) for p in (ROOT / "predictions").glob("*.json")}


def live_probs(rows: pd.DataFrame) -> np.ndarray:
    return race_softmax(MODEL.score(rows[MODEL.cols].to_numpy(float)), np.zeros(len(rows), dtype=int))


def _winner(key):
    w = DF[(DF.race_key == key) & (DF.won == 1)]
    return None if w.empty else w.driver.iloc[0]


@app.get("/api/summary")
def summary():
    m = pd.read_csv(ROOT / "reports" / "metrics_overall.csv").set_index("model")
    pick = lambda n: {k: round(float(m.loc[n, k]), 3) for k in ("top1", "top3", "logloss")}
    return {"races": int(m.loc["ens_post", "races"]), "grid_only": pick("grid_logit"),
            "before_quali": pick("ens_pre"), "full_model": pick("ens_post_cal")}


@app.get("/api/races")
def races():
    r = DF.drop_duplicates("race_key")[["race_key", "season", "round", "race_name", "date"]]
    out = []
    for x in r.itertuples():
        out.append(dict(key=x.race_key, season=int(x.season), name=x.race_name, date=str(x.date.date()),
                        winner=_winner(x.race_key), backtested=x.race_key in set(BT.race_key), frozen=x.race_key in FROZEN))
    return out[::-1]


@app.get("/api/race/{key}")
def race(key: str):
    r = DF[DF.race_key == key].reset_index(drop=True)
    if r.empty:
        raise HTTPException(404, "unknown race")
    bt = BT[BT.race_key == key]
    fz = FROZEN.get(key)
    if fz:  # the pre-race call that was timestamped and committed before the race
        fp = {x["driver"]: x["p_win"] for x in fz["probabilities"]}
        probs = {x.driver_id: fp.get(x.driver, 0.0) for x in r.itertuples()}
        tot = sum(probs.values()) or 1.0
        probs = {k: v / tot for k, v in probs.items()}
        source = f"frozen pre-race prediction ({fz['generated_utc'][:16]} UTC, before the race)"
    elif len(bt):  # honest out-of-sample call, made with only earlier races
        p = race_softmax(bt.s_ens_post.to_numpy(float), np.zeros(len(bt), dtype=int))
        probs = dict(zip(bt.driver_id, p))
        source = "walk-forward backtest (trained only on earlier races)"
    else:
        probs = dict(zip(r.driver_id, live_probs(r)))
        done = r.won.notna().any()
        source = ("model trained on all completed races, INCLUDING this one, so not a fair test" if done
                  else "live model (trained on all completed races)")
    drivers = [dict(id=x.driver_id, name=x.driver, team=x.constructor, grid=int(x.grid), p=float(probs.get(x.driver_id, 0)),
                    won=bool(x.won == 1), finish=None if pd.isna(x.pos) else int(x.pos),
                    status=None if pd.isna(x.pos) else str(x.status)) for x in r.itertuples()]
    drivers.sort(key=lambda d: -d["p"])
    return dict(key=key, name=r.race_name.iloc[0], date=str(r.date.iloc[0].date()), source=source,
                winner=_winner(key), drivers=drivers, frozen=FROZEN.get(key))


class WhatIf(BaseModel):
    grid: dict[str, int]  # driver_id -> new grid slot


@app.post("/api/race/{key}/whatif")
def whatif(key: str, body: WhatIf):
    r = DF[DF.race_key == key].copy().reset_index(drop=True)
    if r.empty:
        raise HTTPException(404, "unknown race")
    g = r.driver_id.map(body.grid).fillna(r.grid).astype(float)
    if g.nunique() != len(g):
        raise HTTPException(422, "grid slots must be unique")
    r["grid"], r["qual_pos"] = g, g
    r["grid_rank_sq"], r["front_row"], r["is_pole"] = np.sqrt(g), (g <= 2).astype(int), (g == 1).astype(int)
    r["grid_pen"], r["sprint_rank"] = 0.0, g
    p = live_probs(r)
    out = sorted(({"id": d, "name": n, "grid": int(gg), "p": float(pp)} for d, n, gg, pp in zip(r.driver_id, r.driver, g, p)),
                 key=lambda x: -x["p"])
    return dict(drivers=out, note="Uses a model trained on all completed races, so for past races this is illustrative, not out-of-sample.")


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")

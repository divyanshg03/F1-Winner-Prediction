"""Final model: train on 2016-2024, test on the unseen 2025 season (and 2026 as a second unseen set).

Protocol
  1. Choose hyper-parameters using ONLY 2016-2021 -> 2022-2024 validation.
  2. Fit the temperature for the ensemble on those same validation predictions.
  3. Refit on all of 2016-2024. Score 2025 once with this single fixed model (no per-race refitting).
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from f1pred import backtest as B
from f1pred import features as F
from f1pred.models import BoostedRace, ConditionalLogit, race_softmax, temperature_fit
from f1pred.nn import NeuralRace

REP, MOD = ROOT / "reports", ROOT / "models"
MOD.mkdir(exist_ok=True)
df = F.make().sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
lab = df[df.won.notna()]
TR = lab[(lab.season >= 2016) & (lab.season <= 2024)]
SUB, VAL = TR[TR.season <= 2021], TR[TR.season >= 2022]
T25, T26 = lab[lab.season == 2025], lab[lab.season == 2026]
print(f"train {TR.race_key.nunique()} races (2016-24) | val {VAL.race_key.nunique()} | test 2025 {T25.race_key.nunique()} | test 2026 {T26.race_key.nunique()}", flush=True)


def ll(model, d, cols):
    p = model.predict(d[cols].to_numpy(float), d.race_key.to_numpy())
    return float(-np.log(p[d.won.to_numpy() == 1]).mean())


def fit(make, d, cols):
    return make().fit(d[cols].to_numpy(float), d.won.to_numpy(), d.race_key.to_numpy())


GRIDS = {
    "logit": [dict(l2=v) for v in (0.3, 1, 3, 10, 30)],
    "gbm": [dict(n_estimators=n, num_leaves=l, min_child_samples=m) for n, l, m in
            [(150, 4, 40), (250, 6, 40), (400, 6, 60), (250, 10, 40), (500, 4, 80)]],
    "nn": [dict(hidden=h, wd=w, epochs=e, dropout=0.2) for h, w, e in itertools.product((16, 32), (1e-3, 1e-2), (30, 80))],
}
MAKE = {"logit": lambda **k: ConditionalLogit(**k), "gbm": lambda **k: BoostedRace(**k), "nn": lambda **k: NeuralRace(**k)}
chosen, val_scores = {}, {}
out = {}
for fs_name, cols in (("post", B.POST), ("pre", B.PRE)):
    for algo, grid in GRIDS.items():
        best = None
        for kw in grid:
            m = fit(lambda: MAKE[algo](**kw), SUB, cols)
            v = ll(m, VAL, cols)
            print(f"[{fs_name}] {algo} {kw} val log-loss {v:.4f}", flush=True)
            if best is None or v < best[0]:
                best = (v, kw, m)
        chosen[f"{algo}_{fs_name}"] = best[1]
        val_scores[f"{algo}_{fs_name}"] = best[2].score(VAL[cols].to_numpy(float))
        print(f"  -> chosen {algo}_{fs_name}: {best[1]}  (val {best[0]:.4f})", flush=True)

# ensemble temperature fitted on validation predictions only
temps = {}
for fs in ("post", "pre"):
    s = np.mean([val_scores[f"{a}_{fs}"] for a in ("logit", "gbm", "nn")], axis=0)
    bags = [(s[(VAL.race_key == k).to_numpy()], int(np.argmax(VAL[VAL.race_key == k].won.to_numpy()))) for k in VAL.race_key.unique()]
    temps[fs] = temperature_fit(bags)
print("ensemble temperatures (val):", {k: round(v, 3) for k, v in temps.items()}, flush=True)

# refit on all of 2016-2024 and score the unseen seasons ONCE
final = {}
for fs_name, cols in (("post", B.POST), ("pre", B.PRE)):
    for algo in GRIDS:
        final[f"{algo}_{fs_name}"] = fit(lambda: MAKE[algo](**chosen[f"{algo}_{fs_name}"]), TR, cols)
grid_m = fit(lambda: ConditionalLogit(l2=1.0), TR, B.GRID_ONLY)
joblib.dump(dict(models=final, chosen=chosen, temps=temps, features=dict(post=B.POST, pre=B.PRE)), MOD / "final_2016_2024.joblib")
(REP / "final_hyperparams.json").write_text(json.dumps(dict(chosen=chosen, temps=temps), indent=2, default=float))


def score_frame(d):
    r = d[["race_key", "season", "round", "race_name", "driver", "driver_id", "constructor", "grid", "won"]].copy()
    r["s_grid_logit"] = grid_m.score(d[B.GRID_ONLY].to_numpy(float))
    for fs, cols in (("post", B.POST), ("pre", B.PRE)):
        for a in ("logit", "gbm", "nn"):
            r[f"s_{a}_{fs}"] = final[f"{a}_{fs}"].score(d[cols].to_numpy(float))
        r[f"s_ens_{fs}"] = np.mean([r[f"s_{a}_{fs}"] for a in ("logit", "gbm", "nn")], axis=0) / temps[fs]
    return r


names = ["grid_logit", "logit_pre", "gbm_pre", "nn_pre", "ens_pre", "logit_post", "gbm_post", "nn_post", "ens_post"]
for label, d in (("2025", T25), ("2026", T26)):
    r = score_frame(d)
    r.to_csv(REP / f"final_scores_{label}.csv", index=False)
    pole = r.groupby("race_key").apply(lambda g: float(g.loc[g.grid.idxmin(), "won"]), include_groups=False).mean()
    uni = float(np.mean(np.log(r.groupby("race_key").size())))
    tab = B.summarize(r, names, n_boot=2000)
    print(f"\n===== UNSEEN {label}: {r.race_key.nunique()} races | pole-sitter wins {pole:.3f} | uniform log-loss {uni:.3f}")
    print(tab[["model", "top1", "top1_lo", "top1_hi", "top3", "logloss", "ll_lo", "ll_hi", "brier"]].round(3).to_string(index=False))
    tab.to_csv(REP / f"final_metrics_{label}.csv", index=False)
    for a, b_ in [("ens_post", "grid_logit"), ("ens_pre", "grid_logit"), ("ens_post", "ens_pre"), ("nn_post", "logit_post"), ("gbm_post", "logit_post")]:
        d_ = B.paired_diff(r, a, b_)
        print(f"  {a} vs {b_}: {d_['diff']:+.3f} [{d_['lo']:+.3f},{d_['hi']:+.3f}] nats/race (+ = first better)")
    if label == "2025":
        p = B.probs(r, "ens_post")
        cat = pd.cut(p, [0, .02, .05, .1, .2, .35, .5, .7, 1.01])
        g = pd.DataFrame({"p": p, "y": r.won, "b": cat}).groupby("b", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
        print("\ncalibration (2025, ens_post):"); print(g.round(3).to_string())
        r["p_ens_post"] = p
        r["p_ens_pre"] = B.probs(r, "ens_pre")
        r.to_csv(REP / "final_predictions_2025.csv", index=False)

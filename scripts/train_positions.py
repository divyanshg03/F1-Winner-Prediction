"""Predict the WHOLE finishing order (P1..P_last) of a race, evaluated against the real positions.

Train: every driver of every race 2016-2024 (full finishing order as the target), all feature parameters.
Test : 2025 and 2026 races, ONE fixed model (no refitting), scored against the correct finishing positions.
Hyper-parameters, feature set and ensemble membership are chosen on 2016-21 -> 2022-24 only (by mean Spearman).
Baseline: 'finish where you start' (grid order).
GPU: CatBoost (regression + YetiRank), XGBoost ranker, PyTorch Plackett-Luce. LightGBM / sklearn run on the CPU.
"""
import itertools, json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRanker, CatBoostRegressor, Pool
from scipy.stats import kendalltau, rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from f1pred import backtest as B, features as F
from f1pred.extra_features import GROUPS, add_groups
from f1pred.rank import ELO, PlackettLuce, add_ratings

REP = ROOT / "reports"

# ------------------------------------------------------------------ data + ALL feature parameters
df = add_ratings(add_groups(F.make()), 0.05, 0.2)
rows = []
for f in (ROOT / "data/processed/sessions").glob("*.json"):
    d = json.load(open(f)); w = d.get("weather") or {}
    for drv, v in (d.get("practice", {}).get("drivers") or {}).items():
        rows.append(dict(race_key=f.stem, driver_id=drv, fp_best=v["best"], fp_long=v["longrun"]))
wx = pd.DataFrame([dict(race_key=f.stem, **(json.load(open(f)).get("weather") or {})) for f in (ROOT / "data/processed/sessions").glob("*.json")])
df = df.merge(pd.DataFrame(rows), on=["race_key", "driver_id"], how="left").merge(wx, on="race_key", how="left")
for c in ["fp_best", "fp_long"]:
    df[c] = df[c].fillna(df.groupby("race_key")[c].transform(lambda s: s.quantile(0.9))).fillna(df[c].median())
    df[c + "_rel"] = df[c] - df.groupby("race_key")[c].transform("min")
df["fp_con_best"] = df.groupby(["race_key", "constructor_id"]).fp_best.transform("min")
for c in ["rain", "track", "air", "humidity", "wind"]:
    df[c] = df[c].fillna(df[c].median())
df["rain_x_grid"] = df.rain * df.grid_rank_sq
PACE = ["drv_pace_s", "drv_pace_l", "con_pace_s", "con_pace_l"]
EXTRA = sum(GROUPS.values(), [])
PRAC = ["fp_best", "fp_long", "fp_best_rel", "fp_long_rel", "fp_con_best"]
WX = ["rain", "track", "air", "humidity", "wind", "rain_x_grid"]
FS = {"POST": B.POST, "ALL_PREQ": B.POST + ELO + PACE + EXTRA, "ALL": B.POST + ELO + PACE + EXTRA + PRAC + WX}
FS = {k: list(dict.fromkeys(v)) for k, v in FS.items()}
print({k: len(v) for k, v in FS.items()}, "feature counts", flush=True)

df = df.sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
lab = df[df.won.notna()].copy()
n_race = lab.groupby("race_key").driver_id.transform("size")
lab["ypos"] = (lab.pos - 1) / (n_race - 1)  # 0 = winner .. 1 = last
TR = lab[(lab.season >= 2016) & (lab.season <= 2024)]
SUB, VAL = TR[TR.season <= 2021], TR[TR.season >= 2022]
TEST = {"2025": lab[lab.season == 2025], "2026": lab[lab.season == 2026]}
print(f"train {TR.race_key.nunique()} races / {len(TR)} driver-results | val {VAL.race_key.nunique()} | 2025 {TEST['2025'].race_key.nunique()} | 2026 {TEST['2026'].race_key.nunique()}", flush=True)


# ------------------------------------------------------------------ metrics vs the real finishing positions
def race_table(d, good):
    out = []
    for k, idx in d.groupby("race_key").indices.items():
        pos = d.pos.to_numpy()[idx]; g = good[idx]; ok = ~np.isnan(pos)
        tr = rankdata(pos[ok], method="ordinal"); pr = rankdata(-g[ok], method="ordinal"); n = ok.sum()
        top = lambda r, m: set(np.where(r <= m)[0])
        out.append(dict(race_key=k, exact=np.mean(pr == tr), within1=np.mean(np.abs(pr - tr) <= 1), within3=np.mean(np.abs(pr - tr) <= 3),
                        mae=np.mean(np.abs(pr - tr)), spearman=np.corrcoef(pr, tr)[0, 1], kendall=kendalltau(pr, tr)[0],
                        podium=len(top(pr, 3) & top(tr, 3)) / 3, top10=len(top(pr, 10) & top(tr, 10)) / 10, winner=float(pr[tr == 1][0] == 1)))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ model adapters: .fit(d, cols), .good(d, cols) (higher = better)
def X(d, cols): return np.nan_to_num(d[cols].to_numpy(float))
def grp(d): return d.race_key.to_numpy()


class Reg:
    def __init__(self, est, sign=-1): self.est, self.sign = est, sign
    def fit(self, d, cols): self.est.fit(X(d, cols), d.ypos.to_numpy()); return self
    def good(self, d, cols): return self.sign * self.est.predict(X(d, cols))


class LGBRank:
    def __init__(self, **kw): self.kw = kw
    def fit(self, d, cols):
        n = d.groupby("race_key", sort=False).size().to_numpy()
        rel = np.round((1 - d.ypos.to_numpy()) * 20).astype(int)
        self.m = lgb.LGBMRanker(objective="lambdarank", label_gain=list(range(0, 41)), learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.7, reg_lambda=8.0, verbose=-1, n_jobs=4, random_state=0, **self.kw)
        self.m.fit(X(d, cols), rel, group=n); return self
    def good(self, d, cols): return self.m.predict(X(d, cols))


class XGBRank:
    def __init__(self, n, depth): self.n, self.depth = n, depth
    def fit(self, d, cols):
        qid = pd.factorize(grp(d))[0]
        self.m = xgb.XGBRanker(objective="rank:pairwise", device="cuda", tree_method="hist", n_estimators=self.n, max_depth=self.depth,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_lambda=8.0, random_state=0)
        self.m.fit(X(d, cols), np.round((1 - d.ypos.to_numpy()) * 20).astype(int), qid=qid); return self
    def good(self, d, cols): return self.m.predict(X(d, cols))


class CatRank:
    def __init__(self, it, depth): self.it, self.depth = it, depth
    def fit(self, d, cols):
        pool = Pool(d[cols].to_numpy(float), label=(1 - d.ypos.to_numpy()), group_id=pd.factorize(grp(d))[0])
        self.m = CatBoostRanker(loss_function="YetiRank", iterations=self.it, depth=self.depth, learning_rate=0.05, l2_leaf_reg=5,
                                task_type="GPU", devices="0", verbose=0, random_seed=0).fit(pool); return self
    def good(self, d, cols): return self.m.predict(d[cols].to_numpy(float))


class PL:
    def __init__(self, **kw): self.kw = kw
    def fit(self, d, cols):
        self.m = PlackettLuce(K=22, **self.kw).fit(X(d, cols), d.pos.to_numpy(float), grp(d)); return self
    def good(self, d, cols): return self.m.score(X(d, cols))


cat_reg = lambda it, dp: Reg(CatBoostRegressor(iterations=it, depth=dp, learning_rate=0.05, l2_leaf_reg=5, task_type="GPU", devices="0", verbose=0, random_seed=0))
FAM = {
    "ridge": [lambda a=a: Reg(Ridge(alpha=a)) for a in (1, 10, 100)],
    "lgbm_reg": [lambda n=n, l=l: Reg(lgb.LGBMRegressor(n_estimators=n, num_leaves=l, learning_rate=0.03, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                         colsample_bytree=0.7, reg_lambda=8.0, verbose=-1, n_jobs=4, random_state=0)) for n, l in ((200, 6), (400, 4))],
    "lgbm_rank": [lambda n=n, l=l: LGBRank(n_estimators=n, num_leaves=l, min_child_samples=40) for n, l in ((200, 6), (400, 4))],
    "rf_reg": [lambda l=l: Reg(RandomForestRegressor(n_estimators=300, min_samples_leaf=l, max_features=0.5, n_jobs=-1, random_state=0)) for l in (10, 30)],
    "cat_reg": [lambda it=it, dp=dp: cat_reg(it, dp) for it, dp in ((300, 4), (600, 6))],
    "cat_rank": [lambda it=it, dp=dp: CatRank(it, dp) for it, dp in ((300, 4), (600, 6))],
    "xgb_rank": [lambda n=n, dp=dp: XGBRank(n, dp) for n, dp in ((200, 3), (400, 4))],
    "pl_lin": [lambda l=l: PL(l2=l) for l in (3, 10, 30)],
    "pl_mlp": [lambda w=w: PL(hidden=32, wd=w, epochs=150) for w in (1e-3, 1e-2)],
}

# ------------------------------------------------------------------ validation: pick (feature set, config) per family
chosen, vtable = {}, []
for name, cfgs in FAM.items():
    best = None
    for fs, cols in FS.items():
        for i, mk in enumerate(cfgs):
            t0 = time.time(); m = mk().fit(SUB, cols); rt = race_table(VAL, m.good(VAL, cols)); sp = rt.spearman.mean()
            vtable.append(dict(family=name, fs=fs, cfg=i, spearman=sp, mae=rt.mae.mean(), exact=rt.exact.mean(), secs=time.time() - t0))
            if best is None or sp > best[0]: best = (sp, fs, i)
    chosen[name] = best
    print(f"{name:10s} -> fs {best[1]:9s} cfg #{best[2]} | val Spearman {best[0]:.4f}", flush=True)
pd.DataFrame(vtable).to_csv(REP / "pos_validation.csv", index=False)
base_val = race_table(VAL, -VAL.grid.to_numpy(float)).spearman.mean()
rank = sorted(chosen, key=lambda n: -chosen[n][0])
top3 = rank[:3]
print(f"grid-order baseline val Spearman {base_val:.4f} | ranking {[(n, round(chosen[n][0], 4)) for n in rank]} | ensemble = {top3}", flush=True)

# ------------------------------------------------------------------ fit once on 2016-2024, score 2025 and 2026
fitted = {n: FAM[n][chosen[n][2]]().fit(TR, FS[chosen[n][1]]) for n in FAM}
json.dump({n: dict(fs=chosen[n][1], cfg=chosen[n][2], val_spearman=chosen[n][0]) for n in chosen} | {"ensemble": top3}, open(REP / "pos_settings.json", "w"), indent=2)
rng = np.random.default_rng(0)


def boot(v, nb=3000):
    idx = rng.integers(0, len(v), (nb, len(v))); b = v[idx].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)


for year, d in TEST.items():
    goods = {"grid_order": -d.grid.to_numpy(float)}
    for n in FAM: goods[n] = fitted[n].good(d, FS[chosen[n][1]])
    ranks = lambda g: np.concatenate([rankdata(-g[i], method="ordinal") for i in d.groupby("race_key", sort=False).indices.values()])
    # ensemble = mean of within-race predicted ranks of the 3 families chosen on validation
    order_idx = np.concatenate(list(d.groupby("race_key", sort=False).indices.values()))
    goods["ens_rank"] = np.zeros(len(d))
    avg = np.mean([ranks(goods[n]) for n in top3], axis=0)
    goods["ens_rank"][order_idx] = -avg
    tabs = {n: race_table(d, g) for n, g in goods.items()}
    print(f"\n===== UNSEEN {year}: {d.race_key.nunique()} races, {len(d)} driver-results scored against the real finishing positions")
    out = []
    for n, t in tabs.items():
        row = dict(model=n)
        for m in ["exact", "within1", "within3", "mae", "spearman", "kendall", "podium", "top10", "winner"]: row[m] = t[m].mean()
        for m in ["spearman", "mae"]:
            dlt = (t[m].to_numpy() - tabs["grid_order"][m].to_numpy()) * (1 if m == "spearman" else -1)  # + = better than grid
            lo, hi = boot(dlt); row[f"{m}_vs_grid"], row[f"{m}_lo"], row[f"{m}_hi"] = dlt.mean(), lo, hi
        out.append(row)
    o = pd.DataFrame(out).sort_values("spearman", ascending=False)
    pd.set_option("display.width", 250)
    print(o[["model", "exact", "within1", "within3", "mae", "spearman", "kendall", "podium", "top10", "winner"]].round(3).to_string(index=False))
    print("paired vs grid order (+ = better):")
    print(o[["model", "spearman_vs_grid", "spearman_lo", "spearman_hi", "mae_vs_grid", "mae_lo", "mae_hi"]].round(3).to_string(index=False), flush=True)
    o.to_csv(REP / f"pos_metrics_{year}.csv", index=False)
    pr = d[["race_key", "race_name", "driver", "constructor", "grid", "pos"]].copy()
    pr["pred_rank_ens"] = ranks(goods["ens_rank"]) if False else rankdata(-goods["ens_rank"], method="ordinal")
    pr.to_csv(REP / f"pos_predictions_{year}.csv", index=False)
print("DONE")

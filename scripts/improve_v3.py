"""v3: trees (decision tree, random forest, CatBoost-GPU), Plackett-Luce full-order training, Elo ratings, chaos split.

Protocol (set before running):
  * Hyper-parameters AND each model's confidence temperature are chosen on 2016-21 -> 2022-24 only.
  * Stage 1  (validation year): fit on 2016-2024, score 2025.   2025 was already seen once by the v2 final model, so it is
    now a VALIDATION year: with this many candidates, treat any 2025 'win' as provisional.
  * Stage 2  (fresh test): refit on 2016-2025 with the same settings, score the 2026 races (15 so far; never used before).
  * ens_v3 = mean of the 3 best models by 2022-24 validation log-loss (chosen without looking at 2025).
GPU: PyTorch (Plackett-Luce) and CatBoost run on the GPU. LightGBM, sklearn trees/forests are CPU-only.
"""
import itertools, json, sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from f1pred import backtest as B, features as F
from f1pred.models import BoostedRace, ConditionalLogit, race_softmax, temperature_fit
from f1pred.nn import NeuralRace
from f1pred.rank import ELO, Chaos, PlackettLuce, SkBinary, add_ratings

REP = ROOT / "reports"
base = F.make()
old = json.load(open(REP / "final_hyperparams.json"))
POST = B.POST


def val_ll(score, d):
    p = race_softmax(score, d.race_key.to_numpy()); return float(-np.log(p[d.won.to_numpy() == 1]).mean())


def temp_for(score, d):
    bags = [(score[(d.race_key == k).to_numpy()], int(np.argmax(d[d.race_key == k].won.to_numpy()))) for k in d.race_key.unique()]
    return temperature_fit(bags)


# ---------------- Elo parameters (chosen on validation, logit + ratings) ----------------
best_elo = None
for kd, kc in itertools.product((0.05, 0.1), (0.1, 0.2)):
    d = add_ratings(base, kd, kc); d = d[d.won.notna()]
    sub, va = d[(d.season >= 2016) & (d.season <= 2021)], d[(d.season >= 2022) & (d.season <= 2024)]
    m = ConditionalLogit(l2=10).fit(sub[POST + ELO].to_numpy(float), sub.won.to_numpy(), sub.race_key.to_numpy())
    s = m.score(va[POST + ELO].to_numpy(float)); T = temp_for(s, va); v = val_ll(s / T, va)
    print(f"elo kd={kd} kc={kc}: val {v:.4f}", flush=True)
    if best_elo is None or v < best_elo[0]: best_elo = (v, kd, kc)
print("chosen elo params", best_elo[1:], flush=True)
df = add_ratings(base, best_elo[1], best_elo[2])
lab = df[df.won.notna()]
SUB, VAL = lab[(lab.season >= 2016) & (lab.season <= 2021)], lab[(lab.season >= 2022) & (lab.season <= 2024)]
TR1, T25 = lab[(lab.season >= 2016) & (lab.season <= 2024)], lab[lab.season == 2025]
TR2, T26 = lab[(lab.season >= 2016) & (lab.season <= 2025)], lab[lab.season == 2026]

# ---------------- model families ----------------
def plain(make, cols): return dict(kind="plain", make=make, cols=cols)
FAM = {}
FAM["logit_post"] = [plain(lambda l=l: ConditionalLogit(l2=l), POST) for l in (3, 10, 30)]
FAM["logit_elo"] = [plain(lambda l=l: ConditionalLogit(l2=l), POST + ELO) for l in (3, 10, 30)]
gb = old["chosen"]["gbm_post"]
FAM["gbm_post"] = [plain(lambda: BoostedRace(**gb), POST)]
FAM["gbm_elo"] = [plain(lambda: BoostedRace(**gb), POST + ELO)]
nnk = old["chosen"]["nn_post"]
FAM["nn_post"] = [plain(lambda: NeuralRace(**nnk), POST)]
cat = lambda it, dp: SkBinary(CatBoostClassifier(iterations=it, depth=dp, learning_rate=0.05, l2_leaf_reg=5, task_type="GPU", devices="0", verbose=0, random_seed=0), nan_fill=False)
FAM["cat_post"] = [plain(lambda it=it, dp=dp: cat(it, dp), POST) for it, dp in itertools.product((150, 400), (3, 5))]
FAM["cat_elo"] = [plain(lambda it=it, dp=dp: cat(it, dp), POST + ELO) for it, dp in itertools.product((150, 400), (3, 5))]
FAM["rf_post"] = [plain(lambda l=l, mf=mf: SkBinary(RandomForestClassifier(n_estimators=400, min_samples_leaf=l, max_features=mf, n_jobs=-1, random_state=0)), POST)
                  for l, mf in itertools.product((10, 30), ("sqrt", 0.5))]
FAM["dt_post"] = [plain(lambda dp=dp, l=l: SkBinary(DecisionTreeClassifier(max_depth=dp, min_samples_leaf=l, random_state=0)), POST)
                  for dp, l in itertools.product((2, 3, 4), (30, 80))]
FAM["pl_post"] = [dict(kind="pl", make=lambda K=K, l=l: PlackettLuce(K=K, l2=l), cols=POST) for K, l in itertools.product((1, 2, 3, 6), (3, 10))]
FAM["pl_elo"] = [dict(kind="pl", make=lambda K=K, l=l: PlackettLuce(K=K, l2=l), cols=POST + ELO) for K, l in itertools.product((1, 2, 3, 6), (3, 10))]
FAM["plmlp_post"] = [dict(kind="pl", make=lambda K=K, w=w: PlackettLuce(K=K, hidden=32, wd=w, epochs=150), cols=POST) for K, w in itertools.product((3, 6), (1e-3, 1e-2))]
FAM["chaos_post"] = [dict(kind="chaos", make=lambda l=l: Chaos(POST, l2=l), cols=POST) for l in (3, 10)]
FAM["chaos_elo"] = [dict(kind="chaos", make=lambda l=l: Chaos(POST + ELO, l2=l), cols=POST + ELO) for l in (3, 10)]


def fit_spec(sp, d):
    m = sp["make"]()
    if sp["kind"] == "chaos": return m.fit(d)
    if sp["kind"] == "pl": return m.fit(d[sp["cols"]].to_numpy(float), d.pos.to_numpy(float), d.race_key.to_numpy())
    return m.fit(d[sp["cols"]].to_numpy(float), d.won.to_numpy(), d.race_key.to_numpy())


def score_spec(sp, m, d):
    return m.score(d) if sp["kind"] == "chaos" else m.score(d[sp["cols"]].to_numpy(float))


# ---------------- validation: choose config + temperature per family ----------------
chosen, temps, valscore, ncand = {}, {}, {}, 0
for name, grid in FAM.items():
    best = None
    for i, sp in enumerate(grid):
        t0 = time.time(); m = fit_spec(sp, SUB); s = score_spec(sp, m, VAL); T = temp_for(s, VAL); v = val_ll(s / T, VAL); ncand += 1
        if best is None or v < best[0]: best = (v, i, T, s / T)
    chosen[name], temps[name], valscore[name] = best[1], best[2], best[3]
    print(f"{name:11s} chosen cfg #{best[1]} of {len(grid)} | temperature {best[2]:.2f} | val log-loss (calibrated) {best[0]:.4f}", flush=True)
rank = sorted(valscore, key=lambda n: val_ll(valscore[n], VAL))
top3 = rank[:3]
ens_val = np.mean([valscore[n] for n in top3], axis=0); T_ens = temp_for(ens_val, VAL)
print("val ranking:", [(n, round(val_ll(valscore[n], VAL), 4)) for n in rank[:8]], "\nens_v3 = mean of", top3, f"T={T_ens:.2f}", flush=True)

# old final ensemble (logit + gbm + nn, post), for a like-for-like comparison
OLD_T = old["temps"]["post"]


def stage(train, test, label):
    fitted = {n: fit_spec(FAM[n][chosen[n]], train) for n in FAM}
    old_models = {"logit": fit_spec(plain(lambda: ConditionalLogit(l2=old["chosen"]["logit_post"]["l2"]), POST), train)}
    r = test[["race_key", "season", "round", "race_name", "driver", "driver_id", "constructor", "grid", "won"]].copy()
    gm = ConditionalLogit(l2=1.0).fit(train[B.GRID_ONLY].to_numpy(float), train.won.to_numpy(), train.race_key.to_numpy())
    r["s_grid_logit"] = gm.score(test[B.GRID_ONLY].to_numpy(float))
    for n in FAM:
        r[f"raw_{n}"] = score_spec(FAM[n][chosen[n]], fitted[n], test); r[f"s_{n}"] = r[f"raw_{n}"] / temps[n]
    r["s_ens_old"] = np.mean([r["raw_logit_post"] if False else r["raw_" + n] for n in ("logit_post", "gbm_post", "nn_post")], axis=0) / OLD_T
    r["s_ens_v3"] = np.mean([r[f"s_{n}"] for n in top3], axis=0) / T_ens
    r.to_csv(REP / f"v3_scores_{label}.csv", index=False)
    names = ["grid_logit", "ens_old", "ens_v3"] + list(FAM)
    tab = B.summarize(r, names, n_boot=1500).sort_values("logloss")
    print(f"\n===== {label}: {r.race_key.nunique()} races | candidate configs tried: {ncand} | pole-sitter wins "
          f"{r.groupby('race_key').apply(lambda g: float(g.loc[g.grid.idxmin(), 'won']), include_groups=False).mean():.3f}")
    rows = []
    for n in tab.model:
        d_old = B.paired_diff(r, n, "ens_old", n_boot=2000); d_grid = B.paired_diff(r, n, "grid_logit", n_boot=2000)
        rows.append(dict(model=n, **tab[tab.model == n].iloc[0][["top1", "top3", "logloss", "ll_lo", "ll_hi"]].to_dict(),
                         vs_old=d_old["diff"], vs_old_lo=d_old["lo"], vs_old_hi=d_old["hi"], vs_grid=d_grid["diff"], vs_grid_lo=d_grid["lo"], vs_grid_hi=d_grid["hi"]))
    out = pd.DataFrame(rows); out.to_csv(REP / f"v3_metrics_{label}.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)


stage(TR1, T25, "2025_validation")
stage(TR2, T26, "2026_fresh")
json.dump(dict(elo=best_elo[1:], chosen=chosen, temps=temps, top3=top3, T_ens=T_ens), open(REP / "v3_settings.json", "w"), indent=2, default=float)
print("DONE")

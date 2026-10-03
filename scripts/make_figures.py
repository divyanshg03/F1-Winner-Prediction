"""Publication figures -> reports/figures/*.png  (light surface, validated 3-slot categorical palette)."""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1pred import backtest as B
from f1pred import features as F

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.axisbelow": True,
})
res = pd.read_csv(ROOT / "reports" / "backtest_scores.csv")
N = res.race_key.nunique()


def title(ax, head, sub):
    ax.figure.text(0.06, 0.965, head, fontsize=17, fontweight="bold", ha="left", va="top", color=INK)
    ax.figure.text(0.06, 0.905, sub, fontsize=11.5, ha="left", va="top", color=INK2)


def foot(fig):
    fig.text(0.06, 0.02, f"Walk-forward backtest, {N} races (2018 – Sep 2026), refit before every race. Data: Jolpica/Ergast.",
             fontsize=8.5, color=INK2, ha="left")


# 1 --------------------------------------------------------------------------------
tab = B.summarize(res, ["grid_logit", "ens_pre", "ens_post"])
labels = {"grid_logit": "Starting grid only", "ens_pre": "Before qualifying\n(form model)", "ens_post": "After qualifying\n(full model)"}
fig, ax = plt.subplots(figsize=(9, 5.4))
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.25, right=0.93)
y = np.arange(len(tab))[::-1]
cols = [INK2, AQUA, BLUE]
for yi, (_, r), c in zip(y, tab.iterrows(), cols):
    ax.barh(yi, r.logloss, color=c, height=0.5)
    ax.plot([r.ll_lo, r.ll_hi], [yi, yi], color=INK, lw=1.4)
    ax.text(r.ll_hi + 0.05, yi, f"{r.logloss:.2f}   ({r.top1:.0%} right)", va="center", fontsize=11, color=INK)
ax.axvline(np.log(20), color=ORANGE, lw=1.6, ls=(0, (4, 3)))
ax.text(np.log(20) - 0.04, y[0] + 0.52, "pure guessing = 3.0", color=ORANGE, ha="right", fontsize=10)
ax.set_yticks(y, [labels[m] for m in tab.model], fontsize=11)
ax.set_xlim(0, 3.7)
ax.set_xlabel("Average penalty on the true winner (log-loss, lower = better)")
ax.grid(axis="y", visible=False)
title(ax, "The starting grid is worth a year of form", "Grid alone ties the form-only model; combining everything cuts the error by about a fifth")
ax.text(3.68, y[-1] - 0.42, "whiskers = 95% interval over races", ha="right", fontsize=9, color=INK2)
foot(fig)
fig.savefig(FIG / "01_headline.png", dpi=200)

# 2 rolling edge -----------------------------------------------------------------------
pm = {}
for nm in ["grid_logit", "ens_post"]:
    m = B.race_metrics(res, B.probs(res, nm)).set_index("race_key")
    pm[nm] = m.logloss
d = pd.DataFrame(pm)
dates = res.drop_duplicates("race_key").set_index("race_key")
d["x"] = pd.to_datetime(res.drop_duplicates("race_key").set_index("race_key").season.astype(str) + "-01-01") + pd.to_timedelta(
    res.drop_duplicates("race_key").set_index("race_key")["round"] * 14, unit="D")
roll = d[["grid_logit", "ens_post"]].rolling(25, min_periods=25).mean()
roll["x"] = d["x"]
fig, ax = plt.subplots(figsize=(9, 5.4))
fig.subplots_adjust(top=0.78, bottom=0.14, left=0.09, right=0.84)
ax.plot(roll.x, roll.grid_logit, color=INK2, lw=2)
ax.plot(roll.x, roll.ens_post, color=BLUE, lw=2)
ax.fill_between(roll.x, roll.ens_post, roll.grid_logit, where=roll.grid_logit > roll.ens_post, color=BLUE, alpha=0.10, lw=0)
l = roll.dropna().iloc[-1]
ax.text(roll.x.iloc[-1], l.grid_logit + 0.0, "  Grid only", color=INK2, va="center", fontsize=11)
ax.text(roll.x.iloc[-1], l.ens_post, "  Full model", color=BLUE, va="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Rolling 25-race log-loss (lower = better)")
ax.set_ylim(0.3, 2.1)
title(ax, "The edge was big, then the grid caught up", "Red Bull-era dominance (2022–24) rewarded form; today's close fields leave little beyond the grid")
foot(fig)
fig.savefig(FIG / "02_edge_over_time.png", dpi=200)

# 3 reliability --------------------------------------------------------------------------
p = B.probs(res, "ens_post")
bins = [0, .02, .05, .1, .2, .35, .5, .7, 1.01]
cat = pd.cut(p, bins)
g = pd.DataFrame({"p": p, "y": res.won, "b": cat}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
fig, ax = plt.subplots(figsize=(7.2, 6.6))
fig.subplots_adjust(top=0.80, bottom=0.12, left=0.12, right=0.95)
ax.plot([0, 1], [0, 1], color=GRID, lw=1.6)
ax.plot(g.p, g.y, color=BLUE, lw=2, marker="o", ms=8, mfc=BLUE, mec=SURF, mew=2)
for i, (_, r) in enumerate(g.iterrows()):
    if i < 3:
        continue
    else:
        ax.text(r.p + 0.015, r.y - 0.04, f"n={int(r.n)}", fontsize=8.5, color=INK2)
ax.plot([0.03, 0.09], [0.84, 0.84], color=GRID, lw=2); ax.text(0.10, 0.835, "perfect calibration", color=INK2, fontsize=10)
ax.text(0.03, 0.70, "bins under 10%: n = 2,946 / 248 / 178", fontsize=8.5, color=INK2)
ax.plot([0.03, 0.09], [0.78, 0.78], color=BLUE, lw=2); ax.text(0.10, 0.775, "model, walk-forward", color=INK2, fontsize=10)
ax.set_xlabel("Probability the model gave a driver")
ax.set_ylabel("How often that driver actually won")
ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
title(ax, "When it says 40%, does it win 40%?", "Close. The top bin runs slightly hot (82% called, 75% won, n=57)")
foot(fig)
fig.savefig(FIG / "03_calibration.png", dpi=200)
print(g.round(3))

# 4 feature-group ablation ------------------------------------------------------------------
from f1pred.models import ConditionalLogit
df = F.make()
d = df[df.won.notna()].sort_values(["date", "round"])
tr, te = d[d.season <= 2021], d[d.season >= 2022]
groups = {
    "Grid & qualifying order": ["grid", "grid_rank_sq", "front_row", "qual_pos", "is_pole", "grid_pen"],
    "This weekend's pace gaps": ["qual_gap", "con_qual_best", "con_qual_mean", "con_qual_best_rel", "sprint_rank"],
    "Team form": ["con_perf_s", "con_perf_l", "con_win", "con_podium", "con_qgap", "con_perf_xs", "con_qgap_xs", "con_win_xs",
                  "con_perf_s_rel", "con_win_rel", "con_qgap_rel"],
    "Driver form": ["drv_perf_s", "drv_perf_l", "drv_win", "drv_podium", "drv_qgap", "drv_gain", "drv_perf_s_rel", "drv_win_rel",
                    "drv_qgap_rel", "drv_rookie"],
    "Circuit history": ["drv_circ", "con_circ"],
    "Reliability (DNF risk)": ["drv_dnf", "con_mech"],
}


def _ll(cols):
    m = ConditionalLogit(l2=3.0).fit(tr[cols].to_numpy(float), tr.won.to_numpy(), tr.race_key.to_numpy())
    t = te.assign(p=m.predict(te[cols].to_numpy(float), te.race_key.to_numpy()))
    return -np.log(t[t.won == 1].p).mean()


full = _ll(B.POST)
ab = pd.Series({g: _ll([c for c in B.POST if c not in cols]) - full for g, cols in groups.items()}).sort_values()
print("ablation (log-loss increase when removed):"); print(ab.round(3))
fig, ax = plt.subplots(figsize=(9, 5.2))
fig.subplots_adjust(top=0.78, bottom=0.15, left=0.30, right=0.93)
ax.barh(ab.index, ab.values, color=[BLUE if v > 0 else ORANGE for v in ab.values], height=0.58)
ax.axvline(0, color=INK2, lw=1)
for yi, v in enumerate(ab.values):
    ax.text(v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=10.5)
ax.set_xlim(-0.08, 0.15)
ax.set_xlabel("Log-loss change when the group is removed   (blue = it was helping, orange = it was hurting)")
ax.grid(axis="y", visible=False)
title(ax, "What the model actually leans on", "Grid and driver form carry it; team form is redundant once you have qualifying")
fig.text(0.06, 0.02, "Conditional-logit ablation, trained on 2014–2021, tested on 2022–Sep 2026.", fontsize=8.5, color=INK2)
fig.savefig(FIG / "04_what_matters.png", dpi=200)

# 5 forward prediction ----------------------------------------------------------------------
pf = sorted((ROOT / "predictions").glob("*.json"))
if pf:
    rec = json.loads(pf[-1].read_text(encoding="utf8"))
    top = pd.DataFrame(rec["probabilities"]).head(7)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.27, right=0.93)
    ax.barh(top.driver + "  (P" + top.grid.astype(str) + ")", top.p_win * 100, color=BLUE, height=0.6)
    for yi, v in enumerate(top.p_win * 100):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=11, color=INK)
    ax.set_xlabel("Win probability (%)")
    ax.grid(axis="y", visible=False)
    title(ax, f"Frozen before lights-out: {rec['race']}", f"Generated {rec['generated_utc'][:16]} UTC from qualifying, commit {rec['git_commit']}. Scored publicly after the race.")
    fig.savefig(FIG / "05_next_race.png", dpi=200)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))

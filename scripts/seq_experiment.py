"""Pre-registered sequence-model check (GRU / Transformer) vs the tabular ensemble.

Protocol: same 188 test races as reports/backtest_scores.csv; refit every 8 test races on races
strictly before the block's first race; early-stop on the last 15% of TRAINING races; 3 seeds averaged
in score space. Variants fixed in advance:
  gru_q   : GRU history + QUALI_FEATS
  trf_q   : Transformer history + QUALI_FEATS
  gru_qf  : GRU history + QUALI_FEATS + FORM_FEATS
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from f1pred import backtest as B
from f1pred import features as F
from f1pred import seq

REFIT_EVERY, SEEDS = 8, (0, 1, 2)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = {
    "gru_q": ("gru", F.QUALI_FEATS),
    "trf_q": ("transformer", F.QUALI_FEATS),
    "gru_qf": ("gru", F.QUALI_FEATS + F.FORM_FEATS),
}

t0 = time.time()
df = pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])
bt = pd.read_csv(ROOT / "reports" / "backtest_scores.csv")
test_keys = bt.drop_duplicates("race_key").race_key.tolist()
rows = {name: [] for name in VARIANTS}
for name, (arch, cols) in VARIANTS.items():
    races = seq.build_races(df, cols)
    all_keys = list(races)  # date ordered
    for b in range(0, len(test_keys), REFIT_EVERY):
        block = test_keys[b:b + REFIT_EVERY]
        train = all_keys[: all_keys.index(block[0])]
        sc = np.mean([seq.fit_predict(races, train, block, arch, True, s, dev) for s in SEEDS], axis=0)
        for i, k in enumerate(block):
            for j, drv in enumerate(races[k]["drivers"]):
                rows[name].append((k, drv, float(sc[i, j])))
        print(f"{name} block {b // REFIT_EVERY + 1}/{-(-len(test_keys) // REFIT_EVERY)}  {time.time() - t0:.0f}s", flush=True)

res = bt.copy()
for name, r in rows.items():
    s = pd.DataFrame(r, columns=["race_key", "driver_id", f"s_{name}"])
    res = res.merge(s, on=["race_key", "driver_id"], how="left")
assert res[[f"s_{n}" for n in VARIANTS]].notna().all().all()
runtime = time.time() - t0

names = ["grid_logit", "ens_post"] + list(VARIANTS)
hold = res[res.season >= 2024]
out = []
for label, g in (("all", res), ("2024-26", hold)):
    t = B.summarize(g, names, n_boot=2000)
    t["slice"] = label
    out.append(t)
summ = pd.concat(out)


def paired(g, a, b, n_boot=5000, seed=0):
    ma = B.race_metrics(g, B.probs(g, a)).logloss.to_numpy()
    mb = B.race_metrics(g, B.probs(g, b)).logloss.to_numpy()
    d = mb - ma  # positive: a better than b
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


pr = []
for label, g in (("all", res), ("2024-26", hold)):
    for v in VARIANTS:
        for ref in ("ens_post", "grid_logit"):
            m, lo, hi = paired(g, v, ref)
            pr.append(dict(slice=label, variant=v, vs=ref, nats_better=m, lo=lo, hi=hi))
pr = pd.DataFrame(pr)
summ.to_csv(ROOT / "reports" / "seq_experiment.csv", index=False)
pr.to_csv(ROOT / "reports" / "seq_experiment_paired.csv", index=False)
res.to_csv(ROOT / "reports" / "seq_scores.csv", index=False)

f = lambda x: f"{x:.3f}"
md = ["# Sequence-model check (pre-registered)", "",
      f"188 walk-forward test races (same as the tabular backtest), refit every {REFIT_EVERY} test races on strictly earlier "
      f"races, early stopping on the last 15% of *training* races, {len(SEEDS)} seeds averaged in score space, "
      f"device={dev}, runtime {runtime / 60:.1f} min.", "",
      "Variants: `gru_q` = GRU over last 10 races + qualifying features; `trf_q` = Transformer (CLS token) + qualifying features; "
      "`gru_qf` = GRU + qualifying + all tabular form features.", ""]
for label in ("all", "2024-26"):
    s = summ[summ["slice"] == label]
    md += [f"## {label} ({int(s.races.iloc[0])} races)", "", "| model | top1 [95% CI] | top3 | log-loss [95% CI] | Brier |", "|---|---|---|---|---|"]
    for r in s.itertuples():
        md.append(f"| {r.model} | {r.top1:.1%} [{r.top1_lo:.1%}, {r.top1_hi:.1%}] | {r.top3:.1%} | {f(r.logloss)} [{f(r.ll_lo)}, {f(r.ll_hi)}] | {f(r.brier)} |")
    md += ["", "Paired difference in log-loss (nats/race; positive = sequence variant better):", "",
           "| variant | vs | diff [95% CI] |", "|---|---|---|"]
    for r in pr[pr["slice"] == label].itertuples():
        md.append(f"| {r.variant} | {r.vs} | {r.nats_better:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] |")
    md.append("")
open(ROOT / "reports" / "seq_experiment.md", "w", encoding="utf8").write("\n".join(md))
print("\n".join(md))

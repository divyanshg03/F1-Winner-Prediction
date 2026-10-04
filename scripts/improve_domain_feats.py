"""Domain-feature groups (circuit geometry/history, driver/team, tyre proxy): walk-forward test vs current model."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from f1pred import backtest as B, features as F
from f1pred.extra_features import GROUPS, add_groups
from f1pred.models import BoostedRace, ConditionalLogit

df = add_groups(F.make())
assert df.won.notna().sum() > 0
allx = sum(GROUPS.values(), [])
for k, cols in {"A": GROUPS["A"], "B": GROUPS["B"], "C": GROUPS["C"], "ABC": allx}.items():
    B.MODELS[f"logit_{k}"] = (lambda: ConditionalLogit(l2=3.0), B.POST + cols)
B.MODELS["gbm_ABC"] = (lambda: BoostedRace(), B.POST + allx)
res = B.walk_forward(df, 2018, verbose=False)
res["s_ens_ABC"] = 0.5 * res.s_logit_ABC + 0.5 * res.s_gbm_ABC
res.to_csv(ROOT / "reports" / "domain_feature_scores.csv", index=False)
names = ["ens_post", "logit_post", "logit_A", "logit_B", "logit_C", "logit_ABC", "gbm_ABC", "ens_ABC"]
for lab, m in [("DEV 2018-23", res.season <= 2023), ("HOLDOUT 2024-26", res.season >= 2024)]:
    t = B.summarize(res[m], names, n_boot=200); print(lab); print(t[["model", "races", "top1", "top3", "logloss"]].round(3).to_string(index=False))
    for n in ["logit_A", "logit_B", "logit_C", "ens_ABC"]:
        ref = "logit_post" if n.startswith("logit") else "ens_post"
        d = B.paired_diff(res, n, ref, m); print(f"  {n} vs {ref}: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]")

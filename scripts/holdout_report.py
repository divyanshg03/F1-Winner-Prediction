import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from f1pred import backtest as B
res = pd.read_csv(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports" / "backtest_scores.csv")
names = ["grid_logit", "logit_post", "gbm_post", "ens_post", "ens_pre"]
for label, g in (("DEV 2018-2023", res[res.season <= 2023]), ("HOLDOUT 2024-2026", res[res.season >= 2024])):
    t = B.summarize(g, names, n_boot=300)
    print(label, g.race_key.nunique(), "races"); print(t[["model","top1","top3","logloss","brier"]].round(3).to_string(index=False))

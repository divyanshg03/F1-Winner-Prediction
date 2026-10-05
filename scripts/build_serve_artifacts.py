"""Train the demo's live model on every completed race (2016-) and save small, torch-free artifacts to deploy/serve/.

Run after new races are added:  python scripts/build_serve_artifacts.py
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1pred import backtest as B
from f1pred import features as F
from f1pred.models import BoostedRace, ConditionalLogit

hp = json.load(open(ROOT / "reports" / "final_hyperparams.json"))["chosen"]
df = F.make()
tr = df[(df.won.notna()) & (df.season >= 2016)]
cols = B.POST
X, y, g = tr[cols].to_numpy(float), tr.won.to_numpy(), tr.race_key.to_numpy()
logit = ConditionalLogit(**hp["logit_post"]).fit(X, y, g)
gbm = BoostedRace(**hp["gbm_post"]).fit(X, y, g)
out = ROOT / "deploy" / "serve"
out.mkdir(parents=True, exist_ok=True)
(out / "logit.json").write_text(json.dumps(dict(cols=cols, mu=logit.std.mu.tolist(), sd=logit.std.sd.tolist(), w=logit.w.tolist(),
                                                 trained_on_races=int(tr.race_key.nunique()), l2=hp["logit_post"]["l2"])), encoding="utf8")
gbm.m.booster_.save_model(str(out / "gbm.txt"))
print(f"saved artifacts trained on {tr.race_key.nunique()} races to {out}")

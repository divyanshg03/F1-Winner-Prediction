"""Polymarket pre-race win probabilities vs my model, on the same races (paired bootstrap).

Market snapshot = last hourly price at least 10 min before lights-out (so, like the model, it is a post-qualifying call).
Only races with >=18 priced drivers are scored (partial 2024 markets would need renormalising over an unknown field).
"""
import sys, unicodedata, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, requests
from f1pred import backtest as B

od = pd.read_csv(ROOT / "data/raw/odds/polymarket_f1.csv")
bt = pd.read_csv(ROOT / "reports/backtest_scores.csv")
norm = lambda s: "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)).lower().replace("-", " ").strip()

# race start times (UTC) from Jolpica
sched = []
for y in (2024, 2025):
    for r in requests.get(f"https://api.jolpi.ca/ergast/f1/{y}.json?limit=40", timeout=30).json()["MRData"]["RaceTable"]["Races"]:
        sched.append(dict(season=y, round=int(r["round"]), date=pd.Timestamp(r["date"]), start=pd.Timestamp(f'{r["date"]}T{r.get("time","13:00:00Z")}')))
sched = pd.DataFrame(sched)
sched["start"] = sched.start.dt.tz_localize(None)

rows, unmatched = [], set()
for (ev, end), g in od.groupby(["event", "end_date"]):
    e = pd.Timestamp(end); s = sched.iloc[(sched.date - e).abs().argsort()[:1]].iloc[0]
    if abs((s.date - e).days) > 2 or s.season not in (2024, 2025): continue
    t0 = (s.start - pd.Timedelta(minutes=10)).timestamp()
    key = f"{s.season}-{s['round']:02d}"
    b = bt[bt.race_key == key]
    if b.empty: continue
    names = {norm(n.split()[-1]) if n.split() else "": n for n in b.driver}
    snap = {}
    for out, h in g.groupby("outcome"):
        h = h[h.ts_utc <= t0]
        if h.empty: continue
        sur = norm(str(out).split()[-1])
        did = b[b.driver.map(lambda d: norm(d.split()[-1])) == sur]
        if len(did) != 1: unmatched.add((key, out)); continue
        snap[did.driver_id.iloc[0]] = float(h.sort_values("ts_utc").price.iloc[-1])
    if len(snap) < 18: continue
    tot = sum(snap.values())
    for _, r in b.iterrows():
        rows.append(dict(race_key=key, driver_id=r.driver_id, p_mkt=snap.get(r.driver_id, 0.0) / tot, overround=tot))
mk = pd.DataFrame(rows)
print("races scored:", mk.race_key.nunique(), "| unmatched outcomes:", sorted(unmatched)[:8], "| mean overround", round(mk.overround.mean(), 3))
res = bt.merge(mk, on=["race_key", "driver_id"])
res["s_mkt"] = np.log(res.p_mkt.clip(lower=0.002))
res["s_blend"] = 0.5 * res.s_ens_post + 0.5 * res.s_mkt
t = B.summarize(res, ["grid_logit", "ens_post", "mkt", "blend"], n_boot=1000) if False else None
for nm, col in [("mkt", "s_mkt"), ("blend", "s_blend")]:
    pass
names = ["grid_logit", "ens_post", "mkt", "blend"]
tab = B.summarize(res.assign(s_mkt=res.s_mkt, s_blend=res.s_blend), names, n_boot=1000)
print(tab[["model", "races", "top1", "top3", "logloss", "ll_lo", "ll_hi"]].round(3).to_string(index=False))
for a, b_ in [("mkt", "ens_post"), ("mkt", "grid_logit"), ("ens_post", "grid_logit"), ("blend", "ens_post"), ("blend", "mkt")]:
    d = B.paired_diff(res, a, b_); print(f"  {a} vs {b_}: {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] nats/race (+ = first better), n={d['races']}")
res.to_csv(ROOT / "reports/odds_vs_model.csv", index=False)

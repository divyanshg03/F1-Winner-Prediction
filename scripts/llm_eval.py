"""Llama 3.1 8B (local, via Ollama) as a zero-/few-shot race-winner predictor on the unseen 2025 and 2026 races.

The model sees only information known before lights-out (grid, recent form, team form, qualifying gap) and must return
win probabilities as JSON. No fine-tuning. Llama 3.1's training data ends ~Dec 2023, so these races are unseen to it too.
Few-shot variant adds 3 worked 2024 examples (race table + actual winner) from the training period.
"""
import json, sys, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, requests
from f1pred import backtest as B, features as F

df = F.make().sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
lab = df[df.won.notna()]
COLS = ["race_key", "race_name", "driver", "driver_id", "constructor", "grid", "drv_perf_s", "con_perf_s", "drv_win", "qual_gap", "won"]

def table(d):
    d = d.sort_values("grid")
    lines = ["grid | driver | team | driver_form(0-1,higher=better) | team_form(0-1) | recent_win_rate | quali_gap_to_pole_%"]
    for r in d.itertuples():
        lines.append(f"P{int(r.grid)} | {r.driver} | {r.constructor} | {r.drv_perf_s:.2f} | {r.con_perf_s:.2f} | {r.drv_win:.2f} | {100 * r.qual_gap:.2f}")
    return "\n".join(lines)

def prompt(d, shots):
    head = ("You are a Formula 1 analyst. Given the pre-race table, estimate each driver's probability of winning the race. "
            "Probabilities must sum to 1. The pole-sitter wins about half of all races. Answer ONLY with JSON mapping driver name to probability.\n\n")
    ex = ""
    for s in shots:
        w = s[s.won == 1].driver.iloc[0]
        ex += f"EXAMPLE ({s.race_name.iloc[0]}):\n{table(s)}\nActual winner: {w}\n\n"
    return head + ex + f"RACE TO PREDICT ({d.race_name.iloc[0]}):\n{table(d)}\n\nJSON:"

def ask(p):
    for _ in range(3):
        try:
            r = requests.post("http://localhost:11434/api/generate", json=dict(model="llama3.1:8b", prompt=p, stream=False, format="json",
                              options=dict(temperature=0, num_ctx=8192, seed=1)), timeout=600).json()["response"]
            return json.loads(r)
        except Exception as e:  # noqa: BLE001
            print("retry", repr(e)[:80], flush=True)
    return {}

shots_races = [lab[lab.race_key == k][COLS] for k in ["2024-05", "2024-09", "2024-14"]]
res_all = []
for label, season in (("2025", 2025), ("2026", 2026)):
    base = pd.read_csv(ROOT / "reports" / f"final_scores_{label}.csv")
    rows = []
    for k in base.race_key.unique():
        d = lab[lab.race_key == k][COLS].reset_index(drop=True)
        for variant, shots in (("zero", []), ("few", shots_races)):
            t0 = time.time(); out = ask(prompt(d, shots))
            norm = {str(a).lower().strip(): float(v) for a, v in out.items() if isinstance(v, (int, float))}
            p = np.array([norm.get(x.lower(), 0.0) for x in d.driver]); hit = (p > 0).sum()
            p = np.clip(p, 0.003, None); p = p / p.sum()
            for dr, pp in zip(d.driver_id, p): rows.append(dict(race_key=k, driver_id=dr, variant=variant, p=pp, named=hit))
            print(label, k, variant, "named", hit, "/", len(d), f"{time.time() - t0:.0f}s", flush=True)
    L = pd.DataFrame(rows)
    for v in ("zero", "few"):
        m = L[L.variant == v][["race_key", "driver_id", "p"]].rename(columns={"p": f"p_{v}"})
        base = base.merge(m, on=["race_key", "driver_id"], how="left")
        base[f"s_llm_{v}"] = np.log(base[f"p_{v}"].fillna(1e-3))
    base["s_llm_blend"] = 0.5 * base.s_ens_post + 0.5 * base.s_llm_few
    base.to_csv(ROOT / "reports" / f"llm_scores_{label}.csv", index=False)
    tab = B.summarize(base, ["grid_logit", "ens_post", "llm_zero", "llm_few", "llm_blend"], n_boot=1000)
    print(f"\n===== Llama 3.1 8B on unseen {label} ({base.race_key.nunique()} races)")
    print(tab[["model", "top1", "top3", "logloss", "ll_lo", "ll_hi"]].round(3).to_string(index=False))
    for a, b_ in [("llm_few", "ens_post"), ("llm_few", "grid_logit"), ("llm_zero", "grid_logit"), ("llm_blend", "ens_post")]:
        dd = B.paired_diff(base, a, b_); print(f"  {a} vs {b_}: {dd['diff']:+.3f} [{dd['lo']:+.3f},{dd['hi']:+.3f}]")

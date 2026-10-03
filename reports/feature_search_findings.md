# Feature search: can more signal raise accuracy?

Same protocol throughout: walk-forward over the same 188 races, features built from earlier races only, judged on dev (2018-23) first, then the 2024-26 holdout once. Metric: per-race winner log-loss difference vs the current model (nats/race, + = better, 95% bootstrap CI).

| Feature group (from the domain checklist) | Dev 2018-23 | Holdout 2024-26 |
|---|---|---|
| Race-pace history from lap data (driver/team, short+long windows) | -0.005 [-0.031, +0.016] | -0.010 [-0.036, +0.015] |
| Circuit geometry + learned circuit history (length, altitude, twistiness, straights, corners; overtaking difficulty, pole conversion, SC rate, DNF rate, degradation) | -0.001 [-0.010, +0.009] | +0.001 [-0.014, +0.015] |
| Driver/team: experience, championship standing, teammate comparison | -0.002 [-0.020, +0.015] | -0.022 [-0.051, +0.005] |
| Tyre-management proxy (circuit degradation x team race-vs-quali pace gap) | -0.002 [-0.006, +0.001] | -0.007 [-0.013, -0.003] |
| Practice pace (FP2, else FP1: best lap, long-run) | **-0.064 [-0.123, -0.019]** | -0.002 [-0.008, +0.004] |
| Race-day weather (rain, temps, humidity, wind) | -0.001 [-0.013, +0.011] | +0.011 [-0.008, +0.032] |
| Everything blended (ensemble) | -0.012 / -0.028 | +0.003 / +0.003 |

**Verdict: none of it helps.** Practice pace significantly hurts on dev (noisy long-run proxies from short sessions). Nothing improves the holdout.

Caveats: weather is actual race-day weather (optimistic vs a forecast), so its null is if anything generous. Circuit geometry is approximate (GeoJSON layouts; FastF1 corner counts are manually annotated upstream). Tyre information is a proxy built from race laps, not compound-level degradation per car. DRS-zone counts were not available from any source found. Data: bacinger/f1-circuits (GeoJSON), Jolpica, FastF1.

Why: the starting grid already encodes this weekend's pace, and with ~190 scored races the model cannot learn small extra effects reliably.

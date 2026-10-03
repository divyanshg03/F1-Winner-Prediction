# Race simulator (audit recommendation 5): findings

Lap-level Monte Carlo (`src/f1pred/sim.py`) fed by a race-pace model (`pace.py`) and FastF1 lap data for 188 races (2018-2026; the 2021 Belgian GP is excluded, it had 3 laps behind the safety car).
Physics knobs were calibrated on 2019-21 only, then frozen; 2022+ (107 races) scored walk-forward. **This is the pre-registered v1 result.**

| | log-loss | 95% CI |
|---|---|---|
| grid-only logit | 1.485 | 1.232 - 1.790 |
| tabular ensemble (`ens_post`) | 1.108 | 0.913 - 1.318 |
| **simulator alone** | 1.480 | 1.391 - 1.574 |
| 50/50 blend (sim + ensemble) | 1.180 | 1.051 - 1.320 |

Paired (nats/race, + = first is better): sim vs grid-only +0.006 [-0.198, +0.238] (tie); sim vs ensemble **-0.371 [-0.518, -0.204]**; blend vs ensemble -0.071 [-0.142, +0.010] (no gain).

## What worked
- **Pace model:** race-pace RMSE 0.00584 vs 0.00903 (qualifying only) vs 0.00976 (field mean). Race pace is ~35% more predictable than qualifying alone suggests.
- Safety-car frequency matches reality (sim 59% of races vs 55% actual).

## What did not
- **The simulator does not beat the tabular ensemble and adds nothing when blended.**
- **It is not realistic about the starting grid:** simulated pole-sitters win 28% of races vs 57% actual (2022+).

## Why (diagnostics on the 2019-21 calibration races only)
Add-back ablation, pole-win rate (actual = 47.5%):

| ingredients switched on | pole wins |
|---|---|
| nothing random, no stops | 99.7% |
| + independent pit stops | 48.5% |
| + pit stops with shared strategy (0.9) | 65.2% |
| + safety cars (alone) | 87.4% |
| + retirements (alone) | 84.6% |
| + start noise (alone) | 45.2% |
| + pace shocks (alone) | 99.0% |
| everything (v1) | 34.2% |

Pit-stop randomness is the dominant culprit: independent strategies and stop timing reshuffle the order far more than real teams allow (they cover each other, pit for clean air, undercut deliberately). Calibration compensated by making passing almost impossible (`pass_thr` = 0.035), which is physically implausible. **The missing piece is reactive strategy modelling, not more calibration.** Not done.

## Disclosures
- The calibration grid was widened twice (after a smoke test on one 2024 race, then because the optimum sat on the grid edge); both decisions used calibration-set evidence, before any 2022+ score was printed.
- A `strategy_share` option (cars copy the race's main strategy) was added after the diagnostics and is **not** used in the v1 numbers above. A post-hoc v2 would be exploratory, not pre-registered.
- Reproduce: `python scripts/run_sim_backtest.py`, `scripts/sim_diagnose.py`, `scripts/sim_addback.py` (lap data via `python -m f1pred.ingest_laps 2018 2026`).

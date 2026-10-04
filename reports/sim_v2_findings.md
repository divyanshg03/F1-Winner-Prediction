# Race simulator v2: reactive strategy (exploratory)

**Status: exploratory.** The 2022+ races were already scored once by sim v1, so v2 is not a clean pre-registered test. Settings were tuned on 2019-21 only (16 combinations), frozen, then 2022+ scored once.

v2 adds: shared pit-window timing across the field, car-to-car strategy copying, undercut (a car stuck within 2 s pits early) and cover (the car ahead pits when the car behind just pitted). All new options default to off, so v1 is reproducible.

| 2022+ (107 races) | log-loss [95% CI] | top-1 |
|---|---|---|
| grid-only | 1.485 [1.233, 1.780] | 57% |
| tabular ensemble | 1.108 [0.923, 1.317] | 62% |
| sim v1 | 1.480 [1.389, 1.575] | 54% |
| **sim v2** | **1.425 [1.314, 1.535]** | 53% |

Paired (nats/race, + = first better): v2 vs v1 **+0.054 [+0.034, +0.074]**; v2 vs grid-only +0.060 [-0.131, +0.278] (tie); v2 vs tabular **-0.317 [-0.458, -0.155]**. On 2024+ only: v2 vs v1 +0.065 [+0.040, +0.089], v2 vs tabular -0.163 [-0.350, +0.035].

## What happened
- **Improvement over v1 is real but small** (+0.054, CI excludes zero).
- **It still loses to the tabular model**, and simulated pole-sitters win **31%** of races vs 57% in reality (v1: 28%). The realism gap barely moved.
- **The reactive behaviours were NOT what helped.** Calibration on 2019-21 chose `p_under = 0` and `p_cover = 0`: turning undercut and cover on made the calibration log-loss worse in every combination. The gain came from shared pit timing and strategy copying.
- So my diagnosis ("teams react to each other") was only partly right. The pit-stop machinery is still the likeliest source of the lost grid advantage (see `sim_findings.md` add-back table), but simple reactive rules don't fix it.

## Caveats
Grid was small (16 combos, 4 knobs); undercut/cover rules are crude (fixed 2 s / 3 s thresholds, no tyre-compound choice); the lap-1 start and pace-shock models were not touched. Reproduce: `python scripts/run_sim_v2.py` (needs lap data: `python -m f1pred.ingest_laps 2018 2026`).

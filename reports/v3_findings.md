# v3: more models, full-finishing-order training, Elo ratings, chaos split

**Protocol.** Hyper-parameters and each model's confidence temperature chosen on 2016-21 -> 2022-24 only. Stage 1: fit on 2016-2024, score 2025 (now a *validation* year: it had already been scored once). Stage 2: refit on 2016-2025, score the 2026 races (15, never used before) = the fresh test. `ens_v3` = mean of the 3 best models by 2022-24 validation (rf_post, gbm_post, logit_elo), picked before looking at 2025. 17 candidate model families were compared. GPU: PyTorch (Plackett-Luce) and CatBoost; LightGBM and sklearn trees/forests are CPU-only.

Log-loss (lower = better); "vs old" = paired nats/race against the previous final ensemble (+ = better), 95% CI.

| model | 2025 (24, validation) | vs old | 2026 (15, fresh) | vs old |
|---|---|---|---|---|
| previous ensemble (ens_old) | 0.952 | - | 1.249 | - |
| **ens_v3** | 0.907 | +0.045 [-0.01, +0.10] | 1.219 | +0.030 [-0.06, +0.12] |
| random forest | 0.941 | +0.011 [-0.08, +0.10] | 1.080 | +0.169 [+0.01, +0.35] |
| LightGBM | 0.843 | **+0.110 [+0.05, +0.17]** | 1.374 | -0.126 [-0.37, +0.10] |
| CatBoost (GPU) | 0.942 | +0.011 | 1.295 | -0.046 |
| decision tree | 1.023 | -0.071 | 1.593 | -0.344 |
| Plackett-Luce (full-order idea) | 1.023 | -0.070 | 1.271 | -0.022 |
| chaos split (finish x win-if-finish) | 1.023 | -0.070 | 1.271 | -0.022 |
| Elo-rating features (logit) | 1.001 | vs logit_post +0.006 | 1.375 | vs logit_post -0.194 |
| grid-only | 1.111 | -0.159 | 1.410 | -0.161 |

## Reading
- **Nothing improves accuracy in a way that holds up.** LightGBM looked significantly better on 2025 (+0.110) but *reversed* on the fresh 2026 races (-0.126). That is the classic selection effect: pick the best of 17 on one year and it regresses. Treat 2025 "wins" as provisional.
- **ens_v3 is small and consistent in sign** (+0.045 on 2025, +0.030 on 2026) but neither interval excludes zero. It is the most defensible upgrade, not a proven one.
- **Random forest** was best on 2022-24 validation, tied on 2025 and best on 2026 (+0.169, CI just above zero). One to watch; one of 17 comparisons, so it may be luck.
- **Full finishing order (Plackett-Luce):** validation chose K=1, i.e. winner-only. Training on more places did not help; it matched the winner-only logit. Lower places are dominated by retirements and lapped cars, which are not the signal for who wins.
- **Chaos split** (P(finish) x P(win | finish)) and **Elo ratings** added nothing measurable.
- **Decision tree** is poor (as expected: a single tree is too coarse); **CatBoost on the GPU** ties the other trees.
- Not done: forecast weather and cleaner practice signals; odds as an input (history too thin).

Reproduce: `python scripts/improve_v3.py`.

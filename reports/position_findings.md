# Predicting the whole finishing order (P1 to last) from 2016-2024, tested on 2025 and 2026

Training: every driver of every race 2016-2024 (190 races, 3,841 driver-results), full finishing order as the target, all feature parameters (37 / 67 / 78-feature sets: base, +Elo/race-pace/circuit/teammate/tyre-proxy, +practice/weather). Model family, feature set and ensemble membership chosen on 2016-21 -> 2022-24 only (by Spearman). ONE fit on 2016-2024, scored on 2025 (24 races) and 2026 (15 races) against the real finishing positions. Baseline: "finish where you start" (grid order). GPU: CatBoost (regression + YetiRank), XGBoost ranker, PyTorch Plackett-Luce; LightGBM and sklearn trees/forests are CPU-only.

| 2025 (24 races) | exact P | within +/-1 | MAE (places) | Spearman | podium set | winner |
|---|---|---|---|---|---|---|
| grid order (baseline) | 18.4% | 38.4% | 3.34 | 0.651 | 75% | 67% |
| XGBoost ranker (GPU) | 16.9% | 38.4% | 3.27 | 0.667 | 72% | 54% |
| LightGBM ranker | 17.3% | 36.6% | 3.30 | 0.663 | 71% | 58% |
| ensemble (3 best on validation) | 16.1% | 36.8% | 3.31 | 0.661 | 75% | 46% |
| random forest | 14.6% | 35.5% | 3.34 | 0.660 | 74% | 38% |

| 2026 (15 races) | exact P | within +/-1 | MAE | Spearman | podium set | winner |
|---|---|---|---|---|---|---|
| grid order (baseline) | 14.8% | 41.2% | 3.50 | 0.639 | 60% | 67% |
| ridge | 13.9% | 36.4% | 3.47 | 0.662 | 58% | 60% |
| ensemble | 13.0% | 35.2% | 3.53 | 0.654 | 58% | 40% |

Paired vs grid order (Spearman, + = better): best models +0.010 to +0.016 on 2025 (CIs about [-0.02, +0.04]) and +0.012 to +0.022 on 2026 (CIs about [-0.02, +0.05]). **No model is significantly better than grid order on either year.** Validation (2022-24, where choices were made) showed +0.05 to +0.06 over grid (0.678 vs 0.618), which shrank to about +0.01 to +0.02 on genuinely unseen years: the usual selection effect.

## Reading
- A finishing-order model can learn something, but not much beyond the starting grid. Exact-position hit rate is about 14-17% (grid order: 15-18%); average error about 3.3 places either way.
- Race-day chaos dominates the result: a retirement puts a driver at the back regardless of pace, and the models cannot know who will retire. Classified positions for retired drivers were kept as the target as the task asks.
- The winner pick is worse for these order models than the dedicated winner model (e.g. ensemble 46% vs grid 67% in 2025): optimising the whole order trades away the top of it.
- Not significant at 24 and 15 races: intervals are about +/-0.03 Spearman and +/-0.2 places.
- Weather in the feature set is the actual race-day weather (optimistic); the best models mostly chose the feature sets without it.

Reproduce: `python scripts/train_positions.py`.

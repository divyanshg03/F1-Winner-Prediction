# Prediction-market odds vs the model

Source: Polymarket race-winner markets (public API, no account; `scripts/fetch_market_odds.py`). Market snapshot = last hourly price at least 10 minutes before lights-out, normalised over the field (mean overround 1.06). Like the model, it is a post-qualifying call. Scored only on races with >=18 priced drivers: **17 races, all 2025** (19 2025 markets exist; partial 2024 markets list only 5-11 drivers, so they were excluded; no 2026 history was available).

| | log-loss | top-1 | top-3 |
|---|---|---|---|
| grid-only logit | 1.075 | 71% | 94% |
| my model (ens_post) | 0.934 | 65% | 100% |
| **market** | 0.961 | 53% | 100% |
| 50/50 blend (model + market) | 0.910 | 65% | 100% |

Paired (nats/race, + = first better): market vs model **-0.027 [-0.192, +0.137]**; market vs grid-only +0.114 [-0.151, +0.360]; blend vs model +0.024 [-0.059, +0.107]; blend vs market +0.050 [-0.032, +0.136].

**Reading:** the model is statistically indistinguishable from the betting market on these 17 races. Both are better than the grid on log-loss, but not significantly at n=17. The blend is nominally best but its gain is well inside the noise. This is a tie, not a win: with 17 races the intervals are about +/-0.15 nats, so only a large gap could have shown up.

Caveats: 2025 only (one season, one regulation set); the blend weight was fixed at 0.5 (not fitted); market prices are thin-liquidity exchange prices, not a sharp bookmaker line; the market's top-1 is lower mostly because a probability-weighted market sometimes backs a non-pole driver. Not tested: adding market price as a model feature (too few races to train on).

# Final model: trained on 2016-2024, tested on the unseen 2025 season

Protocol: hyper-parameters chosen on 2016-21 -> 2022-24 validation only; ensemble temperature fitted on those validation predictions; then ONE fit on all of 2016-2024 and ONE scoring pass over 2025 (24 races, no per-race refitting). 2026 (15 races, new regulations) is a second unseen set. 2025 data was never used to train, tune or choose anything.

## 2025 (24 races, unseen)
Pole-sitter won 67% of 2025 races. Uniform guessing = 2.99 log-loss.

| model | log-loss [95% CI] | top-1 | top-3 |
|---|---|---|---|
| grid-only logit | 1.111 [0.866, 1.380] | 67% | 96% |
| **ens_post (after qualifying)** | **0.946 [0.722, 1.171]** | 62% | 100% |
| gbm_post | 0.921 [0.741, 1.115] | 62% | 100% |
| logit_post | 1.081 [0.867, 1.312] | 58% | 96% |
| nn_post (own neural net) | 1.107 [0.916, 1.308] | 62% | 100% |
| ens_pre (before qualifying) | 1.624 [1.400, 1.896] | 21% | 92% |

Paired (nats/race, + = first better): ens_post vs grid-only **+0.166 [+0.042, +0.283]** (significant); ens_post vs ens_pre +0.678 [+0.361, +1.005]; gbm_post vs logit_post +0.160 [+0.047, +0.282]; nn_post vs logit_post -0.026 [-0.107, +0.060] (tie); ens_pre vs grid-only **-0.512 [-0.922, -0.118]** (significantly worse).

## 2026 (15 races, unseen, new regulations)
ens_post 1.270 [0.920, 1.651] vs grid-only 1.416 (paired +0.146 [-0.325, +0.908], not significant); ens_pre 2.235, much worse. Wide intervals (15 races).

## Honest reading
- **After qualifying, the model gives well-calibrated probabilities that beat the grid on log-loss in an unseen year** (+0.166, CI excludes zero). Its top-1 accuracy (62%) is *not* better than "pole wins" (67% in 2025): the gain is in probability quality, not in naming the winner.
- **Before qualifying, the model is poor on unseen years** (21% top-1 in 2025, 7% in 2026): form alone mis-ranks a field whose pecking order changed (2025: McLaren; 2026: new regulations). Use the post-qualifying model for race-week predictions.
- LightGBM was the best single model; the hand-written neural net tied the logit. Ensembling was close to the best single model, not better.
- Calibration (2025, ens_post): predicted 26% -> won 28%; 41% -> 38%; 59% -> 77% (n=13); bins under 10% (n=421) saw 0 wins vs ~12 expected-ish at the low end (slightly over-confident on long shots). Bins above 35% are small (n=24).
- 24 races means intervals of roughly +/-0.2 nats; modest differences are not detectable.

Reproduce: `python scripts/train_final.py` (model saved to `models/final_2016_2024.joblib`).

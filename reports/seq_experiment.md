# Sequence-model check (pre-registered)

188 walk-forward test races (same as the tabular backtest), refit every 8 test races on strictly earlier races, early stopping on the last 15% of *training* races, 3 seeds averaged in score space, device=cuda, runtime 7.0 min.

Variants: `gru_q` = GRU over last 10 races + qualifying features; `trf_q` = Transformer (CLS token) + qualifying features; `gru_qf` = GRU + qualifying + all tabular form features.

## all (188 races)

| model | top1 [95% CI] | top3 | log-loss [95% CI] | Brier |
|---|---|---|---|---|
| grid_logit | 53.2% [46.3%, 60.1%] | 87.2% | 1.522 [1.335, 1.733] | 0.652 |
| ens_post | 59.0% [52.1%, 66.0%] | 89.9% | 1.221 [1.058, 1.400] | 0.570 |
| gru_q | 55.9% [48.9%, 62.8%] | 89.9% | 1.256 [1.113, 1.415] | 0.592 |
| trf_q | 54.3% [46.8%, 61.2%] | 86.2% | 1.324 [1.168, 1.501] | 0.610 |
| gru_qf | 56.4% [48.9%, 63.8%] | 89.4% | 1.261 [1.120, 1.421] | 0.587 |

Paired difference in log-loss (nats/race; positive = sequence variant better):

| variant | vs | diff [95% CI] |
|---|---|---|
| gru_q | ens_post | -0.036 [-0.111, +0.042] |
| gru_q | grid_logit | +0.265 [+0.150, +0.396] |
| trf_q | ens_post | -0.104 [-0.174, -0.033] |
| trf_q | grid_logit | +0.197 [+0.047, +0.358] |
| gru_qf | ens_post | -0.041 [-0.101, +0.021] |
| gru_qf | grid_logit | +0.260 [+0.110, +0.423] |

## 2024-26 (63 races)

| model | top1 [95% CI] | top3 | log-loss [95% CI] | Brier |
|---|---|---|---|---|
| grid_logit | 58.7% [46.0%, 71.4%] | 90.5% | 1.380 [1.077, 1.786] | 0.597 |
| ens_post | 54.0% [42.8%, 65.1%] | 88.9% | 1.321 [1.064, 1.604] | 0.624 |
| gru_q | 60.3% [47.6%, 71.5%] | 87.3% | 1.257 [1.006, 1.508] | 0.588 |
| trf_q | 55.6% [42.9%, 68.3%] | 84.1% | 1.409 [1.149, 1.675] | 0.634 |
| gru_qf | 58.7% [46.0%, 71.4%] | 87.3% | 1.329 [1.097, 1.580] | 0.617 |

Paired difference in log-loss (nats/race; positive = sequence variant better):

| variant | vs | diff [95% CI] |
|---|---|---|
| gru_q | ens_post | +0.063 [-0.059, +0.193] |
| gru_q | grid_logit | +0.122 [-0.057, +0.352] |
| trf_q | ens_post | -0.088 [-0.217, +0.043] |
| trf_q | grid_logit | -0.029 [-0.266, +0.262] |
| gru_qf | ens_post | -0.008 [-0.079, +0.069] |
| gru_qf | grid_logit | +0.051 [-0.183, +0.331] |

## Verdict

- **Sequence models do not beat the tabular ensemble.** Over all 188 races `gru_q` and `gru_qf` tie `ens_post` (-0.036 and -0.041 nats/race, CIs include 0); the Transformer is significantly *worse* (-0.104 [-0.174, -0.033]).
- All three clearly beat grid-only over the full sample (+0.20 to +0.27 nats/race, CIs exclude 0), so history sequences carry real signal, but no more than the hand-built tabular features.
- On the 2024-26 holdout (63 races) every comparison has a CI that includes 0; `gru_q` is nominally best (+0.063 vs `ens_post`) but that is within noise and is one of six comparisons, so it is not evidence of an edge.
- Hypothesis "the data is thin for sequence models": supported for the Transformer, a tie for the GRU.

## Deviations and caveats

- No change to the pre-registered protocol after seeing results.
- Refit every 8 test races (as registered), so models in a block are up to 7 races staler than the tabular backtest (refit every race). This slightly favours the tabular ensemble.
- Qualifying features (`F.QUALI_FEATS`) are fed to all variants as static inputs, as registered; `gru_qf` adds `F.FORM_FEATS`.
- Hyperparameters (hidden 48, dropout 0.2, AdamW lr 2e-3 / wd 1e-2, 60-epoch cap, patience 8) were fixed once and never tuned.

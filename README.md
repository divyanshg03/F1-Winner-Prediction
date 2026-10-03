# F1 Winner Predictor

**Calibrated, backtested probabilities for who wins a Formula 1 Grand Prix, and an honest account of how much anyone can know.**

![headline](reports/figures/01_headline.png)

## The short version

I tested whether a model can beat the obvious bet ("pole-sitter wins") over **188 real races (2018 to Sep 2026)**, refitting before every race so nothing ever sees the future.

| | Winner log-loss ↓ | Top-1 | Top-3 |
|---|---|---|---|
| Pure guessing (20 drivers) | 3.00 | 5% | 15% |
| Pole-sitter / starting grid only | 1.52 | 53% | 87% |
| **Before qualifying** (form only) | 1.53 | 45% | 77% |
| **After qualifying** (full model) | **1.22** | **59%** | **90%** |

What survives scrutiny:

- **The probabilities are good.** Paired against the grid-only model, the full model improves log-loss by **0.30 nats/race (95% CI 0.14 to 0.47)**. Calibration is close to the diagonal ([figure](reports/figures/03_calibration.png)).
- **The accuracy edge over "pole wins" is *not* statistically significant** (59% vs 53%, intervals overlap). Don't read it as "the model picks winners better than the grid does".
- **The edge is era-dependent.** It was large while one team dominated (+0.42 nats/race, 2018–23) and **unproven in 2024–26** (+0.06, CI −0.20 to +0.36, 63 races). When the field is close, the grid already tells you almost everything ([figure](reports/figures/02_edge_over_time.png)).
- **Before qualifying, form alone is no better than knowing the grid** and collapses under regulation resets (2026: 3 of 15 right).

![edge](reports/figures/02_edge_over_time.png)

## Forward predictions

`scripts/predict_next.py` trains on every completed race, scores the next one, and freezes a timestamped JSON (with git commit) into [`predictions/`](predictions/) **before the race**. Those files are the audit trail: the repo records the call, then the result.

## What was wrong with v1 (and why this exists)

The first version (2025, kept in [`legacy/`](legacy/)) reported **95.7% accuracy**. That number was meaningless: predicting *"nobody wins"* scores 95.0%, because only 1 in 20 driver-rows is a winner. The audit that led to v2 found:

- `driver_dnf_rate` was **always zero**: it searched for the string "DNF", which never appears in the data (statuses are `Retired`, `Collision`, `Engine`...).
- The Transformer trained on **NaN loss** for every epoch and reported accuracy from garbage weights.
- Random train/test splits over **overlapping 5-race windows** (leakage), early stopping on the test set, and prediction-time grid positions filled with a driver's *historical average*.
- Only 183 races, 36 of them in the test set, with no baseline comparison.

## What v2 does differently

| | v1 | v2 |
|---|---|---|
| Framing | 20 independent yes/no problems | one softmax over the field per race |
| Evaluation | random split, accuracy | walk-forward refit before every race; log-loss, Brier, top-k, paired bootstrap CIs |
| Baselines | none | uniform, pole, grid-only logit |
| Data | 183 races ending mid-2025 | 2014–2026, results + qualifying + sprints (267 races) |
| Leakage guard | none | 4 tests, including *shuffling a race's own result and checking its features don't move* |
| Models | XGBoost, MLP, (broken) Transformer | conditional logit + LightGBM, ensembled; temperature calibration tested |
| Reproducible | notebooks | `./run_all.sh` |

Deliberate non-goal: sequence models (RNN/Transformer) are **not** included. With ~270 races the data is thin for them, but that is a hypothesis I have *not* tested here (v1's Transformer trained on NaN loss, so it is no evidence either way).

## What the model uses

![what matters](reports/figures/04_what_matters.png)

Ablation (train 2014–21, test 2022–26): grid and driver form carry the model. Team form is redundant once qualifying is known, and removing it slightly *improves* the logit. I reported this rather than retuning on it.

## Honest limitations

- **No odds benchmark.** Bookmaker odds are the real gold standard and aren't in this data. Beating the grid is a much lower bar.
- **Small samples.** 188 scored races; the 2024–26 holdout is 63 races and 2026 is 15. Intervals are wide on purpose.
- **Calibration at the top end runs slightly hot** (82% predicted, 75% observed, n=57), so treat 70%+ calls with care.
- **Missing signals:** long-run practice pace, tyre strategy, weather, safety-car risk, post-qualifying penalties for future races (qualifying order is used as the grid until it is official).
- **Selection discipline:** I added one family of features (this weekend's team pace, faster constructor form) after seeing early results, developing on 2018–23 only. It changed nothing (log-loss 1.170 vs 1.169 on dev; 1.321 vs 1.318 on holdout). Both are reported.
- **2026 is a regulation reset.** Features that lean on car history are discounted at season starts (`RESET_SEASONS`), but 15 races is too few to judge.

## Run it

```bash
pip install -r requirements.txt
./run_all.sh            # ingest -> tests -> backtest -> figures -> next-race prediction
```

Or step by step:

```bash
cd src && python -m f1pred.ingest 2014 2026   # cached Jolpica pulls (~15 min first time)
python -m pytest tests -q
python scripts/run_backtest.py                # writes reports/backtest_scores.csv, metrics_*.csv
python scripts/holdout_report.py              # dev (2018-23) vs holdout (2024-26)
python scripts/make_figures.py
python scripts/predict_next.py --refresh      # after qualifying, before the race
```

## Layout

```
src/f1pred/    ingest.py  features.py  models.py  backtest.py
scripts/       run_backtest.py  holdout_report.py  make_figures.py  predict_next.py
tests/         test_leakage.py
predictions/   frozen pre-race predictions (the audit trail)
reports/       backtest scores, metrics tables, figures
articles/      Substack essay + LinkedIn post
legacy/        v1 notebooks and data (kept on purpose)
```

## Data and licence

Race data from the [Jolpica](https://github.com/jolpica/jolpica-f1) Ergast-compatible API. Code is MIT-licensed. For education and research; not betting advice.

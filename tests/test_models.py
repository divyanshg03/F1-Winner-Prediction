import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1pred import backtest as B
from f1pred.models import ConditionalLogit, race_softmax, temperature_fit


def test_race_softmax_sums_to_one_per_race():
    s = np.random.default_rng(0).normal(size=12)
    r = np.repeat([0, 1, 2], 4)
    p = race_softmax(s, r)
    for k in range(3):
        assert abs(p[r == k].sum() - 1) < 1e-9


def test_conditional_logit_recovers_a_planted_signal():
    rng = np.random.default_rng(1)
    n_races, d = 300, 10
    X = rng.normal(size=(n_races * d, 2))
    rid = np.repeat(np.arange(n_races), d)
    true = 2.0 * X[:, 0]  # only feature 0 matters
    y = np.zeros(len(X))
    for r in range(n_races):
        m = rid == r
        pr = np.exp(true[m]) / np.exp(true[m]).sum()
        y[np.where(m)[0][rng.choice(d, p=pr)]] = 1
    m = ConditionalLogit(l2=0.1).fit(X, y, rid)
    assert m.w[0] > 1.0 and abs(m.w[1]) < 0.4


def test_uniform_probabilities_score_log_n():
    res = pd.DataFrame({"race_key": ["a"] * 20, "won": [1.0] + [0.0] * 19})
    m = B.race_metrics(res, np.full(20, 0.05))
    assert abs(m.logloss.iloc[0] - np.log(20)) < 1e-9


def test_temperature_identity_when_scores_are_already_calibrated():
    rng = np.random.default_rng(2)
    data = []
    for _ in range(400):
        s = rng.normal(size=8)
        p = np.exp(s) / np.exp(s).sum()
        data.append((s, int(rng.choice(8, p=p))))
    assert 0.8 < temperature_fit(data) < 1.25

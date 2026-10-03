"""The backtest is only credible if features for race N cannot see race N or later."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from f1pred import features as F
from f1pred.ingest import PROC

pytestmark = pytest.mark.skipif(not (PROC / "results.csv").exists(), reason="run f1pred.ingest first")


@pytest.fixture(scope="module")
def raw():
    return F.load_raw()


def test_future_results_do_not_change_past_features(raw):
    res, qua, spr = raw
    full = F.build(res, qua, spr)
    cutoff = full.drop_duplicates("race_key").race_key.iloc[-30]
    s, r = map(int, cutoff.split("-"))
    # destroy everything from the cutoff race onward (results of that race and later)
    keep = (res.season < s) | ((res.season == s) & (res["round"] < r))
    trunc_res = res[keep]
    trunc = F.build(trunc_res, qua, spr)
    a = full[full.race_key.isin(trunc.race_key.unique())].sort_values(["race_key", "driver_id"]).reset_index(drop=True)
    b = trunc.sort_values(["race_key", "driver_id"]).reset_index(drop=True)
    cols = F.FORM_FEATS + F.QUALI_FEATS
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_corrupting_a_races_own_result_does_not_change_its_features(raw):
    res, qua, spr = raw
    base = F.build(res, qua, spr)
    key = base.race_key.iloc[-200:].drop_duplicates().iloc[5]
    s, r = map(int, key.split("-"))
    bad = res.copy()
    m = (bad.season == s) & (bad["round"] == r)
    bad.loc[m, "position"] = bad.loc[m, "position"].sample(frac=1, random_state=1).values  # shuffle finishing order
    other = F.build(bad, qua, spr)
    cols = F.FORM_FEATS + F.QUALI_FEATS
    a = base[base.race_key == key].sort_values("driver_id")[cols].reset_index(drop=True)
    b = other[other.race_key == key].sort_values("driver_id")[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_label_is_not_a_feature():
    assert not {"won", "pos", "dnf", "mech", "status"} & set(F.FORM_FEATS + F.QUALI_FEATS)


def test_exactly_one_winner_per_race(raw):
    df = F.build(*raw)
    w = df.groupby("race_key").won.sum()
    assert (w == 1).all(), w[w != 1]

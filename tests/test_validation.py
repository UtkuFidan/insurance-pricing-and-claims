import numpy as np
import pandas as pd
import pytest

from ins_ml.calibrate import Recalibrator
from ins_ml.cv import fold_ids, summarise
from ins_ml.metrics import gamma_deviance, loss_ratio_table


def test_fold_ids_cover_all_rows_and_keep_groups_together():
    groups = np.repeat(np.arange(300), 3)  # three policy-years per holder
    f = fold_ids(len(groups), 5, seed=1, groups=groups)
    assert set(f) == set(range(5))
    assert all(len(set(f[groups == g])) == 1 for g in range(300))
    plain = fold_ids(1000, 5, seed=1)
    assert np.bincount(plain).min() == 200  # KFold sizes are equal when n divides evenly


def test_summarise_mean_and_sample_sd():
    rows = [{"fold": i, "model": "a", "x": v} for i, v in enumerate([1.0, 2.0, 3.0])]
    out = summarise(rows)
    assert out.loc[0, "x_mean"] == pytest.approx(2.0)
    assert out.loc[0, "x_sd"] == pytest.approx(1.0)  # ddof=1


def test_balance_factor_restores_total_and_isotonic_is_monotone():
    rng = np.random.default_rng(0)
    n = 5000
    expo = rng.uniform(0.2, 1, n)
    true = rng.gamma(2, 50, n)
    actual = rng.poisson(true * expo * 0.5) * 1.0  # model over-predicts 2x
    pred = true
    bal = Recalibrator("balance").fit(pred, actual, expo)
    assert (bal.transform(pred) * expo).sum() == pytest.approx(actual.sum(), rel=1e-9)
    assert bal.factor == pytest.approx(0.5, rel=0.05)
    iso = Recalibrator("isotonic").fit(pred, actual, expo)
    grid = np.sort(pred)
    assert np.all(np.diff(iso.transform(grid)) >= -1e-12)
    assert (iso.transform(pred) * expo).sum() == pytest.approx(actual.sum(), rel=1e-6)


def test_gamma_deviance_hand_value_and_zero_at_truth():
    y, mu, w = np.array([2.0, 4.0]), np.array([1.0, 4.0]), np.array([1.0, 3.0])
    # 2 * [1*(-ln 2 + 1)] / 4
    assert gamma_deviance(y, mu, w) == pytest.approx(2 * (1 - np.log(2)) / 4)
    assert gamma_deviance(y, y, w) == pytest.approx(0.0)


def test_loss_ratio_table_hand_computed():
    loss = np.array([10.0, 10.0, 30.0, 50.0])
    prem = np.array([25.0, 25.0, 25.0, 25.0])
    rank = np.array([1.0, 2.0, 3.0, 4.0])
    t = loss_ratio_table(loss, prem, rank, np.ones(4), n=2)
    assert t["loss_ratio"].tolist() == pytest.approx([0.4, 1.6])
    assert t["index"].tolist() == pytest.approx([0.4 / 1.0, 1.6 / 1.0])  # portfolio LR is 1.0
    assert t["exposure"].sum() == 4


def test_recalibrator_rejects_unknown_kind():
    with pytest.raises(ValueError):
        Recalibrator("nope").fit([1.0], [1.0], [1.0])


def test_summarise_keeps_group_order():
    rows = [{"fold": f, "model": m, "x": 1.0} for f in range(2) for m in ("b", "a")]
    assert summarise(rows)["model"].tolist() == ["b", "a"]
    assert isinstance(summarise(rows), pd.DataFrame)


def test_isotonic_never_predicts_zero_even_when_most_policies_have_no_loss():
    rng = np.random.default_rng(2)
    n = 4000
    pred = rng.gamma(2, 20, n)
    actual = (rng.random(n) < 0.03) * rng.gamma(2, 500, n)
    out = Recalibrator("isotonic").fit(pred, actual, np.ones(n)).transform(pred)
    assert out.min() > 0.0
    assert out.min() >= 0.04 * actual.sum() / n

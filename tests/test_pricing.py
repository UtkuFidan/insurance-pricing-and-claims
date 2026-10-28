import numpy as np
import pytest

from ins_ml import glm
from ins_ml.features import Design
from ins_ml.metrics import calibration_table, double_lift_table, gini, poisson_deviance


def test_intercept_only_poisson_equals_claims_per_exposure_year(tiny):
    d = Design(factors=[]).fit(tiny)
    fit = glm.fit_frequency(tiny, d)
    expected = tiny.ClaimNb.sum() / tiny.Exposure.sum()  # 4 / 3.5
    assert np.exp(fit.params[0]) == pytest.approx(expected, rel=1e-6)


def test_offset_scales_expected_claims_with_exposure(tiny):
    d = Design(factors=["VehGas"]).fit(tiny)
    fit = glm.fit_frequency(tiny, d)
    rate = fit.predict_rate(tiny)
    half, full = tiny.assign(Exposure=0.5), tiny.assign(Exposure=1.0)
    # the model predicts a rate per exposure year, so it must not depend on the Exposure column
    assert fit.predict_rate(half) == pytest.approx(fit.predict_rate(full))
    # fitted expected claims reproduce observed total (Poisson score equation with intercept)
    assert (rate * tiny.Exposure).sum() == pytest.approx(tiny.ClaimNb.sum(), rel=1e-6)


def test_exposure_not_confused_with_claims_in_rates():
    """Same claim rate, different exposure: a correct offset gives identical fitted rates."""
    import pandas as pd

    base = {
        "VehAge": 1,
        "VehPower": 6,
        "BonusMalus": 50,
        "Density": 100,
        "VehBrand": "B1",
        "VehGas": "D",
        "Region": "R1",
        "Area": "A",
        "LogDensity": np.log(100.0),
    }
    rows = [
        dict(ClaimNb=k, Exposure=e, DrivAge=a, **base)
        for a, k, e in [(20, 1, 1.0), (20, 2, 2.0), (50, 1, 2.0), (50, 1, 2.0)]
    ]
    df = pd.DataFrame(rows).assign(Exposure=lambda x: x.Exposure.clip(upper=1.0))
    # ClaimNb/Exposure: young = 3 / 2 exposure-years (after clip) -> rates are ratio of sums
    d = Design(factors=["DrivAgeBand"]).fit(df)
    fit = glm.fit_frequency(df, d)
    r = fit.predict_rate(df)
    assert r[0] == pytest.approx(3 / 2, rel=1e-5)
    assert r[2] == pytest.approx(2 / 2, rel=1e-5)


def test_pure_premium_is_frequency_times_severity():
    pp = glm.pure_premium(np.array([0.1, 0.2]), np.array([1000.0, 500.0]), np.array([1.0, 0.5]))
    assert pp == pytest.approx([100.0, 50.0])


def test_poisson_deviance_hand_value():
    y, mu = np.array([0, 1, 2]), np.array([0.5, 1.0, 1.5])
    # terms: 2*(0-(0-0.5)) = 1 ; 0 ; 2*(2*ln(2/1.5) - 0.5)
    expected = (1.0 + 0.0 + 2 * (2 * np.log(2 / 1.5) - 0.5)) / 3
    assert poisson_deviance(y, mu) == pytest.approx(expected)


def test_gini_hand_computed_and_properties():
    loss, expo = np.array([0.0, 0, 0, 4]), np.ones(4)
    assert gini(loss, np.array([1, 2, 3, 4.0]), expo) == pytest.approx(0.75)
    assert gini(loss, np.array([4, 3, 2, 1.0]), expo) < 0  # reversed ranking is worse than random
    assert gini(loss, np.ones(4), expo) == pytest.approx(0.0)  # no discrimination


def test_gini_is_exposure_weighted():
    loss = np.array([0.0, 4.0])
    # second policy carries 3x the exposure: Lorenz (0,0),(0.25,0),(1,1) -> area 0.375
    assert gini(loss, np.array([1.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(1 - 0.75)


def test_calibration_and_double_lift_shapes(synth):
    df, _ = synth
    pred = np.full(len(df), df.ClaimNb.sum() / df.Exposure.sum())
    cal = calibration_table(df.ClaimNb.to_numpy(), pred, df.Exposure.to_numpy(), 10)
    assert len(cal) == 1 or len(cal) == 10
    assert cal.exposure.sum() == pytest.approx(df.Exposure.sum())
    dl = double_lift_table(df.ClaimNb.to_numpy(), pred, pred * 1.1, df.Exposure.to_numpy(), 5)
    assert len(dl) <= 5 and dl.exposure.sum() == pytest.approx(df.Exposure.sum())


def test_glm_recovers_synthetic_direction(synth):
    df, _ = synth
    fit = glm.fit_frequency(df)
    rel = glm.relativities(fit, df, ["DrivAgeBand", "BonusMalusBand"])
    r = rel.set_index(["factor", "level"]).relativity
    assert r[("DrivAgeBand", "18-21")] > r[("DrivAgeBand", "41-50")]
    assert r[("BonusMalusBand", "126+")] > 1.0


def test_gamma_and_tweedie_fit_run(synth):
    df, _ = synth
    s = glm.fit_severity(df)
    assert np.all(s.predict_rate(df) > 0) and s.scale > 0
    t = glm.fit_tweedie(df)
    assert (t.predict_rate(df) * df.Exposure).sum() == pytest.approx(df.Loss.sum(), rel=0.15)

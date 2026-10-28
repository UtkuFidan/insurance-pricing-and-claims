import numpy as np
import pandas as pd

from ins_ml import glm
from ins_ml.fraud import screen_claims


def _claims(amounts, pred=1000.0, exposure=1.0, nb=1):
    n = len(amounts)
    return pd.DataFrame(
        {
            "IDpol": range(n),
            "ClaimAmount": amounts,
            "pred_sev": pred,
            "Exposure": exposure,
            "ClaimNb": nb,
        }
    )


def test_extreme_claim_is_flagged_with_reason_codes():
    scr = screen_claims(_claims([900.0, 1100.0, 80_000.0]), dispersion=2.0)
    assert list(scr.flag) == [False, False, True]
    assert "SEV_TAIL" in scr.reason_codes.iloc[2] and "SEV_RATIO" in scr.reason_codes.iloc[2]
    assert scr.reason_codes.iloc[0] == ""


def test_short_exposure_and_multiclaim_need_two_reasons():
    d = _claims([2500.0], exposure=0.05, nb=3)  # 2.5x pred: short-exposure + multi-claim only
    scr = screen_claims(d, dispersion=2.0)
    assert set(scr.reason_codes.iloc[0].split(";")) == {"SHORT_EXPOSURE_LARGE", "MULTI_CLAIM"}
    assert bool(scr.flag.iloc[0])
    single = screen_claims(_claims([2500.0], exposure=0.05), dispersion=2.0)
    assert not bool(single.flag.iloc[0])


def test_injected_outliers_dominate_flags_on_synthetic(synth):
    df, sev = synth
    fit = glm.fit_severity(df)
    c = sev.merge(df, on="IDpol")
    c["pred_sev"] = fit.predict_rate(c)
    scr = screen_claims(c[["IDpol", "ClaimAmount", "pred_sev", "Exposure", "ClaimNb"]], fit.scale)
    assert 0 < scr.flag.sum() < 0.05 * len(scr)
    assert scr[scr.flag].ClaimAmount.median() > 5 * scr.ClaimAmount.median()
    assert np.isfinite(scr.tail_prob).all()

import numpy as np
import pandas as pd
import pytest

from ins_ml import gbm, spain
from ins_ml.data import build_policy_table, synthetic_portfolio


@pytest.fixture(scope="module")
def policies():
    freq, sev = synthetic_portfolio(5000, seed=11)
    return build_policy_table(freq, sev)


def test_poisson_gbm_predicts_a_rate_not_a_count(policies):
    f = gbm.fit_gbm(policies, "poisson", gbm.FRENCH, {"max_depth": 3}, rounds=40)
    r1 = f.predict_rate(policies.assign(Exposure=1.0))
    r2 = f.predict_rate(policies.assign(Exposure=0.25))
    assert np.allclose(r1, r2)  # exposure enters only as the offset, so the rate is unchanged
    assert (r1 * policies["Exposure"]).sum() == pytest.approx(policies["ClaimNb"].sum(), rel=0.15)


def test_native_categoricals_and_unseen_level(policies):
    f = gbm.fit_gbm(policies, "tweedie", gbm.FRENCH.with_exposure(), rounds=30)
    new = policies.head(20).assign(VehBrand="NEVER_SEEN")
    assert np.all(np.isfinite(f.predict_rate(new)))
    assert f.contributions(policies.head(10)).shape == (10, len(gbm.FRENCH.with_exposure().names))


def test_gamma_gbm_uses_only_claim_rows(policies):
    f = gbm.fit_gbm(policies, "gamma", gbm.FRENCH, rounds=30)
    assert np.all(f.predict_rate(policies.head(50)) > 0)


def test_tune_returns_hyperparameters_and_trial_table(policies):
    hp, trials = gbm.tune(policies, "poisson", gbm.FRENCH, n_trials=2, n_splits=2, rounds=30)
    assert {"max_depth", "eta", "min_child_weight"} <= set(hp)
    assert len(trials) == 2


def test_spain_prepare_maps_schema_and_drops_zero_exposure():
    raw = spain.synthetic_raw(2000)
    raw.loc[:9, "liability_exposure"] = 0.0
    d = spain.prepare(raw)
    assert len(d) == 1990
    assert {"ClaimNb", "Exposure", "Premium", "Loss", "licence_years", "brand_grp"} <= set(
        d.columns
    )
    assert "policy_status" not in d.columns  # leaks exposure information
    assert (d["Loss"] <= spain.LOSS_CAP).all()
    assert (d["licence_years"] == raw.loc[10:, "vehicle_age"].to_numpy()).all()


def test_download_rejects_a_file_with_the_wrong_hash(tmp_path):
    (tmp_path / "motor_insurance_portfolio.csv").write_text("not the dataset")
    with pytest.raises(RuntimeError, match="SHA-256"):
        spain.download(tmp_path)
    assert not (tmp_path / "motor_insurance_portfolio.csv").exists()


def test_spain_pipeline_runs_offline_and_keeps_holders_out_of_two_folds(tmp_path):
    res = spain.run_spain(tmp_path, None, synthetic=True, folds=2, trials=1, rounds=30)
    cmp = res["comparison"]
    assert cmp["market"].tolist() == ["Spain 2022-24 (liability)"]
    lr = res["spain_loss_ratio_by_predicted_decile"]
    assert len(lr) == 10 and lr["exposure"].sum() == pytest.approx(
        res["meta"]["exposure_years"] * 0.8, rel=0.2
    )
    assert isinstance(res["spain_frequency_cv"], pd.DataFrame)

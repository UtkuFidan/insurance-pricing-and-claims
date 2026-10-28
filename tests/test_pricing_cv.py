import json

import pandas as pd
import pytest

from ins_ml.data import build_policy_table, synthetic_portfolio
from ins_ml.pricing import run_pricing


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    freq, sev = synthetic_portfolio(9000, seed=5)
    out = tmp_path_factory.mktemp("res")
    res = run_pricing(
        build_policy_table(freq, sev), out, tmp_path_factory.mktemp("fig"),
        folds=3, trials=1, rounds=40,
    )  # fmt: skip
    return res, out


def test_cv_tables_report_mean_and_sd_per_model(result):
    res, _ = result
    t = res["pure_premium_cv"]
    assert {"tweedie_deviance_mean", "tweedie_deviance_sd", "gini_loss_mean"} <= set(t.columns)
    assert len(t) == 5 and t["tweedie_deviance_sd"].notna().all()
    assert res["meta"]["n_tuning"] + res["meta"]["n_cv"] == res["meta"]["n_policies"]


def test_balance_correction_restores_the_pooled_total(result):
    res, _ = result
    y = res["y"]
    for model, pred in res["oof"]["pp"]["balance"].items():
        total = (pred * y["exposure"]).sum()
        assert total == pytest.approx(y["loss"].sum(), rel=0.05), model


def test_loss_ratio_deciles_and_artifacts(result):
    res, out = result
    lr = res["loss_ratio_by_challenger_decile"]
    assert len(lr) == 10
    # the incumbent premium is loaded to 70% of its balance-corrected pure premium
    overall = lr["loss"].sum() / lr["premium"].sum()
    assert overall == pytest.approx(res["meta"]["target_loss_ratio"], rel=0.1)
    for f in ("tuned_params.json", "optuna_trials.csv", "pricing_meta.json", "frequency_cv.csv"):
        assert (out / f).exists()
    hp = json.loads((out / "tuned_params.json").read_text())
    assert set(hp) == {"freq_claims", "freq_amount", "severity", "pure_premium"}
    assert isinstance(pd.read_csv(out / "frequency_cv.csv"), pd.DataFrame)

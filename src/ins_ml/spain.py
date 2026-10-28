"""Spanish motor portfolio 2022-2024 as a second market for the same frequency pipeline.

Source: Espinosa, Lledo and Atance (2026), Data in Brief, doi:10.1016/j.dib.2026.112883; data on
Mendeley Data, doi:10.17632/sw4jmdb2sm.1, CC BY 4.0. One row per policy-year with exposure, premium,
claim counts and incurred cost by coverage. The third-party liability coverage is the closest
counterpart to freMTPL2 (French MTPL), so ``liability_*`` columns are used.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import gbm, glm, plots
from .cv import fold_ids, summarise
from .features import BANDS, Design
from .metrics import (
    calibration_table,
    gini,
    loss_ratio_table,
    poisson_deviance,
    tweedie_deviance,
)

FILE_URL = (
    "https://data.mendeley.com/public-files/datasets/sw4jmdb2sm/files/"
    "3dc64675-b6bd-4e8e-bfba-b64e585c6599/file_downloaded"
)
FILE_SHA256 = "6a47d19d5278a049ea0aeaf39c955cc26068639bdc58cb4523b201e740f0faf4"
LOSS_CAP = 100_000.0  # cap on incurred per policy-year (claim-level amounts are not available)

SPAIN_BANDS = {
    "DrivAgeBand": ("driver_age", *BANDS["DrivAgeBand"][1:]),
    "LicenceBand": (
        "licence_years",
        [-1, 5, 10, 20, 30, 40, 200],
        ["0-5", "6-10", "11-20", "21-30", "31-40", "41+"],
    ),
    "PowerBand": (
        "power_to_weight",
        [-1, 9, 11, 13, 15, 18, 1000],
        ["<=9", "9-11", "11-13", "13-15", "15-18", ">18"],
    ),
    "ValueBand": (
        "vehicle_value",
        [0, 10000, 15000, 20000, 25000, 35000, 50000, 1e12],
        ["<=10k", "10-15k", "15-20k", "20-25k", "25-35k", "35-50k", ">50k"],
    ),
    "SeatsBand": ("seats", [0, 2, 4, 5, 6, 100], ["<=2", "3-4", "5", "6", "7+"]),
}
SPAIN_FACTORS = [
    *SPAIN_BANDS,
    "bonus_score",
    "fuel_type",
    "brand_grp",
    "municipality_type",
    "circulation_area",
]
SPAIN_SPEC = gbm.FeatureSpec(
    numeric=("driver_age", "licence_years", "power_to_weight", "log_value", "seats"),
    categorical=("bonus_score", "fuel_type", "brand_grp", "municipality_type", "circulation_area"),
)
SHARED = {"DrivAgeBand"}  # rating factors with the same meaning in both markets


def download(data_dir: str | Path) -> Path:
    """Fetch the CSV (95 MB) into ``data_dir`` and verify its SHA-256."""
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    f = p / "motor_insurance_portfolio.csv"
    if not f.exists():
        req = urllib.request.Request(FILE_URL, headers={"User-Agent": "ins-ml/0.2 (research code)"})
        with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 (fixed https URL)
            f.write_bytes(r.read())
    if hashlib.sha256(f.read_bytes()).hexdigest() != FILE_SHA256:
        f.unlink()
        raise RuntimeError("Mendeley file failed its SHA-256 check; delete the cache and retry")
    return f


def prepare(raw: pd.DataFrame, min_brand_share: float = 0.01) -> pd.DataFrame:
    """Map the raw columns onto the common schema used by the pricing code.

    Policy-years with zero exposure are dropped; the 2 missing licence-years and 513 missing
    vehicle values are set to the median (no target information is used).
    ``policy_status`` is not used as a feature: a cancelled flag is only known after the fact and
    carries exposure information.
    ``vehicle_age`` in the file equals ``driver_age - age_driving_licence`` in 91% of rows and rises
    by one each year, so it behaves as years since the licence was issued and is named
    ``licence_years`` here. The file has no true vehicle age.
    """
    d = raw[raw["liability_exposure"] > 0].copy()
    out = pd.DataFrame(
        {
            "insured_id": d["insured_id"].to_numpy(),
            "year": d["year"].to_numpy(),
            "ClaimNb": d["liability_claims"].astype(int).clip(upper=4).to_numpy(),
            "Exposure": d["liability_exposure"].clip(upper=1.0).to_numpy(),
            "Premium": d["liability_premium"].to_numpy(),
            "LossRaw": d["liability_incurred"].to_numpy(),
            "driver_age": d["driver_age"].clip(upper=90).to_numpy(),
            "licence_years": d["vehicle_age"].fillna(d["vehicle_age"].median()).to_numpy(),
            "power_to_weight": d["power_to_weight_ratio"].to_numpy(),
            "vehicle_value": d["vehicle_value"].fillna(d["vehicle_value"].median()).to_numpy(),
            "seats": d["seats"].to_numpy(),
            "bonus_score": d["bonus_score"].astype(str).to_numpy(),
            "fuel_type": d["fuel_type"].fillna("NA").astype(str).to_numpy(),
            "municipality_type": d["municipality_type"].astype(str).to_numpy(),
            "circulation_area": d["circulation_area"].astype(str).to_numpy(),
        }
    )
    out["Loss"] = out["LossRaw"].clip(upper=LOSS_CAP)
    out["log_value"] = np.log(out["vehicle_value"])
    share = d["vehicle_brand"].astype(str).value_counts(normalize=True)
    keep = set(share[share >= min_brand_share].index)
    out["brand_grp"] = (
        d["vehicle_brand"].astype(str).where(d["vehicle_brand"].isin(keep), "OTHER").to_numpy()
    )
    return out.reset_index(drop=True)


def synthetic_raw(n: int = 12_000, seed: int = 0) -> pd.DataFrame:
    """Small synthetic table with the Mendeley column names, for offline tests and CI."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, n // 2, n)
    age = np.clip(rng.normal(49, 12, n).round(), 18, 90)
    expo = np.clip(rng.beta(2, 1.2, n), 0.02, 1.0)
    bonus = rng.choice(["G", "N", "B"], n, p=[0.96, 0.025, 0.015])
    rate = 0.13 * np.exp(
        0.9 * np.exp(-(age - 18) / 8) + 0.6 * (bonus == "B") + 0.3 * (bonus == "N")
    )
    claims = rng.poisson(expo * rate)
    inc = claims * rng.gamma(2.0, 600.0, n) * (rng.random(n) < 0.7)
    return pd.DataFrame(
        {
            "insured_id": ids,
            "year": rng.choice([2022, 2023, 2024], n),
            "driver_age": age,
            "vehicle_age": np.clip(age - rng.integers(18, 26, n), 0, 70),
            "age_driving_licence": rng.integers(18, 26, n),
            "fuel_type": rng.choice(["D", "G", None], n, p=[0.6, 0.39, 0.01]),
            "vehicle_value": np.exp(rng.normal(10, 0.4, n)),
            "seats": rng.choice([2, 4, 5, 7], n),
            "power_to_weight_ratio": rng.normal(12, 2.5, n).clip(4, 30),
            "vehicle_brand": rng.choice([f"BRAND{i}" for i in range(8)], n),
            "municipality_type": rng.choice(["I", "C", "IS"], n),
            "circulation_area": rng.choice(["U", "R"], n),
            "bonus_score": bonus,
            "liability_premium": 100 + 30 * rate * expo * 10,
            "liability_claims": claims,
            "liability_incurred": inc,
            "liability_exposure": expo,
        }
    )


def _design(df: pd.DataFrame) -> Design:
    return Design(factors=SPAIN_FACTORS, bands=SPAIN_BANDS).fit(df)


def _cv_models(df, fid, groups, hp, kind: str, rounds, seed):
    """OOF GLM and XGBoost predictions for ``kind`` = 'freq' (Poisson) or 'tweedie' (loss rate)."""
    oof = {n: np.empty(len(df)) for n in ("GLM", "XGBoost")}
    for k in range(fid.max() + 1):
        tr, te = df[fid != k].reset_index(drop=True), df[fid == k].reset_index(drop=True)
        g_tr = groups[fid != k]
        if kind == "freq":
            oof["GLM"][fid == k] = glm.fit_frequency(tr, _design(tr)).predict_rate(te)
            fit = gbm.fit_gbm(
                tr, "poisson", SPAIN_SPEC, hp, "ClaimNb", rounds=rounds, seed=seed + k, groups=g_tr
            )
        else:
            oof["GLM"][fid == k] = glm.fit_tweedie(tr, _design(tr)).predict_rate(te)
            fit = gbm.fit_gbm(
                tr, "tweedie", SPAIN_SPEC, hp, rounds=rounds, seed=seed + k, groups=g_tr
            )
        oof["XGBoost"][fid == k] = fit.predict_rate(te)
    return oof


def _freq_rows(df, fid, preds: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for k in range(fid.max() + 1):
        m = fid == k
        y, e = df["ClaimNb"].to_numpy()[m], df["Exposure"].to_numpy()[m]
        for n, v in preds.items():
            rows.append(
                {
                    "fold": k,
                    "model": n,
                    "poisson_deviance": poisson_deviance(y, v[m] * e),
                    "gini": gini(y, v[m], e),
                    "pred_over_actual": float((v[m] * e).sum() / y.sum()),
                }
            )
    return rows


def run_spain(
    out_dir: str | Path,
    fig_dir: str | Path | None = None,
    synthetic: bool = False,
    data_dir: str | Path = "data/mendeley",
    folds: int = 5,
    trials: int = 25,
    rounds: int = 1200,
    seed: int = 42,
    reuse_params: bool = False,
    french: dict | None = None,
) -> dict:
    """Same frequency pipeline as the French study, plus premium adequacy on the real premiums.

    ``french`` optionally carries the French results (``frequency_cv`` table, tuned parameters,
    relativities, full policy table) so the cross-market comparison can be written in one pass.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = synthetic_raw() if synthetic else pd.read_csv(download(data_dir), sep=";")
    df = prepare(raw)
    groups = df["insured_id"].to_numpy()

    # tuning slice: whole policyholders, never used in the evaluation folds
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    is_tune = (rng.random(inv.max() + 1) < 0.2)[inv]
    tune_df, cv_df = df[is_tune].reset_index(drop=True), df[~is_tune].reset_index(drop=True)
    g_tune, g_cv = groups[is_tune], groups[~is_tune]
    pj = out / "tuned_params_spain.json"
    if reuse_params and pj.exists():
        hps = json.loads(pj.read_text())
    else:
        hps = {}
        trial_tables = []
        for name, (obj, count) in {
            "freq": ("poisson", "ClaimNb"),
            "pure_premium": ("tweedie", "ClaimNb"),
        }.items():
            hp, t = gbm.tune(
                tune_df,
                obj,
                SPAIN_SPEC,
                count,
                n_trials=trials,
                seed=seed,
                groups=g_tune,
                timeout=900,
            )
            hps[name] = hp
            trial_tables.append(t.assign(study=name))
        pd.concat(trial_tables).to_csv(
            out / "optuna_trials_spain.csv", index=False, float_format="%.5g"
        )
        pj.write_text(json.dumps(hps, indent=2))

    fid = fold_ids(len(cv_df), folds, seed, g_cv)
    y, e = cv_df["ClaimNb"].to_numpy(), cv_df["Exposure"].to_numpy()

    # frequency, Spanish-tuned hyper-parameters
    oof = _cv_models(cv_df, fid, g_cv, hps["freq"], "freq", rounds, seed)
    preds = {
        "Null (portfolio mean)": np.empty(len(cv_df)),
        "Poisson GLM": oof["GLM"],
        "XGBoost Poisson": oof["XGBoost"],
    }
    for k in range(folds):
        tr = fid != k
        preds["Null (portfolio mean)"][~tr] = (
            cv_df["ClaimNb"][tr].sum() / cv_df["Exposure"][tr].sum()
        )
    freq_cv = summarise(_freq_rows(cv_df, fid, preds))

    # do the French hyper-parameters carry over?
    transfer_hp = None
    if french and french.get("hps"):
        o2 = _cv_models(cv_df, fid, g_cv, french["hps"]["freq_claims"], "freq", rounds, seed)[
            "XGBoost"
        ]
        transfer_hp = summarise(
            _freq_rows(cv_df, fid, {"XGBoost Poisson, French hyper-parameters": o2})
        )

    # out-of-time: fit 2022-2023, score 2024
    tr_t, te_t = (
        cv_df[cv_df["year"] < 2024].reset_index(drop=True),
        cv_df[cv_df["year"] == 2024].reset_index(drop=True),
    )
    oot_rows = []
    if len(tr_t) and len(te_t):
        g_t = g_cv[cv_df["year"].to_numpy() < 2024]
        f_g = glm.fit_frequency(tr_t, _design(tr_t))
        f_x = gbm.fit_gbm(
            tr_t,
            "poisson",
            SPAIN_SPEC,
            hps["freq"],
            "ClaimNb",
            rounds=rounds,
            seed=seed,
            groups=g_t,
        )
        yt, et = te_t["ClaimNb"].to_numpy(), te_t["Exposure"].to_numpy()
        for n, v in (
            ("Poisson GLM", f_g.predict_rate(te_t)),
            ("XGBoost Poisson", f_x.predict_rate(te_t)),
        ):
            oot_rows.append(
                {
                    "model": n,
                    "poisson_deviance": poisson_deviance(yt, v * et),
                    "gini": gini(yt, v, et),
                    "pred_over_actual": float((v * et).sum() / yt.sum()),
                }
            )
    oot = pd.DataFrame(oot_rows)

    # pooled calibration and interpretation on the final model
    cal_glm = calibration_table(y, preds["Poisson GLM"], e)
    cal_gbm = calibration_table(y, preds["XGBoost Poisson"], e)
    design = _design(cv_df)
    f_glm = glm.fit_frequency(cv_df, design)
    rel = glm.relativities(f_glm, cv_df)
    f_gbm = gbm.fit_gbm(
        cv_df, "poisson", SPAIN_SPEC, hps["freq"], "ClaimNb", rounds=rounds, seed=seed, groups=g_cv
    )
    _, _, imp = gbm.shap_importance(f_gbm, cv_df)

    # premium adequacy on the premiums actually charged (liability cover)
    t_oof = _cv_models(cv_df, fid, g_cv, hps["pure_premium"], "tweedie", rounds, seed)
    loss, prem = cv_df["Loss"].to_numpy(), cv_df["Premium"].to_numpy()
    pp_dev = {n: tweedie_deviance(loss, v * e, e) for n, v in t_oof.items()}
    lr_pred = loss_ratio_table(loss, prem, t_oof["XGBoost"], e, predicted=t_oof["XGBoost"] * e)
    lr_prem = loss_ratio_table(loss, prem, prem / e, e)

    # shared-feature transfer: driver age only, trained on France, scored on Spain
    transfer = None
    if french and french.get("policies") is not None:
        transfer = _shared_transfer(french["policies"], cv_df, fid)

    comparison = _comparison(french, cv_df, freq_cv, imp, rel)
    tables = {
        "spain_frequency_cv": freq_cv,
        "spain_out_of_time": oot,
        "spain_calibration_glm": cal_glm,
        "spain_calibration_xgb": cal_gbm,
        "spain_relativities_frequency": rel,
        "spain_shap_importance": imp,
        "spain_loss_ratio_by_predicted_decile": lr_pred,
        "spain_loss_ratio_by_premium_decile": lr_prem,
        "market_comparison": comparison,
    }
    if transfer_hp is not None:
        tables["spain_hyperparameter_transfer"] = transfer_hp
    if transfer is not None:
        tables["spain_shared_feature_transfer"] = transfer
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.5g")

    meta = {
        "rows_raw": int(len(raw)),
        "rows_used": int(len(df)),
        "policyholders": int(pd.Series(groups).nunique()),
        "exposure_years": float(df["Exposure"].sum()),
        "claims": int(df["ClaimNb"].sum()),
        "claim_frequency": float(df["ClaimNb"].sum() / df["Exposure"].sum()),
        "liability_premium": float(df["Premium"].sum()),
        "liability_incurred_capped": float(df["Loss"].sum()),
        "portfolio_loss_ratio": float(df["Loss"].sum() / df["Premium"].sum()),
        "tweedie_deviance_by_model": pp_dev,
        "n_tuning": int(len(tune_df)),
        "n_cv": int(len(cv_df)),
        "folds": folds,
        "seed": seed,
    }
    (out / "spain_meta.json").write_text(json.dumps(meta, indent=2))
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plots.loss_ratio(
            {"By predicted pure-premium decile": lr_pred, "By charged-premium decile": lr_prem},
            Path(fig_dir) / "spain_loss_ratio_deciles.png",
        )
    return {"meta": meta, "hps": hps, "comparison": comparison, **tables}


def _shared_transfer(fr: pd.DataFrame, es: pd.DataFrame, fid: np.ndarray) -> pd.DataFrame:
    """Poisson GLM on driver age only: fitted on France and scored on Spain, vs fitted on Spain."""
    bands = {"DrivAgeBand": BANDS["DrivAgeBand"]}
    es = es.assign(DrivAge=es["driver_age"])  # the French column name carries the shared meaning
    mk = lambda d: Design(factors=["DrivAgeBand"], bands=bands).fit(d)  # noqa: E731
    fr_fit = glm.fit_frequency(fr, mk(fr))
    v_fr = fr_fit.predict_rate(es)
    y, e = es["ClaimNb"].to_numpy(), es["Exposure"].to_numpy()
    cv_pred = np.empty(len(es))
    for k in range(fid.max() + 1):
        tr, te = es[fid != k].reset_index(drop=True), es[fid == k].reset_index(drop=True)
        cv_pred[fid == k] = glm.fit_frequency(tr, mk(tr)).predict_rate(te)
    rows = []
    for n, v in (
        ("Fitted on France, scored on Spain", v_fr),
        ("Fitted on Spain (5-fold)", cv_pred),
    ):
        rows.append(
            {
                "model": n,
                "gini": gini(y, v, e),
                "pred_over_actual": float((v * e).sum() / y.sum()),
                "poisson_deviance": poisson_deviance(y, v * e),
            }
        )
    return pd.DataFrame(rows)


def _comparison(
    french, es: pd.DataFrame, freq_cv: pd.DataFrame, imp: pd.DataFrame, rel: pd.DataFrame
) -> pd.DataFrame:
    def row(market, n, expo, claims, cv, top):
        g = cv.set_index("model")
        null = g.loc["Null (portfolio mean)", "poisson_deviance_mean"]
        glm_d = g.loc["Poisson GLM", "poisson_deviance_mean"]
        x_d = g.loc["XGBoost Poisson", "poisson_deviance_mean"]
        return {
            "market": market,
            "rows": n,
            "exposure_years": expo,
            "claims": claims,
            "claim_frequency": claims / expo,
            "glm_dev_gain_vs_null_pct": 100 * (1 - glm_d / null),
            "xgb_dev_gain_vs_glm_pct": 100 * (1 - x_d / glm_d),
            "gini_glm": g.loc["Poisson GLM", "gini_mean"],
            "gini_glm_sd": g.loc["Poisson GLM", "gini_sd"],
            "gini_xgb": g.loc["XGBoost Poisson", "gini_mean"],
            "gini_xgb_sd": g.loc["XGBoost Poisson", "gini_sd"],
            "top_shap_features": ", ".join(top),
        }

    rows = []
    if french and french.get("frequency_cv") is not None:
        p = french["policies"]
        rows.append(
            row(
                "France (freMTPL2)",
                len(p),
                p["Exposure"].sum(),
                int(p["ClaimNb"].sum()),
                french["frequency_cv"],
                french["shap"]["feature"].head(3),
            )
        )
    rows.append(
        row(
            "Spain 2022-24 (liability)",
            len(es),
            es["Exposure"].sum(),
            int(es["ClaimNb"].sum()),
            freq_cv,
            imp["feature"].head(3),
        )
    )
    return pd.DataFrame(rows)

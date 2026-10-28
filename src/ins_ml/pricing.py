"""Frequency-severity pricing study: GLMs vs tuned gradient boosting, 5-fold cross-validated.

Design.
* A random 20% tuning slice is set aside. Optuna tunes the boosting hyper-parameters on it only.
* The other 80% is split into K folds. Every model is refitted on K-1 folds and scored on the held
  out fold; all tables report mean and sd across folds, and the pooled out-of-fold predictions feed
  the calibration, double-lift, Lorenz and loss-ratio views.
* The pure premium chain uses the number of claims that carry an amount (``NSev``), so frequency and
  severity describe the same events; see ``data.reconcile_claims``.
* Totals are corrected with a balance factor or isotonic autocalibration fitted on out-of-fold
  predictions of the other folds, never on the fold being scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import gbm, glm, plots
from .calibrate import Recalibrator
from .cv import cv_frequency, fold_ids, summarise
from .features import GLM_FACTORS, GLM_FACTORS_AMOUNT, KEY_FACTORS, Design
from .metrics import (
    calibration_table,
    double_lift_table,
    gamma_deviance,
    gini,
    lorenz_curve,
    loss_ratio_table,
    poisson_deviance,
    tweedie_deviance,
)

PDP_GRIDS = {
    "DrivAge": list(range(18, 91, 4)),
    "BonusMalus": list(range(50, 151, 10)),
    "VehPower": list(range(4, 10)),
    "LogDensity": list(np.linspace(0, 10, 11)),
}
TARGET_LOSS_RATIO = 0.70  # loading used to turn the incumbent's pure premium into a premium
STUDIES = {
    "freq_claims": ("poisson", "ClaimNb", gbm.FRENCH),
    "freq_amount": ("poisson", "NSev", gbm.FRENCH.with_exposure()),
    "severity": ("gamma", "NSev", gbm.FRENCH),
    "pure_premium": ("tweedie", "ClaimNb", gbm.FRENCH.with_exposure()),
}


def tune_all(tune_df: pd.DataFrame, trials: int, seed: int, out: Path) -> dict:
    hps, tables = {}, []
    for name, (obj, count, spec) in STUDIES.items():
        hp, t = gbm.tune(tune_df, obj, spec, count, n_trials=trials, seed=seed, timeout=900)
        hps[name] = hp
        tables.append(t.assign(study=name))
    pd.concat(tables).to_csv(out / "optuna_trials.csv", index=False, float_format="%.5g")
    (out / "tuned_params.json").write_text(json.dumps(hps, indent=2))
    return hps


def _pp_metrics(loss, v, expo):
    return {
        "tweedie_deviance": tweedie_deviance(loss, v * expo, expo),
        "gini_loss": gini(loss, v, expo),
        "pred_over_actual": float((v * expo).sum() / loss.sum()),
    }


def run_pricing(
    df: pd.DataFrame,
    out_dir: str | Path,
    fig_dir: str | Path | None = None,
    seed: int = 42,
    folds: int = 5,
    trials: int = 25,
    tune_frac: float = 0.2,
    rounds: int = 1200,
    reuse_params: bool = False,
) -> dict:
    """Tune on a side slice, cross-validate everything else, write tables and figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    is_tune = rng.random(len(df)) < tune_frac
    tune_df, cv_df = df[is_tune].reset_index(drop=True), df[~is_tune].reset_index(drop=True)
    pj = out / "tuned_params.json"
    hps = (
        json.loads(pj.read_text())
        if reuse_params and pj.exists()
        else tune_all(tune_df, trials, seed, out)
    )

    fid = fold_ids(len(cv_df), folds, seed)
    expo_all, loss_all, claims_all = (
        cv_df["Exposure"].to_numpy(),
        cv_df["Loss"].to_numpy(),
        cv_df["ClaimNb"].to_numpy(),
    )
    basic = lambda d: Design(GLM_FACTORS).fit(d)  # noqa: E731
    amount = lambda d: Design(GLM_FACTORS_AMOUNT).fit(d)  # noqa: E731

    # 1. claim frequency (all reported claims), the cross-market benchmark
    freq_oof, freq_rows = cv_frequency(
        cv_df, fid, basic, gbm.FRENCH, hps["freq_claims"], rounds=rounds, seed=seed
    )

    # 2. amount-bearing frequency, severity and pure premium
    names_pp = [
        "Null (portfolio mean)",
        "Freq x Sev GLM",
        "Tweedie GLM (p=1.5)",
        "XGBoost freq x XGBoost sev",
        "XGBoost Tweedie (p=1.5)",
    ]
    pp_oof = {k: np.empty(len(cv_df)) for k in names_pp}
    amt_rows, sev_rows = [], []
    for k in range(folds):
        tr, te = cv_df[fid != k].reset_index(drop=True), cv_df[fid == k].reset_index(drop=True)
        e, nsev = te["Exposure"].to_numpy(), te["NSev"].to_numpy()
        n_glm = glm.fit_frequency(tr, amount(tr), "NSev")
        s_glm = glm.fit_severity(tr, basic(tr))
        t_glm = glm.fit_tweedie(tr, amount(tr))
        n_gbm = gbm.fit_gbm(
            tr,
            "poisson",
            gbm.FRENCH.with_exposure(),
            hps["freq_amount"],
            "NSev",
            rounds=rounds,
            seed=seed + k,
        )
        s_gbm = gbm.fit_gbm(tr, "gamma", gbm.FRENCH, hps["severity"], rounds=rounds, seed=seed + k)
        t_gbm = gbm.fit_gbm(
            tr,
            "tweedie",
            gbm.FRENCH.with_exposure(),
            hps["pure_premium"],
            rounds=rounds,
            seed=seed + k,
        )

        null_n = tr["NSev"].sum() / tr["Exposure"].sum()
        has = (te["NSev"] > 0) & (te["Loss"] > 0)
        null_s = tr["Loss"].sum() / tr["NSev"].sum()
        nr = {
            "Null (portfolio mean)": np.full(len(te), null_n),
            "Poisson GLM": n_glm.predict_rate(te),
            "XGBoost Poisson": n_gbm.predict_rate(te),
        }
        for n, v in nr.items():
            amt_rows.append(
                {
                    "fold": k,
                    "model": n,
                    "poisson_deviance": poisson_deviance(nsev, v * e),
                    "gini": gini(nsev, v, e),
                    "pred_over_actual": float((v * e).sum() / nsev.sum()),
                }
            )
        sr = {
            "Null (portfolio mean)": np.full(len(te), null_s),
            "Gamma GLM": s_glm.predict_rate(te),
            "XGBoost Gamma": s_gbm.predict_rate(te),
        }
        y_s, w_s = (
            (te["Loss"] / te["NSev"].where(te["NSev"] > 0, 1))[has].to_numpy(),
            te["NSev"][has].to_numpy(),
        )
        for n, v in sr.items():
            sev_rows.append(
                {
                    "fold": k,
                    "model": n,
                    "gamma_deviance": gamma_deviance(y_s, v[has.to_numpy()], w_s),
                    "pred_over_actual": float(
                        (v[has.to_numpy()] * w_s).sum() / te["Loss"][has].sum()
                    ),
                }
            )
        pp = {
            names_pp[0]: np.full(len(te), tr["Loss"].sum() / tr["Exposure"].sum()),
            names_pp[1]: nr["Poisson GLM"] * sr["Gamma GLM"],
            names_pp[2]: t_glm.predict_rate(te),
            names_pp[3]: nr["XGBoost Poisson"] * sr["XGBoost Gamma"],
            names_pp[4]: t_gbm.predict_rate(te),
        }
        for n, v in pp.items():
            pp_oof[n][fid == k] = v

    def per_fold(preds: dict[str, np.ndarray]) -> list[dict]:
        rows = []
        for k in range(folds):
            m = fid == k
            for n, v in preds.items():
                rows.append({"fold": k, "model": n, **_pp_metrics(loss_all[m], v[m], expo_all[m])})
        return rows

    # 3. balance correction / autocalibration, cross-fitted on the other folds' predictions
    variants: dict[str, dict[str, np.ndarray]] = {"raw": pp_oof}
    for kind in ("balance", "isotonic"):
        corrected = {n: np.empty(len(cv_df)) for n in names_pp}
        for k in range(folds):
            tr_m, te_m = fid != k, fid == k
            for n in names_pp:
                cal = Recalibrator(kind).fit(pp_oof[n][tr_m], loss_all[tr_m], expo_all[tr_m])
                corrected[n][te_m] = cal.transform(pp_oof[n][te_m])
        variants[kind] = corrected
    ba_rows = []
    for variant, preds in variants.items():
        ba_rows += [{**r, "variant": variant} for r in per_fold(preds)]
    ba = summarise(ba_rows, ["model", "variant"])

    # 4. summary tables (mean and sd over folds)
    freq_tbl = summarise(freq_rows.to_dict("records"))
    amt_tbl = summarise(amt_rows)
    sev_tbl = summarise(sev_rows)
    pp_tbl = summarise(per_fold(pp_oof))

    # 5. pooled out-of-fold views
    cal_glm = calibration_table(claims_all, freq_oof["Poisson GLM"], expo_all)
    cal_gbm = calibration_table(claims_all, freq_oof["XGBoost Poisson"], expo_all)
    dl_freq = double_lift_table(
        claims_all, freq_oof["Poisson GLM"], freq_oof["XGBoost Poisson"], expo_all
    )
    inc, chal = "Freq x Sev GLM", "XGBoost Tweedie (p=1.5)"
    dl_pp = double_lift_table(
        loss_all, variants["balance"][inc], variants["balance"][chal], expo_all
    )
    cal_pp = pd.DataFrame({"decile": calibration_table(loss_all, pp_oof[chal], expo_all)["decile"]})
    for variant in ("raw", "balance", "isotonic"):
        t = calibration_table(loss_all, variants[variant][chal], expo_all)
        cal_pp[f"{variant}_predicted_rate"] = t["predicted_rate"]
        if variant == "raw":
            cal_pp["actual_rate"] = t["actual_rate"]
        cal_pp[f"{variant}_pred_to_actual"] = t["pred_to_actual"]

    # 6. premium adequacy: incumbent tariff = balance-corrected Freq x Sev GLM, loaded to 70%
    premium = variants["balance"][inc] * expo_all / TARGET_LOSS_RATIO
    lr_by_challenger = loss_ratio_table(loss_all, premium, variants["balance"][chal], expo_all)
    lr_by_incumbent = loss_ratio_table(loss_all, premium, variants["balance"][inc], expo_all)

    # 7. interpretation artefacts: final models on all CV rows
    design = basic(cv_df)
    f_glm = glm.fit_frequency(cv_df, design, "ClaimNb")
    s_glm = glm.fit_severity(cv_df, design)
    f_gbm = gbm.fit_gbm(
        cv_df, "poisson", gbm.FRENCH, hps["freq_claims"], "ClaimNb", rounds=rounds, seed=seed
    )
    rel = glm.relativities(f_glm, cv_df)
    sev_rel = glm.relativities(s_glm, cv_df, KEY_FACTORS)
    sv, Xs, imp = gbm.shap_importance(f_gbm, cv_df)
    pdp = {
        feat: {
            "XGBoost Poisson": gbm.partial_dependence(f_gbm.predict_rate, cv_df, feat, grid),
            "Poisson GLM": gbm.partial_dependence(f_glm.predict_rate, cv_df, feat, grid),
        }
        for feat, grid in PDP_GRIDS.items()
    }

    tables = {
        "frequency_cv": freq_tbl,
        "frequency_amount_cv": amt_tbl,
        "severity_cv": sev_tbl,
        "pure_premium_cv": pp_tbl,
        "pure_premium_balance_cv": ba,
        "calibration_glm": cal_glm,
        "calibration_xgb": cal_gbm,
        "calibration_pure_premium_xgb": cal_pp,
        "double_lift_frequency": dl_freq,
        "double_lift_pure_premium": dl_pp,
        "loss_ratio_by_challenger_decile": lr_by_challenger,
        "loss_ratio_by_incumbent_decile": lr_by_incumbent,
        "relativities_frequency": rel,
        "relativities_severity": sev_rel,
        "shap_importance": imp,
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False, float_format="%.5g")

    meta = {
        "n_policies": int(len(df)),
        "n_tuning": int(len(tune_df)),
        "n_cv": int(len(cv_df)),
        "folds": folds,
        "optuna_trials": trials,
        "exposure_years": float(df["Exposure"].sum()),
        "claim_frequency": float(df["ClaimNb"].sum() / df["Exposure"].sum()),
        "claims_in_freq_table": int(df["ClaimNb"].sum()),
        "claims_with_amount": int(df["NSev"].sum()),
        "severity_dispersion": s_glm.scale,
        "target_loss_ratio": TARGET_LOSS_RATIO,
        "seed": seed,
    }
    (out / "pricing_meta.json").write_text(json.dumps(meta, indent=2))

    if fig_dir is not None:
        fig = Path(fig_dir)
        fig.mkdir(parents=True, exist_ok=True)
        lor = {
            k: lorenz_curve(loss_all, v, expo_all)
            for k, v in variants["balance"].items()
            if not k.startswith("Null")
        }
        pooled_gini = {k: gini(loss_all, v, expo_all) for k, v in variants["balance"].items()}
        plots.lorenz(lor, pooled_gini, fig / "lorenz_curve.png")
        plots.double_lift(
            {
                "Claim frequency": (dl_freq, "Poisson GLM", "XGBoost Poisson"),
                "Pure premium": (dl_pp, "Freq x Sev GLM", "XGBoost Tweedie"),
            },
            fig / "double_lift.png",
        )
        plots.relativities(rel, KEY_FACTORS, fig / "relativities.png")
        plots.shap_summary(sv, Xs, fig / "shap_summary.png")
        plots.partial_dependence(pdp, fig / "partial_dependence.png")
        plots.calibration(
            {"Poisson GLM": cal_glm, "XGBoost Poisson": cal_gbm}, fig / "calibration.png"
        )
        plots.balance_before_after(cal_pp, fig / "balance_before_after.png")
        plots.loss_ratio(
            {"By challenger decile": lr_by_challenger, "By incumbent decile": lr_by_incumbent},
            fig / "loss_ratio_deciles.png",
        )
    arrays = {"loss": loss_all, "exposure": expo_all, "claims": claims_all}
    return {
        "meta": meta,
        "hps": hps,
        "oof": {"freq": freq_oof, "pp": variants},
        "y": arrays,
        **tables,
    }

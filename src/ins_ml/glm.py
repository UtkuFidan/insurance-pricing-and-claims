"""Frequency (Poisson), severity (Gamma) and pure-premium (Tweedie) GLMs via statsmodels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import Design, add_bands


@dataclass
class GLMFit:
    design: Design
    params: np.ndarray
    bse: np.ndarray
    scale: float
    kind: str

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        """exp(X beta): expected value per unit of exposure (claims per year, severity, ...)."""
        return np.exp(self.design.transform(df) @ self.params)


def fit_frequency(
    df: pd.DataFrame, design: Design | None = None, target: str = "ClaimNb"
) -> GLMFit:
    """Poisson GLM, log link, ``log(Exposure)`` as offset: E[count] = Exposure * exp(X beta).

    ``target`` is ``ClaimNb`` (all reported claims) or ``NSev`` (claims that carry an amount).
    """
    design = design or Design().fit(df)
    X = design.transform(df)
    res = sm.GLM(
        df[target].to_numpy(),
        X,
        family=sm.families.Poisson(),
        offset=np.log(df["Exposure"].to_numpy()),
    ).fit()
    return GLMFit(design, res.params, res.bse, 1.0, "poisson")


def fit_severity(df: pd.DataFrame, design: Design | None = None) -> GLMFit:
    """Gamma GLM, log link, on mean claim size per policy weighted by the number of claims."""
    c = df[(df["NSev"] > 0) & (df["Loss"] > 0)]
    design = design or Design().fit(df)
    X = design.transform(c)
    y = (c["Loss"] / c["NSev"]).to_numpy()
    res = sm.GLM(
        y, X, family=sm.families.Gamma(sm.families.links.Log()), var_weights=c["NSev"].to_numpy()
    ).fit()
    return GLMFit(design, res.params, res.bse, float(res.scale), "gamma")


def fit_tweedie(df: pd.DataFrame, design: Design | None = None, power: float = 1.5) -> GLMFit:
    """Tweedie GLM (compound Poisson-gamma) on loss per exposure year, weighted by exposure."""
    design = design or Design().fit(df)
    X = design.transform(df)
    fam = sm.families.Tweedie(var_power=power, link=sm.families.links.Log())
    y = (df["Loss"] / df["Exposure"]).to_numpy()
    start = np.zeros(X.shape[1])
    start[0] = np.log(df["Loss"].sum() / df["Exposure"].sum())
    res = sm.GLM(y, X, family=fam, var_weights=df["Exposure"].to_numpy()).fit(
        start_params=start, maxiter=200
    )
    return GLMFit(design, res.params, res.bse, float(res.scale), "tweedie")


def pure_premium(freq_rate: np.ndarray, severity: np.ndarray, exposure: np.ndarray | float = 1.0):
    """Pure premium = expected claim frequency x expected severity (x exposure)."""
    return np.asarray(freq_rate) * np.asarray(severity) * np.asarray(exposure)


def relativities(fit: GLMFit, df: pd.DataFrame, factors: list[str] | None = None) -> pd.DataFrame:
    """Multiplicative relativities vs the reference level (largest exposure) with 95% CI."""
    d = add_bands(df, fit.design.bands)
    rows = []
    idx = {n: i for i, n in enumerate(fit.design.names)}
    for f in factors or fit.design.factors:
        expo = d.groupby(f)["Exposure"].sum()
        for lv in fit.design.levels[f]:
            if lv == fit.design.ref[f]:
                b, se = 0.0, 0.0
            else:
                i = idx[f"{f}={lv}"]
                b, se = fit.params[i], fit.bse[i]
            rows.append(
                {
                    "factor": f,
                    "level": lv,
                    "relativity": np.exp(b),
                    "ci_low": np.exp(b - 1.96 * se),
                    "ci_high": np.exp(b + 1.96 * se),
                    "exposure": float(expo.get(lv, 0.0)),
                    "is_reference": lv == fit.design.ref[f],
                }
            )
    return pd.DataFrame(rows)

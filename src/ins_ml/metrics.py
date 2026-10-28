"""Pricing metrics: deviances, Lorenz/Gini of ordered risk, double lift and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Mean Poisson deviance 2/n * sum(y*log(y/mu) - (y - mu)), with 0*log(0) = 0."""
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return float(2.0 * np.mean(term - (y - mu)))


def tweedie_deviance(
    loss: np.ndarray, pred_loss: np.ndarray, exposure: np.ndarray, power: float = 1.5
) -> float:
    """Exposure-weighted mean Tweedie deviance on loss rates (loss per exposure year)."""
    return float(
        mean_tweedie_deviance(
            np.asarray(loss) / exposure,
            np.asarray(pred_loss) / exposure,
            sample_weight=exposure,
            power=power,
        )
    )


def lorenz_curve(actual: np.ndarray, pred_rate: np.ndarray, exposure: np.ndarray):
    """Lorenz curve of ordered risk.

    Policies are sorted by predicted rate (ascending; ties merged). x = cumulative exposure share,
    y = cumulative share of the actual amount (claims or loss).
    """
    g = (
        pd.DataFrame(
            {
                "p": np.asarray(pred_rate, float),
                "e": np.asarray(exposure, float),
                "a": np.asarray(actual, float),
            }
        )
        .groupby("p", sort=True)[["e", "a"]]
        .sum()
    )
    x = np.concatenate([[0.0], g["e"].cumsum().to_numpy()])  # tied predictions merge into one step
    y = np.concatenate([[0.0], g["a"].cumsum().to_numpy()])
    return x / x[-1], y / y[-1]


def gini(actual: np.ndarray, pred_rate: np.ndarray, exposure: np.ndarray) -> float:
    """Gini = 1 - 2 * area under the ordered Lorenz curve (0 = random, higher = better ranking)."""
    x, y = lorenz_curve(actual, pred_rate, exposure)
    return float(1.0 - 2.0 * np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def _exposure_buckets(key: np.ndarray, exposure: np.ndarray, n: int) -> np.ndarray:
    """Assign equal-exposure buckets 0..n-1 after sorting by ``key``."""
    order = np.argsort(key, kind="stable")
    share = (np.cumsum(exposure[order]) - exposure[order] / 2) / exposure.sum()
    out = np.empty(len(key), dtype=int)
    out[order] = np.minimum((share * n).astype(int), n - 1)
    return out


def calibration_table(
    actual: np.ndarray, pred_rate: np.ndarray, exposure: np.ndarray, n: int = 10
) -> pd.DataFrame:
    """Predicted vs actual rate by exposure-weighted decile of predicted rate."""
    exposure = np.asarray(exposure, float)
    b = _exposure_buckets(np.asarray(pred_rate), exposure, n)
    d = pd.DataFrame(
        {"b": b, "act": actual, "pred": np.asarray(pred_rate) * exposure, "expo": exposure}
    )
    g = d.groupby("b").sum()
    out = pd.DataFrame(
        {
            "decile": g.index + 1,
            "exposure": g["expo"].to_numpy(),
            "actual_rate": (g["act"] / g["expo"]).to_numpy(),
            "predicted_rate": (g["pred"] / g["expo"]).to_numpy(),
        }
    )
    out["pred_to_actual"] = out["predicted_rate"] / out["actual_rate"]
    return out


def double_lift_table(
    actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, exposure: np.ndarray, n: int = 10
) -> pd.DataFrame:
    """Double lift: sort by pred_b / pred_a (challenger / incumbent), equal-exposure buckets.

    Reports actual and both models' average rates per bucket. If B is better, actual follows B
    across the buckets where the two models disagree.
    """
    exposure = np.asarray(exposure, float)
    b = _exposure_buckets(np.asarray(pred_b) / np.asarray(pred_a), exposure, n)
    d = pd.DataFrame(
        {
            "b": b,
            "act": actual,
            "a": np.asarray(pred_a) * exposure,
            "bb": np.asarray(pred_b) * exposure,
            "expo": exposure,
        }
    )
    g = d.groupby("b").sum()
    return pd.DataFrame(
        {
            "bucket": g.index + 1,
            "exposure": g["expo"].to_numpy(),
            "actual": (g["act"] / g["expo"]).to_numpy(),
            "model_a": (g["a"] / g["expo"]).to_numpy(),
            "model_b": (g["bb"] / g["expo"]).to_numpy(),
        }
    )


def gamma_deviance(y: np.ndarray, mu: np.ndarray, weight: np.ndarray) -> float:
    """Weighted mean Gamma deviance 2 * sum w (-ln(y/mu) + (y - mu)/mu) / sum w."""
    y, mu, w = (np.asarray(a, float) for a in (y, mu, weight))
    return float(2.0 * np.sum(w * (-np.log(y / mu) + (y - mu) / mu)) / w.sum())


def loss_ratio_table(
    loss: np.ndarray,
    premium: np.ndarray,
    rank_key: np.ndarray,
    exposure: np.ndarray,
    n: int = 10,
    predicted: np.ndarray | None = None,
) -> pd.DataFrame:
    """Loss ratio by equal-exposure decile of ``rank_key`` (the risk ranking being tested).

    ``index`` is the decile loss ratio relative to the portfolio loss ratio (portfolio = 1.00); a
    rising index across deciles means the premium under-charges the risks the ranking calls worse.
    """
    exposure = np.asarray(exposure, float)
    b = _exposure_buckets(np.asarray(rank_key, float), exposure, n)
    g = pd.DataFrame({"b": b, "loss": loss, "prem": premium, "expo": exposure}).groupby("b").sum()
    lr = g["loss"] / g["prem"]
    return pd.DataFrame(
        {
            "decile": g.index + 1,
            "exposure": g["expo"].to_numpy(),
            "premium": g["prem"].to_numpy(),
            "loss": g["loss"].to_numpy(),
            "loss_ratio": lr.to_numpy(),
            "index": (lr / (g["loss"].sum() / g["prem"].sum())).to_numpy(),
        }
    )

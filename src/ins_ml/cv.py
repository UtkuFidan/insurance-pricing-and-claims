"""Cross-validation helpers shared by the French and Spanish studies."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import gbm, glm
from .features import Design
from .metrics import gini, poisson_deviance


def fold_ids(n: int, k: int, seed: int, groups: np.ndarray | None = None) -> np.ndarray:
    """Fold index per row. With ``groups`` (e.g. policyholder id) a group never spans two folds."""
    ids = np.empty(n, dtype=int)
    if groups is None:
        splitter = KFold(k, shuffle=True, random_state=seed).split(np.arange(n))
    else:
        splitter = GroupKFold(k, shuffle=True, random_state=seed).split(np.arange(n), groups=groups)
    for i, (_, te) in enumerate(splitter):
        ids[te] = i
    return ids


def summarise(rows: list[dict], by: str | list[str] = "model") -> pd.DataFrame:
    """Mean and sd (ddof=1) of every numeric column across folds, one row per ``by`` group."""
    d = pd.DataFrame(rows).drop(columns=["fold"], errors="ignore")
    keys = [by] if isinstance(by, str) else by
    g = d.groupby(keys, sort=False)
    mean, sd = g.mean(numeric_only=True), g.std(ddof=1, numeric_only=True)
    out = mean.add_suffix("_mean").join(sd.add_suffix("_sd"))
    cols = [c for m in mean.columns for c in (f"{m}_mean", f"{m}_sd")]
    return out[cols].reset_index()


def cv_frequency(
    df: pd.DataFrame,
    folds: np.ndarray,
    design_factory: Callable[[pd.DataFrame], Design],
    spec: gbm.FeatureSpec,
    hp: dict,
    groups: np.ndarray | None = None,
    target: str = "ClaimNb",
    rounds: int = 1200,
    seed: int = 1,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Out-of-fold Poisson frequency predictions (rate per exposure year) for null, GLM and XGBoost.

    Returns ``{model: oof rate}`` and one row of metrics per model and fold.
    """
    names = ("Null (portfolio mean)", "Poisson GLM", "XGBoost Poisson")
    oof = {k: np.empty(len(df)) for k in names}
    rows = []
    for k in range(folds.max() + 1):
        tr, te = df[folds != k].reset_index(drop=True), df[folds == k].reset_index(drop=True)
        g_tr = None if groups is None else groups[folds != k]
        f_glm = glm.fit_frequency(tr, design_factory(tr), target)
        f_gbm = gbm.fit_gbm(
            tr, "poisson", spec, hp, target, rounds=rounds, seed=seed + k, groups=g_tr
        )
        pred = {
            names[0]: np.full(len(te), tr[target].sum() / tr["Exposure"].sum()),
            names[1]: f_glm.predict_rate(te),
            names[2]: f_gbm.predict_rate(te),
        }
        y, e = te[target].to_numpy(), te["Exposure"].to_numpy()
        for n, v in pred.items():
            oof[n][folds == k] = v
            rows.append(
                {
                    "fold": k,
                    "model": n,
                    "poisson_deviance": poisson_deviance(y, v * e),
                    "gini": gini(y, v, e),
                    "pred_over_actual": float((v * e).sum() / y.sum()),
                }
            )
    return oof, pd.DataFrame(rows)

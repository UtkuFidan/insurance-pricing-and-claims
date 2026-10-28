"""Gradient boosting (XGBoost): Poisson, Gamma and Tweedie objectives, Optuna tuning, SHAP, PDP.

Categoricals use XGBoost's native categorical splits (no ordinal codes). Poisson models take the
claim count as label and ``log(Exposure)`` as base margin, so predictions are rates per year.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

TWEEDIE_POWER = 1.5
NTHREAD = 0  # all cores; heavy runs are serialised by the caller (lockf)
BASE_PARAMS = {
    "tree_method": "hist",
    "nthread": NTHREAD,
    "seed": 1,
    "max_cat_to_onehot": 8,
    "max_bin": 128,
}


@dataclass(frozen=True)
class FeatureSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def names(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def with_exposure(self) -> FeatureSpec:
        return FeatureSpec((*self.numeric, "Exposure"), self.categorical)


FRENCH = FeatureSpec(
    numeric=("DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity"),
    categorical=("Area", "VehBrand", "VehGas", "Region"),
)


def encode(df: pd.DataFrame, spec: FeatureSpec, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Numeric columns as float; categoricals as pandas categories with the training levels."""
    X = df[list(spec.numeric)].astype(float).copy()
    for c in spec.categorical:
        X[c] = pd.Categorical(df[c].astype(str), categories=categories[c])
    return X[spec.names]


def _dmatrix(X: pd.DataFrame, **kw) -> xgb.DMatrix:
    return xgb.DMatrix(X, enable_categorical=True, **kw)


def _labels(df: pd.DataFrame, objective: str, count_col: str, margin0: float = 0.0):
    """(row mask, label, weight, base_margin) for the objective.

    Poisson: label = count, base margin = log(exposure) + margin0, where margin0 is the log of the
    portfolio rate, so boosting starts from the right level instead of from a rate of 1.
    """
    expo = df["Exposure"].to_numpy()
    if objective == "poisson":
        return np.ones(len(df), bool), df[count_col].to_numpy(float), None, np.log(expo) + margin0
    if objective == "tweedie":
        return np.ones(len(df), bool), (df["Loss"] / df["Exposure"]).to_numpy(), expo, None
    if objective == "gamma":
        m = ((df["NSev"] > 0) & (df["Loss"] > 0)).to_numpy()
        n = df["NSev"].to_numpy(float)
        return m, (df["Loss"].to_numpy() / np.where(m, n, 1.0)), n, None
    raise ValueError(objective)


def _xgb_params(objective: str, hp: dict) -> dict:
    p = {**BASE_PARAMS, **hp}
    if objective == "poisson":
        p.update(objective="count:poisson", eval_metric="poisson-nloglik")
    elif objective == "gamma":
        p.update(objective="reg:gamma", eval_metric="gamma-nloglik")
    else:
        p.update(
            objective="reg:tweedie",
            tweedie_variance_power=TWEEDIE_POWER,
            eval_metric=f"tweedie-nloglik@{TWEEDIE_POWER}",
        )
    return p


DEFAULT_HP = {
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
}


@dataclass
class GBMFit:
    booster: xgb.Booster
    spec: FeatureSpec
    categories: dict[str, list[str]]
    objective: str
    best_iteration: int
    hp: dict = field(default_factory=dict)
    margin0: float = 0.0

    def matrix(self, df: pd.DataFrame, **kw) -> xgb.DMatrix:
        return _dmatrix(encode(df, self.spec, self.categories), **kw)

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Expected value per unit exposure (frequency, severity or loss rate)."""
        return self.booster.predict(
            self.matrix(df, **self._kw(len(df))), iteration_range=(0, self.best_iteration)
        )

    def _kw(self, n: int) -> dict:
        return {"base_margin": np.full(n, self.margin0)} if self.objective == "poisson" else {}

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values on the log scale (last column dropped = bias term)."""
        c = self.booster.predict(self.matrix(df, **self._kw(len(df))), pred_contribs=True)
        return c[:, :-1]


def fit_gbm(
    df: pd.DataFrame,
    objective: str,
    spec: FeatureSpec,
    hp: dict | None = None,
    count_col: str = "ClaimNb",
    valid_frac: float = 0.1,
    rounds: int = 1200,
    seed: int = 1,
    groups: np.ndarray | None = None,
    early_stopping: int = 40,
) -> GBMFit:
    """Fit on ``df``; early stopping on a random ``valid_frac`` slice (whole groups if given)."""
    hp = {**DEFAULT_HP, **(hp or {})}
    cats = {c: sorted(df[c].astype(str).unique()) for c in spec.categorical}
    rng = np.random.default_rng(seed)
    if groups is None:
        is_val = rng.random(len(df)) < valid_frac
    else:
        _, inv = np.unique(groups, return_inverse=True)
        is_val = (rng.random(inv.max() + 1) < valid_frac)[inv]
    X = encode(df, spec, cats)
    margin0 = (
        float(np.log(df[count_col].sum() / df["Exposure"].sum())) if objective == "poisson" else 0.0
    )
    keep, y, w, bm = _labels(df, objective, count_col, margin0)

    def mk(mask):
        m = mask & keep
        return _dmatrix(
            X[m],
            label=y[m],
            weight=None if w is None else w[m],
            base_margin=None if bm is None else bm[m],
        )

    dtr, dva = mk(~is_val), mk(is_val)
    bst = xgb.train(
        _xgb_params(objective, hp),
        dtr,
        rounds,
        evals=[(dva, "valid")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    return GBMFit(bst, spec, cats, objective, bst.best_iteration + 1, hp, margin0)


def suggest_hp(trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "eta": trial.suggest_float("eta", 0.04, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 3, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def tune(
    df: pd.DataFrame,
    objective: str,
    spec: FeatureSpec,
    count_col: str = "ClaimNb",
    n_trials: int = 25,
    n_splits: int = 3,
    seed: int = 42,
    groups: np.ndarray | None = None,
    rounds: int = 600,
    timeout: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Optuna (TPE) search of boosting hyper-parameters on ``df``.

    Objective: mean validation negative log-likelihood (XGBoost's own metric for the objective) over
    ``n_splits`` folds, each with early stopping. ``df`` is a tuning slice that never enters the
    cross-validated evaluation, so the reported fold metrics carry no selection optimism.
    """
    import optuna
    from sklearn.model_selection import GroupKFold, KFold

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cats = {c: sorted(df[c].astype(str).unique()) for c in spec.categorical}
    X = encode(df, spec, cats)
    margin0 = (
        float(np.log(df[count_col].sum() / df["Exposure"].sum())) if objective == "poisson" else 0.0
    )
    keep, y, w, bm = _labels(df, objective, count_col, margin0)
    if groups is None:
        folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    else:
        folds = list(GroupKFold(n_splits, shuffle=True, random_state=seed).split(X, groups=groups))

    def mk(idx):
        i = idx[keep[idx]]
        return _dmatrix(
            X.iloc[i],
            label=y[i],
            weight=None if w is None else w[i],
            base_margin=None if bm is None else bm[i],
        )

    mats = [(mk(tr), mk(va)) for tr, va in folds]

    def obj(trial):
        hp = suggest_hp(trial)
        scores, iters = [], []
        for dtr, dva in mats:
            b = xgb.train(
                _xgb_params(objective, hp),
                dtr,
                rounds,
                evals=[(dva, "valid")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            scores.append(b.best_score)
            iters.append(b.best_iteration + 1)
        trial.set_user_attr("rounds", int(np.mean(iters)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, timeout=timeout, n_jobs=1)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return dict(study.best_params), trials


def shap_importance(fit: GBMFit, df: pd.DataFrame, n: int = 3000, seed: int = 0):
    """SHAP values (log scale) on a sample: (values, encoded X, mean |SHAP| table)."""
    sample = df.sample(min(n, len(df)), random_state=seed)
    sv = fit.contributions(sample)
    X = encode(sample, fit.spec, fit.categories)
    Xn = X.copy()
    for c in fit.spec.categorical:
        Xn[c] = X[c].cat.codes.astype(float)  # codes only for the beeswarm colouring
    imp = pd.DataFrame({"feature": fit.spec.names, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return sv, Xn, imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def partial_dependence(
    predict_rate, df: pd.DataFrame, feature: str, grid, n: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Manual partial dependence: mean predicted rate with ``feature`` set to each grid value."""
    sample = df.sample(min(n, len(df)), random_state=seed).copy()
    vals = []
    for g in grid:
        s = sample.copy()
        s[feature] = g
        if feature == "LogDensity":
            s["Density"] = np.exp(g)
        vals.append(float(np.mean(predict_rate(s))))
    return pd.DataFrame({feature: list(grid), "avg_rate": vals})

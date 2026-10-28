"""Balance correction and autocalibration of rate predictions.

A log-link Gamma or Tweedie GLM, and a boosted model stopped early, do not reproduce the portfolio
total. Two corrections are provided, both fitted on out-of-fold training predictions only:

* balance: one multiplicative factor, sum(actual) / sum(predicted).
* isotonic (autocalibration, Denuit, Charpentier and Trufin 2021): E[Y | prediction] = prediction,
  estimated by exposure-weighted isotonic regression of the observed rate on the predicted rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Recalibrator:
    kind: str  # "balance" | "isotonic"
    factor: float = 1.0
    iso: IsotonicRegression | None = field(default=None, repr=False)

    def fit(self, pred_rate, actual, exposure) -> Recalibrator:
        pred_rate, actual, exposure = (np.asarray(a, float) for a in (pred_rate, actual, exposure))
        if self.kind == "balance":
            self.factor = float(actual.sum() / (pred_rate * exposure).sum())
        elif self.kind == "isotonic":
            # Isotonic fits put long runs of policies at zero when claims are rare. A zero forecast
            # on a policy that then claims has an unbounded deviance, so the fitted rate is floored
            # at 5% of the portfolio average rate.
            floor = 0.05 * actual.sum() / exposure.sum()
            self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=floor)
            self.iso.fit(pred_rate, actual / exposure, sample_weight=exposure)
            # a monotone fit still has to reproduce the training total
            fitted = self.iso.predict(pred_rate)
            self.factor = float(actual.sum() / (fitted * exposure).sum())
        else:
            raise ValueError(self.kind)
        return self

    def transform(self, pred_rate) -> np.ndarray:
        pred_rate = np.asarray(pred_rate, float)
        if self.kind == "balance":
            return pred_rate * self.factor
        out = self.iso.predict(pred_rate) * self.factor
        return np.maximum(out, 1e-9)  # keep strictly positive for deviance and ratios

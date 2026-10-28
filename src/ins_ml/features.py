"""Rating-factor banding and a small explicit dummy-coding design for the GLMs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANDS: dict[str, tuple[str, list[float], list[str]]] = {
    "DrivAgeBand": (
        "DrivAge",
        [17, 21, 25, 30, 40, 50, 60, 70, 200],
        ["18-21", "22-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    ),
    "VehAgeBand": (
        "VehAge",
        [-1, 0, 2, 5, 10, 15, 200],
        ["0", "1-2", "3-5", "6-10", "11-15", "16+"],
    ),
    "BonusMalusBand": (
        "BonusMalus",
        [0, 50, 60, 70, 80, 100, 125, 1000],
        ["50", "51-60", "61-70", "71-80", "81-100", "101-125", "126+"],
    ),
    "DensityBand": (
        "Density",
        [0, 50, 200, 1000, 4000, 1e12],
        ["<=50", "51-200", "201-1000", "1001-4000", ">4000"],
    ),
    # Share of claims that carry an amount rises with exposure (see data.reconcile_claims), so the
    # amount-based models need exposure as a covariate; it is set to a full year for pricing.
    "ExposureBand": (
        "Exposure",
        [0, 0.1, 0.25, 0.5, 0.75, 1.0001],
        ["<=0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", ">0.75"],
    ),
}
CATEGORICAL = ["VehPowerCat", "VehBrand", "VehGas", "Region"]
GLM_FACTORS = [b for b in BANDS if b != "ExposureBand"] + CATEGORICAL
GLM_FACTORS_AMOUNT = GLM_FACTORS + ["ExposureBand"]
KEY_FACTORS = ["DrivAgeBand", "BonusMalusBand", "VehPowerCat", "DensityBand"]


def add_bands(df: pd.DataFrame, bands: dict | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with banded rating factors (as strings).

    ``bands`` maps a new column to ``(source column, edges, labels)``; the default is the French
    MTPL banding above, which also derives ``VehPowerCat``.
    """
    out = df.copy()
    for name, (src, edges, labels) in (BANDS if bands is None else bands).items():
        out[name] = pd.cut(out[src], bins=edges, labels=labels).astype(str)
    if bands is None:
        out["VehPowerCat"] = np.where(out["VehPower"] >= 9, "9+", out["VehPower"].astype(str))
    return out


@dataclass
class Design:
    """Treatment-coded dummy design. Reference level = largest exposure share."""

    factors: list[str] = field(default_factory=lambda: list(GLM_FACTORS))
    levels: dict[str, list[str]] = field(default_factory=dict)
    ref: dict[str, str] = field(default_factory=dict)
    bands: dict | None = None  # None = the French MTPL banding

    def _bands(self) -> dict:
        return BANDS if self.bands is None else self.bands

    def fit(self, df: pd.DataFrame) -> Design:
        d = add_bands(df, self.bands)
        for f in self.factors:
            expo = d.groupby(f)["Exposure"].sum()
            self.ref[f] = str(expo.idxmax())
            present = set(map(str, expo.index))
            order = self._bands()[f][2] if f in self._bands() else sorted(present)
            self.levels[f] = [lv for lv in order if lv in present]  # natural band order
        return self

    @property
    def names(self) -> list[str]:
        cols = ["Intercept"]
        for f in self.factors:
            cols += [f"{f}={lv}" for lv in self.levels[f] if lv != self.ref[f]]
        return cols

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        d = add_bands(df, self.bands)
        cols = [np.ones(len(d))]
        for f in self.factors:
            vals = d[f].astype(str).to_numpy()
            unseen = set(vals) - set(self.levels[f])
            if unseen:
                raise ValueError(f"unseen level(s) {sorted(unseen)} for factor {f}")
            cols += [(vals == lv).astype(float) for lv in self.levels[f] if lv != self.ref[f]]
        return np.column_stack(cols)

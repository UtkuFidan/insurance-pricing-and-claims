"""Severity-anomaly screening with reason codes.

IMPORTANT: this is a screening heuristic, not a fraud model. The public data contain no fraud
labels, so nothing here is validated against confirmed fraud. A flag means "unusual relative to
the model-predicted severity and a few simple red-flag rules - worth a second look", nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

REASONS = {
    "SEV_TAIL": "Amount in the extreme upper tail of the GLM-implied Gamma severity distribution",
    "SEV_RATIO": "Amount more than 5x the model-predicted severity",
    "SHORT_EXPOSURE_LARGE": "Large claim (>2x predicted) on a policy with < 0.15 years of exposure",
    "MULTI_CLAIM": "Policy with 3 or more claims in the exposure period",
}


def screen_claims(
    claims: pd.DataFrame, dispersion: float, tail_p: float = 1e-3, ratio_threshold: float = 5.0
) -> pd.DataFrame:
    """Score claim rows and attach reason codes.

    ``claims`` needs: ClaimAmount, pred_sev (GLM mean), Exposure, ClaimNb. Under the fitted Gamma
    model amount ~ Gamma(shape=1/phi, scale=phi*pred_sev); ``tail_prob`` is P(X >= amount).
    A claim is flagged if SEV_TAIL fires, or if two or more reason codes fire.
    """
    d = claims.copy()
    d["ratio"] = d["ClaimAmount"] / d["pred_sev"]
    d["tail_prob"] = stats.gamma.sf(
        d["ClaimAmount"], a=1.0 / dispersion, scale=dispersion * d["pred_sev"]
    )
    fired = {
        "SEV_TAIL": d["tail_prob"] < tail_p,
        "SEV_RATIO": d["ratio"] > ratio_threshold,
        "SHORT_EXPOSURE_LARGE": (d["Exposure"] < 0.15) & (d["ratio"] > 2.0),
        "MULTI_CLAIM": d["ClaimNb"] >= 3,
    }
    for k, v in fired.items():
        d[k] = v.to_numpy()
    d["n_reasons"] = sum(v.astype(int) for v in fired.values())
    names = list(fired)
    mat = np.column_stack([fired[k].to_numpy() for k in names])
    d["reason_codes"] = [
        ";".join(n for n, hit in zip(names, row, strict=True) if hit) for row in mat
    ]
    d["flag"] = fired["SEV_TAIL"] | (d["n_reasons"] >= 2)
    return d


def run_fraud(df: pd.DataFrame, sev: pd.DataFrame, out_dir, seed: int = 42, folds: int = 5) -> dict:
    """Screen every claim row against a severity GLM that never saw its policy.

    The Gamma severity GLM is fitted on K-1 folds of policies and used to predict the claims of the
    held-out fold, so a claim is not judged by a model that was fitted to it.
    """
    from pathlib import Path

    from . import glm
    from .cv import fold_ids
    from .features import Design

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fid = pd.Series(fold_ids(len(df), folds, seed), index=df["IDpol"].to_numpy())
    c = sev[["IDpol", "ClaimAmount"]].copy()
    c["IDpol"] = c["IDpol"].astype("int64")
    c = c.merge(df, on="IDpol", how="inner")
    c["fold"] = fid.reindex(c["IDpol"]).to_numpy()
    c["pred_sev"] = np.nan
    scales = []
    for k in range(folds):
        train = df[fid.to_numpy() != k]
        fit = glm.fit_severity(train, Design().fit(train))
        m = c["fold"] == k
        c.loc[m, "pred_sev"] = fit.predict_rate(c[m])
        scales.append(fit.scale)
    dispersion = float(np.mean(scales))
    scr = screen_claims(c[["IDpol", "ClaimAmount", "pred_sev", "Exposure", "ClaimNb"]], dispersion)
    reasons = pd.DataFrame(
        {
            "reason": list(REASONS),
            "claims": [int(scr[k].sum()) for k in REASONS],
            "description": list(REASONS.values()),
        }
    )
    top = (
        scr[scr.flag]
        .sort_values("tail_prob")
        .head(10)[["IDpol", "ClaimAmount", "pred_sev", "ratio", "tail_prob", "reason_codes"]]
    )
    reasons.to_csv(out / "fraud_reason_counts.csv", index=False)
    top.to_csv(out / "fraud_top_flags.csv", index=False, float_format="%.5g")
    summary = {
        "claims_screened": int(len(scr)),
        "flagged": int(scr.flag.sum()),
        "flag_rate": float(scr.flag.mean()),
        "dispersion": dispersion,
        "share_of_amount_flagged": float(
            scr.loc[scr.flag, "ClaimAmount"].sum() / scr.ClaimAmount.sum()
        ),
    }
    import json

    (out / "fraud_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "reasons": reasons, "top": top, "screened": scr}

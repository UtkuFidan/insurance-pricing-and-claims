"""Claims reserving: chain-ladder, Mack standard errors, Bornhuetter-Ferguson, bootstrap ODP.

Triangles are the sample data shipped with the ``chainladder`` package (GenIns = Taylor & Ashe
1983 cumulative claims; CLRD = CAS Loss Reserving Database, Schedule P), so everything runs offline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

from . import plots


def triangle_array(tri) -> np.ndarray:
    """Cumulative triangle as a 2-D array (origin x development), NaN where unobserved."""
    return np.array(tri.values[0, 0], dtype=float)


def volume_weighted_ldfs(arr: np.ndarray) -> np.ndarray:
    """All-year volume-weighted age-to-age factors, computed directly with numpy."""
    n = arr.shape[1]
    f = np.empty(n - 1)
    for j in range(n - 1):
        ok = np.isfinite(arr[:, j + 1])
        f[j] = arr[ok, j + 1].sum() / arr[ok, j].sum()
    return f


def link_ratios(arr: np.ndarray) -> np.ndarray:
    out = arr[:, 1:] / arr[:, :-1]
    return np.pad(out, ((0, 0), (0, 1)), constant_values=np.nan)


def mack_table(tri) -> tuple[pd.DataFrame, float, float]:
    """Chain-ladder reserve by origin with Mack standard errors; also total IBNR and total s.e."""
    m = cl.MackChainladder().fit(tri)
    t = m.summary_.to_frame().fillna(0.0)
    t.index = [str(i.year) for i in t.index]
    t.columns = ["latest", "ibnr", "ultimate", "mack_se"]
    t["cv"] = np.where(t.ibnr > 0, t.mack_se / t.ibnr, np.nan)
    return t, float(t.ibnr.sum()), float(m.total_mack_std_err_.values.squeeze())


def odp_dispersion(arr: np.ndarray) -> float:
    """Pearson scale parameter of the over-dispersed Poisson model (England & Verrall)."""
    f = volume_weighted_ldfs(arr)
    n_o, n_d = arr.shape
    fitted = np.full_like(arr, np.nan)
    for i in range(n_o):
        last = n_d - 1 - i
        fitted[i, last] = arr[i, last]
        for j in range(last - 1, -1, -1):
            fitted[i, j] = fitted[i, j + 1] / f[j]
    inc_fit = np.diff(fitted, axis=1, prepend=0.0)
    inc_act = np.diff(arr, axis=1, prepend=0.0)
    mask = np.isfinite(inc_act) & (inc_fit > 0)
    resid = (inc_act[mask] - inc_fit[mask]) / np.sqrt(inc_fit[mask])
    p = 2 * n_o - 1
    return float(np.sum(resid**2) / (mask.sum() - p))


def bootstrap_odp(tri, n_sims: int = 2000, seed: int = 42) -> dict:
    """Bootstrap ODP (England-Verrall): residual resampling for parameter error, plus gamma
    process error with variance = scale * mean for each origin year's reserve."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sims = cl.BootstrapODPSample(n_sims=n_sims, random_state=seed).fit_transform(tri)
        ibnr = cl.Chainladder().fit(sims).ibnr_.values[:, 0, :, 0]  # (sims, origin)
    ibnr = np.nan_to_num(ibnr)
    phi = odp_dispersion(triangle_array(tri))
    rng = np.random.default_rng(seed)
    mean = np.clip(ibnr, 1e-6, None)
    with_pv = rng.gamma(shape=mean / phi, scale=phi)
    with_pv = np.where(ibnr > 0, with_pv, 0.0)
    total = with_pv.sum(axis=1)
    pcts = {f"p{q}": float(np.percentile(total, q)) for q in (5, 25, 50, 75, 90, 95, 99)}
    return {
        "total": total,
        "by_origin": with_pv,
        "phi": phi,
        "pcts": pcts,
        "mean": float(total.mean()),
        "std": float(total.std(ddof=1)),
    }


def bornhuetter_ferguson_demo(lob: str = "comauto", n_mature: int = 5) -> pd.DataFrame:
    """Chain-ladder vs BF on CAS CLRD net paid losses for one line of business (all companies).

    A-priori loss ratio = mean chain-ladder ultimate loss ratio of the ``n_mature`` oldest origin
    years. This is a stated assumption, not a pricing view; BF shrinks immature years towards it.
    """
    c = cl.load_sample("clrd")
    x = c[c["LOB"] == lob].sum()
    paid, prem = x["CumPaidLoss"], x["EarnedPremNet"].latest_diagonal
    dev = cl.Development().fit_transform(paid)
    chain = cl.Chainladder().fit(dev)
    prem_v = prem.values.squeeze()
    elr = float((chain.ultimate_.values.squeeze() / prem_v)[:n_mature].mean())
    bf = cl.BornhuetterFerguson(apriori=elr).fit(dev, sample_weight=prem)
    latest = paid.latest_diagonal.values.squeeze()
    out = pd.DataFrame(
        {
            "origin": [str(i.year) for i in paid.origin.to_timestamp()],
            "premium": prem_v,
            "paid_to_date": latest,
            "pct_developed": latest / chain.ultimate_.values.squeeze(),
            "cl_ibnr": np.nan_to_num(chain.ibnr_.values.squeeze()),
            "bf_ibnr": np.nan_to_num(bf.ibnr_.values.squeeze()),
        }
    )
    out.attrs["apriori_elr"] = elr
    return out


def run_reserving(
    out_dir: str | Path, fig_dir: str | Path | None = None, n_sims: int = 2000, seed: int = 42
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tri = cl.load_sample("genins")
    arr = triangle_array(tri)
    ldf = volume_weighted_ldfs(arr)
    table, total_ibnr, total_se = mack_table(tri)
    boot = bootstrap_odp(tri, n_sims, seed)
    bf = bornhuetter_ferguson_demo()

    pd.DataFrame(
        {"period": [f"{12 * (j + 1)}-{12 * (j + 2)}" for j in range(len(ldf))], "ldf": ldf}
    ).to_csv(out / "genins_ldfs.csv", index=False, float_format="%.5f")
    table.to_csv(out / "genins_chainladder_mack.csv", float_format="%.1f")
    bf.to_csv(out / "clrd_comauto_cl_vs_bf.csv", index=False, float_format="%.4g")
    summary = {
        "triangle": "GenIns (Taylor & Ashe 1983)",
        "total_ibnr": total_ibnr,
        "mack_total_se": total_se,
        "mack_total_cv": total_se / total_ibnr,
        "bootstrap_mean": boot["mean"],
        "bootstrap_std": boot["std"],
        "bootstrap_percentiles": boot["pcts"],
        "odp_dispersion": boot["phi"],
        "n_sims": n_sims,
        "bf_apriori_elr": bf.attrs["apriori_elr"],
        "bf_total_ibnr": float(bf.bf_ibnr.sum()),
        "cl_total_ibnr_clrd": float(bf.cl_ibnr.sum()),
    }
    (out / "reserving_summary.json").write_text(json.dumps(summary, indent=2))

    if fig_dir is not None:
        fig = Path(fig_dir)
        plots.triangle_heatmap(
            arr,
            link_ratios(arr),
            fig / "triangle_heatmap.png",
            "GenIns cumulative claims triangle (origin 2001-2010)",
            origins=[str(i.year) for i in tri.origin.to_timestamp()],
            devs=[str(d) for d in tri.development],
        )
        plots.ibnr_figure(table.ibnr, table.mack_se, boot["total"], boot["pcts"], fig / "ibnr.png")
    return {"ldf": ldf, "mack": table, "summary": summary, "bf": bf}

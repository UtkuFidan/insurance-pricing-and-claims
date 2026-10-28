"""Matplotlib figures (small PNGs committed under docs/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402,F401

C = ["#1f5fa8", "#e07b00", "#2a9d5c", "#8e44ad", "#c0392b"]
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 100,
    }
)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def lorenz(curves: dict, ginis: dict, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random ordering")
    for (name, (x, y)), c in zip(curves.items(), C, strict=False):
        ax.plot(x, y, color=c, lw=1.6, label=f"{name} (Gini {ginis[name]:.3f})")
    ax.set(
        xlabel="Cumulative share of exposure (low to high predicted risk)",
        ylabel="Cumulative share of loss",
        title="Lorenz curve of ordered risk (test set)",
    )
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, path)


def double_lift(panels: dict, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4))
    for ax, (title, (t, a, b)) in zip(np.atleast_1d(axes), panels.items(), strict=True):
        ax.plot(t.bucket, t.actual, "o-", color="black", label="Actual")
        ax.plot(t.bucket, t.model_a, "s--", color=C[0], label=a)
        ax.plot(t.bucket, t.model_b, "^--", color=C[1], label=b)
        ax.set(
            xlabel=f"Bucket of {b} / {a} (equal exposure)",
            ylabel="Average rate per exposure year",
            title=f"Double lift: {title}",
        )
        ax.legend(fontsize=7)
    _save(fig, path)


def relativities(rel, factors, path):
    fig, axes = plt.subplots(1, len(factors), figsize=(3.3 * len(factors), 3.6))
    for ax, f in zip(axes, factors, strict=True):
        d = rel[rel.factor == f].reset_index(drop=True)
        ax.errorbar(
            range(len(d)),
            d.relativity,
            yerr=[d.relativity - d.ci_low, d.ci_high - d.relativity],
            fmt="o-",
            color=C[0],
            capsize=2,
        )
        ax.axhline(1, color="grey", ls="--", lw=0.8)
        ax.set_xticks(range(len(d)), d.level, rotation=60, fontsize=7)
        ax.set(
            title=f.replace("Band", "").replace("Cat", ""),
            ylabel="Relativity (95% CI)" if f == factors[0] else "",
        )
    fig.suptitle("Poisson GLM frequency relativities (reference level = 1)", fontsize=10)
    _save(fig, path)


def shap_summary(sv, X, path):
    import shap

    plt.figure(figsize=(6, 4))
    shap.summary_plot(sv, X, show=False, plot_size=None, max_display=9)
    plt.title("SHAP summary: XGBoost Poisson (log-frequency)", fontsize=10)
    _save(plt.gcf(), path)


def partial_dependence(pdp: dict, path):
    fig, axes = plt.subplots(1, len(pdp), figsize=(3.3 * len(pdp), 3.4))
    for ax, (feat, models) in zip(axes, pdp.items(), strict=True):
        for (name, d), c in zip(models.items(), C, strict=False):
            ax.plot(d[feat], d.avg_rate, "o-", ms=3, color=c, label=name)
        ax.set(xlabel=feat, ylabel="Avg claim frequency" if feat == list(pdp)[0] else "")
    axes[0].legend(fontsize=7)
    fig.suptitle("Partial dependence of claim frequency (GLM vs GBM)", fontsize=10)
    _save(fig, path)


def calibration(tables: dict, path):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    for (name, t), c in zip(tables.items(), C, strict=False):
        ax.plot(t.decile, t.predicted_rate / t.actual_rate, "o-", color=c, label=name)
    ax.axhline(1, color="grey", ls="--")
    ax.set(
        xlabel="Decile of predicted frequency",
        ylabel="Predicted / actual",
        title="Calibration by decile (pooled out-of-fold)",
    )
    ax.legend(fontsize=7)
    _save(fig, path)


def balance_before_after(cal: pd.DataFrame, path):
    """Predicted / actual loss by decile of predicted rate, before and after the corrections."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    labels = {"raw": "Raw", "balance": "Balance factor", "isotonic": "Isotonic autocalibration"}
    for (v, lab), c in zip(labels.items(), C, strict=False):
        ax.plot(cal.decile, cal[f"{v}_pred_to_actual"], "o-", ms=4, color=c, label=lab)
    ax.axhline(1, color="grey", ls="--")
    ax.set(
        xlabel="Decile of predicted pure premium",
        ylabel="Predicted / actual loss",
        title="XGBoost Tweedie: before and after correction",
    )
    ax.legend(fontsize=7)
    _save(fig, path)


def loss_ratio(tables: dict, path):
    """Loss ratio index (portfolio = 1) by decile for one or more rankings."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for (name, t), c in zip(tables.items(), C, strict=False):
        ax.plot(t.decile, t["index"], "o-", ms=4, color=c, label=name)
    ax.axhline(1, color="grey", ls="--")
    ax.set(xlabel="Decile (1 = lowest predicted risk)", ylabel="Loss ratio / portfolio loss ratio")
    ax.set_title("Premium adequacy by predicted-risk decile")
    ax.legend(fontsize=7)
    _save(fig, path)


def triangle_heatmap(cum, ratios, path, title="Development triangle", origins=None, devs=None):
    """cum: 2-D array (origin x dev) with NaN below the diagonal; ratios: age-to-age factors."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, data, fmt, name, cmap in (
        (axes[0], cum / 1e3, "{:,.0f}", "Cumulative claims (thousands)", "Blues"),
        (axes[1], ratios, "{:.3f}", "Age-to-age development factors (age j to j+12)", "Oranges"),
    ):
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=6.5)
        ax.set(title=name, xlabel="Development age (months)", ylabel="Origin year")
        if origins is not None:
            ax.set_xticks(range(data.shape[1]), devs, fontsize=7)
            ax.set_yticks(range(data.shape[0]), origins, fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title, fontsize=10)
    _save(fig, path)


def ibnr_figure(ibnr, se, sims_total, pcts, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    x = np.arange(len(ibnr))
    ax.bar(
        x,
        ibnr.values / 1e6,
        color=C[0],
        yerr=1.96 * se.values / 1e6,
        capsize=2,
        label="Chain-ladder IBNR",
    )
    ax.set_xticks(x, ibnr.index.astype(str), rotation=45)
    ax.set(ylabel="Millions", title="IBNR by origin year (+/- 1.96 Mack s.e.)")
    ax = axes[1]
    ax.hist(sims_total / 1e6, bins=40, color=C[2], alpha=0.8)
    for lab, v, c in (
        ("P50", pcts["p50"], "black"),
        ("P75", pcts["p75"], C[1]),
        ("P95", pcts["p95"], C[4]),
    ):
        ax.axvline(v / 1e6, color=c, ls="--", label=f"{lab}: {v / 1e6:.1f}m")
    ax.set(
        xlabel="Total reserve (millions)",
        ylabel="Simulations",
        title="Bootstrap ODP: total reserve distribution",
    )
    ax.legend(fontsize=8)
    _save(fig, path)

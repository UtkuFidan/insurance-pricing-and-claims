"""Data loading and cleaning for the French MTPL portfolio (freMTPL2freq / freMTPL2sev).

Source: CASdatasets (Dutang & Charpentier), mirrored on OpenML (ids 41214 and 41215, CC0).
Raw files are downloaded on first use and cached under ``data/`` (gitignored).
A seeded synthetic generator with the same schema is provided for offline tests and CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FREQ_ID, SEV_ID = 41214, 41215
LOSS_CAP = 100_000.0  # per-claim cap used for severity / pure-premium modelling

FEATURES_RAW = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Density",
    "Region",
    "Area",
]


def load_openml(data_home: str | Path = "data/openml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read from cache) the raw frequency and severity tables."""
    from sklearn.datasets import fetch_openml

    Path(data_home).mkdir(parents=True, exist_ok=True)
    freq = fetch_openml(data_id=FREQ_ID, as_frame=True, parser="auto", data_home=str(data_home))
    sev = fetch_openml(data_id=SEV_ID, as_frame=True, parser="auto", data_home=str(data_home))
    return freq.frame.copy(), sev.frame.copy()


def build_policy_table(
    freq: pd.DataFrame, sev: pd.DataFrame, cap: float = LOSS_CAP
) -> pd.DataFrame:
    """Clean the raw tables and attach per-policy claim amounts.

    Cleaning follows the common practice for this dataset: ClaimNb capped at 4, Exposure at 1
    year, VehPower at 9, VehAge at 20, DrivAge at 90, BonusMalus at 150. Individual claims are
    capped at ``cap`` for severity and pure-premium work (``Loss``); ``LossRaw`` is uncapped.
    Policies without a severity record get ``NSev = 0`` and ``Loss = 0``.
    """
    df = freq.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    df["ClaimNb"] = df["ClaimNb"].astype(int).clip(upper=4)
    df["Exposure"] = df["Exposure"].astype(float).clip(upper=1.0)
    df["VehPower"] = df["VehPower"].astype(int).clip(upper=9)
    df["VehAge"] = df["VehAge"].astype(int).clip(upper=20)
    df["DrivAge"] = df["DrivAge"].astype(int).clip(upper=90)
    df["BonusMalus"] = df["BonusMalus"].astype(int).clip(upper=150)
    df["VehGas"] = df["VehGas"].astype(str).str.strip("'")
    for c in ("Area", "VehBrand", "Region"):
        df[c] = df[c].astype(str)
    df["LogDensity"] = np.log(df["Density"].astype(float))

    s = sev.copy()
    s["IDpol"] = s["IDpol"].astype("int64")
    s["Capped"] = s["ClaimAmount"].clip(upper=cap)
    agg = s.groupby("IDpol").agg(
        NSev=("ClaimAmount", "size"), Loss=("Capped", "sum"), LossRaw=("ClaimAmount", "sum")
    )
    df = df.merge(agg, how="left", left_on="IDpol", right_index=True)
    df[["NSev", "Loss", "LossRaw"]] = df[["NSev", "Loss", "LossRaw"]].fillna(0.0)
    df["NSev"] = df["NSev"].astype(int)
    return df.reset_index(drop=True)


def reconcile_claims(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reconcile the claim counts in the frequency table with the amounts in the severity table.

    Severity rows are aggregated to one record per policy and joined on ``IDpol``. Each policy is
    then classified by how its ``ClaimNb`` relates to the number of amounts it has. Returns the
    classification table (policies, claims, amounts, mean exposure) and a dict of counts that
    cannot be joined (severity rows whose policy is not in the frequency table).
    """
    f = freq[["IDpol", "ClaimNb", "Exposure"]].copy()
    f["IDpol"] = f["IDpol"].astype("int64")
    s = sev[["IDpol", "ClaimAmount"]].copy()
    s["IDpol"] = s["IDpol"].astype("int64")
    agg = s.groupby("IDpol")["ClaimAmount"].agg(NSev="size", Amount="sum")
    m = f.merge(agg, how="left", left_on="IDpol", right_index=True).fillna({"NSev": 0, "Amount": 0})
    m["ClaimNb"], m["NSev"] = m["ClaimNb"].astype(int), m["NSev"].astype(int)
    status = np.select(
        [m["ClaimNb"] == 0, m["NSev"] == 0, m["NSev"] == m["ClaimNb"], m["NSev"] < m["ClaimNb"]],
        [
            "no claim",
            "claims, none with an amount",
            "every claim has an amount",
            "some claims lack an amount",
        ],
        default="more amounts than claims",
    )
    tbl = (
        m.assign(status=status)
        .groupby("status")
        .agg(
            policies=("IDpol", "size"),
            claims_in_freq=("ClaimNb", "sum"),
            amounts_in_sev=("NSev", "sum"),
            mean_exposure=("Exposure", "mean"),
        )
        .reset_index()
    )
    orphans = ~s["IDpol"].isin(f["IDpol"])
    info = {
        "policies": int(len(f)),
        "claims_in_freq": int(m["ClaimNb"].sum()),
        "severity_rows": int(len(s)),
        "severity_rows_without_policy": int(orphans.sum()),
        "amount_without_policy": float(s.loc[orphans, "ClaimAmount"].sum()),
        "amounts_joined": int(m["NSev"].sum()),
    }
    return tbl, info


def amount_share_by_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Share of claim-bearing policies that have an amount, by exposure band."""
    c = df[df["ClaimNb"] > 0]
    band = pd.cut(c["Exposure"], [0, 0.1, 0.25, 0.5, 0.75, 1.0], include_lowest=True)
    g = c.groupby(band, observed=True).agg(
        policies=("IDpol", "size"), with_amount=("NSev", lambda x: (x > 0).mean())
    )
    return g.reset_index().rename(columns={"Exposure": "exposure_band"})


def split(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split (policy level, seeded)."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_frac
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def synthetic_portfolio(n: int = 20_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic raw ``(freq, sev)`` tables with the freMTPL2 schema and known true effects.

    True frequency: log-linear in driver age (young drivers worse), bonus-malus and log-density,
    times exposure. True severity: Gamma with mean rising mildly with vehicle power. As in the real
    data, some claim-bearing policies have no amount (more often when exposure is short) and a few
    severity rows belong to policies that are not in the frequency table.
    """
    rng = np.random.default_rng(seed)
    exposure = np.clip(rng.beta(2.0, 1.2, n), 0.02, 1.0)
    drivage = np.clip(rng.normal(45, 14, n).round(), 18, 90).astype(int)
    bm = np.clip(50 + rng.gamma(1.2, 12, n).round(), 50, 230).astype(int)
    density = np.exp(rng.normal(6.0, 1.8, n)).round().clip(1, 27000).astype(int)
    vehpower = rng.integers(4, 13, n)
    vehage = rng.integers(0, 25, n)
    brand = rng.choice([f"B{i}" for i in range(1, 7)], n)
    gas = rng.choice(["Diesel", "Regular"], n)
    region = rng.choice([f"R{i}" for i in range(1, 6)], n)
    area = np.clip(np.digitize(np.log(density), [3, 4.5, 6, 7.5, 9]), 0, 5)
    area = np.array(list("ABCDEF"))[area]

    log_rate = (
        -2.6
        + 0.9 * np.exp(-(drivage - 18) / 8.0)
        + 0.012 * (np.minimum(bm, 150) - 50)
        + 0.08 * np.log(density)
        + 0.15 * (gas == "Diesel")
    )
    claims = rng.poisson(exposure * np.exp(log_rate))
    freq = pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1, dtype=float),
            "ClaimNb": claims,
            "Exposure": exposure,
            "Area": area,
            "VehPower": vehpower,
            "VehAge": vehage,
            "DrivAge": drivage,
            "BonusMalus": bm,
            "VehBrand": brand,
            "VehGas": gas,
            "Density": density,
            "Region": region,
        }
    )
    ids = np.repeat(freq["IDpol"].to_numpy().astype("int64"), claims)
    vp = np.repeat(vehpower, claims)
    mean_sev = 1500 * np.exp(0.05 * (np.minimum(vp, 9) - 6))
    amount = rng.gamma(2.0, mean_sev / 2.0)
    big = rng.random(len(ids)) < 0.01  # a few injected extreme claims for the screening module
    amount[big] *= rng.uniform(15, 40, big.sum())
    sev = pd.DataFrame({"IDpol": ids, "ClaimAmount": amount.round(2)})
    p_missing = 0.35 - 0.25 * exposure  # whole policies lose their amounts, as in freMTPL2sev
    lost = (rng.random(n) < p_missing)[
        np.searchsorted(freq["IDpol"].to_numpy().astype("int64"), ids)
    ]
    sev = sev[~lost].reset_index(drop=True)
    orphans = pd.DataFrame(
        {"IDpol": n + 1 + np.arange(5), "ClaimAmount": [900.0, 1200, 1500, 2500, 800]}
    )
    return freq, pd.concat([sev, orphans], ignore_index=True)

import pandas as pd

from ins_ml.data import amount_share_by_exposure, build_policy_table, reconcile_claims


def _tables():
    freq = pd.DataFrame(
        {
            "IDpol": [1, 2, 3, 4, 5],
            "ClaimNb": [0, 1, 2, 1, 3],
            "Exposure": [1.0, 0.5, 1.0, 0.05, 1.0],
        }
    )
    sev = pd.DataFrame(
        {
            "IDpol": [
                2,
                3,
                3,
                5,
                99,
            ],  # 4 has a claim but no amount; 5 has 1 of 3; 99 has no policy
            "ClaimAmount": [1000.0, 500.0, 700.0, 900.0, 123.0],
        }
    )
    return freq, sev


def test_reconcile_classifies_policies_and_counts_orphans():
    t, info = reconcile_claims(*_tables())
    by = t.set_index("status")
    assert by.loc["no claim", "policies"] == 1
    assert by.loc["every claim has an amount", "policies"] == 2  # policies 2 and 3
    assert by.loc["claims, none with an amount", "policies"] == 1  # policy 4
    assert by.loc["some claims lack an amount", "policies"] == 1  # policy 5
    assert info["claims_in_freq"] == 7 and info["amounts_joined"] == 4
    assert info["severity_rows_without_policy"] == 1
    assert info["amount_without_policy"] == 123.0


def test_policy_table_aggregates_severity_per_policy_and_ignores_orphans():
    freq, sev = _tables()
    freq = freq.assign(
        Area="A", VehPower=6, VehAge=3, DrivAge=40, BonusMalus=50, VehBrand="B1",
        VehGas="Diesel", Density=100, Region="R1",
    )  # fmt: skip
    df = build_policy_table(freq, sev).set_index("IDpol")
    assert df.loc[3, "NSev"] == 2 and df.loc[3, "Loss"] == 1200.0
    assert df.loc[4, "NSev"] == 0 and df.loc[4, "Loss"] == 0.0
    assert df["Loss"].sum() == 1000 + 1200 + 900  # the orphan amount is not attached to anyone


def test_amount_share_by_exposure_rises_in_fixture():
    freq, sev = _tables()
    freq = freq.assign(NSev=[0, 1, 2, 0, 1])
    t = amount_share_by_exposure(freq)
    assert t["with_amount"].iloc[0] == 0.0  # the 0.05-year policy has no amount
    assert t["with_amount"].iloc[-1] == 1.0

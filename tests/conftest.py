import numpy as np
import pandas as pd
import pytest

from ins_ml.data import build_policy_table, synthetic_portfolio


@pytest.fixture(scope="session")
def synth():
    freq, sev = synthetic_portfolio(6000, seed=3)
    return build_policy_table(freq, sev), sev


@pytest.fixture
def tiny():
    """Hand-built portfolio: two risk classes, known exposures and claim counts."""
    return pd.DataFrame(
        {
            "ClaimNb": [1, 0, 0, 2, 0, 1],
            "Exposure": [0.5, 1.0, 0.25, 1.0, 0.25, 0.5],
            "DrivAge": [20, 20, 20, 50, 50, 50],
            "VehAge": [1] * 6,
            "VehPower": [6] * 6,
            "BonusMalus": [50] * 6,
            "Density": [100] * 6,
            "VehBrand": ["B1"] * 6,
            "VehGas": ["Diesel"] * 6,
            "Region": ["R1"] * 6,
            "Area": ["A"] * 6,
            "LogDensity": np.log([100.0] * 6),
        }
    )

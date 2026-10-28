import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from ins_ml.reserving import mack_table, odp_dispersion, triangle_array, volume_weighted_ldfs

# Hand-computed 3x3 triangle:
#   A: 100 150 165 | B: 110 168 | C: 120
# f1 = (150+168)/(100+110) = 318/210 ; f2 = 165/150 = 1.1
# ultimates: A=165, B=168*1.1=184.8, C=120*(318/210)*1.1
F1, F2 = 318 / 210, 1.1
ULT = np.array([165.0, 168 * F2, 120 * F1 * F2])
RAW = np.array([[100, 150, 165], [110, 168, np.nan], [120, np.nan, np.nan]])


@pytest.fixture
def tri():
    rows = []
    for i, origin in enumerate([2020, 2021, 2022]):
        for j in range(3 - i):
            rows.append({"origin": origin, "dev": origin + j, "paid": RAW[i, j]})
    return cl.Triangle(
        pd.DataFrame(rows), origin="origin", development="dev", columns="paid", cumulative=True
    )


def test_numpy_volume_weighted_factors_match_hand_values():
    assert volume_weighted_ldfs(RAW) == pytest.approx([F1, F2])


def test_chainladder_package_factors_and_ultimates_match_hand_values(tri):
    ldf = cl.Development(average="volume").fit(tri).ldf_.values.squeeze()
    assert ldf == pytest.approx([F1, F2])
    ult = cl.Chainladder().fit(tri).ultimate_.values.squeeze()
    assert ult == pytest.approx(ULT)
    ibnr = cl.Chainladder().fit(tri).ibnr_.values.squeeze()
    assert np.nansum(ibnr) == pytest.approx(ULT.sum() - (165 + 168 + 120))


def test_genins_total_ibnr_matches_published_value():
    tri = cl.load_sample("genins")
    table, total, se = mack_table(tri)
    assert total == pytest.approx(18_680_856, rel=1e-4)  # classic Taylor-Ashe chain-ladder reserve
    assert se == pytest.approx(2_441_364, rel=1e-3)  # Mack total standard error
    assert (table.ibnr >= 0).all()


def test_odp_dispersion_matches_england_verrall():
    phi = odp_dispersion(triangle_array(cl.load_sample("genins")))
    assert phi == pytest.approx(52_601, rel=1e-3)

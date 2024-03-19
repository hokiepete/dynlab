import numpy as np
from dynlab.diagnostics import FTLE, LCS
from dynlab.flows import double_gyre


def test_ftle():
    x = [0, 1, 2]
    y = [0, 1]

    expected_ftle = np.array([
        [0.04001625, 0.16099239, 0.000000],
        [0.00000000, 0.16099239, 0.226121]
    ])
    assert np.allclose(
        expected_ftle,
        FTLE(num_threads=1).compute(x, y, double_gyre, (2, 0))
    )


def test_lcs():
    expected_lcs = [
        np.array([
            [1.01249658, 0.55555556],
            [1.00954249, 0.66666667],
            [1.00900913, 0.77777778],
            [1.00888040, 0.88888889],
            [1.00885959, 1.00000000]
        ]),
        np.array([
            [0.00000000, 0.96183275],
            [0.05301184, 1.00000000]
        ]),
        np.array([
            [1.94690241, 1.00000000],
            [2.00000000, 0.96177941]
        ])
    ]

    x = np.linspace(0, 2, 20)
    y = np.linspace(0, 1, 10)

    attracting_lcs = LCS().compute(x, y, f=double_gyre, t=(0.1, 0))

    assert all([np.isclose(x, y).all() for x, y in zip(expected_lcs, attracting_lcs)])

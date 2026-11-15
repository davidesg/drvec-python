"""As the C: .pre files that do not share one rescaling factor are warned of
(bugs/py-pre-mixed-refactors-silent)."""
import warnings

import numpy as np
import pytest

fue = pytest.importorskip("fue")

from drvec.data import MIXED_REFACTORS, read_pre


def _pre(tmp_path, name, y, factor):
    ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name=name)
    m = fue.Model(ts, d=1, D=0, boxlam=0.0, ar=[[0.1]], ar_free=[[True]],
                  refactor=float(factor))
    m.fit()
    p = tmp_path / f"{name}_{factor:g}.pre"
    m.write_pre(str(p))
    return str(p)


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(8)
    c = np.cumsum(rng.normal(0, 0.01, 120))
    u = np.zeros(120)
    for t in range(1, 120):
        u[t] = 0.7 * u[t - 1] + rng.normal(0, 0.004)
    return 120 * np.exp(c + u), 100 * np.exp(c)


def _avisos(files):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        read_pre(files)
    return [str(x.message) for x in w if "refactor" in str(x.message)]


def test_mixed_factors_are_warned_of(tmp_path, series):
    y, x = series
    got = _avisos([_pre(tmp_path, "Y", y, 100), _pre(tmp_path, "X", x, 1)])
    assert got == [MIXED_REFACTORS]


def test_a_common_factor_is_silent(tmp_path, series):
    y, x = series
    assert _avisos([_pre(tmp_path, "Y", y, 100), _pre(tmp_path, "X", x, 100)]) == []
    assert _avisos([_pre(tmp_path, "Y", y, 1), _pre(tmp_path, "X", x, 1)]) == []

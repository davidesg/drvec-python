"""The VEC in levels, forecast by drvarma (levels.to_levels_varma and
drvarma.forecast.forecast_given), against drvec's own -f on every golden case
with a .forecast (tests/data/oracle: the C's).

Two comparisons:

* AGAINST drvec's -f, in the model's units (the level before the Box-Cox is
  inverted, and its s.e.), on the same fit: relative TOL_LEVELS = 1e-12 on
  every platform. Not byte identity, because the arithmetic is not the same
  one: drvec recurs on Ybar = [nabla Y2 ; W] and cumulates, drvarma recurs on
  the levels with p + 1 lags; drvec's band is the cumulated Ybar psi weights
  through level_error_map, drvarma's the psi weights of the levels system
  with Sigma_e = T0^-1 Sigma T0^-T. The two are equal in exact arithmetic and
  round differently: the largest gap over the ten cases is 1.7e-14 (dfc's
  points), 3.1e-15 in an s.e., so 1e-12 keeps a margin of about 60.
* AGAINST the oracle's <base>.forecast, the file drvec writes rebuilt from
  drvarma's numbers (Level, Low, High from drvarma; the s.e. column is drvec's
  delta method, bt_jac, applied to drvarma's s.d.): on Linux x86_64 byte for
  byte, as drvec's own; elsewhere drvec's tolerance (1e-3 relative), since the
  fit itself moves there.

The band's multiplier: drvec's is the C's 1.959964 (at 95%), drvarma's ladder
uses 1.96. The conversion passes drvec's, and test_the_ladders_multiplier
pins what drvarma's default gives instead.
"""
import os

import numpy as np
import pytest

from conftest import E1, EXACT, FIXTURES, PRE
from oracle import out_text

import drvec
from drvec.forecast import bt_jac
from drvec.levels import to_levels_varma

fg = pytest.importorskip("drvarma.forecast")
if not hasattr(fg, "forecast_given"):
    pytest.skip("drvarma without forecast_given (drvarma 0.3.0, branch prevision-niveles)",
                allow_module_level=True)

TOL_LEVELS = 1e-12


def _fx(*names):
    return [os.path.join(FIXTURES, n) for n in names]


MU = _fx("mmpre.muskrat.pre", "minkmu.pre")
MD = _fx("mmdet.muskrat.pre", "mmdet.mink.pre")
MDU = _fx("mmdet.muskrat.pre", "mmdet.minkmu.pre")
MON = _fx("mmmon.muskrat.pre", "mmmon.mink.pre")
E1S = _fx("e1seas.invest.pre", "e1seas.income.pre", "e1seas.cons.pre")
B40 = _fx("mmpre.muskrat.pre", "mmb40.mink.pre")
# every oracle case with a .forecast (SOURCE), as test_forecast and test_m5 fit them
CASES = {
    "fc201": (PRE, (2, 0, 1), dict(case=2, forecast=8)),
    "fc211": (PRE, (2, 1, 1), dict(case=2, forecast=6)),
    "fcpre": (MU, (2, 0, 1), dict(case="pre", forecast=5)),
    "ewin": (PRE, (2, 0, 1), dict(case=2, forecast=4, estwin=50)),
    "e1fc": (E1, (2, 0, 1), dict(case=2, forecast=8, estwin=70)),
    "dfc": (MD, (2, 0, 1), dict(case=2, forecast=8)),
    "dfcmon": (MON, (2, 0, 1), dict(case=2, forecast=12)),
    "e1sfc": (E1S, (2, 0, 1), dict(case=2, forecast=8, estwin=70)),
    "dcpfc": (MDU, (2, 0, 1), dict(case="pre", forecast=5)),
    "b40": (B40, (2, 0, 1), dict(case=2, forecast=8)),
}
# beyond the goldens: the corners of the conversion (p = 1, r = 0, cases 1 and 3)
EXTRA = {
    "r0c3": (PRE, (2, 0, 0), dict(case=3, forecast=5)),
    "r0c1": (PRE, (2, 1, 0), dict(case=1, forecast=5)),
    "p1": (PRE, (1, 0, 1), dict(case=2, forecast=5)),
    "p1c3": (PRE, (1, 1, 1), dict(case=3, forecast=5)),
}
_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, (p, q, r), kw = {**CASES, **EXTRA}[label]
        _FITS[label] = drvec.fit(files, p, q, r, **kw)
    return _FITS[label]


def _drvec_arrays(res, H):
    """drvec's own -f in the model's units: the level and its theoretical s.e.
    before the back-transform (FitResult.forecasts, the very numbers
    forecast_vec prints the .out and the .forecast from)."""
    fc = res.forecasts(H)
    return fc.model_level, fc.model_se


def _rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


@pytest.mark.parametrize("label", list(CASES) + list(EXTRA))
def test_the_levels_varma_forecasts_as_drvec(label):
    res = _fit(label)
    H = res.forecast_h
    out = to_levels_varma(res).forecast(H)
    w = np.array([o["w"] for o in out]).T
    sd = np.array([o["sd"] for o in out]).T
    lev, se = _drvec_arrays(res, H)
    assert _rel(w, lev) <= TOL_LEVELS
    assert _rel(sd, se) <= TOL_LEVELS


def _forecast_file(res, H, out, conf=0.95):
    """drvec's <base>.forecast body (forecast_vec's format) from drvarma's
    numbers; the s.e. column is drvec's delta method on drvarma's s.d."""
    from drvec.forecast import _date
    data = res.data
    lv = to_levels_varma(res)
    det = lv.det(H)
    f = []
    for i in range(lv.M):
        f.append("Series %d (%s):" % (i + 1, data.names[i]))
        f.append("  %-9s %14s %14s %14s %12s" % ("date", "Level", "Low", "High", "s.e."))
        for h in range(1, H + 1):
            per, sub = _date(data.start, data.freq, lv.origin + h)
            d = ("  %3d/%-5d" % (sub, per)) if data.freq > 1 else ("  %-9d" % per)
            o = out[i]
            se = bt_jac(lv.lambdas[i], lv.refactors[i], o["w"][h - 1], det[i][h - 1]) * o["sd"][h - 1]
            f.append(d + " %14.6f %14.6f %14.6f %12.6f" % (
                o["level"][h - 1], o["low"][h - 1], o["high"][h - 1], se))
        f.append("")
    return "\n".join(f) + "\n"


def _body(text):
    return text.split("\n\n", 1)[1]


@pytest.mark.parametrize("label", list(CASES))
def test_the_oracles_forecast_file(label):
    res = _fit(label)
    H = res.forecast_h
    mine = _forecast_file(res, H, to_levels_varma(res).forecast(H))
    theirs = _body(out_text(label + ".forecast"))
    if EXACT:
        assert mine == theirs
    else:
        from test_forecast import _close
        _close(mine, theirs)


def test_the_unit_roots():
    """Phi(1) has rank r: M - r singular values at zero, r away from it."""
    for label in ("fc201", "e1fc", "r0c3"):
        lv = to_levels_varma(_fit(label))
        sv = np.linalg.svd(np.eye(lv.M) - lv.phi.sum(axis=0), compute_uv=False)
        assert np.all(sv[lv.r:] < 1e-12) and np.all(sv[:lv.r] > 1e-6), (label, sv)


def test_the_ladders_multiplier():
    """drvarma's default band is the ladder's +/- 1.96 s.d.; drvec's, the C's
    +/- 1.959964. Same points and s.d., the band edges apart by 3.6e-5 s.d."""
    from fue.forecast import _inv_boxcox
    res = _fit("e1fc")
    lv = to_levels_varma(res)
    a = lv.forecast(8)
    b = lv.forecast(8, z=1.96)
    det = lv.det(8)
    for i, (x, y) in enumerate(zip(a, b)):
        assert np.array_equal(x["level"], y["level"]) and np.array_equal(x["sd"], y["sd"])
        assert not np.array_equal(x["high"], y["high"])
        ys = x["w"] + det[i]
        assert np.array_equal(y["high"], [_inv_boxcox(v + 1.96 * e, lv.lambdas[i], lv.refactors[i])
                                          for v, e in zip(ys, x["sd"])])


def test_refusals():
    res = drvec.fit(MU, 2, 0, 1, case="pre", trend=True)
    with pytest.raises(ValueError, match="-trend"):
        to_levels_varma(res)

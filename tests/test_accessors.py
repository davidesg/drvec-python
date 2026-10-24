"""A1: the accessors (STUDY-vec §4.2 items 1-6) equal the numbers the .out
prints, on every golden single-fit case of the suite (tests/accessors.py
rebuilds each block from the accessor in the .out's format and finds it in the
port's .out; on Linux x86_64 that .out is the oracle's byte for byte, which the
case's own test checks). The fits are the other test modules' own, shared
through their caches where they keep one."""
import os

import pytest

from conftest import E1, FIXTURES, PRE, STRICT
from oracle import golden_filter, out_text

import accessors
import drvec

import test_fdhess
import test_forecast
import test_gate
import test_m5
import test_options
import test_restrictions

MM = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "mmpre.mink.pre")]

# label -> a function returning the fit
FITS = {}
for _mod in (test_options, test_restrictions, test_forecast, test_fdhess, test_gate):
    for _lab in _mod.CASES:
        FITS[_lab] = (lambda m, l: (lambda: m._fit(l)))(_mod, _lab)
for _lab in test_m5.FITS:
    FITS[_lab] = (lambda l: (lambda: test_m5._run(l)))(_lab)

_OWN = {}


def _own(label, *a, **kw):
    if label not in _OWN:
        _OWN[label] = drvec.fit(*a, **kw)
    return _OWN[label]


FITS.update({
    "proute": lambda: _own("proute", PRE, 2, 1, 1, case=2),
    "e1fit": lambda: _own("e1fit", E1, 2, 1, 1, case=2),
    "ms201": lambda: _own("ms201", PRE, 2, 0, 1, case=2, multistart=4),
    "ms211": lambda: _own("ms211", PRE, 2, 1, 1, case=2, multistart=5),
    "xlink": lambda: _own("xlink", MM, 2, 0, 1, case=2,
                          xpre=[os.path.join(FIXTURES, "xrev.pre")],
                          xlink=os.path.join(FIXTURES, "xnet1.txt")),
    "irfboot": lambda: _own("irfboot", MM, 2, 0, 1, case=2, irfboot=40, engine="python"),
})

FORECASTS = set(test_forecast.CASES) | {k for k, v in test_m5.FITS.items()
                                        if v[2].get("forecast")}
RESTRICTED = {"fixb2row", "xsys", "weakex1", "weakex2", "e1alpha", "wxfix", "wxrow",
              "dxsys", "xarma", "xarma1", "xar0", "xdet", "xmon", "xlink", "gzero", "gzeroa"}


@pytest.mark.parametrize("label", sorted(FITS))
def test_accessors_equal_the_out(label):
    res = FITS[label]()
    out = res.out_text(label)
    assert accessors.check(res, label, out) >= 8
    assert (res.restriction_lr is not None) == (label in RESTRICTED)
    if STRICT:
        # and so the oracle's: the port's .out is its byte for byte
        assert golden_filter(out) == golden_filter(out_text(label + ".out"))


@pytest.mark.parametrize("label", sorted(FORECASTS))
def test_forecast_arrays_equal_the_file(label):
    res = FITS[label]()
    assert accessors.check_forecasts(res, label) == res.spec.M


def test_irf_band_arrays_equal_the_file():
    res = FITS["irfboot"]()
    assert accessors.check_irf_bands(res, "irfboot") == 2 * 2 * 2 * 21
    b = res.irf_bands()
    assert b.nok == 40 and b.nfail == 0 and b.irf.shape == (21, 2, 2)
    assert res.provenance.engine == "python"


def test_restriction_lr_kinds():
    assert FITS["fixb2row"]().restriction_lr.kind == "fixb2row"
    assert FITS["wxrow"]().restriction_lr.kind == "fixb2row+alpha"
    assert FITS["weakex1"]().restriction_lr.kind == "alpha"
    assert FITS["xsys"]().restriction_lr.kind == "xsys"
    x = FITS["xlink"]().restriction_lr
    assert x.kind == "xlink" and x.ll_bridge is not None
    # -fixb2 alone: the C prints no LR, and neither does the accessor
    assert FITS["fixb2v"]().restriction_lr is None


def test_shapes_and_conventions():
    res = FITS["proute"]()
    M, r = 2, 1
    assert res.alpha.shape == (M, r) and res.beta.shape == (M, r)
    assert res.beta[M - r:, :].tolist() == [[1.0]]
    assert res.gamma.shape == (1, M, M) and res.theta.shape == (1, M, M)
    assert (res.Q == res.Q.T).all()
    assert res.residuals.shape == (61, M) and res.residual_names == ["muskrat", "ec1"]
    assert res.level_irf(5).shape == (6, M, M)
    assert res.provenance.engine == "python"
    with pytest.raises(ValueError, match="horizon"):
        res.forecasts()

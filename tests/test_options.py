"""The remaining PRE_CASES of the oracle's tools/golden.sh, each against the
oracle's .out (tests/data/oracle, SOURCE): logL to 1e-6, every estimate and
s.e. to 1e-4 (to the printed digits on Linux x86_64), and the .out byte for
byte after golden.sh's filter."""
import os

import pytest

from conftest import EXACT, FIXTURES, STRICT, TOL_EST, TOL_LL, TOL_SE
from oracle import golden_filter, number_after, out_text, param_table

import drvec

MM = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "mmpre.mink.pre")]
MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]

CASES = {
    "fixb2row": (MM, (2, 1, 1), dict(case=2, fixb2row={1: 0.0})),
    "nodrift": (MM, (2, 0, 1), dict(case=3, nodrift=[1])),
    "casepre": (MU, (2, 0, 1), dict(case="pre")),
    "trend": (MU, (2, 0, 1), dict(case="pre", trend=True)),
    "xsys": (MM, (3, 0, 1), dict(case=2, xsys=[1])),
}

_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, (p, q, r), kw = CASES[label]
        _FITS[label] = drvec.fit(files, p, q, r, **kw)
    return _FITS[label]


@pytest.mark.parametrize("label", list(CASES))
def test_logl_and_estimates(label):
    res = _fit(label)
    out = out_text(label + ".out")
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    table = param_table(out)
    for p in res.params:
        est, se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if STRICT else TOL_EST), p.label
        if se is None:                       # an MA root on the circle: no s.e.
            assert p.kind == "theta" and res.ma_on_boundary
            continue
        assert abs(p.se - se) <= (5.01e-7 if STRICT else TOL_SE), p.label


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_fit(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


def test_fixb2row_lr_against_the_free_model():
    res = _fit("fixb2row")
    out = out_text("fixb2row.out")
    assert abs(res.lr_free - number_after(out, "logL H(r)  free        :")) < (1e-6 if EXACT else TOL_LL)
    assert res.spec.npar() == 13


def test_held_drift_rows():
    assert "E[D.muskrat]" not in [p.label for p in _fit("nodrift").params]
    assert [p.label for p in _fit("trend").params][:2] == ["E[ec1]", "trend[ec1]"]


def test_refusals():
    with pytest.raises(ValueError, match="-nodrift holds a drift of case 3"):
        drvec.fit(MM, 2, 0, 1, case=2, nodrift=[1])
    with pytest.raises(ValueError, match="-fixb2row needs r >= 1"):
        drvec.fit(MM, 2, 0, 0, case=2, fixb2row={1: 0.0})
    with pytest.raises(ValueError, match="B2 has 1 row"):
        drvec.fit(MM, 2, 0, 1, case=2, fixb2row={2: 0.0})
    with pytest.raises(ValueError, match="-trend"):
        drvec.fit(MM, 2, 0, 1, case=2, trend=True)


@pytest.fixture(scope="module")
def irfboot():
    return drvec.fit(MM, 2, 0, 1, case=2, irfboot=40)


def test_irfboot_out_and_file(irfboot):
    """-irfboot 40: the .out with its band section and <base>.irfboot, whole."""
    res = irfboot
    assert res.irfboot.nok == 40 and res.irfboot.nfail == 0
    mine = golden_filter(res.out_text("irfboot"))
    theirs = golden_filter(out_text("irfboot.out"))
    fb = res.irfboot_text("irfboot")
    if STRICT:
        assert mine == theirs
        assert fb == out_text("irfboot.irfboot")
    else:
        rows = [l.split(",") for l in fb.splitlines()[2:]]
        crow = [l.split(",") for l in out_text("irfboot.irfboot").splitlines()[2:]]
        assert len(rows) == len(crow)
        for a, b in zip(rows, crow):
            assert a[:4] == b[:4]
            assert max(abs(float(u) - float(v)) for u, v in zip(a[4:], b[4:])) < 1e-4


def test_irfboot_refusal():
    with pytest.raises(ValueError, match="-irfboot bootstraps the responses"):
        drvec.fit(MU, 2, 0, 1, case="pre", trend=True, irfboot=10)


def test_xsys():
    res = _fit("xsys")
    out = out_text("xsys.out")
    assert abs(res.lr_free - number_after(out, "logL VEC free          :")) < 1e-6
    assert res.xs_df == 3
    assert [p.label for p in res.params if p.kind == "psi"] == ["psi[1 <- ec1]"]


def test_xsys_refusals():
    """The gaps the C declares, with its words: q > 0, the Y1 block, every series."""
    with pytest.raises(ValueError, match="a moving average \\(q > 0, not"):
        drvec.fit(MM, 2, 1, 1, case=2, xsys=[1])
    with pytest.raises(ValueError, match="is in the Y1 block"):
        drvec.fit(MM, 2, 0, 1, case=2, xsys=[2])
    with pytest.raises(ValueError, match="every series an input"):
        drvec.fit(MM, 2, 0, 0, case=2, xsys=[1, 2])


def test_xlink():
    res = drvec.fit(MM, 2, 0, 1, case=2, xpre=[os.path.join(FIXTURES, "xrev.pre")],
                    xlink=os.path.join(FIXTURES, "xnet1.txt"))
    out = out_text("xlink.out")
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    assert abs(res.xl_base_ll - number_after(out, "logL VEC without inputs      :")) < 1e-6
    assert res.xl_bridge_ll == res.xl_base_ll or not EXACT     # omega = 0 is the VEC
    table = param_table(out)
    for p in res.params:
        est, se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if STRICT else TOL_EST), p.label
    mine = golden_filter(res.out_text("xlink"))
    if STRICT:
        assert mine == golden_filter(out)


def test_the_gaps_the_c_declares():
    xr = [os.path.join(FIXTURES, "xrev.pre")]
    net = os.path.join(FIXTURES, "xnet1.txt")
    with pytest.raises(ValueError, match="-lrtest, -specs, -rungs, -matest, -artest, -eval, -f"):
        drvec.fit(MM, 2, 0, 1, case=2, xpre=xr, xlink=net, forecast=5)
    with pytest.raises(ValueError, match="-xlink needs the .pre route"):
        drvec.rank_test(MM, 2, 0, case=2, xpre=xr, xlink=net)
    with pytest.raises(ValueError, match="-xsys embeds an input"):
        drvec.rank_test(MM, 2, 0, case=2, xsys=[1])
    with pytest.raises(ValueError, match="-trend \\(case 4\\)"):
        drvec.rank_test(MU, 2, 0, case=3, trend=True)
    with pytest.raises(ValueError, match="-xlink needs a free mean of W"):
        drvec.fit(MM, 2, 0, 1, case=1, xpre=xr, xlink=net)
    with pytest.raises(ValueError, match="-xpre and -xlink go together"):
        drvec.fit(MM, 2, 0, 1, case=2, xpre=xr)

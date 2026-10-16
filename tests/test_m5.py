"""Milestone 5: deterministic variables in a .pre, and -xpre inputs with an
ARMA model, against the oracle (tests/data/oracle/SOURCE, milestone 5; the
fixtures are tests/data/fixtures/make_m5.py's, see SOURCE there).

The C subtracts each file's deterministic component, the EXACT rational filter
omega(B)/delta(B) on the file's own dates (build_det_component; BUG-40 closed
WON'T FIX), from w before anything else, keeps it over the horizon to bring
the forecasts back to levels, and does the same to an -xpre input, whose
presample it backcasts with the input's own ARMA (xl_backcast).

On the oracle's platform every product is byte-identical (the first line of a
.forecast/.recursive names the program and is not compared). Elsewhere: the
logL to TOL_LL and the estimates to TOL_EST for the well-defined fits; for the
ill-defined ones (ddet211, dstep: four starts spread over 4-5 logL units, the
best stopping on criterion 3) the declaration only, as for E1."""
import os

import pytest

from conftest import EXACT, FIXTURES, STRICT, TOL_EST, TOL_LL, TOL_LR
from oracle import boot_rows, golden_filter, number_after, out_text, param_table

import drvec
from drvec import prefile


def _fx(*names):
    return [os.path.join(FIXTURES, n) for n in names]


MM = _fx("mmpre.muskrat.pre", "mmpre.mink.pre")
MD = _fx("mmdet.muskrat.pre", "mmdet.mink.pre")
MDU = _fx("mmdet.muskrat.pre", "mmdet.minkmu.pre")
E1S = _fx("e1seas.invest.pre", "e1seas.income.pre", "e1seas.cons.pre")
MON = _fx("mmmon.muskrat.pre", "mmmon.mink.pre")


def _x(pre, net):
    return dict(xpre=_fx(pre), xlink=os.path.join(FIXTURES, net))


FITS = {
    "ddet201": (_fx("mmpre.muskrat.pre", "mmdet.mink.pre"), (2, 0, 1), dict(case=2)),
    "ddet211": (MD, (2, 1, 1), dict(case=2)),
    "dstep": (_fx("mmpre.muskrat.pre", "mmpre.step.pre"), (2, 1, 1), dict(case=2)),
    "e1seas": (E1S, (2, 0, 1), dict(case=2)),
    "mon201": (MON, (2, 0, 1), dict(case=2)),
    "dfc": (MD, (2, 0, 1), dict(case=2, forecast=8)),
    "dfcmon": (MON, (2, 0, 1), dict(case=2, forecast=12)),
    "e1sfc": (E1S, (2, 0, 1), dict(case=2, forecast=8, estwin=70)),
    "dcpfc": (MDU, (2, 0, 1), dict(case="pre", forecast=5)),
    "dfdh": (MD, (2, 0, 1), dict(case=2, fdhess=True)),
    "dcpre": (MDU, (2, 0, 1), dict(case="pre")),
    "dxsys": (MD, (3, 0, 1), dict(case=2, xsys=[1])),
    "xarma": (MM, (2, 0, 1), dict(case=2, **_x("xrevarma.pre", "xnet2.txt"))),
    "xarma1": (MM, (2, 0, 1), dict(case=2, **_x("xrevarma.pre", "xnet1.txt"))),
    "xar0": (MM, (2, 0, 1), dict(case=2, **_x("xrevar0.pre", "xnet2.txt"))),
    "xdet": (MD, (2, 0, 1), dict(case=2, **_x("xrevdet.pre", "xnet2.txt"))),
    "xmon": (MON, (2, 0, 1), dict(case=2, **_x("xmon.pre", "xnetm.txt"))),
}
ILL = {"ddet211", "dstep"}
RANKS = {
    "drank": (MD, (2, 0), dict(case=2)),
    "e1srank": (E1S, (2, 0), dict(case=2)),
    "dcprank": (MDU, (2, 0), dict(case="pre")),
}
_DONE = {}


def _fit(label):
    if label not in _DONE:
        files, (p, q, r), kw = FITS[label]
        _DONE[label] = drvec.fit(files, p, q, r, **kw)
    return _DONE[label]


def _rank(label):
    if label not in _DONE:
        files, (p, q), kw = RANKS[label]
        _DONE[label] = drvec.rank_test(files, p, q, **kw)
    return _DONE[label]


def _tail(text):
    return text.split("\n", 1)[1]


@pytest.mark.parametrize("label", list(FITS))
def test_the_fit(label):
    res = _fit(label)
    out = out_text(label + ".out")
    if label in ILL:
        assert res.termcode == 3
        assert "NOT a convergence" in out
        if not EXACT:
            return
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    table = param_table(out)
    for p in res.params:
        est, _se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if STRICT else TOL_EST), p.label


@pytest.mark.parametrize("label", list(FITS))
def test_the_products(label):
    res = _fit(label)
    mine = golden_filter(res.out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())
    if res.forecast_h:
        fc = res.forecast_text(label)
        if STRICT:
            assert _tail(fc) == _tail(out_text(label + ".forecast"))
        else:
            assert len(fc.splitlines()) == len(out_text(label + ".forecast").splitlines())
    rec = res.recursive_text()
    if rec is not None and STRICT:
        assert _tail(rec) == _tail(out_text(label + ".recursive"))


@pytest.mark.parametrize("label", list(RANKS))
def test_the_rank_test(label):
    t = _rank(label)
    mine = golden_filter(t.out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())
    body = theirs.split("  r    npar        logL         AIC         BIC")[1]
    for line in body.splitlines()[2:2 + t.spec.M]:
        r, npar, ll = line.split()[:3]
        assert t.npar[int(r)] == int(npar)
        assert abs(t.ll[int(r)] - float(ll)) <= 5.01e-5 + (0 if EXACT else TOL_LL)


def test_the_bootstrap(tmp_path):
    """-lrtest -bootstrap 10 on the deterministic pair: replication by
    replication (the simulated data are in w - det units, as the C's)."""
    path = tmp_path / "dboot.boot"
    t = drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, boot_path=str(path))
    theirs = boot_rows("dboot.boot")
    mine = []
    for line in open(path):
        if line[0].isdigit():
            h0, rep, st, lr = line.strip().split(",")
            mine.append((int(h0), int(rep), st, float(lr) if lr else None))
    assert [m[:3] for m in mine] == [c[:3] for c in theirs]
    for a, b in zip(mine, theirs):
        if b[3] is None:
            assert a[3] is None
        elif EXACT:
            assert a[3] == b[3]
        else:
            assert abs(a[3] - b[3]) <= TOL_LR
    if STRICT:
        assert golden_filter(t.out_text("dboot")) == golden_filter(out_text("dboot.out"))


# ---- the reader and the filter, piece by piece ---------------------------------

def test_the_reader_agrees_with_fue_where_both_read():
    """prefile.read is the C's reader; on every fixture it reads the series,
    calendar, lambda and mean flag fue.load reads."""
    import glob
    import fue
    for f in sorted(glob.glob(os.path.join(FIXTURES, "*.pre"))):
        pf = prefile.read(f)
        ts, m = fue.load(f)
        assert [float(v) for v in ts.data] == pf.data, f
        assert ts.name == pf.name and tuple(ts.start) == (pf.begyear, pf.begtime), f
        assert int(ts.freq or 1) == pf.freq and float(m.boxlam) == pf.boxlam, f
        assert bool(m.estimate_mu) == (pf.imu != 0), f
        assert len(m.interventions) == pf.ndet, f
        for iv, om, de in zip(m.interventions, pf.omega, pf.delta):
            assert list(iv.omega) == om and list(iv.delta) == de, f


def test_dated_variables_on_each_calendar():
    """gen_detvar: annual dates are the year alone, the others `sub year'."""
    v = prefile.gen_detvar("impulse 3 1970", 4, 1960, 1, 92)
    assert v.index(1.0) == 43 and sum(v) == 1.0                     # 1970Q3 is obs 43
    v = prefile.gen_detvar("step 1900", 1, 1850, 1, 62)
    assert v[51] == 1.0 and v[50] == 0.0 and v[62] == 1.0
    v = prefile.gen_detvar("compimp 1870", 1, 1850, 1, 62)
    assert (v[21], v[22]) == (1.0, -1.0)
    v = prefile.gen_detvar("ramp 6 1992", 12, 1990, 1, 40)
    assert (v[29], v[30], v[31]) == (0.0, 1.0, 2.0)
    v = prefile.gen_detvar("alter", 4, 1960, 1, 4)
    assert v[1:] == [-1.0, 1.0, -1.0, 1.0]
    # easter: 1991 is March 31 (all of March), 1992 April 19 (April)
    v = prefile.gen_detvar("easter", 12, 1991, 1, 24)
    assert v[3] == 1.0 and v[4] == 0.0 and v[16] == 1.0 and sum(v) == 2.0
    assert prefile.gen_detvar("easter", 4, 1991, 1, 8) is None     # monthly only
    assert prefile.gen_detvar("dummy7", 4, 1991, 1, 8) is None
    assert prefile.gen_detvar("impulse 1970", 4, 1960, 1, 8) is None


def test_the_rational_filter_is_exact():
    """A step through omega/(1 - delta B): omega (1 - delta^k)/(1 - delta) after
    k periods, never truncated."""
    pf = prefile.read(os.path.join(FIXTURES, "mmdet.mink.pre"))
    pf.detspec, pf.omega, pf.delta = ["step 1851"], [[0.3]], [[0.95]]
    det = prefile.det_component(pf, 200)
    for k in (1, 10, 100, 199):
        assert abs(det[k] - 0.3 * (1 - 0.95 ** k) / 0.05) < 1e-12


def test_an_unknown_variable_is_refused(tmp_path):
    src = open(os.path.join(FIXTURES, "mmpre.step.pre")).read()
    bad = tmp_path / "bad.pre"
    bad.write_text(src.replace("step 1900", "dummy 1900"))
    with pytest.raises(ValueError, match="no reconocida"):
        drvec.fit([MM[0], str(bad)], 2, 0, 1, case=2)


def test_the_factors_multiply_out():
    """expand_ar_factors: two first-order factors are not one of second order."""
    pf = prefile.read(os.path.join(FIXTURES, "xrevar0.pre"))
    assert pf.ar1 == [[0.5], [-0.2]]
    assert prefile.expand_ar(pf) == [0.5 + -0.2, -(0.5 * -0.2)]
    pm = prefile.read(os.path.join(FIXTURES, "xmon.pre"))
    ar, ma = prefile.expand_ar(pm), prefile.expand_ma(pm)
    assert len(ar) == 3 and len(ma) == 14 and ma[11] != 0.0 and ma[10] == 0.0


def test_the_backcast_without_arma_is_the_mean():
    from drvec.xlink import backcast
    assert backcast([1.0, 2.0, 4.0], [], [], 0.25, 3) == [0.25, 0.25, 0.25]

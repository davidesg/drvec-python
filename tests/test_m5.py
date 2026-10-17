"""Milestone 5: deterministic variables in a .pre, and -xpre inputs with an
ARMA model, against the oracle (tests/data/oracle/SOURCE, milestone 5; the
fixtures are tests/data/fixtures/make_m5.py's, see SOURCE there).

The reader and the univariate cast are FUE's (drvec.data.levels_cast: fue.load
and cast_us_py at the .pre's parameters, the non-stationary operator emptied),
as in drtran-python. The deterministic component follows fue's rule
everywhere: nu(B) = omega(B)/delta(B) truncated at 40 lags for a rational term,
every variable summed, one subtraction (atsw-gui lib/fuedet; fue >= 0.1.18
carries the same routine, and the oracle, drvec C 0.13, calls it: BUG-40 is
closed by agreeing with fue). The port's w is fue's own cast bit for bit, the
oracle's is the same, and every oracle case is byte-identical on Linux x86_64
with GSL -- b40 and b40rank included, the step whose tail past lag 40 reaches
0.46 (docs/PORT.md §13). The transfer functions of an -xlink input keep the
exact, untruncated filter, in the C and in the port.
"""
import copy
import os

import numpy as np
import pytest

from conftest import EXACT, FIXTURES, STRICT, TOL_EST, TOL_LL, TOL_LR
from oracle import boot_rows, golden_filter, number_after, out_text, param_table

import drvec
import drvec.data


def _fx(*names):
    return [os.path.join(FIXTURES, n) for n in names]


MM = _fx("mmpre.muskrat.pre", "mmpre.mink.pre")
MD = _fx("mmdet.muskrat.pre", "mmdet.mink.pre")
MDU = _fx("mmdet.muskrat.pre", "mmdet.minkmu.pre")
E1S = _fx("e1seas.invest.pre", "e1seas.income.pre", "e1seas.cons.pre")
MON = _fx("mmmon.muskrat.pre", "mmmon.mink.pre")
B40 = _fx("mmpre.muskrat.pre", "mmb40.mink.pre")


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
    "b40": (B40, (2, 0, 1), dict(case=2, forecast=8)),
}
RANKS = {
    "drank": (MD, (2, 0), dict(case=2)),
    "e1srank": (E1S, (2, 0), dict(case=2)),
    "dcprank": (MDU, (2, 0), dict(case="pre")),
    "b40rank": (B40, (2, 0), dict(case=2)),
}
ALL = sorted(set(FITS) | set(RANKS))
# ill-defined: several starts over some logL units, the C's best a criterion-3 stop
ILL = {"ddet211", "dstep"}
_DONE = {}


def _run(label):
    """The fit or rank test, and its products (the forecasts read Data.det_path
    when they are written)."""
    if label not in _DONE:
        if label in FITS:
            files, (p, q, r), kw = FITS[label]
            res = drvec.fit(files, p, q, r, **kw)
            res.texts = (res.out_text(label),
                         res.forecast_text(label) if res.forecast_h else None,
                         res.recursive_text())
        else:
            files, (p, q), kw = RANKS[label]
            res = drvec.rank_test(files, p, q, **kw)
            res.texts = (res.out_text(label), None, None)
        _DONE[label] = res
    return _DONE[label]


def _tail(text):
    return text.split("\n", 1)[1]


def _diff(theirs, mine):
    """The lines that differ, for an off-platform failure: the oracle's
    first, then the port's."""
    import difflib
    return "\n".join(difflib.unified_diff(theirs.splitlines(), mine.splitlines(),
                                           "oracle", "port", n=1, lineterm=""))


def _verdicts(text):
    """[(r, LR, verdict)] of the rank test's table."""
    body = text.split("  r    M-r        LR      10%      5%      1%")[1]
    out = []
    for line in body.splitlines()[2:]:
        tok = line.split()
        if not tok:
            break
        out.append((int(tok[0]), float(tok[2]), " ".join(tok[6:])))
    return out


# ---- every case: the port's w is fue's, and the products are the oracle's ----------

@pytest.mark.parametrize("label", ALL)
def test_w_is_fues_cast(label):
    """Every column of the port's levels is fue's own cast: differenced as fue
    differences it (cast_us_py on the .pre's model, untouched), it is fue's
    stationary w bit for bit."""
    import fue
    from fue.cast_us import _build_initial_x, build_est_spec, cast_us_py
    files = (FITS[label] if label in FITS else RANKS[label])[0]
    data = drvec.read_pre(files)
    for i, f in enumerate(files):
        m = fue.load(f)[1]
        w = cast_us_py(_build_initial_x(m), build_est_spec(m))[5]
        col = data.raw[:, i]
        z = col if m.d == 0 else np.diff(col)
        off = len(w) - len(z)                        # every file here ends together
        assert np.array_equal(z, w[off:]), f


@pytest.mark.parametrize("label", sorted(FITS))
def test_fit(label):
    res = _run(label)
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


@pytest.mark.parametrize("label", sorted(RANKS))
def test_rank_test(label):
    """The rank test's npar and logL per rank, and its verdicts. b40rank is
    BUG-40's case: the step's tail past lag 40 moved LR(0) across the 10%
    critical value between the old oracle (exact filter, 13.2989, "H0 not
    rejected") and fue's rule; the oracle now applies fue's rule too."""
    t = _run(label)
    theirs = golden_filter(out_text(label + ".out"))
    body = theirs.split("  r    npar        logL         AIC         BIC")[1]
    for line in body.splitlines()[2:2 + t.spec.M]:
        r, npar, ll = line.split()[:3]
        assert t.npar[int(r)] == int(npar)
        assert abs(t.ll[int(r)] - float(ll)) <= 5.01e-5 + (0 if EXACT else TOL_LL)
    vm, vt = _verdicts(golden_filter(t.texts[0])), _verdicts(theirs)
    assert [(v[0], v[2]) for v in vm] == [(v[0], v[2]) for v in vt]
    if label == "b40rank":
        assert vt[0][1:] == (13.7873, "reject H0 at 10%")


@pytest.mark.parametrize("label", ALL)
def test_products(label):
    """.out, .forecast and .recursive byte-identical (golden.sh's filter) on
    the oracle's platform; elsewhere the same number of lines."""
    out, fc, rec = _run(label).texts
    mine = golden_filter(out)
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        if len(mine.splitlines()) != len(theirs.splitlines()):
            print(_diff(theirs, mine))       # in full: pytest truncates the message
        assert len(mine.splitlines()) == len(theirs.splitlines())
    if fc is not None:
        if STRICT:
            assert _tail(fc) == _tail(out_text(label + ".forecast"))
        else:
            assert len(fc.splitlines()) == len(out_text(label + ".forecast").splitlines())
    if rec is not None and STRICT:
        assert _tail(rec) == _tail(out_text(label + ".recursive"))


def test_the_bootstrap(tmp_path):
    """-lrtest -bootstrap 10 on the deterministic pair, replication by
    replication."""
    theirs = boot_rows("dboot.boot")
    path = tmp_path / "dboot.boot"
    t = drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, boot_path=str(path))
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


# ---- fue's reader and cast, piece by piece ----------------------------------------

def _indicator(model, itv, T):
    """fue's indicator of a deterministic variable over T periods (index 0 is
    the presample slot)."""
    from fue.cast_us import _build_indicator
    ts = model.series
    freq = int(ts.freq) if ts.freq else 1
    return _build_indicator(itv, T, freq, ts.start[1] if ts.start else 1,
                            ts.start[0] if ts.start else None)


def test_dated_variables_on_each_calendar():
    """fue's indicators place a dated term as the C's gen_detvar does: annual
    dates are the year alone, the others `sub year'."""
    import fue
    m = fue.load(_fx("e1seas.invest.pre")[0])[1]
    v = _indicator(m, m.interventions[3], 92)              # impulse 3 1970
    assert list(v).index(1.0) == 43 and sum(v) == 1.0
    m = fue.load(_fx("mmpre.step.pre")[0])[1]
    v = _indicator(m, m.interventions[0], 62)              # step 1900
    assert v[51] == 1.0 and v[50] == 0.0 and v[62] == 1.0
    m = fue.load(_fx("mmdet.mink.pre")[0])[1]
    v = _indicator(m, m.interventions[2], 62)              # compimp 1870
    assert (v[21], v[22]) == (1.0, -1.0)
    m = fue.load(_fx("mmmon.mink.pre")[0])[1]
    v = _indicator(m, m.interventions[1], 40)              # ramp 3 1993
    assert (v[38], v[39], v[40]) == (0.0, 1.0, 2.0)
    m = fue.load(_fx("mmdet.muskrat.pre")[0])[1]
    v = _indicator(m, m.interventions[2], 4)               # alter
    assert list(v[1:]) == [-1.0, 1.0, -1.0, 1.0]
    m = fue.load(_fx("mmmon.muskrat.pre")[0])[1]                # easter from 1990/1
    v = _indicator(m, m.interventions[0], 24)
    assert v[4] == 1.0 and v[15] == 1.0 and sum(v) == 2.0       # 1990 Apr 15, 1991 Mar 31


def test_the_40_lag_rule():
    """A step through 0.3/(1 - 0.95 B): the component is the exact filter up
    to 40 periods after the step, and from then on falls short by the tail
    0.3 (0.95^41 - 0.95^(k+1)) / 0.05 -- fue's rule, nu(B) truncated at 40
    lags, which the oracle (drvec C 0.13, lib/fuedet) applies too: BUG-40
    closed by agreeing with fue. b40/b40rank are this file."""
    import fue
    m = fue.load(_fx("mmb40.mink.pre")[0])[1]
    mine = np.array(drvec.data.det_path(m, 0, 70))
    k0 = m.interventions[0].at                       # 0-based row of the step
    for k in (0, 10, 40, 41, 51, 59):
        exact = 0.3 * (1 - 0.95 ** (k + 1)) / 0.05
        tail = 0.3 * (0.95 ** 41 - 0.95 ** (k + 1)) / 0.05 if k > 40 else 0.0
        assert abs(mine[k0 + k] - (exact - tail)) < 1e-12, k


def test_an_unknown_variable_is_refused_as_fue_refuses_it(tmp_path):
    """fue reads an unknown word as a non-standard variable and asks for its
    data column; `time' (the C's synonym of trend) is one of them."""
    src = open(_fx("mmpre.step.pre")[0]).read()
    for word in ("dummy 1900", "time"):
        bad = tmp_path / "bad.pre"
        bad.write_text(src.replace("step 1900", word))
        with pytest.raises(ValueError, match="se esperaban 2 columnas"):
            drvec.fit([MM[0], str(bad)], 2, 0, 1, case=2)


def test_a_nonstandard_variable_is_refused_as_the_c_refuses_it(tmp_path):
    """With its data column fue reads it; drvec refuses it by design (a
    transfer input is drtran's job), with the C's message."""
    lines = open(_fx("mmpre.step.pre")[0]).read().split("\n")
    i = next(k for k, x in enumerate(lines) if x.startswith("** Series"))
    out = [x.replace("step 1900", "dummy") for x in lines[:i + 1]]
    out += [x + "  0.0" if x.strip() else x for x in lines[i + 1:]]
    bad = tmp_path / "custom.pre"
    bad.write_text("\n".join(out))
    with pytest.raises(ValueError, match="NO ESTÁNDAR"):
        drvec.fit([MM[0], str(bad)], 2, 0, 1, case=2)


def test_easter_off_monthly_data_is_refused(tmp_path):
    """fue would carry it as an identically null regressor; the C refuses it."""
    src = open(_fx("mmpre.step.pre")[0]).read()
    bad = tmp_path / "e.pre"
    bad.write_text(src.replace("step 1900", "easter"))
    with pytest.raises(ValueError, match="easter is monthly only"):
        drvec.fit([MM[0], str(bad)], 2, 0, 1, case=2)


def test_the_factors_multiply_out():
    """fue's cast multiplies the factors out: two first-order factors are not
    one of second order; the annual MA lands at lag 12, the fixed-frequency
    AR(2)/MA(2) add two lags each."""
    import fue
    phi, theta, mu, _z = drvec.data.levels_cast(fue.load(_fx("xrevar0.pre")[0])[1])
    assert phi == [0.5 + -0.2, -(0.5 * -0.2)] and theta == [0.4] and mu == 10.6
    phi, theta, _mu, _z = drvec.data.levels_cast(fue.load(_fx("xmon.pre")[0])[1])
    assert len(phi) == 3 and len(theta) == 14 and theta[11] != 0.0 and theta[10] == 0.0


def test_the_levels_cast_leaves_the_model_alone():
    """levels_cast casts a COPY with the operator emptied (drtran's
    _alt_est_specs pattern): the model fue read keeps its d."""
    import fue
    m = fue.load(_fx("mmdet.muskrat.pre")[0])[1]
    before = copy.deepcopy((m.d, m.D, m.ifadf))
    drvec.data.levels_cast(m)
    assert (m.d, m.D, m.ifadf) == before


def test_the_backcast_without_arma_is_the_mean():
    from drvec.xlink import backcast
    assert backcast([1.0, 2.0, 4.0], [], [], 0.25, 3) == [0.25, 0.25, 0.25]

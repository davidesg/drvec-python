"""Milestone 5: deterministic variables in a .pre, and -xpre inputs with an
ARMA model, against the oracle (tests/data/oracle/SOURCE, milestone 5; the
fixtures are tests/data/fixtures/make_m5.py's, see SOURCE there).

The reader and the univariate cast are FUE's (drvec.data.levels_cast: fue.load
and cast_us_py at the .pre's parameters, the non-stationary operator emptied),
as in drtran-python. The C drvec builds the deterministic component its own
way: the exact omega(B)/delta(B) by its recursion, the variables summed and
then subtracted. fue convolves with nu(B), truncated at 40 lags for a rational
term (BUG-40, WON'T FIX in the C), and subtracts one variable at a time. So the
oracle cases fall in two groups (docs/PORT.md §13):

* BYTE: every file of the case has no deterministic variable or a single one
  with no delta. fue's w and the C's are the same bits, and every product is
  byte-identical to the oracle on Linux x86_64 with GSL.
* FUE: some file has two or more variables or a rational one. Its w differs
  from the C's: by the BUG-40 tail sum_{k>40} nu_k I_{t-k} where a delta term
  reaches past lag 40 (mmdet.muskrat's cos with delta, ~1e-12; mmb40.mink's
  step with delta 0.95, up to 0.46; mmmon.muskrat's step over dfcmon's
  horizon, ~1e-5), and elsewhere by rounding only (a few ulps). An ulp in w is
  enough to move the optimiser's path, so the .out is not byte-identical even
  where the numbers agree to every printed digit but the last. Tested instead:
  (a) the w the port uses is fue's own cast z, bit for bit; (b) the oracle's
  component minus fue's is the BUG-40 tail plus rounding; (c) fed the oracle's
  w (tests/cdet.py), the port reproduces the oracle byte for byte -- so the
  component is the only difference --; and on fue's w the .out has the
  oracle's sections, npar, ranks and decisions, with the logL to 1e-8 where the
  tail is below 1e-11 (measured: 7e-11 at most, the optimiser's stop), and on
  b40 within the tail's first-order effect plus a second-order re-optimisation.
"""
import copy
import os

import numpy as np
import pytest

import cdet
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
# fue's w is the C's, bit for bit: no deterministic variable, or one with no delta
BYTE = {"dstep", "xarma", "xarma1", "xar0"}
FUE = set(ALL) - BYTE
# a delta term reaching past lag 40 in the sample or the horizon (BUG-40's tail)
BUG40 = {"ddet211", "dfc", "dcpfc", "dfdh", "dcpre", "dxsys", "xdet", "drank",
         "dcprank", "dfcmon", "b40", "b40rank"}
# ill-defined: four starts over 4-5 logL units, the C's best a criterion-3 stop
ILL = {"ddet211", "dstep"}
_DONE = {}


def _run(label, oracle_det=False):
    """The fit or rank test, and its products written under the same component
    (the forecasts read Data.det_path when they are written)."""
    key = (label, oracle_det)
    if key not in _DONE:
        with pytest.MonkeyPatch.context() as mp:
            if oracle_det:
                cdet.oracle_det(mp)
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
        _DONE[key] = res
    return _DONE[key]


def _tail(text):
    return text.split("\n", 1)[1]


def _sections(text):
    return [line for line in text.splitlines() if line.startswith("===")]


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


def _byte_checks(res, label):
    out, fc, rec = res.texts
    mine = golden_filter(out)
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())
    if fc is not None:
        if STRICT:
            assert _tail(fc) == _tail(out_text(label + ".forecast"))
        else:
            assert len(fc.splitlines()) == len(out_text(label + ".forecast").splitlines())
    if rec is not None and STRICT:
        assert _tail(rec) == _tail(out_text(label + ".recursive"))


def _models(label):
    import fue
    files, _o, kw = FITS[label] if label in FITS else RANKS[label]
    return [fue.load(f)[1] for f in list(files) + list(kw.get("xpre", []))]


def _horizon(label):
    return FITS[label][2].get("forecast", 0) if label in FITS else 0


# ---- the classification itself --------------------------------------------------

@pytest.mark.parametrize("label", ALL)
def test_the_groups(label):
    """BYTE iff every file's fue w is the C's bit for bit; BUG40 iff a delta
    term's tail is non-zero over the sample or the forecast horizon."""
    models = _models(label)
    same_bits = all(np.array_equal(drvec.data.levels_cast(m)[3], cdet.exact_w(m))
                    for m in models)
    assert same_bits == (label in BYTE)
    tail = max(float(np.abs(cdet.bug40_tail(m, int(m.series.nobs) + _horizon(label)))
                     .max(initial=0.0)) for m in models)
    assert (tail > 0.0) == (label in BUG40)


# ---- BYTE: byte for byte ---------------------------------------------------------

@pytest.mark.parametrize("label", sorted(BYTE))
def test_byte_group_fit(label):
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


@pytest.mark.parametrize("label", sorted(BYTE))
def test_byte_group_products(label):
    _byte_checks(_run(label), label)


# ---- FUE: (a) fue's w, (b) the tail, (c) the oracle on the oracle's w --------------

@pytest.mark.parametrize("label", sorted(FUE))
def test_fue_group_w_is_fues_cast(label):
    """(a) Every column of the port's levels is fue's own cast: differenced as
    fue differences it (cast_us_py on the .pre's model, untouched), it is fue's
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


@pytest.mark.parametrize("label", sorted(FUE))
def test_fue_group_the_component_differs_by_the_tail(label):
    """(b) The oracle's component minus fue's, over the sample and the horizon:
    the BUG-40 tail, plus rounding (a few ulps of the component)."""
    for m in _models(label):
        if not m.interventions:
            continue
        T = int(m.series.nobs) + _horizon(label)
        exact = np.array(cdet.exact_det(m, T))
        mine = np.array(drvec.data.det_path(m, 0, T))
        tail = cdet.bug40_tail(m, T)
        assert np.abs((exact - mine) - tail).max() <= 1e-14 * max(1.0, np.abs(exact).max())


@pytest.mark.parametrize("label", sorted(FUE))
def test_fue_group_on_the_oracles_w_is_the_oracle(label):
    """(c) Fed the oracle's component, the port is the oracle: every product
    byte-identical. What moves on fue's w moves because of the w alone."""
    res = _run(label, oracle_det=True)
    if label in ILL:
        assert res.termcode == 3
        assert "NOT a convergence" in res.texts[0]
    _byte_checks(res, label)


@pytest.mark.parametrize("label", sorted(set(FITS) & FUE))
def test_fue_group_fit(label):
    """(c) On fue's w: the oracle's sections and npar, and the logL."""
    res = _run(label)
    theirs = out_text(label + ".out")
    assert _sections(golden_filter(res.texts[0])) == _sections(golden_filter(theirs))
    assert len(res.params) == len(param_table(theirs))
    ll_c = number_after(theirs, "logelf           :")
    if label in ILL:
        # a ridge: an ulp in w changes which start wins; the port's is not lower
        assert res.loglik >= ll_c - 1e-6
    elif label == "b40":
        _b40_logl(res, ll_c)
    else:
        assert abs(res.loglik - ll_c) <= (1e-8 if EXACT else TOL_LL)


def _b40_logl(res, ll_c):
    """BUG-40 at its size: the tail reaches 0.46 (0.3 (0.95^41 - 0.95^(m+1)) /
    0.05). By the envelope theorem the logL moves, to first order, by the
    change of the data at the C's optimum, l(x_C; w_fue) - l(x_C; w_C); the
    re-optimisation adds a second-order, non-negative term."""
    from drvec.loglik import loglik
    c = _run("b40", oracle_det=True)
    first = loglik(c.spec, res.data.at(c.spec.r), c.x)[0] - c.loglik
    second = res.loglik - (c.loglik + first)
    assert 0.40 < first < 0.50                      # measured 0.4630
    assert -1e-9 <= second < 0.02                   # measured 0.0093
    assert abs(c.loglik - ll_c) < 1e-9


@pytest.mark.parametrize("label", sorted(set(RANKS) & FUE))
def test_fue_group_rank_test(label):
    """(c) The rank test on fue's w: the oracle's sections, npar and logL per
    rank, and its verdicts -- but on b40rank, where the tail flips one."""
    t = _run(label)
    mine = golden_filter(t.texts[0])
    theirs = golden_filter(out_text(label + ".out"))
    assert _sections(mine) == _sections(theirs)
    body = theirs.split("  r    npar        logL         AIC         BIC")[1]
    for line in body.splitlines()[2:2 + t.spec.M]:
        r, npar, ll = line.split()[:3]
        assert t.npar[int(r)] == int(npar)
        if label != "b40rank":
            assert abs(t.ll[int(r)] - float(ll)) <= 5.01e-5 + (0 if EXACT else TOL_LL)
    vm, vt = _verdicts(mine), _verdicts(theirs)
    assert [v[0] for v in vm] == [v[0] for v in vt]
    if label == "b40rank":
        # the tail moves LR(0) from 13.2989 to 13.7873, across the 10% critical
        # value 13.75: here BUG-40 flips the verdict at 10%
        assert vt[0][2] == "H0 not rejected" and vm[0][2] == "reject H0 at 10%"
        assert 0.4 < vm[0][1] - vt[0][1] < 0.6
    else:
        assert [v[2] for v in vm] == [v[2] for v in vt]


def test_the_bootstrap(tmp_path):
    """-lrtest -bootstrap 10 on the deterministic pair (FUE, BUG40): on the
    oracle's w, replication by replication; on fue's, the same statuses and
    each LR* within 1e-6."""
    theirs = boot_rows("dboot.boot")
    for oracle_det in (True, False):
        path = tmp_path / f"dboot{int(oracle_det)}.boot"
        with pytest.MonkeyPatch.context() as mp:
            if oracle_det:
                cdet.oracle_det(mp)
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
            elif EXACT and oracle_det:
                assert a[3] == b[3]
            else:
                assert abs(a[3] - b[3]) <= (TOL_LR if oracle_det else 1e-6)
        if STRICT and oracle_det:
            assert golden_filter(t.out_text("dboot")) == golden_filter(out_text("dboot.out"))


# ---- fue's reader and cast, piece by piece ----------------------------------------

def test_dated_variables_on_each_calendar():
    """fue's indicators place a dated term as the C's gen_detvar does: annual
    dates are the year alone, the others `sub year'."""
    import fue
    m = fue.load(_fx("e1seas.invest.pre")[0])[1]
    v = cdet._indicator(m, m.interventions[3], 92)              # impulse 3 1970
    assert list(v).index(1.0) == 43 and sum(v) == 1.0
    m = fue.load(_fx("mmpre.step.pre")[0])[1]
    v = cdet._indicator(m, m.interventions[0], 62)              # step 1900
    assert v[51] == 1.0 and v[50] == 0.0 and v[62] == 1.0
    m = fue.load(_fx("mmdet.mink.pre")[0])[1]
    v = cdet._indicator(m, m.interventions[2], 62)              # compimp 1870
    assert (v[21], v[22]) == (1.0, -1.0)
    m = fue.load(_fx("mmmon.mink.pre")[0])[1]
    v = cdet._indicator(m, m.interventions[1], 40)              # ramp 3 1993
    assert (v[38], v[39], v[40]) == (0.0, 1.0, 2.0)
    m = fue.load(_fx("mmdet.muskrat.pre")[0])[1]
    v = cdet._indicator(m, m.interventions[2], 4)               # alter
    assert list(v[1:]) == [-1.0, 1.0, -1.0, 1.0]
    m = fue.load(_fx("mmmon.muskrat.pre")[0])[1]                # easter from 1990/1
    v = cdet._indicator(m, m.interventions[0], 24)
    assert v[4] == 1.0 and v[15] == 1.0 and sum(v) == 2.0       # 1990 Apr 15, 1991 Mar 31


def test_bug40_is_fues_truncation():
    """A step through 0.3/(1 - 0.95 B): fue's component is the exact one up to
    40 periods after the step, and from then on falls short by the tail
    0.3 (0.95^41 - 0.95^(k+1)) / 0.05 (BUG-40, WON'T FIX in the C)."""
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

"""-fdhess: the standard errors from the finite-difference Hessian at the
optimum (exact_hessian_se, drvec.c:1910), against the oracle's .out
(tests/data/oracle, SOURCE):

  fdh201  mink-muskrat 2 0 1 -case 2 -fdhess     the Hessian is used
  e1fdh   E1           2 0 1 -case 2 -fdhess     the Hessian is used
  fdh211  mink-muskrat 2 1 1 -mean -case 2 -fdhess
          the optimum is ON the MA invertibility wall (BUG-49): 42 finite-
          difference steps leave the admissible region and the BFGS s.e. are
          kept, with the .out saying why (BUG-34)

On the oracle's platform the .out is byte-identical. Elsewhere the s.e. are
compared on the relative tier TOL_FDSE_REL (conftest.py) -- they are path
independent but carry a numerical noise floor, measured below -- and the
DECISION (Hessian used or not) is asserted everywhere."""
import numpy as np
import pytest

from conftest import E1, EXACT, PRE, STRICT, TOL_EST, TOL_FDSE_REL, TOL_LL
from oracle import golden_filter, number_after, out_text, param_table

import drvec

CASES = {
    "fdh201": (PRE, (2, 0, 1)),
    "e1fdh": (E1, (2, 0, 1)),
    "fdh211": (PRE, (2, 1, 1)),
}
_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, (p, q, r) = CASES[label]
        _FITS[label] = drvec.fit(files, p, q, r, case=2, fdhess=True)
    return _FITS[label]


@pytest.mark.parametrize("label", ["fdh201", "e1fdh"])
def test_the_hessian_standard_errors(label):
    res = _fit(label)
    out = out_text(label + ".out")
    assert res.fdhess.ok and res.fdhess.nrej == 0
    assert "Standard errors  : finite-difference Hessian at the optimum (-fdhess)" in out
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    table = param_table(out)
    for p in res.params:
        est, se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if STRICT else TOL_EST), p.label
        if STRICT:
            assert abs(p.se - se) <= 5.01e-7, p.label
        else:
            assert abs(p.se - se) <= TOL_FDSE_REL * se + 5.01e-7, p.label


def test_on_the_boundary_the_bfgs_ones_are_kept():
    """fdh211: the decision, everywhere; the count of refused steps and the
    s.e. themselves only on the oracle's platform."""
    res = _fit("fdh211")
    assert not res.fdhess.ok and res.fdhess.nrej > 0
    assert "on the boundary" in res.fdhess.why
    assert "42 of the finite-difference" in res.fdhess.warning(res.npar) or not EXACT
    plain = drvec.fit(PRE, 2, 1, 1, case=2)
    if EXACT:
        assert res.se == plain.se                   # the BFGS ones, untouched


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_fit(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


def test_noise_floor():
    """The measurement behind TOL_FDSE_REL: x moved by ~1 ulp moves the s.e. by
    up to a few 1e-3 relative, and no more than the tier allows."""
    from drvec.fdhess import exact_hessian_se
    res = _fit("e1fdh")
    rd = res.data.at(1)
    base = np.array(res.se)
    worst = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = np.array(res.x) * (1 + 4e-16 * rng.standard_normal(len(res.x)))
        fh = exact_hessian_se(res.spec, rd, list(x))
        assert fh.ok
        worst = max(worst, float(np.max(np.abs(fh.dev - base) / base)))
    assert 1e-5 < worst < TOL_FDSE_REL / 2

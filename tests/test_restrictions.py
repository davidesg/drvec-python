"""Restrictions on alpha and beta, with their LR tests against the free model as
the C prints them (tests/data/oracle, SOURCE):

  weakex1, weakex2  mink-muskrat 2 0 1 -case 2 -weakex i
  e1alpha           E1 2 0 1 -case 2 -alpha e1alpha.txt (alpha_invest = alpha_income)
  fixb2v            mink-muskrat 2 0 1 -case 2 -fixb2 0      (the `ols-b2' start)
  fixb2s            mink-muskrat 2 0 1 -case 2 -fixb2        (B2 at static OLS)
  wxfix             mink-muskrat 2 0 1 -case 2 -weakex 1 -fixb2 0
  wxrow             E1 2 0 1 -case 2 -weakex 3 -fixb2row 1 0  (the joint LR)

Every one of them is a well-defined optimum: all the starts of the search end
at the same logL (to the printed six decimals)."""
import os

import pytest

from conftest import E1, EXACT, FIXTURES, PRE, STRICT, TOL_EST, TOL_LL, TOL_SE
from oracle import golden_filter, number_after, out_text, param_table

import drvec

ALPHA = os.path.join(FIXTURES, "e1alpha.txt")
CASES = {
    "weakex1": (PRE, dict(weakex=1)),
    "weakex2": (PRE, dict(weakex=2)),
    "e1alpha": (E1, dict(alpha=ALPHA)),
    "fixb2v": (PRE, dict(fixb2=0.0)),
    "fixb2s": (PRE, dict(fixb2=True)),
    "wxfix": (PRE, dict(weakex=1, fixb2=0.0)),
    "wxrow": (E1, dict(weakex=3, fixb2row={1: 0.0})),
}
_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, kw = CASES[label]
        _FITS[label] = drvec.fit(files, 2, 0, 1, case=2, **kw)
    return _FITS[label]


@pytest.mark.parametrize("label", list(CASES))
def test_logl_and_estimates(label):
    res = _fit(label)
    out = out_text(label + ".out")
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    table = param_table(out)
    assert len(table) == res.npar
    for p in res.params:
        est, se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if STRICT else TOL_EST), p.label
        if EXACT:
            assert abs(p.se - se) <= 5.01e-7, p.label


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_fit(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


@pytest.mark.parametrize("label,key,df", [
    ("weakex1", "logL H(r)  free        :", 1), ("weakex2", "logL H(r)  free        :", 1),
    ("e1alpha", "logL H(r)  free        :", 1), ("wxfix", "logL H(r)  free        :", 1),
    ("wxrow", "logL H(r)  free        :", 2)])
def test_the_lr_against_the_free_model(label, key, df):
    res = _fit(label)
    out = out_text(label + ".out")
    assert abs(res.lr_free - number_after(out, key)) < (1e-6 if EXACT else TOL_LL)
    lr = 2.0 * (res.lr_free - res.loglik)
    assert abs(lr - number_after(out, "LR = 2(free - restr.)  :")) < (1e-6 if EXACT else 2 * TOL_LL)
    assert ("degrees of freedom     : %d " % df) in out


def test_the_restricted_vector():
    """weakex1: alpha's muskrat row is zero; fixb2: B2 is out of the vector."""
    res = _fit("weakex1")
    assert [p.label for p in res.params if p.kind == "psi"] == ["psi[1 <- ec1]"]
    assert res.spec.alpha == ((0.0,), (1.0,))
    assert res.npar == 9 and _fit("fixb2v").npar == 9
    assert not [p for p in _fit("fixb2s").params if p.kind == "beta"]
    assert _fit("fixb2v").search.candidates[2].label == "ols-b2"
    assert [c.label for c in _fit("fixb2s").search.candidates] == ["seed", "cold"]


def test_refusals():
    with pytest.raises(ValueError, match="-weakex 3 is outside 1..2"):
        drvec.fit(PRE, 2, 0, 1, case=2, weakex=3)
    with pytest.raises(ValueError, match="means nothing with r = 0"):
        drvec.fit(PRE, 2, 0, 0, case=2, weakex=1)
    with pytest.raises(ValueError, match="-xsys embeds an input"):
        drvec.fit(PRE, 3, 0, 1, case=2, xsys=[1], weakex=2)
    with pytest.raises(ValueError, match="cannot be\n       combined with -fixb2"):
        drvec.fit(PRE, 2, 0, 1, case=2, fixb2=0.0, fixb2row={1: 0.0})
    with pytest.raises(ValueError, match="A does not have rank 2"):
        drvec.fit(E1, 2, 0, 1, case=2, alpha=[[1, 1], [1, 1], [0, 0]])
    with pytest.raises(ValueError, match="must start with 'M sa'"):
        drvec.fit(PRE, 2, 0, 1, case=2, alpha=ALPHA)

"""Milestone 4: the entry gate of a PLAIN fit at the diagonal rung (r = 0 with
-diagar -diagma -diagcov), "=== Entry gate: the factorisation contract ===",
with its optimality certificate (the logL at the values brought in against the
fit's), against the oracle's dgate, dgate2, dgateq, dgatepq and e1gate
(tests/data/oracle/SOURCE, milestone 4)."""
import os

import pytest

from conftest import E1, EXACT, FIXTURES, STRICT, TOL_LL
from oracle import golden_filter, number_after, out_text

import drvec

MM = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "mmpre.mink.pre")]
MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]
DIAG = dict(diag_ar=True, ma="diag", diag_cov=True)

CASES = {
    "dgate": (MU, (2, 0), "pre"),
    "dgate2": (MM, (2, 0), 2),
    "dgateq": (MM, (2, 1), 2),
    "dgatepq": (MU, (2, 1), "pre"),
    "e1gate": (E1, (2, 1), 2),
}

_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, (p, q), case = CASES[label]
        _FITS[label] = drvec.fit(files, p, q, 0, case=case, **DIAG)
    return _FITS[label]


def _section(text):
    a = text.index("=== Entry gate: the factorisation contract ===")
    return text[a:text.index("=====", a + 50)]


@pytest.mark.parametrize("label", list(CASES))
def test_the_gate_and_its_certificate(label):
    res = _fit(label)
    g = res.gate
    sec = _section(out_text(label + ".out"))
    tol = 5.01e-11 if EXACT else TOL_LL
    assert abs(g.sum - number_after(sec, "sum  =")) <= tol
    assert abs(g.joint - number_after(sec, "joint=")) <= tol
    assert abs(g.ll_start - number_after(sec, "logL AT the values brought in   =")) <= tol
    assert abs(g.fit_ll - number_after(sec, "logL of the diagonal fit        =")) <= tol
    # the decisions: the .pre route's start (init_guess, not the .pre models)
    # is not the univariate optimum, and the contract holds -- except on E1,
    # whose diagonal fit stops on criterion 3 (the line search) and misses the
    # gate by a hair in the C itself (-4.120e-06 against 1e-6, as e1rungs' gate,
    # TODO.md): there the oracle's verdict on its platform, and elsewhere the
    # verdict the port's own gap gives, which rounding can flip
    if label == "e1gate":
        assert g.verified == (abs(g.gap) < 1e-6)
        if EXACT:
            assert res.termcode == 3
            assert not g.verified and "*** NOT VERIFIED ***" in sec
    else:
        assert g.verified and "VERIFIED" in sec
    assert g.fit_ll - g.ll_start > 1e-3 and "A SPECIFICATION" in sec


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_fit(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    elif label == "e1gate" and _fit(label).gate.verified:
        # the verdict flipped off the oracle's platform: the C's paragraph on
        # a failed contract (7 lines) is not written
        assert len(mine.splitlines()) == len(theirs.splitlines()) - 7
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


def test_no_gate_off_the_diagonal_rung():
    """Only at r = 0 with all three flags: the factorisation holds nowhere else."""
    assert drvec.fit(MM, 2, 0, 0, case=2, diag_ar=True, diag_cov=True).gate is None
    assert drvec.fit(MM, 2, 0, 1, case=2, **DIAG).gate is None

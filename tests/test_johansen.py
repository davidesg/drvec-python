"""Johansen's canonical seed: the C solves its eigenproblem with GSL's
gsl_eigen_gensymmv. The port calls the same routine from drvarma's compiled
engine (drvec._gsl) and the seed is the C's bit for bit; without that engine,
scipy's eigh is a
few ulps off, and the effect is measured and bounded here."""
import pytest

from conftest import EXACT, PRE, fragile
from oracle import est_pairs

import drvec
from drvec._gsl import backend
from drvec.fit import est_run
from drvec.start import canonical_b2, init_guess

DATA = drvec.read_pre(PRE)
GSL = backend() == "gsl"


def test_canonical_b2():
    for q in (0, 1):
        spec = drvec.Spec(2, 2, q, 1, 2)
        b2 = canonical_b2(spec, DATA.at(1))[0][0]
        if GSL and EXACT:
            assert b2 == 0.4194721834180301                # the C's
        else:
            assert abs(b2 - 0.4194721834180301) < 1e-12    # a few ulps off


@fragile
def test_the_johansen_fits():
    """case 1 (pair 14) and case 2, r = 1 (pair 10). With scipy: 9.5e-8 in logL
    on case 1 (a tc-3 stop amplifies the 2 ulps) and ~6e-14 on case 2."""
    for run, k, q, tol in (("proute", 14, 1, 1e-6), ("rankboot", 10, 0, 1e-12)):
        spec = drvec.Spec(2, 2, q, 1, 2)
        start, ll_c, _s, _i, x_c = est_pairs(run)[k]
        mine = init_guess(spec, DATA.at(1), seedjoh=True)
        er = est_run(spec, DATA.at(1), mine)
        if GSL:
            assert (mine, er.x, er.ll) == (start, x_c, ll_c)
        else:
            assert abs(er.ll - ll_c) < tol
            assert max(abs(a - b) for a, b in zip(er.x, x_c)) < 1e-6


@pytest.mark.skipif(not GSL, reason="drvarma's engine has no GSL: scipy's eigh is in use")
def test_scipy_is_still_within_a_few_ulps(monkeypatch):
    monkeypatch.setenv("DRVEC_EIGEN", "scipy")
    b2 = canonical_b2(drvec.Spec(2, 2, 0, 1, 2), DATA.at(1))[0][0]
    assert abs(b2 - 0.4194721834180301) < 1e-12

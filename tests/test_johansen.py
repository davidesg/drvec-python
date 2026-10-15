"""The one piece that is not bit-exact: Johansen's canonical seed, solved with
scipy's eigh where the C uses GSL's gsl_eigen_gensymmv. Measured, and bounded."""
from conftest import PRE
from oracle import est_pairs

import drvec
from drvec.fit import est_run
from drvec.start import canonical_b2, init_guess

DATA = drvec.read_pre(PRE)


def test_canonical_b2_is_within_a_few_ulps():
    for q in (0, 1):
        spec = drvec.Spec(2, 2, q, 1, 2)
        b2 = canonical_b2(spec, DATA.at(1))[0][0]
        assert abs(b2 - 0.4194721834180301) < 1e-15       # the C's (2 ulps off)


def test_the_johansen_fits_land_within_tolerance():
    """case 1 (pair 14): 9.5e-8 in logL (a tc-3 stop amplifies the 2 ulps);
    case 2, r = 1 (pair 10): ~6e-14. Neither start wins its search."""
    for run, k, q, tol in (("proute", 14, 1, 1e-6), ("rankboot", 10, 0, 1e-12)):
        spec = drvec.Spec(2, 2, q, 1, 2)
        _start, ll_c, _s, _i, x_c = est_pairs(run)[k]
        er = est_run(spec, DATA.at(1), init_guess(spec, DATA.at(1), seedjoh=True))
        assert abs(er.ll - ll_c) < tol
        assert max(abs(a - b) for a, b in zip(er.x, x_c)) < 1e-6

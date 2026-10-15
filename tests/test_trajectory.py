"""(b) est() and the cold start against the C, call by call: the same start
gives the same final vector, bit for bit (drvarma's raxopt + the cast)."""
import pytest

from conftest import PRE
from oracle import est_pairs, specs

import drvec
from drvec.fit import est_run
from drvec.start import init_guess

DATA = drvec.read_pre(PRE)
PROUTE = est_pairs("proute")
SP = specs("proute")


def _spec(k):
    r, q, ma, dar, dcov = SP[str(k)]
    return drvec.Spec(2, 2, q, r, 2, ma, dar, dcov)


# a few of each kind: rung 0, rung 2, the r = 1 winner (66 iterations), a tc-3
# stop, the matri and marow classes and the q = 0 class of the nested chain
@pytest.mark.parametrize("k", [0, 7, 13, 12, 15, 17, 19])
def test_est_from_the_c_start(k):
    spec = _spec(k)
    start, ll_c, _s2, _ifa, x_c = PROUTE[k]
    er = est_run(spec, DATA.at(spec.r), start)
    assert er.ifault == 0
    assert er.x == x_c
    assert er.ll == ll_c


@pytest.mark.parametrize("k", [0, 1, 4, 5, 8, 9, 13, 15, 17, 19])
def test_cold_start_is_the_c_one(k):
    """Pairs whose start is init_guess (the C's cold start of that model)."""
    spec = _spec(k)
    assert init_guess(spec, DATA.at(spec.r)) == PROUTE[k][0]

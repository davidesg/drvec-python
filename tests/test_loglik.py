"""(a) The cast + drvarma's elf at the oracle's full-precision optima: the logL
the C printed for every est() call, exactly."""
import pytest

from conftest import PRE, same
from oracle import est_pairs, specs

import drvec
from drvec.loglik import loglik

DATA = drvec.read_pre(PRE)


def _cases():
    for run in ("proute", "rankboot"):
        pairs = est_pairs(run)
        for k, sp in specs(run).items():
            if sp == "hold":
                continue
            yield run, int(k), sp, pairs[int(k)]


@pytest.mark.parametrize("run,k,sp,pair", list(_cases()),
                         ids=[f"{c[0]}-{c[1]}" for c in _cases()])
def test_logl_at_the_c_optimum(run, k, sp, pair):
    r, q, ma, dar, dcov = sp
    spec = drvec.Spec(2, 2, q, r, 2, ma, dar, dcov)
    _start, ll_c, s2_c, _ifa, x = pair
    assert len(x) == spec.npar()
    ll, s2, ifa = loglik(spec, DATA.at(r), x)
    assert ifa == 0
    assert same(ll, ll_c, 1e-9)           # bit for bit on Linux x86_64
    assert same(s2, s2_c, 1e-12)


def test_printed_estimates_give_the_printed_logl_to_rounding():
    """The table's 6-decimal estimates: 1.4e-6 from 6.4786201604, which is the
    rounding of the table, not a defect (docs/PORT.md §1)."""
    x = [7.662021, 0.833411, 0.651188, 0.458269, -0.542023, -0.793693, 0.033748,
         -0.042477, -0.907771, -1.030168, 0.103040, 1.329699, 0.513584, -0.240516]
    ll, _s2, _ = loglik(drvec.Spec(2, 2, 1, 1, 2), DATA.at(1), x)
    assert abs(ll - 6.4786201604) < 5e-6

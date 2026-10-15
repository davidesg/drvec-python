"""-multistart n: n - 1 jittered starts added to the search, from the C's own
generator (fit_search, drvec.c:5202-5218), against the oracle (SOURCE):

  ms201   mink-muskrat 2 0 1 -case 2 -multistart 4
          a well-defined optimum: the six starts end at the same logL to the
          printed decimals, and `jitter 1' wins by ~1e-10 -- a near-tie that
          only the oracle's platform is guaranteed to break the same way
  ms211   mink-muskrat 2 1 1 -mean -case 2 -multistart 5
          case 1 on the MA wall: no jitter beats `cold'
  msrank  mink-muskrat 2 0 1 -case 2 -lrtest -multistart 3 (every rank)

The jitters are the 64-bit stream; the C's `unsigned long' is 32-bit on
Windows, where its own jitters differ (the bootstrap's D4, docs/PORT.md)."""
import pytest

from conftest import EXACT, PRE, STRICT, TOL_EST, TOL_LL
from oracle import golden_filter, number_after, out_text, param_table

import drvec


@pytest.fixture(scope="module")
def ms201():
    return drvec.fit(PRE, 2, 0, 1, case=2, multistart=4)


def test_well_defined(ms201):
    out = out_text("ms201.out")
    assert [c.label for c in ms201.search.candidates] == [
        "seed", "cold", "johansen", "jitter 1", "jitter 2", "jitter 3"]
    assert max(c.ll for c in ms201.search.candidates) - min(
        c.ll for c in ms201.search.candidates) < 1e-6
    assert ms201.search.best.label == "jitter 1" or not EXACT
    assert abs(ms201.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    table = param_table(out)
    for p in ms201.params:
        assert abs(p.value - table[p.label][0]) <= (5.01e-7 if STRICT else TOL_EST), p.label
    if STRICT:
        assert golden_filter(ms201.out_text("ms201")) == golden_filter(out)


def test_on_the_wall_cold_still_wins():
    res = drvec.fit(PRE, 2, 1, 1, case=2, multistart=5)
    assert len(res.search.candidates) == 8 and res.search.best.label == "cold"
    out = out_text("ms211.out")
    assert abs(res.loglik - number_after(out, "logelf           :")) < (1e-6 if EXACT else TOL_LL)
    if STRICT:
        assert golden_filter(res.out_text("ms211")) == golden_filter(out)


def test_every_rank_of_the_rank_test():
    t = drvec.rank_test(PRE, 2, 0, case=2, multistart=3)
    assert all(len(f.candidates) >= 3 for f in t.fits)
    out = out_text("msrank.out")
    mine = golden_filter(t.out_text("msrank"))
    if STRICT:
        assert mine == golden_filter(out)
    else:
        assert len(mine.splitlines()) == len(golden_filter(out).splitlines())

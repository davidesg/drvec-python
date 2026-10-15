"""Run against the source tree without installing (src/ on sys.path).

THE PLATFORM CONVENTION (as in fue and drtran). The oracle is the C on Linux
x86_64, and the port is bit-identical to it THERE: drvarma's compiled
likelihood, libm and the BLAS round the same way the oracle's build did. On any
other platform (Linux aarch64, macOS, Windows) the last bits move -- the
compiled AS 311 rounds differently, and Windows' libm moves the generator's
Box-Muller in the last ulp -- and the optimiser's trajectory amplifies them on
a flat likelihood (case 1 sits on the MA invertibility wall). So:

* EXACT (Linux x86_64): every test as written, bit for bit and byte for byte.
* elsewhere, the same tests with the TOLERANCE TIER below, and the tests that
  replay single trajectories (start by start, call by call) skipped as fragile
  off the oracle's platform. What must still agree everywhere is every
  DECISION: the winning start, the rank test's verdicts, the bootstrap's
  replication statuses and p-values.

The tier was chosen from the first CI run (38736500814): off Linux x86_64 the
largest moves were 1.17e-6 in logL and 7.1e-7 in an estimate (case 1), 5.7e-14
in a rank-test logL and below 1e-9 in every bootstrap LR*. TOL_LL = 1e-5 and
TOL_EST = 1e-4 keep a 10x margin over that while staying inside the port's own
homologation target for estimates; TOL_SE = 1e-3, because the BFGS standard
errors are read off the optimiser's last Hessian update, which is trajectory
dependent; TOL_LR = 1e-6 for each bootstrap LR*.
"""
import os
import platform
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
DATA = os.path.join(HERE, "data")
FIXTURES = os.path.join(DATA, "fixtures")
PRE = [os.path.join(FIXTURES, "mmpre.muskrat.pre"), os.path.join(FIXTURES, "mmpre.mink.pre")]
E1 = [os.path.join(FIXTURES, f"e1{n}.pre") for n in ("invest", "income", "cons")]

EXACT = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")

TOL_LL = 1e-5
TOL_EST = 1e-4
TOL_SE = 1e-3
TOL_LR = 1e-6

fragile = pytest.mark.skipif(
    not EXACT, reason="replays one trajectory bit for bit: fragile off the oracle's "
                      "platform (Linux x86_64)")


def same(a, b, tol):
    """a == b on the oracle's platform; |a - b| <= tol elsewhere."""
    if EXACT:
        return a == b
    return abs(a - b) <= tol

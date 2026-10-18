"""Run against the source tree without installing (src/ on sys.path), or,
with DRVEC_TEST_INSTALLED=1, against the installed drvec (a wheel or an sdist
install: the sdist carries the tests and their data).

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
# The suite tests the PORT: the bootstraps run in Python whatever drvec C is
# installed (engine="auto" would take it). The C engine's own tests ask for
# it by name (tests/test_c_engine.py).
os.environ.setdefault("DRVEC_ENGINE", "python")
INSTALLED = os.environ.get("DRVEC_TEST_INSTALLED") == "1"
if not INSTALLED:
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
DATA = os.path.join(HERE, "data")
FIXTURES = os.path.join(DATA, "fixtures")
PRE = [os.path.join(FIXTURES, "mmpre.muskrat.pre"), os.path.join(FIXTURES, "mmpre.mink.pre")]
E1 = [os.path.join(FIXTURES, f"e1{n}.pre") for n in ("invest", "income", "cons")]

EXACT = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")

# The C solves Johansen's eigenproblem and the eigenvalues it PRINTS (Pi's, the
# operator roots) with GSL; the port calls the same GSL from drvarma's compiled
# engine (drvec._gsl). Without that engine (scipy/numpy) the johansen start lands a few ulps
# away -- and where it wins (xsys) the standard errors move by ~2e-4 -- and the
# printed eigenvalues may come in another order: the byte-for-byte .out tests
# and the printed-digit tests need both.
from drvec._gsl import backend as _backend          # noqa: E402

GSL = _backend() == "gsl"
STRICT = EXACT and GSL

TOL_LL = 1e-5
TOL_EST = 1e-4
TOL_SE = 1e-3
TOL_LR = 1e-6

# An estimate along a FLAT direction of the likelihood moves, off the oracle's
# platform, by more than TOL_EST in absolute terms while the logL stays inside
# TOL_LL: the same optimum, an ill-conditioned coordinate. The measure of
# "flat" is the parameter's own standard error, so the tier for an estimate is
# max(TOL_EST, TOL_EST_SE * s.e.), the s.e. the oracle printed. A move of k
# standard errors along a parameter costs about k^2 / 2 in logL; at k = 1e-3
# that is 5e-7, inside TOL_LL, so this tier admits nothing the logL tier does
# not already admit (and it only widens TOL_EST where the s.e. exceeds 0.1).
# Seen in CI: xdet's E[ec1] on macOS (run 39054125424), 1.75e-4 from the
# oracle with an s.e. of 134 (1.3e-6 s.e.); xar0's E[ec1] on Windows (run
# 39055641897), 5.04e-4 with an s.e. of 6.75 (7.5e-5 s.e.). Not used on the
# oracle's platform (STRICT).
TOL_EST_SE = 1e-3


def tol_est(se):
    """The off-platform tier for one estimate whose oracle s.e. is `se`
    (None where the .out prints none)."""
    return TOL_EST if se is None else max(TOL_EST, TOL_EST_SE * se)

# -fdhess's standard errors (milestone 3). They do not follow the optimiser's
# path, but they are NOT bit-stable either: the objective carries ~1e-14
# relative noise (AS 311's sums at |Sigma| ~ 1e-11), and the C's step,
# macheps^(1/3) * max(|x|, 1) ~ 6e-6, divides it by ~4e-11. Measured on Linux
# x86_64 (tests/test_fdhess.py, test_noise_floor): moving x by one or two
# ulps moves an s.e. by up to 4.5e-3 RELATIVE (E1, D.invest <- ec1) and 1.3e-3
# on mink-muskrat (6e-3 absolute on E[ec1], whose s.e. is 6.9). Off the
# oracle's platform the compiled likelihood rounds differently, which is the
# same kind of perturbation: the tier is relative, 2e-2, a 4x margin.
TOL_FDSE_REL = 2e-2

fragile = pytest.mark.skipif(
    not EXACT, reason="replays one trajectory bit for bit: fragile off the oracle's "
                      "platform (Linux x86_64)")


def same(a, b, tol):
    """a == b on the oracle's platform; |a - b| <= tol elsewhere."""
    if EXACT:
        return a == b
    return abs(a - b) <= tol


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the tests marked slow (bootstraps of minutes)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("DRVEC_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: run with --runslow (or DRVEC_SLOW=1)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

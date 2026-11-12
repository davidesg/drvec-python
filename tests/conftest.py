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

* EXACT (Linux x86_64, MEASURED to reproduce the oracle's bytes: below):
  every test as written, bit for bit and byte for byte.
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

PLATFORM = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")

# Linux x86_64 is the oracle's platform, but not every Linux x86_64 reproduces
# its bytes: they are also those of the GSL the oracle runs on (2.7.1, built
# for baseline x86-64). Ubuntu 26.04's apt GSL is built for x86-64-v3 (FMA,
# AVX2), and with it drvarma's eigensolver and a drvec C linked against it
# move in the 12th digit: 60 byte comparisons failed, every decision and every
# tolerance held (bugs/py-tests-exact-gsl-v3). So EXACT is measured, not
# assumed: the port must write the oracle's e1fit.out byte for byte (one fit,
# about a second; e1fit is among the files that move with that GSL; and two
# more for the libc, below), and a
# drvec C must write the oracle's proute.out (c_exact below). Where it does
# not, the suite runs its tolerance tier, as off Linux x86_64.
#
# The libc moves them too (bugs/py-tests-exact-libc): glibc 2.28 (AlmaLinux 8)
# and musl (Alpine) move e1fit and the rest; glibc 2.34 (Rocky 9) writes e1fit
# as the oracle does, but not the Pi eigenvalues e1fdh and e1seas print (the
# two zero eigenvalues of a rank-1 Pi on three series come out as -0.000000,
# or in another order). So the probe is those three fits, about 5 s more.
#
# DRVEC_REQUIRE_EXACT=1 turns the measurement off and asserts the oracle's
# bytes on Linux x86_64, as before: CI's reference job (tests.yml,
# ubuntu-24.04, the oracle's own Ubuntu) sets it, so a regression that moves
# the bytes there fails instead of falling into the tolerance tier.
REQUIRE_EXACT = os.environ.get("DRVEC_REQUIRE_EXACT", "").strip() == "1"


E1S = [os.path.join(FIXTURES, f"e1seas.{n}.pre") for n in ("invest", "income", "cons")]
_PROBES = {"e1fit": (E1, (2, 1, 1), {}),
           "e1fdh": (E1, (2, 0, 1), dict(fdhess=True)),
           "e1seas": (E1S, (2, 0, 1), {})}


def _port_writes_the_oracle():
    from oracle import golden_filter, out_text      # needs DATA, defined above
    import drvec
    try:
        for label, (files, (p, q, r), kw) in _PROBES.items():
            res = drvec.fit(files, p, q, r, case=2, engine="python", **kw)
            if golden_filter(res.out_text(label)) != golden_filter(out_text(label + ".out")):
                return False
        return True
    except Exception:                                # noqa: BLE001
        return False


PORT_EXACT = PLATFORM and (REQUIRE_EXACT or _port_writes_the_oracle())
EXACT = PORT_EXACT

_C_EXACT = {}


def c_exact(path):
    """Does the drvec C at PATH write the oracle's bytes? Linux x86_64 and,
    unless DRVEC_REQUIRE_EXACT=1, measured on proute (one fit, well under a
    second). Cached per binary."""
    if not PLATFORM or not path:
        return False
    if REQUIRE_EXACT:
        return True
    if path not in _C_EXACT:
        import re
        import shutil
        import subprocess
        import tempfile
        from oracle import golden_filter, out_text
        ver = re.compile(r"(?i)\bdrvec 0\.1[23]\b")
        norm = lambda t: ver.sub("DRVEC <v>", golden_filter(t))     # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            for f in ("mmpre.muskrat.pre", "mmpre.mink.pre"):
                shutil.copy(os.path.join(FIXTURES, f), d)
            try:
                cp = subprocess.run([path, "mmpre.muskrat.pre", "mmpre.mink.pre", "2", "1",
                                     "1", "-mean", "-case", "2", "-name", "proute"],
                                    cwd=d, capture_output=True, text=True, timeout=120)
                mine = open(os.path.join(d, "proute.out")).read() if cp.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                mine = ""
        _C_EXACT[path] = bool(mine) and norm(mine) == norm(out_text("proute.out"))
    return _C_EXACT[path]

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

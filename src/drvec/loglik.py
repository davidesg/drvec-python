"""The exact likelihood of the cast, on drvarma's compiled AS 311 (`elf_c`).

`elf` is called as the C calls it (drvmlest.c:90, 135, 167): sigma2 = 1, and the
scale concentrated out of the two factors it returns,

    logL   = -0.5 m n (LOG2PI - log m - log n + 1) - 0.5 n (m log f1 + log f2)
    sigma2 = f1 / (n m)

with the C's TRUNCATED constant LOG2PI = 1.837877066 (4.1e-10 below log 2 pi;
drtran-python uses log(2 pi): do not copy that). `xitol` is +1e-3 in every fit
(the xi sequence truncated, the default `-m 1`) and -1e-3 in the entry gate.
"""
from __future__ import annotations

import math

from drvarma._engine import elf_c

from .cast import shootx

LOG2PI = 1.837877066
XITOL = 1.0e-3


def concentrated(m, n, f1, f2):
    return (-0.5 * m * n * (LOG2PI - math.log(m) - math.log(n) + 1.0)
            - 0.5 * n * (m * math.log(f1) + math.log(f2)))


def elf(spec, rd, x, xitol=XITOL, atf=False, hold=None):
    """(f1, f2, a, ifault). ifault 6 = the cast refused the point (Sigma)."""
    mu, phi, theta, qq, w, ifc = shootx(spec, rd, x, hold)
    if ifc:
        return None, None, None, 6
    _lg, f1, f2, a, ifa = elf_c(spec.M, rd.n, spec.p, spec.q, mu, phi, theta, qq, w,
                                1.0, xitol, atf)
    return f1, f2, a, ifa


def loglik(spec, rd, x, xitol=XITOL, hold=None):
    """(logL, sigma2, ifault) at x."""
    f1, f2, _a, ifa = elf(spec, rd, x, xitol, True, hold)
    if ifa:
        return float("-inf"), float("nan"), ifa
    return concentrated(spec.M, rd.n, f1, f2), f1 / (rd.n * spec.M), 0

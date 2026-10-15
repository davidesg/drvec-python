"""Standard errors from the BFGS factor (drvmlest.c:117-125).

`raxopt` leaves in `b` the lower Cholesky factor of the BFGS Hessian of the
normalised objective; the C's covariance is

    cov[:, i] = 2 fk (b b')^{-1} e_i / n,      dev_i = sqrt(cov[i][i])

with fk the final value of the objective of THAT run (the winner's, in the
search). The finite-difference Hessian (`-fdhess`, exact_hessian_se 1910) is not
in milestone 1.
"""
from __future__ import annotations

import math

import numpy as np
from drvarma._qnewt import cholsol


def bfgs_covariance(b, fk, npar, n):
    """b is 1-indexed (npar+1, npar+1), as _qnewt returns it. -> (dev, cov),
    0-based numpy."""
    cov = np.zeros((npar, npar))
    dev = np.zeros(npar)
    for i in range(1, npar + 1):
        v = np.zeros(npar + 1)
        v[i] = 1.0
        cholsol(b, npar, v)
        for j in range(1, npar + 1):
            cov[j - 1, i - 1] = (2.0 * fk * v[j]) / n
        c = cov[i - 1, i - 1]
        dev[i - 1] = math.sqrt(c) if c >= 0.0 else float("nan")
    return dev, cov

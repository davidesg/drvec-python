"""GSL's eigensolvers, from drvarma's compiled engine: the C's own
`gsl_eigen_gensymmv` + `gsl_eigen_gensymmv_sort` (canonical_b2, drvec.c:1265-1272),
so that Johansen's seed is the C's bit for bit, and `gsl_eigen_nonsymm` (eigenqr,
nlatools.c:270), so that the eigenvalues the reports print come in the C's order.

Why (docs/PORT.md, decision D2 revised): scipy's `eigh` is a few ulps off GSL,
and on the two-series cases that only moved the `johansen` start by 1e-7 in
logL. On the three-series case (Lutkepohl's E1) it moves that start's optimum by
0.47 in logL and one bootstrap LR* by 3.5e-5 -- beyond the 1e-6 tolerance.
With GSL the whole case is byte-identical (the fit's search table, the 40
replications of the .boot).

Where GSL comes from (decided 2026-10-15, TODO.md): drvarma >= 0.3.0 exports
both routines from its compiled engine (`drvarma._engine.gensymmv`,
`eigen_nonsymm`), which links GSL on every platform it ships a binary wheel for.
Before, this module looked for a system libgsl through ctypes, which Windows
and macOS without Homebrew do not have; it no longer does.

When drvarma's engine is unavailable (its pure-Python fallback, or
DRVARMA_NO_ENGINE), the seed is scipy's `eigh` and the eigenvalues numpy's, as
before without libgsl: the same values to rounding, possibly in another order.
`DRVEC_EIGEN=scipy` forces scipy, `DRVEC_EIGEN=gsl` makes a missing engine an
error; the default (`auto`) uses GSL whenever drvarma's engine provides it.
"""
from __future__ import annotations

import os

import numpy as np


def _engine():
    """drvarma's `_engine` module when it can run GSL's routines, else None."""
    try:
        from drvarma import _engine as E
        return E if E.gsl_available() else None
    except (ImportError, AttributeError):       # AttributeError: drvarma < 0.3.0
        return None


def backend():
    """'gsl' or 'scipy': what canonical_b2 will use."""
    mode = os.environ.get("DRVEC_EIGEN", "auto").lower()
    if mode == "scipy":
        return "scipy"
    if _engine() is not None:
        return "gsl"
    if mode == "gsl":
        raise RuntimeError("DRVEC_EIGEN=gsl and drvarma's compiled engine cannot run "
                           "GSL's eigensolvers (drvarma >= 0.3.0 with its engine)")
    return "scipy"


def gensymmv_desc(A, B):
    """Eigenvectors of A v = l B v, columns sorted by DECREASING eigenvalue, or
    None when the problem cannot be solved (B not positive definite)."""
    if backend() == "scipy":
        import scipy.linalg as sl
        try:
            _ev, V = sl.eigh(A, B)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return V[:, ::-1]
    try:
        _ev, V = _engine().gensymmv(A, B, sort="val_desc")   # GSL_EIGEN_SORT_VAL_DESC
    except np.linalg.LinAlgError:
        return None
    return V


def eigen_nonsymm(A):
    """eigenqr (nlatools.c:270): the eigenvalues of a real general matrix in
    GSL's order (balanced, gsl_eigen_nonsymm), as (wr, wi) lists. Without GSL,
    numpy's eigvals: the same values to rounding, possibly in another order."""
    A = np.array(A, float)
    if backend() == "gsl":
        try:
            wr, wi = _engine().eigen_nonsymm(A, balance=True)
            return [float(x) for x in wr], [float(x) for x in wi]
        except np.linalg.LinAlgError:       # QR without convergence: the C aborts
            pass
    ev = np.linalg.eigvals(A)
    return [float(e.real) for e in ev], [float(e.imag) for e in ev]

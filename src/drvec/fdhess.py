"""-fdhess: standard errors from the finite-difference Hessian AT THE OPTIMUM.

C: `exact_hessian_se` (drvec.c:1910-2007) with its objective `fdh_obj` (1877),
and `fdhess` (qnewtopt.c:404), which is drvarma's `_qnewt.fdhess` line for line.

The BFGS covariance (stderr.py) is read off the optimiser's last Hessian
update and so depends on the trajectory; this one depends only on the point.
The objective is the concentrated likelihood normalised AT THE OPTIMUM,

    g(x) = (f1(x) / f1(x*))^m (f2(x) / f2(x*)),   g(x*) = 1,

the Hessian of g by Dennis and Schnabel's A5.6.2 with eta = macheps, and

    cov = 2 g(x*) H^{-1} / n          (the same formula as est's)

A point the cast or AS 311 refuses counts as 1e10 and as a rejection. Any
rejection means the optimum is ON the boundary of the admissible region (an MA
root on the unit circle, BUG-49): the BFGS standard errors are kept and the
.out says why (BUG-34). So does a Hessian that fails choldcp with every
evaluation admissible (the optimiser did not stop at a maximum).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from drvarma import _qnewt

from .cast import shootx
from .loglik import XITOL
from .nla import choldcp

REFUSED = 1.0e10


@dataclass
class FdHess:
    ok: bool                 # the s.e. are the Hessian's
    nrej: int                # finite-difference evaluations refused (boundary)
    nneg: int = 0            # non-positive eigenvalues of H (when not ok, nrej == 0)
    ratio: float = 0.0       # -min/max eigenvalue
    dev: np.ndarray = None
    cov: np.ndarray = None

    @property
    def why(self):
        """The reason written to the .out when the s.e. could not be used."""
        return ("the optimum is on the boundary (finite-difference steps left the "
                "admissible region)" if self.nrej > 0 else
                "the Hessian at the optimum is not positive definite")

    def warning(self, npar):
        """The C's stderr warning (drvec.c:1975-1991)."""
        if self.ok:
            return None
        if self.nrej > 0:
            return ("WARNING: -fdhess: the optimum lies ON the boundary of the admissible\n"
                    "         region -- %d of the finite-difference evaluations fell\n"
                    "         outside it -- so the unconstrained Hessian is not defined\n"
                    "         there and the BFGS standard errors are kept.  See the roots\n"
                    "         of the AR and MA operators reported above: a modulus at one\n"
                    "         identifies the binding direction.\n" % self.nrej)
        return ("WARNING: -fdhess: the Hessian at the reported optimum is not positive\n"
                "         definite: %d of %d eigenvalues are non-positive, the most\n"
                "         negative being %.3g times the largest positive one.  Every\n"
                "         evaluation was admissible, so the optimiser did not stop at\n"
                "         a maximum.  The BFGS standard errors are kept.\n"
                % (self.nneg, npar, self.ratio))


def _f12(spec, rd, x, xitol):
    mu, phi, theta, qq, w, ifc = shootx(spec, rd, x)
    if ifc > 0:
        return None
    from drvarma._engine import elf_c
    _lg, f1, f2, _a, ifa = elf_c(spec.M, rd.n, spec.p, spec.q, mu, phi, theta, qq, w,
                                 1.0, xitol, False)
    if ifa > 0:
        return None
    return f1, f2


def exact_hessian_se(spec, rd, x, xitol=XITOL):
    """-> FdHess. `rd` is the estimation sample (n = rd.n is the C's nobs)."""
    npar = len(x)
    m = spec.M
    base = _f12(spec, rd, list(x), xitol)
    if base is None:
        return FdHess(ok=False, nrej=0)
    n1, n2 = base
    nrej = [0]

    def obj(xk1):
        v = _f12(spec, rd, xk1[1:].tolist(), xitol)
        if v is None:
            nrej[0] += 1
            return REFUSED
        return pow(v[0] / n1, m) * (v[1] / n2)

    xk = np.zeros(npar + 1)
    xk[1:] = x
    f = obj(xk)                                     # = 1 by construction
    nrej[0] = 0
    H = np.zeros((npar + 1, npar + 1))
    _qnewt.fdhess(obj, npar, xk, f, _qnewt.MACHEPS, H)
    Hs = H[1:, 1:].copy()
    ev = np.linalg.eigvals(Hs).real                 # eigenqr, for the warning only
    mx = max(0.0, float(ev.max())) if npar else 0.0
    mn = min(0.0, float(ev.min())) if npar else 0.0
    nneg = int((ev <= 0.0).sum())
    ratio = -mn / mx if mx > 0.0 else 0.0
    if nrej[0] > 0:
        return FdHess(ok=False, nrej=nrej[0], nneg=nneg, ratio=ratio)
    L, ifc = choldcp([[float(v) for v in row] for row in Hs])
    if ifc > 0:
        return FdHess(ok=False, nrej=0, nneg=nneg, ratio=ratio)
    Lk = np.zeros((npar + 1, npar + 1))
    Lk[1:, 1:] = np.array(L)
    cov = np.zeros((npar, npar))
    dev = np.zeros(npar)
    for i in range(1, npar + 1):
        e = np.zeros(npar + 1)
        e[i] = 1.0
        _qnewt.cholsol(Lk, npar, e)
        for j in range(1, npar + 1):
            cov[j - 1, i - 1] = (2.0 * f * e[j]) / rd.n
        c = cov[i - 1, i - 1]
        dev[i - 1] = math.sqrt(c if c > 0.0 else 0.0)
    return FdHess(ok=True, nrej=0, nneg=nneg, ratio=ratio, dev=dev, cov=cov)

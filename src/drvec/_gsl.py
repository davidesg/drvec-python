"""GSL's generalized symmetric eigensolver, through ctypes, when the library is
on the system: the C's own `gsl_eigen_gensymmv` + `gsl_eigen_gensymmv_sort`
(canonical_b2, drvec.c:1265-1272), so that Johansen's seed is the C's bit for bit.

Why (docs/PORT.md, decision D2 revised): scipy's `eigh` is a few ulps off GSL,
and on the two-series cases that only moved the `johansen` start by 1e-7 in
logL. On the three-series case (Lutkepohl's E1) it moves that start's optimum by
0.47 in logL and one bootstrap LR* by 3.5e-5 -- beyond the 1e-6 tolerance.
With GSL the whole case is byte-identical (the fit's search table, the 40
replications of the .boot).

`DRVEC_EIGEN=scipy` forces scipy, `DRVEC_EIGEN=gsl` makes a missing GSL an
error; the default (`auto`) uses GSL when it can be loaded.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os

import numpy as np

_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    names = [ctypes.util.find_library("gsl"), "libgsl.so.27", "libgsl.so", "libgsl.dylib",
             "libgsl-27.dll", "gsl.dll"]
    cblas = [ctypes.util.find_library("gslcblas"), "libgslcblas.so.0", "libgslcblas.so",
             "libgslcblas.dylib", "libgslcblas-0.dll", "gslcblas.dll"]
    for c in cblas:
        if not c:
            continue
        try:
            ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            pass
    for n in names:
        if not n:
            continue
        try:
            lib = ctypes.CDLL(n)
        except OSError:
            continue
        try:
            P, S, D = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double
            for f, res, args in (
                    ("gsl_matrix_alloc", P, [S, S]), ("gsl_vector_alloc", P, [S]),
                    ("gsl_eigen_gensymmv_alloc", P, [S]),
                    ("gsl_matrix_set", None, [P, S, S, D]), ("gsl_matrix_get", D, [P, S, S]),
                    ("gsl_vector_get", D, [P, S]),
                    ("gsl_eigen_gensymmv", ctypes.c_int, [P, P, P, P, P]),
                    ("gsl_eigen_gensymmv_sort", ctypes.c_int, [P, P, ctypes.c_int]),
                    ("gsl_eigen_gensymmv_free", None, [P]), ("gsl_matrix_free", None, [P]),
                    ("gsl_vector_free", None, [P]), ("gsl_set_error_handler_off", P, [])):
                fn = getattr(lib, f)
                fn.restype = res
                fn.argtypes = args
        except AttributeError:
            continue
        lib.gsl_set_error_handler_off()     # a failure is a return code, not abort()
        _lib = lib
        break
    return _lib


def backend():
    """'gsl' or 'scipy': what canonical_b2 will use."""
    mode = os.environ.get("DRVEC_EIGEN", "auto").lower()
    if mode == "scipy":
        return "scipy"
    if _load() is not None:
        return "gsl"
    if mode == "gsl":
        raise RuntimeError("DRVEC_EIGEN=gsl and libgsl cannot be loaded")
    return "scipy"


_SORT_VAL_DESC = 1


def gensymmv_desc(A, B):
    """Eigenvectors of A v = l B v, columns sorted by DECREASING eigenvalue, or
    None when the problem cannot be solved (B not positive definite)."""
    M = A.shape[0]
    if backend() == "scipy":
        import scipy.linalg as sl
        try:
            _ev, V = sl.eigh(A, B)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return V[:, ::-1]
    g = _lib
    Ag, Bg = g.gsl_matrix_alloc(M, M), g.gsl_matrix_alloc(M, M)
    V, ev = g.gsl_matrix_alloc(M, M), g.gsl_vector_alloc(M)
    ws = g.gsl_eigen_gensymmv_alloc(M)
    try:
        for i in range(M):
            for j in range(M):
                g.gsl_matrix_set(Ag, i, j, float(A[i, j]))
                g.gsl_matrix_set(Bg, i, j, float(B[i, j]))
        if g.gsl_eigen_gensymmv(Ag, Bg, ev, V, ws) != 0:
            return None
        g.gsl_eigen_gensymmv_sort(ev, V, _SORT_VAL_DESC)
        return np.array([[g.gsl_matrix_get(V, i, j) for j in range(M)] for i in range(M)])
    finally:
        g.gsl_eigen_gensymmv_free(ws)
        g.gsl_vector_free(ev)
        g.gsl_matrix_free(V)
        g.gsl_matrix_free(Bg)
        g.gsl_matrix_free(Ag)

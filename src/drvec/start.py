"""Starting values: the cold start and Johansen's canonical seed.

C: `prelim_b2` (drvec.c:1337), `init_guess` (3715; the default path 3865-4290)
and `canonical_b2` (1143-1330). Transcribed loop for loop: the cold start is
bit-identical to the C's (tests/test_trajectory.py), which matters because the
case-1 optimum is reached from it and from nowhere else.
"""
from __future__ import annotations

import numpy as np

from ._gsl import gensymmv_desc
from .nla import zeros, ludcp, lusol


def prelim_b2(spec, rd):
    """B2 by static OLS of each Y1 on [1, Y2 levels]; B2 = -slope."""
    M, r, s, n = spec.M, spec.r, spec.s, rd.n
    dm, y2 = rd.dm, rd.y2
    B2 = zeros(s, max(r, 1))
    for j in range(r):
        nc = s + 1
        XX = zeros(nc, nc)
        Xy = [0.0] * nc
        for i in range(nc):
            for ii in range(nc):
                ss = 0.0
                for t in range(n):
                    xi = 1.0 if i == 0 else y2[t][i - 1]
                    xii = 1.0 if ii == 0 else y2[t][ii - 1]
                    ss += xi * xii
                XX[i][ii] = ss
            acc = 0.0
            for t in range(n):
                xi = 1.0 if i == 0 else y2[t][i - 1]
                acc += xi * dm[t][s + j]
            Xy[i] = acc
        ip = ludcp(XX, nc)
        lusol(XX, Xy, nc, ip)
        for i in range(s):
            B2[i][j] = -Xy[i + 1]
    return B2


def canonical_b2(spec, rd):
    """Johansen's reduced-rank B2 (canonical_b2), or None when it cannot be
    formed. The generalized symmetric eigenproblem is the C's own GSL
    `gsl_eigen_gensymmv` through ctypes when libgsl is on the system, and
    scipy's `eigh` otherwise (`_gsl.py`). scipy's is a few ulps off: 2 ulps
    in B2 on mink-muskrat, which moves the `johansen` start by up to 9.5e-8
    in logL there, and by 0.47 on Lutkepohl's E1 (tests/test_johansen.py)."""
    M, r, s, p, n = spec.M, spec.r, spec.s, spec.p, rd.n
    nf = spec.nf
    T = n - p
    nd = nf * M
    if r <= 0 or T <= nd + M + 1:
        return None
    dm, y2 = rd.dm, rd.y2
    R0 = zeros(T, M)
    R1 = zeros(T, M)
    for t in range(p, n):
        row = t - p
        for j in range(r):
            R0[row][j] = dm[t][s + j] - dm[t - 1][s + j]
            R1[row][j] = dm[t - 1][s + j]
        for i in range(s):
            R0[row][r + i] = dm[t][i]
            R1[row][r + i] = y2[t - 1][i]
    nc = nd + 1
    D = zeros(T, nc)
    for t in range(p, n):
        row = t - p
        c = 0
        D[row][c] = 1.0; c += 1
        for k in range(1, nf + 1):
            for j in range(r):
                D[row][c] = dm[t - k][s + j] - dm[t - k - 1][s + j]; c += 1
            for i in range(s):
                D[row][c] = dm[t - k][i]; c += 1
    XtX = zeros(nc, nc)
    for c in range(nc):
        for c2 in range(nc):
            ss = 0.0
            for t in range(T):
                ss += D[t][c] * D[t][c2]
            XtX[c][c2] = ss
    ip = ludcp(XtX, nc)
    for e in range(M):
        for RR in (R0, R1):
            Xty = [0.0] * nc
            for c in range(nc):
                acc = 0.0
                for t in range(T):
                    acc += D[t][c] * RR[t][e]
                Xty[c] = acc
            lusol(XtX, Xty, nc, ip)
            for t in range(T):
                fit = 0.0
                for c in range(nc):
                    fit += Xty[c] * D[t][c]
                RR[t][e] -= fit
    S00 = zeros(M, M); S01 = zeros(M, M); S11 = zeros(M, M)
    for i in range(M):
        for j in range(M):
            a0 = a1 = a2 = 0.0
            for t in range(T):
                a0 += R0[t][i] * R0[t][j]
                a1 += R0[t][i] * R1[t][j]
                a2 += R1[t][i] * R1[t][j]
            S00[i][j] = a0 / T; S01[i][j] = a1 / T; S11[i][j] = a2 / T
    cp = [row[:] for row in S00]
    ip2 = ludcp(cp, M)
    S00i = zeros(M, M)
    for j in range(M):
        col = [1.0 if i == j else 0.0 for i in range(M)]
        lusol(cp, col, M, ip2)
        for i in range(M):
            S00i[i][j] = col[i]
    A = zeros(M, M)
    for i in range(M):
        for j in range(M):
            ss = 0.0
            for k in range(M):
                for t in range(M):
                    ss += S01[k][i] * S00i[k][t] * S01[t][j]
            A[i][j] = ss
    As = np.array([[0.5 * (A[i][j] + A[j][i]) for j in range(M)] for i in range(M)])
    Bs = np.array([[0.5 * (S11[i][j] + S11[j][i]) for j in range(M)] for i in range(M)])
    V = gensymmv_desc(As, Bs)                       # GSL_EIGEN_SORT_VAL_DESC
    if V is None:
        return None
    bet = [[float(V[i, j]) for j in range(r)] for i in range(M)]
    Bt = [[bet[j][i] for j in range(r)] for i in range(r)]      # Btop'
    ip3 = ludcp(Bt, r)
    if any(abs(Bt[i][i]) < 1.0e-12 for i in range(r)):
        return None
    B2 = zeros(s, max(r, 1))
    for i in range(M):
        z = bet[i][:]
        lusol(Bt, z, r, ip3)
        if i >= r:
            for j in range(r):
                B2[i - r][j] = z[j]
    return B2


def init_guess(spec, rd, seedjoh=False, consistent=True):
    """The conditional (Johansen-style) start: B2 by static OLS (or the canonical
    one), Lambda and F by the conditional regression
        nabla Y_t = sum F_k nabla Y_{t-k} - Lambda (W_{t-1} - E[W]) + e_t,
    Sigma from its residuals divided by their [1][1], Theta at zero."""
    M, r, s, p, q, n = spec.M, spec.r, spec.s, spec.p, spec.q, rd.n
    nf = spec.nf
    dm, y2 = rd.dm, rd.y2
    B2 = prelim_b2(spec, rd)
    if seedjoh and r > 0:
        Bc = canonical_b2(spec, rd)
        if Bc is not None:
            B2 = Bc
    if spec.b2_fix is not None and spec.fixb2_given and r > 0 and consistent:
        # -fixb2 v: the rest of the seed from the W that is fitted (BUG-33);
        # consistent=False is the search's `ols-b2' start (drvec.c:5143)
        for i in range(s):
            for j in range(r):
                B2[i][j] = float(spec.b2_value(i, j))
    elif spec.b2_hold and r > 0:                # -fixb2row: held rows BEFORE W (BUG-33)
        for i in range(s):
            v = spec.b2_held(i)
            if v is not None:
                for j in range(r):
                    B2[i][j] = float(v)
    W = zeros(n, max(r, 1))
    dY = zeros(n, M)                    # internal order [nabla Y1 ; nabla Y2]
    for t in range(n):
        for j in range(r):
            w = dm[t][s + j]
            for i in range(s):
                w += B2[i][j] * y2[t][i]
            W[t][j] = w
        for j in range(r):
            dY[t][j] = dm[t][s + j] - dm[t - 1][s + j] if t > 0 else 0.0
        for i in range(s):
            dY[t][r + i] = dm[t][i]
    ET = [0.0] * max(r, 1)
    if spec.trend:                              # -trend: W detrended on centred t
        tc = 0.5 * (n + 1)
        stt = 0.0
        for t in range(1, n + 1):
            stt += (t - tc) * (t - tc)
        for j in range(r):
            swt = 0.0
            for t in range(1, n + 1):
                swt += W[t - 1][j] * (t - tc)
            ET[j] = swt / stt if stt > 0.0 else 0.0
            for t in range(1, n + 1):
                W[t - 1][j] -= ET[j] * (t - tc)
    EW = [0.0] * max(r, 1)
    EdY2 = [0.0] * s
    for j in range(r):
        sm = 0.0
        for t in range(n):
            sm += W[t][j]
        EW[j] = sm / n
    for i in range(s):
        sm = 0.0
        for t in range(n):
            sm += dY[t][r + i]
        EdY2[i] = sm / n
    nreg = r + nf * M
    T = n - p
    if T < 1:
        T = 1
    X = zeros(T, max(nreg, 1))
    Yd = zeros(T, M)
    for t in range(p, n):
        row = t - p
        col = 0
        for j in range(r):
            X[row][col] = W[t - 1][j] - EW[j]; col += 1
        for k in range(1, nf + 1):
            for i in range(M):
                X[row][col] = dY[t - k][i]; col += 1
        for i in range(M):
            Yd[row][i] = dY[t][i]
    Lam = zeros(M, max(r, 1))
    F = [zeros(M, M) for _ in range(nf)]
    if nreg > 0:
        XtX = zeros(nreg, nreg)
        for i in range(nreg):
            for j in range(nreg):
                ss = 0.0
                for t in range(T):
                    ss += X[t][i] * X[t][j]
                XtX[i][j] = ss
        ip = ludcp(XtX, nreg)
        for eq in range(M):
            Xty = [0.0] * nreg
            for i in range(nreg):
                acc = 0.0
                for t in range(T):
                    acc += X[t][i] * Yd[t][eq]
                Xty[i] = acc
            lusol(XtX, Xty, nreg, ip)
            for j in range(r):
                Lam[eq][j] = -Xty[j]
            for k in range(nf):
                for i in range(M):
                    F[k][eq][i] = Xty[r + k * M + i]
    E = zeros(T, M)
    for t in range(T):
        for i in range(M):
            ei = Yd[t][i]
            for c in range(nreg):
                bi = -Lam[i][c] if c < r else F[(c - r) // M][i][(c - r) % M]
                ei -= bi * X[t][c]
            E[t][i] = ei
    Sig = zeros(M, M)
    for i in range(M):
        for j in range(M):
            ss = 0.0
            for t in range(T):
                ss += E[t][i] * E[t][j]
            Sig[i][j] = ss / T if T > 0 else 1.0
    x = []
    if spec.case == 2:
        x += [EW[j] for j in range(r)]
    elif spec.case == 3:
        x += [EdY2[i] for i in range(s) if spec.mu_free(i)] + [EW[j] for j in range(r)]
    if spec.trend:
        x += [ET[j] for j in range(r)]
    sa = spec.alpha_sa
    if sa:
        # psi = (A'A)^-1 A' Lambda_ols: the free seed projected onto the
        # restriction (drvec.c:4176-4205)
        from .params import RowOrder
        l2i = RowOrder(M, r).lam2inp
        A = spec.alpha_A()
        AtA = zeros(sa, sa)
        for a in range(sa):
            for b in range(sa):
                acc = 0.0
                for i in range(M):
                    acc += A[i][a] * A[i][b]
                AtA[a][b] = acc
        ip = ludcp(AtA, sa)
        psi = zeros(sa, max(r, 1))
        for j in range(r):
            AtL = [0.0] * sa
            for a in range(sa):
                acc = 0.0
                for i in range(M):
                    acc += A[l2i(i)][a] * Lam[i][j]
                AtL[a] = acc
            lusol(AtA, AtL, sa, ip)
            for a in range(sa):
                psi[a][j] = AtL[a]
        x += [psi[a][j] for a in range(sa) for j in range(r)]
    else:
        x += [Lam[i][j] for i in range(M) for j in range(r)]
    for k in range(nf):
        x += [F[k][i][j] for i in range(M) for j in range(M) if spec.f_free(i, j)]
    from .params import theta_len
    x += [0.0] * theta_len(spec.theta_class(), q, M, r)
    s11 = Sig[0][0] if Sig[0][0] > 1.0e-24 else 1.0e-24
    for i in range(M):
        for j in range(M):
            Sig[i][j] /= s11
    Sig[0][0] = 1.0
    x += [Sig[i][i] for i in range(1, M)]
    if not spec.diag_cov:
        x += [Sig[i][j] for i in range(1, M) for j in range(i)]
    x += [B2[i][j] for j in range(r) for i in range(s) if spec.b2_held(i) is None]
    if spec.xlink:                               # omega = delta = 0 (drtran's seeding)
        from .xlink import npar as xl_npar
        x += [0.0] * xl_npar(spec.xlink)
    assert len(x) == spec.npar()
    return x

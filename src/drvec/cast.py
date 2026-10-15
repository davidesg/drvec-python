"""The VEC cast: the parameter vector -> the stationary VARMA on Ybar.

C: `vec_shootx` (drvec.c:4303-4790), default parameterisation (no -warma,
-alpha, -xlink, -trend, -mawarma). Mauricio (2006), eqs. 10-18:

    Cbar   = [0_{s x r}  I_s ; I_r  B2']        Cinv = [-B2'  I_r ; I_s  0]
    Hbar   = [0 0 ; 0 I_r]                      LamBar = [0_{M x s}, Lambda]
    PhiBar_1 = Cinv Hbar - LamBar + F_1 Cinv
    PhiBar_k = F_k Cinv - F_{k-1} Cinv Hbar     (k = 2..p-1)
    PhiBar_p = -F_{p-1} Cinv Hbar
    phi*_k = Cbar PhiBar_k,   theta*_k = Cbar Theta_k Cinv,   qq = Cbar Sigma Cbar'
    mu = [E[nabla Y2] ; E[W]],  w_t = [nabla Y2_t ; Y1_t + B2' Y2_t]

Every product is `nla.mm` (the C's loop order); W is built column by column,
adding B2[i][j] * Y2_i in increasing i, which is the C's order of additions --
so the cast is bit-identical to the C's (docs/PORT.md §1) and costs ~130 us per
evaluation with drvarma's compiled `elf`.

`hold` is the C's `prof_hold` (the route-(B) profile, gate_profile_seed): x then
carries only [mean, Lambda, B2], and F, Theta and Sigma are read from the hold
(internal order, full M x M).
"""
from __future__ import annotations

import numpy as np

from .nla import mm, zeros, chol_ok


def unpack(spec, x, hold=None):
    """x -> (mu (.inp/Ybar order), Lambda, F, Theta, Sigma, B2) as lists; the walk
    of vec_shootx [3]."""
    M, r, s, q = spec.M, spec.r, spec.s, spec.q
    nf = spec.nf
    idx = 0
    mu = [0.0] * M
    if spec.case == 2:
        for j in range(r):
            mu[s + j] = x[idx]; idx += 1
    elif spec.case == 3:
        for i in range(s):
            if spec.mu_free(i):
                mu[i] = x[idx]; idx += 1
        for j in range(r):
            mu[s + j] = x[idx]; idx += 1
    dtr = [0.0] * max(r, 1)
    if spec.trend:
        for j in range(r):
            dtr[j] = x[idx]; idx += 1
    Lam = zeros(M, max(r, 1))
    sa = spec.alpha_sa
    if sa:                                      # alpha = A psi (-xsys), drvec.c:4448
        from .params import RowOrder
        l2i = RowOrder(M, r).lam2inp
        A = spec.alpha_A()
        psi = zeros(sa, max(r, 1))
        for i in range(sa):
            for j in range(r):
                psi[i][j] = x[idx]; idx += 1
        for i in range(M):
            Ai = A[l2i(i)]
            for j in range(r):
                acc = 0.0
                for kk in range(sa):
                    acc += Ai[kk] * psi[kk][j]
                Lam[i][j] = acc
    else:
        for i in range(M):
            for j in range(r):
                Lam[i][j] = x[idx]; idx += 1
    if hold is not None:
        F = [[row[:] for row in Fk] for Fk in hold["F"][:nf]]
        Th = [[row[:] for row in Tk] for Tk in hold["Th"][:q]]
        S = [row[:] for row in hold["S"]]
    else:
        F = [zeros(M, M) for _ in range(nf)]
        for k in range(nf):
            for i in range(M):
                for j in range(M):
                    if spec.f_free(i, j):
                        F[k][i][j] = x[idx]; idx += 1
        from .params import theta_unpack
        Th, idx = theta_unpack(x, idx, spec.theta_class(), q, M, r)
        S = zeros(M, M)
        S[0][0] = 1.0
        for i in range(1, M):
            S[i][i] = x[idx]; idx += 1
        if not spec.diag_cov:
            for i in range(1, M):
                for j in range(i):
                    S[i][j] = x[idx]; S[j][i] = S[i][j]; idx += 1
    B2 = zeros(s, max(r, 1))
    for j in range(r):
        for i in range(s):
            v = spec.b2_held(i)
            if v is None:
                B2[i][j] = x[idx]; idx += 1
            else:
                B2[i][j] = float(v)
    if spec.xlink:
        from .xlink import npar as xl_npar
        xp = list(x[idx:idx + xl_npar(spec.xlink)])
        idx += len(xp)
    else:
        xp = None
    if idx != len(x):
        raise ValueError(f"the vector has {len(x)} entries and the walk consumed {idx}")
    unpack.xp = xp
    return mu, Lam, F, Th, S, B2, dtr


def shootx(spec, rd, x, hold=None):
    """-> (mu, phi (p,M,M), theta (q,M,M), qq (M,M), w (n,M), ifault) as numpy,
    0-based, ready for `elf_c`. ifault = 1 when Sigma fails choldcp (4563-4577)."""
    M, r, s, p, q = spec.M, spec.r, spec.s, spec.p, spec.q
    mu, Lam, F, Th, S, B2, dtr = unpack(spec, x, hold)
    xp = unpack.xp
    ifault = 0 if chol_ok(S) else 1
    xT = None
    if spec.xlink:
        from .xlink import compute
        xT = compute(spec.xlink, rd.xl, xp, rd.nobs_raw, M)
        if xT is None:
            ifault = 1                          # a delta(B) that is not stable

    Cb = zeros(M, M); Ci = zeros(M, M); H = zeros(M, M); LB = zeros(M, M)
    for i in range(s):
        Cb[i][r + i] = 1.0
    for j in range(r):
        Cb[s + j][j] = 1.0
    for j in range(r):
        for i in range(s):
            Cb[s + j][r + i] = B2[i][j]
    for i in range(r):
        for j in range(s):
            Ci[i][j] = -B2[j][i]
    for i in range(r):
        Ci[i][s + i] = 1.0
    for i in range(s):
        Ci[r + i][i] = 1.0
    for i in range(r):
        H[s + i][s + i] = 1.0
    for i in range(M):
        for j in range(r):
            LB[i][s + j] = Lam[i][j]

    PB = [zeros(M, M) for _ in range(p + 1)]
    T1 = mm(Ci, H)
    for i in range(M):
        for j in range(M):
            PB[1][i][j] = T1[i][j] - LB[i][j]
    if p >= 2:
        T2 = mm(F[0], Ci)
        for i in range(M):
            for j in range(M):
                PB[1][i][j] += T2[i][j]
    for k in range(2, p):                       # PhiBar_k, k = 2..p-1
        T1 = mm(F[k - 1], Ci)
        T3 = mm(mm(F[k - 2], Ci), H)
        for i in range(M):
            for j in range(M):
                PB[k][i][j] = T1[i][j] - T3[i][j]
    if p >= 2:
        T2 = mm(mm(F[p - 2], Ci), H)
        for i in range(M):
            for j in range(M):
                PB[p][i][j] = -T2[i][j]

    phi = np.array([mm(Cb, PB[k]) for k in range(1, p + 1)], float).reshape(p, M, M)
    theta = (np.array([mm(mm(Cb, Th[k]), Ci) for k in range(q)], float).reshape(q, M, M)
             if q else np.zeros((0, M, M)))
    T1 = mm(Cb, S)
    qq = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            ss = 0.0
            for k1 in range(M):
                ss += T1[i][k1] * Cb[j][k1]
            qq[i, j] = ss

    w = rd.dm_np.copy()
    tc = np.arange(1, rd.n + 1, dtype=float) - 0.5 * (rd.n + 1)
    for j in range(r):
        col = w[:, s + j]
        for i in range(s):
            col += B2[i][j] * rd.y2_np[:, i]
        if spec.trend:
            col -= dtr[j] * tc                  # w - dtr (t - (n+1)/2), drvec.c:4766
    if xT is not None:                          # the inputs, by subtraction (4768-4775)
        for i in range(s):
            w[:, i] -= xT[1:, i] - xT[:-1, i]
        for j in range(r):
            dw = xT[1:, s + j].copy()
            for i in range(s):
                dw += B2[i][j] * xT[1:, i]
            w[:, s + j] -= dw
    return np.array(mu, float), phi, theta, qq, w, ifault

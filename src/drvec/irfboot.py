"""-irfboot N: percentile bands for the level responses and the decomposition
by the parametric bootstrap (irf_bootstrap, drvec.c:7540-7760).

N samples are simulated from the fitted model (simulate_h0, the generator of
the rank bootstrap, from its seed), each is refitted as the data were (same
rank, same restrictions, the ladder start and the search), and the responses
of every refit are collected. The bands are Efron's percentiles,
col[ceil(p nok)] of the sorted replications."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .rank import fit_rank, simulate_h0
from .report import banner, ldl_sigma, level_irf_R
from .rng import Rng

PICK = (0, 1, 2, 4, 8, 12, 16, 20, 30, 40)
QP = (0.025, 0.05, 0.50, 0.95, 0.975)


def _irf_at(spec, data, x, K):
    """model_at + level_irf_R: the responses R[k][i][j] of the model x, or None."""
    from .cast import shootx, unpack
    from .loglik import elf
    rd = data.at(spec.r)
    mu, phi, theta, qq, w, ifs = shootx(spec, rd, x)
    if ifs:
        return None
    f1, _f2, _a, ifa = elf(spec, rd, x, atf=True)
    if ifa:
        return None
    sigma2 = f1 / (rd.n * spec.M)
    ld = ldl_sigma(sigma2, qq.tolist(), spec.M)
    if ld is None:
        return None
    P, D = ld
    B2 = unpack(spec, x)[5]
    return level_irf_R(spec.M, spec.r, spec.p, spec.q, phi.tolist(), theta.tolist(), B2, K, P, D)


@dataclass
class IrfBoot:
    N: int
    K: int
    R0: list
    irf: list          # per good replication: R[k][i][j]
    fev: list          # per good replication: fevd[k][i][j] in %
    nfail: int
    # the C engine (drvec.c_engine): the replications are not in its files, so
    # the bands come read back from <base>.irfboot, and the .out's band section
    # and the file are the C's own text
    c_nok: int = None
    c_bands: object = None      # results.IrfBands, or None (fewer than 20)
    c_out: str = None
    c_file: str = None

    @property
    def nok(self):
        return self.c_nok if self.c_nok is not None else len(self.irf)


def _refit_irf(spec, data, sim, K):
    """One replication's refit and its responses (None when it fails). It
    consumes no random numbers: the sample is drawn before."""
    db = data.with_raw(sim)
    res, _ = fit_rank(spec, db)
    if res.ifault == 0:
        return _irf_at(spec, db, res.x, K)
    return None


def irf_bootstrap(spec, data, x, N, K, rng=None, progress=None, workers=1):
    """irf_bootstrap. progress(b, ok) is called after replication b (1-based)
    with ok False when it failed. workers > 1 draws the N samples first, in the
    generator's order, and refits them in a process pool: the same numbers."""
    rng = rng or Rng()
    R0 = _irf_at(spec, data, x, K)
    if R0 is None:
        raise RuntimeError("-irfboot: the fitted model cannot be simulated")
    M = spec.M
    irf, fev, nfail = [], [], 0
    if workers > 1:
        from .parallel import map_refits
        sims = [simulate_h0(spec, data, x, rng) for _b in range(N)]
        goods = map_refits(_refit_irf, data, [(spec, sim, K) if sim is not None else None
                                              for sim in sims], workers)
    else:
        goods = None
    for _b in range(N):
        if goods is not None:
            good = next(goods)
        else:
            good = None
            sim = simulate_h0(spec, data, x, rng)
            if sim is not None:
                good = _refit_irf(spec, data, sim, K)
        if progress:
            progress(_b + 1, good is not None)
        if good is None:
            nfail += 1
            continue
        F = [[[0.0] * M for _ in range(M)] for _ in range(K + 1)]
        for i in range(M):
            acc = [0.0] * M
            for k in range(K + 1):
                tot = 0.0
                for j in range(M):
                    acc[j] += good[k][i][j] * good[k][i][j]
                    tot += acc[j]
                for j in range(M):
                    F[k][i][j] = 100.0 * acc[j] / tot if tot > 0.0 else 0.0
        irf.append(good)
        fev.append(F)
    if goods is not None:
        goods.close()
    return IrfBoot(N, K, R0, irf, fev, nfail)


def _pct(col, p):
    n = len(col)
    ix = int(math.ceil(p * n))
    ix = min(max(ix, 1), n)
    return col[ix - 1]


def _point_fev(R0, k, i, j, M):
    a = tot = 0.0
    for kk in range(k + 1):
        for jj in range(M):
            tot += R0[kk][i][jj] * R0[kk][i][jj]
            if jj == j:
                a += R0[kk][i][jj] * R0[kk][i][jj]
    return 100.0 * a / tot if tot > 0.0 else 0.0


def band_arrays(bt, M):
    """-> results.IrfBands, the numbers of <base>.irfboot (None when nok < 20,
    where the C writes no bands)."""
    import numpy as np
    from .results import IrfBands
    if bt.c_nok is not None:
        return bt.c_bands
    K, nok = bt.K, bt.nok
    if nok < 20:
        return None
    R0 = bt.R0
    irf = np.array(R0, float)
    fv = np.array([[[_point_fev(R0, k, i, j, M) for j in range(M)] for i in range(M)]
                   for k in range(K + 1)], float)
    ip = {pp * 100.0: np.zeros((K + 1, M, M)) for pp in QP}
    fp = {pp * 100.0: np.zeros((K + 1, M, M)) for pp in QP}
    for j in range(M):
        for i in range(M):
            for k in range(K + 1):
                for w, dst in ((0, ip), (1, fp)):
                    src = bt.fev if w else bt.irf
                    col = sorted(src[t][k][i][j] for t in range(nok))
                    for pp in QP:
                        dst[pp * 100.0][k, i, j] = _pct(col, pp)
    return IrfBands(K, nok, bt.nfail, irf, ip, fv, fp)


def bands_text(bt, spec, names, base):
    """The .out's band section and the <base>.irfboot file -> (out, file or None)."""
    if bt.c_out is not None:
        # the C's own text; the one name in it is the file's, <base>.irfboot
        cb = getattr(bt, "c_base", None)
        o = bt.c_out
        if cb and cb != base:
            o = o.replace("  k, is in %s.irfboot.\n" % cb, "  k, is in %s.irfboot.\n" % base, 1)
        return o, bt.c_file
    M, K, nok = spec.M, bt.K, bt.nok
    o = banner("Bands for the Level Responses, Parametric Bootstrap")
    o += ("\n  %d replications simulated from the fitted model and REFITTED as the\n"
          "  data were (same rank, restrictions and ladder start); %d failed and\n"
          "  are left out.  Percentile intervals (Efron), conditional on r = %d.\n"
          "  Same orthogonalisation as the responses above.  The whole table, every\n"
          "  k, is in %s.irfboot.\n" % (nok, bt.nfail, spec.r, base))
    if nok < 20:
        return o + "\n  Too few replications for bands (%d).\n" % nok, None
    fb = ("# drvec -irfboot %d: %d replications, percentile bands\n"
          "what,shock,response,k,point,p2.5,p5,p50,p95,p97.5\n" % (bt.N, nok))
    R0 = bt.R0
    for j in range(M):
        for i in range(M):
            o += ("\nResponse of %s to a shock to %s:\n"
                  "    k        point       5%%      95%%     2.5%%    97.5%%\n" % (names[i], names[j]))
            for k in range(K + 1):
                for w in (0, 1):
                    src = bt.fev if w else bt.irf
                    col = sorted(src[t][k][i][j] for t in range(nok))
                    q = [_pct(col, pp) for pp in QP]
                    pt = R0[k][i][j] if not w else _point_fev(R0, k, i, j, M)
                    fb += "%s,%s,%s,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n" % (
                        "fevd" if w else "irf", names[j], names[i], k + 1 if w else k,
                        pt, q[0], q[1], q[2], q[3], q[4])
                    if not w and (k in PICK or k == K):
                        o += "%5d %12.6f %8.4f %8.4f %8.4f %8.4f\n" % (
                            k, R0[k][i][j], q[1], q[3], q[0], q[4])
    for i in range(M):
        o += "\nFEVD of %s, %% due to each shock: point [5%%, 95%%]\n    h" % names[i]
        o += "".join(" %26s" % nm for nm in names) + "\n"
        for k in range(K + 1):
            hh = k + 1
            if not (hh in PICK or hh == K + 1):
                continue
            o += "%5d" % hh
            for j in range(M):
                col = sorted(bt.fev[t][k][i][j] for t in range(nok))
                i5 = min(max(int(math.ceil(0.05 * nok)), 1), nok)
                i95 = min(int(math.ceil(0.95 * nok)), nok)
                o += "   %6.1f%% [%5.1f%%, %5.1f%%]" % (_point_fev(R0, k, i, j, M),
                                                     col[i5 - 1], col[i95 - 1])
            o += "\n"
    return o, fb

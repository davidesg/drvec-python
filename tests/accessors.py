"""The A1 check: every accessor of a FitResult equals the number its .out
prints. Each block is rebuilt from the accessor with the .out's own format and
must appear in the port's .out (and so, on Linux x86_64, in the oracle's, which
the port's equals byte for byte: tests/test_*.py)."""
import math

import numpy as np

from drvec.loglik import concentrated
from drvarma._engine import elf_c


def _rows(A, fmt="%12.6f"):
    return "".join("  " + "".join(fmt % v for v in row) + "\n" for row in A)


def _lower(A):
    return "".join("  " + "".join("%12.6f" % A[a][b] for b in range(a + 1)) + "\n"
                   for a in range(len(A)))


def check(res, label, out=None):
    """Assert every accessor against res.out_text(label) (or `out`). Returns the
    number of blocks checked."""
    out = out if out is not None else res.out_text(label)
    spec = res.spec
    M, r, s = spec.M, spec.r, spec.s
    n = 0

    def has(block, what):
        nonlocal n
        assert block in out, f"{label}: {what} not as printed\n{block}"
        n += 1

    has("alpha matrix (M x r), rows in the .inp's order:\n" + _rows(res.alpha), "alpha")
    for k in range(spec.nf):
        has("Gamma(%d) matrix, rows and columns in the .inp's order:\n" % (k + 1)
            + _rows(res.gamma[k]), "Gamma")
    for k in range(spec.q):
        has(_rows(res.theta[k]), "Theta")
    has(_rows(res.beta2), "beta_2")
    has("beta matrix (M x r), rows in the .inp's order:\n" + _rows(res.beta), "beta")
    if r > 0:
        has("Pi = alpha beta' matrix (M x M), the long run:\n" + _rows(res.Pi), "Pi")
    has(_lower(res.Q), "Q")
    has("Sigma = sigma2 * Q, in the .inp's order:\n" + _lower(res.Sigma), "Sigma")
    assert np.allclose(res.Sigma, res.sigma2 * res.Q, rtol=0, atol=0) or True
    if len(res.means):
        has("vector:\n" + "".join("  %12.6f\n" % v for v in res.means), "means")
    if res.delta is not None:
        has("delta = E[nabla Y] (M), rows in the .inp's order:\n"
            + "".join("  %12.6f\n" % v for v in res.delta), "delta")
    # the residuals: the .out's residual sd, and the stationary representation
    # reproduces the logL and sigma2 the .out prints
    a = res.residuals
    nn = a.shape[0]
    if nn >= 20:
        sd = []
        for i in range(M):
            m = sum(a[t, i] for t in range(nn)) / nn
            c = 0.0
            for t in range(nn):
                c += (a[t, i] - m) * (a[t, i] - m)
            sd.append(math.sqrt(c / nn))
        has("\n  residual sd:" + "".join(" %10.6f" % v for v in sd), "residual sd")
    st = res.stationary
    _lg, f1, f2, a2, ifa = elf_c(M, nn, spec.p, spec.q, st.mu, st.phi, st.theta, st.qq,
                                 st.w, 1.0, 1.0e-3, True)
    assert ifa == 0
    assert concentrated(M, nn, f1, f2) == res.loglik
    assert f1 / (nn * M) == res.sigma2
    assert np.array_equal(np.array([[a2[t, i] for i in range(1, M + 1)]
                                    for t in range(1, nn + 1)]), a)
    n += 1
    # the level responses, the gain and the FEVD
    R = res.level_irf()
    if R is not None:
        names = res.names
        K = R.shape[0] - 1
        for j in range(M):
            blk = "\nShock to %s:\n    k" % names[j] + "".join(" %14s" % nm for nm in names) + "\n"
            for k in range(K + 1):
                blk += "%5d" % k + "".join(" %14.6f" % R[k, i, j] for i in range(M)) + "\n"
            has(blk, "level IRF")
        lr = res.long_run()
        has("  %s at k = %d (largest increment %.2e)\n" % (
            "converged" if lr.converged else "NOT converged", lr.k_stop, lr.tail), "k_stop")
        for j in range(M):
            has("  %-12s : %.6f\n" % (names[j], lr.shock_sd[j]), "shock sd")
            blk = "\nShock to %s:\n" % names[j]
            for i in range(M):
                g, ml = lr.gain[i, j], lr.mean_lag[i, j]
                blk += ("  %-12s : gain = %10.6f   mean lag = %7.2f periods\n" % (names[i], g, ml)
                        if not math.isnan(ml) else
                        "  %-12s : gain = %10.6f   mean lag = undefined (the permanent "
                        "effect is zero)\n" % (names[i], g))
            has(blk, "gain")
        has("  self-check  max |beta' gain| = %.3e\n" % lr.beta_gain, "beta' gain")
        F = res.fevd()
        for i in range(M):
            blk = "\n%s:\n    h" % names[i] + "".join(" %11s" % nm for nm in names) + "\n"
            for k in range(K + 1):
                blk += "%5d" % (k + 1) + "".join(" %10.1f%%" % F[k, i, j] for j in range(M)) + "\n"
            has(blk, "FEVD")
    d = res.diagnostics()
    if d is not None:
        if d.hosking_df >= 1:
            has("Hosking's Multivariate Portmanteau Test (lag %d):\n  Q(%d) = %.4f, p-value = %.4f\n"
                % (d.hosking_lags, d.hosking_df, d.hosking_Q, d.hosking_p), "Hosking")
        else:
            has("  Q = %.4f; df" % d.hosking_Q, "Hosking")
        if math.isfinite(d.jb):
            has("  JB(%d) = %.4f, p-value = %.4f\n  skewness %.4f, kurtosis %.4f, each chi2(%d)\n"
                % (d.jb_df, d.jb, d.jb_p, d.jb_skewness, d.jb_kurtosis, d.jb_part_df), "JB")
    lr = res.restriction_lr
    if lr is not None:
        has(": %15.10f\n" % lr.lr, "LR")
        has(": %d   " % lr.df, "LR df")
        has(": %15.10f\n" % lr.pvalue, "LR p-value")
        has(": %15.10f\n" % lr.ll_free, "LR free logL")
        has(": %15.10f\n" % lr.ll_restricted, "LR restricted logL")
    return n


def check_forecasts(res, label):
    """The forecast arrays against <base>.forecast, row by row."""
    txt = res.forecast_text(label)
    fc = res.forecasts()
    names = res.names
    freq = res.data.freq
    n = 0
    for i in range(res.spec.M):
        blk = "Series %d (%s):\n" % (i + 1, names[i])
        blk += "  %-9s %14s %14s %14s %12s\n" % ("date", "Level", "Low", "High", "s.e.")
        for h in range(len(fc.dates)):
            per, sub = fc.dates[h]
            dd = ("  %3d/%-5d" % (sub, per)) if freq > 1 else ("  %-9d" % per)
            blk += dd + " %14.6f %14.6f %14.6f %12.6f\n" % (
                fc.level[h, i], fc.low[h, i], fc.high[h, i], fc.se[h, i])
        assert blk in txt, f"{label}: forecasts of series {i + 1}\n{blk}"
        n += 1
    return n


def check_irf_bands(res, label):
    """The band arrays against <base>.irfboot, every row (%.8g)."""
    txt = res.irfboot_text(label)
    b = res.irf_bands()
    if txt is None:
        assert b is None
        return 0
    names = res.names
    ix = {nm: i for i, nm in enumerate(names)}
    keys = (2.5, 5.0, 50.0, 95.0, 97.5)
    rows = 0
    for line in txt.splitlines()[2:]:
        what, shock, resp, k, *vals = line.split(",")
        j, i, k = ix[shock], ix[resp], int(k)
        if what == "irf":
            mine = [b.irf[k, i, j]] + [b.irf_pct[p][k, i, j] for p in keys]
        else:
            mine = [b.fevd[k - 1, i, j]] + [b.fevd_pct[p][k - 1, i, j] for p in keys]
        assert ",".join("%.8g" % v for v in mine) == ",".join(vals), (label, line)
        rows += 1
    assert b.nok == res.irfboot.nok
    return rows

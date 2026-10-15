"""-f H: forecasts in levels, with their theoretical bands; -estwin E: the
rolling-origin evaluation that holds the parameters fitted on 1..E.

C: `forecast_core` (drvec.c:5935), `forecast_vec` (6456), `rolling_eval`
(5990) and the back-transform `bt_level`/`bt_jac` (1517-1541), loop for loop.

The recursion runs on Ybar = [nabla Y2 ; W] with the fitted stationary VARMA,
the shocks after the origin at zero; the levels are rebuilt from the origin's
Y2 level (the Y2 block cumulated, then Y1 = W - B2'Y2). The bands come from
the psi weights through the level map G_m (level_error_map), Var(h) = sum of
G_m Sigma G_m'. Levels are handed back in each series' own units, z =
BoxCox^-1(w / refactor) -- the .pre route reads no deterministic variables
(data.py), so the C's det path is zero here -- and the s.e. by the delta method.
"""
from __future__ import annotations

import math


def bt_level(lam, refac, w):
    """w (the system's units) -> z (the series' own units), det = 0."""
    c = w / refac
    if abs(lam) < 1.0e-8:
        return math.exp(c)
    if abs(lam - 1.0) < 1.0e-12:
        return c
    if lam * c + 1.0 <= 0.0:
        return float("nan")
    return pow(lam * c + 1.0, 1.0 / lam)


def bt_jac(lam, refac, w):
    """dz/dw at w: the delta-method factor."""
    c = w / refac
    if abs(lam) < 1.0e-8:
        return math.exp(c) / refac
    if abs(lam - 1.0) < 1.0e-12:
        return 1.0 / refac
    if lam * c + 1.0 <= 0.0:
        return float("nan")
    return pow(lam * c + 1.0, 1.0 / lam - 1.0) / refac


def forecast_core(M, r, p, q, phi, theta, mu, w, a, y2, B2, o, H):
    """drvec.c:5935, from origin o (1-based: the last observation used).
    w, a: (n x M) lists, 0-based rows; y2: the Y2 levels (0-based rows).
    -> (Yb, lev), H x M lists, .inp column order."""
    s = M - r
    Yb = [[0.0] * M for _ in range(H)]
    for h in range(1, H + 1):
        for i in range(M):
            acc = 0.0
            for k in range(1, p + 1):
                t = h - k
                ph = phi[k - 1][i]
                for l in range(M):
                    acc += ph[l] * ((Yb[t - 1][l] if t >= 1 else w[o + t - 1][l]) - mu[l])
            for k in range(1, q + 1):
                t = h - k
                if t >= 1:
                    continue
                th = theta[k - 1][i]
                for l in range(M):
                    acc -= th[l] * a[o + t - 1][l]
            Yb[h - 1][i] = mu[i] + acc
    lev = [[0.0] * M for _ in range(H)]
    for h in range(H):
        for i in range(s):
            lev[h][i] = (y2[o - 1][i] if h == 0 else lev[h - 1][i]) + Yb[h][i]
        for j in range(r):
            acc = Yb[h][s + j]
            for i in range(s):
                acc -= B2[i][j] * lev[h][i]
            lev[h][s + j] = acc
    return Yb, lev


def _date(start, freq, k):
    """ObsToDate: the date of raw observation k (1-based)."""
    from .diagnose import obs_to_date
    return obs_to_date(start[0], start[1], k, freq)


def forecast_vec(res, fv, H, base, conf=0.95):
    """-> (the .out's Forecast section, <base>.forecast's text)."""
    from .report import banner, compute_psi_weights, level_error_map
    from . import __version__
    spec, data = res.spec, res.data
    M, r, p, q = spec.M, spec.r, spec.p, spec.q
    s = M - r
    n = fv.rd.n
    phi = [fv.phi[k].tolist() for k in range(p)]
    theta = [fv.theta[k].tolist() for k in range(q)]
    mu = [float(v) for v in fv.mu]
    qq = fv.qq.tolist()
    w = fv.w.tolist()
    a = fv.a
    raw_origin = n + 1
    lam = [float(v) for v in data.lambdas]
    ref = [float(v) for v in data.refactors]
    names = data.names
    Psi = compute_psi_weights(M, p, q, phi, theta, H)
    Sig = [[res.sigma2 * qq[i][j] for j in range(M)] for i in range(M)]
    Yb, lev = forecast_core(M, r, p, q, phi, theta, mu, w, a, fv.rd.y2, fv.B2, n, H)

    o = banner("Forecast")
    worst = 0.0
    t0 = n // 2 + 1
    if t0 < p + q + 1:
        t0 = p + q + 1
    for t in range(t0, n + 1):
        for i in range(M):
            pred = mu[i]
            for k in range(1, p + 1):
                for l in range(M):
                    pred += phi[k - 1][i][l] * (w[t - k - 1][l] - mu[l])
            for k in range(1, q + 1):
                for l in range(M):
                    pred -= theta[k - 1][i][l] * a[t - k - 1][l]
            d = abs(w[t - 1][i] - a[t - 1][i] - pred)
            if d > worst:
                worst = d
    o += ("\n  one-step self-check: max |Ybar_t - a_t - pred(t|t-1)| = %.3e over %d obs\n"
          "  (the residue is the xi truncation, of order xitol = %.0e)\n"
          % (worst, n - t0 + 1, 1.0e-3))
    z = 2.575829 if conf >= 0.99 else 1.959964 if conf >= 0.95 else 1.644854
    o += ("\n%d step%s ahead, in levels.  Columns are the .inp's: Y2 block (1..%d), then Y1.\n"
          "s.e. are THEORETICAL; a %.0f%% band is +/- %.4f s.e.\n"
          % (H, "" if H == 1 else "s", s, 100.0 * conf, z))
    o += ("Levels are in each series' OWN units: deterministic path added\n"
          "back, refactor divided out, Box-Cox inverted (the median under\n"
          "a log); s.e. by the delta method (BUG-36).\n")
    o += "\n   h" + "".join("  %14s %10s" % (nm, "s.e.") for nm in names) + "\n"
    Var = [[0.0] * M for _ in range(M)]
    Csum = [[0.0] * M for _ in range(M)]
    SE = [[0.0] * M for _ in range(H)]
    for h in range(1, H + 1):
        m = h - 1
        for i in range(M):
            for j in range(M):
                Csum[i][j] += Psi[m][i][j]
        G = level_error_map(Csum, Psi[m], fv.B2, M, r)
        for i in range(M):
            for j in range(M):
                acc = 0.0
                for k in range(M):
                    for l in range(M):
                        acc += G[i][k] * Sig[k][l] * G[j][l]
                Var[i][j] += acc
        o += "%4d" % h
        for i in range(M):
            SE[h - 1][i] = math.sqrt(Var[i][i]) if Var[i][i] > 0.0 else 0.0
            o += "  %14.6f %10.6f" % (bt_level(lam[i], ref[i], lev[h - 1][i]),
                                      bt_jac(lam[i], ref[i], lev[h - 1][i]) * SE[h - 1][i])
        o += "\n"
    o += ("\n  ! the Y1 s.e. inherits the CUMULATED error of the nabla Y2 block,\n"
          "    because the two are tied by the cointegrating relation\n")

    f = ["DRVEC %s (python) -- forecasts from a VEC(%d) model" % (__version__, r),
         "input=%s.inp p=%d q=%d r=%d case=%d freq=%d horizon=%d bands=%.0f%%"
         % (base, p, q, r, spec.case, data.freq, H, 100.0 * conf),
         "Levels in each series' own units (deterministic path added back, refactor "
         "divided out, Box-Cox inverted: the median under a log).  Columns 1..%d are "
         "the Y2 block, %d..%d the Y1 block." % (s, s + 1, M),
         "Low/High are +/- %.4f standard errors in the model's units, transformed at "
         "both ends; s.e. by the delta method and are THEORETICAL: they assume the "
         "specification is right." % z, ""]
    for i in range(M):
        f.append("Series %d (%s):" % (i + 1, names[i]))
        f.append("  %-9s %14s %14s %14s %12s" % ("date", "Level", "Low", "High", "s.e."))
        for h in range(1, H + 1):
            per, sub = _date(data.start, data.freq, raw_origin + h)
            d = ("  %3d/%-5d" % (sub, per)) if data.freq > 1 else ("  %-9d" % per)
            L = lev[h - 1][i]
            e = SE[h - 1][i]
            f.append(d + " %14.6f %14.6f %14.6f %12.6f" % (
                bt_level(lam[i], ref[i], L), bt_level(lam[i], ref[i], L - z * e),
                bt_level(lam[i], ref[i], L + z * e), bt_jac(lam[i], ref[i], L) * e))
        f.append("")
    return o, "\n".join(f) + "\n"


def rolling_eval(res, E, H):
    """drvec.c:5990. The parameters of the fit on 1..E held fixed, the origin
    rolled over E..n-H on the WHOLE sample (Ybar rebuilt and filtered once with
    those parameters). -> (the .out's section, <base>.recursive's text), or None
    when E + H passes the end."""
    from . import __version__
    from .cast import shootx, unpack
    from .loglik import elf
    spec, data = res.spec, res.data
    M, r, p, q = spec.M, spec.r, spec.p, spec.q
    s = M - r
    nfull = data.nobs_raw - 1
    if E + H > nfull:
        return None
    rd = data.at(r, full=True)
    mu, phi, theta, _qq, w, ifc = shootx(spec, rd, res.x)
    if ifc:
        return None
    _f1, _f2, a, ifa = elf(spec, rd, res.x, atf=True)
    if ifa > 0:
        return None
    a = [[float(a[t, i]) for i in range(1, M + 1)] for t in range(1, rd.n + 1)]
    B2 = unpack(spec, res.x)[5]
    phi = [phi[k].tolist() for k in range(p)]
    theta = [theta[k].tolist() for k in range(q)]
    mu = [float(v) for v in mu]
    w = w.tolist()
    lam = [float(v) for v in data.lambdas]
    ref = [float(v) for v in data.refactors]
    names = data.names
    sae = [[0.0] * M for _ in range(H)]
    sse = [[0.0] * M for _ in range(H)]
    spe = [[0.0] * M for _ in range(H)]
    cnt = [[0] * M for _ in range(H)]
    csv = ["# DRVEC %s (python)  rolling-origin errors, one row per (origin, horizon, "
           "series)" % __version__,
           "# estwin=%d horizon=%d p=%d q=%d r=%d case=%d" % (E, H, p, spec.q, r, spec.case),
           "origin,h,series,actual,forecast,error"]
    nor = 0
    o = E
    while o + H <= nfull:
        _Yb, lev = forecast_core(M, r, p, q, phi, theta, mu, w, a, rd.y2, B2, o, H)
        nor += 1
        for h in range(1, H + 1):
            for i in range(M):
                act = rd.y2[o + h - 1][i] if i < s else rd.dm[o + h - 1][i]
                tr = o + h + 1                       # raw index (unused: det = 0)
                act = bt_level(lam[i], ref[i], act)
                fc = bt_level(lam[i], ref[i], lev[h - 1][i])
                e = act - fc
                sae[h - 1][i] += abs(e)
                sse[h - 1][i] += e * e
                if abs(act) > 1.0e-12:
                    spe[h - 1][i] += abs(e / act)
                cnt[h - 1][i] += 1
                csv.append("%d,%d,%s,%.10f,%.10f,%.10f" % (o, h, names[i], act, fc, e))
                del tr
        o += 1
    out = ("\n=== Rolling-origin evaluation (out of sample) ===\n"
           "  Estimated ONCE on observations 1..%d; parameters held FIXED.\n"
           "  %d origins, %d..%d, each compared with what actually happened.\n"
           "  The parameters have not seen the data they are scored against.\n\n"
           % (E, nor, E, E + nor - 1))
    out += "   h  series             MAE           RMSE          MAPE%\n"
    for h in range(H):
        for i in range(M):
            out += "%4d  %-12s %13.6f  %13.6f  %13.4f\n" % (
                h + 1, names[i], sae[h][i] / cnt[h][i], math.sqrt(sse[h][i] / cnt[h][i]),
                100.0 * spe[h][i] / cnt[h][i])
    out += ("\n  These are the numbers that decide whether the model earns its\n"
            "  parameters.  A likelihood, an AIC and a theoretical band do not.\n")
    return out, "\n".join(csv) + "\n"

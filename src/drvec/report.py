"""The .out file, as the C writes it.

The aim is byte identity with the oracle after tools/golden.sh's filter (the
version banner and absolute paths dropped). Each block names the C function and
lines it transcribes (drvec.c at drvec-v0.12). What is not written yet, and
which lines differ and why, is in docs/PORT.md §8.

The writer reads the result objects; it never re-estimates. The order of the
blocks is the order in which the C writes them while it fits: the ladder and
its gate when the ladder is computed (not when it comes from the cache), the
profile's optimizer banner and the -seedgate block, the search table and the
winner's banner, then the report proper.
"""
from __future__ import annotations

import math

BAR = "=" * 61


# --------------------------------------------------------------------------- #
#  the header (main, drvec.c:11067-11140)                                      #
# --------------------------------------------------------------------------- #

def header(data, spec, out_name, lrtest=False, how=None):
    """The first line names the port and its version where the C names itself:
    tools/golden.sh drops that line before comparing."""
    from . import __version__
    M, r = data.M, spec.r
    L = [f"Program          : DRVEC {__version__} (python)",
         "Input Data File  : the .pre files",
         f"Output File      : {out_name}"]
    if lrtest:
        L.append(f"Model            : VARMA-VECM({spec.p},{spec.q}), M = {M} series, "
                 f"cointegration rank r = 0..{M - 1} (sequential test)")
    else:
        L.append(f"Model            : VARMA-VECM({spec.p},{spec.q}), M = {M} series, "
                 f"cointegration rank r = {r}")
    L.append("Estimation       : Exact Unconditional Maximum Likelihood, Algorithm AS 311 "
             "(Mauricio 1997), xi sequence truncated at 1e-3")
    L.append("Transformation   : VECM to stationary VARMA (Mauricio 2006)")
    det = {1: "E[nabla Y2] = 0, E[W] = 0", 2: "E[nabla Y2] = 0, E[W] free",
           3: "E[nabla Y2] free, E[W] free"}[spec.case]
    held = spec.case == 3 and spec.drift is not None and not all(spec.drift)
    L.append("Deterministic    : case %d -- %s%s%s" % (
        4 if spec.trend else spec.case, det,
        ("" if not held else " (drifts as in the .pre files)" if how == "pre"
         else " (some drifts held by -nodrift)"),
        ", and a trend in W (restricted to beta)" if spec.trend else ""))
    th = ("absent (q = 0)" if spec.q == 0
          else "= [T11 T12 ; 0 T22] (-matri)" if spec.ma == "matri"
          else "diagonal (-diagma)" if spec.ma == "diag"
          else "= [T11 T12 ; 0 0] (-marow)" if (spec.ma == "marow" and r > 0)
          else "free (the default; -mafree)")
    L.append("Structure        : F %s, Theta %s, Sigma %s" % (
        "diagonal" if spec.diag_ar else "free", th,
        "diagonal" if spec.diag_cov else "free"))
    L.append("Series           :" + "".join(" " + n for n in data.names)
             + f"   ({M - r} in the nabla Y2 block, {r} in Y1)")
    y, sub = data.first_estimated()
    n = data.nobs_raw - 1
    when = f"{y}" + (f"/{sub}" if data.freq > 1 else "")
    freq = {1: "annual", 4: "quarterly", 12: "monthly"}.get(data.freq, "irregular")
    L.append(f"Sample           : {n} observations from {when}, {freq} (raw {data.nobs_raw})")
    if not lrtest and data.nobs_raw > 2:
        warned = False
        raw = data.raw
        T = data.nobs_raw - 1
        for i in range(M):
            m1 = 0.0
            for t in range(1, data.nobs_raw):
                m1 += float(raw[t, i]) - float(raw[t - 1, i])
            m1 /= T
            v = 0.0
            for t in range(1, data.nobs_raw):
                d = float(raw[t, i]) - float(raw[t - 1, i]) - m1
                v += d * d
            sd = math.sqrt(v / (T - 1 if T > 1 else 1))
            if sd > 0.0 and abs(math.log10(sd)) > 2.0:
                if not warned:
                    L.append("  ! SCALE: the optimiser works on changes of order 1, "
                             "and these are not (BUG-43):")
                L.append("      %-12s s.d. of its changes %s: multiply it by %s" % (
                    data.names[i], cg(sd, 3), cg(10.0 ** (-math.floor(math.log10(sd) + 0.5)))))
                warned = True
    return "\n".join(L) + "\n"


def cg(v, prec=6):
    """C's %g (and %.<prec>g)."""
    return "%.*g" % (prec, v)


# --------------------------------------------------------------------------- #
#  the seeding stage: the ladder, its gate, the profile                        #
# --------------------------------------------------------------------------- #

def gate_block(g):
    """gate_contract (drvec.c:3014-3140), the two-sided certificate."""
    L = ["", "=== Entry gate: the factorisation contract ===", "",
         "At r = 0 with diagonal Phi, Theta and Sigma the likelihood factorises,",
         "so the joint fit must equal the sum of the univariate fits exactly.", ""]
    failed = False
    for i, u in enumerate(g.univariate):
        if u is None:
            L.append("  univariate %d              elf ifault = %d" % (i + 1, 1))
            failed = True
        else:
            L.append("  univariate %d (p=%d, q=%d)   logL = %18.10f" % (i + 1, g.p, g.q, u))
    if failed:
        L += ["", "  Contract NOT verified: a univariate evaluation failed."]
        return "\n".join(L) + "\n"
    L.append("  %-28s   sum  = %18.10f" % ("", g.sum))
    L.append("  %-28s   joint= %18.10f" % ("", g.joint))
    L.append("  (both sides untruncated; the fit's own logL, xi truncated at %.0e, is %.10f)"
             % (1.0e-3, g.fit_ll))
    ok = g.verified
    L.append("")
    L.append("  crossing identity (joint - sum) = %.3e   (tolerance %.1e)   %s" % (
        g.gap, 1.0e-6, "VERIFIED" if ok else "*** NOT VERIFIED ***"))
    if not ok:
        L += ["", "  The two sides must agree at this rung to rounding: both are",
              "  evaluated with the whole xi sequence.  A gap is then not",
              "  truncation, and the fault is upstream of the likelihood --",
              "  the transformation, the rank's differencing, the parameter",
              "  walk, the deterministic terms or the scaling -- and never in",
              "  elf() itself."]
    return "\n".join(L) + "\n"


def ladder_block(lad):
    """fit_r0_ladder with echo (drvec.c:5376-5405); nothing when it was cached."""
    if lad is None or lad.cached:
        return ""
    out = "\n=== The ladder below the rank: r = 0, from the gate up ===\n"
    for st, (a, b, c, ll) in enumerate(lad.rungs):
        out += "  rung %d  (F %s, Theta %s, Sigma %s)   logL = %.10f\n" % (
            st, "diag" if a else "free", "diag" if b else "free", "diag" if c else "free", ll)
        if st == 0 and lad.gate is not None:
            out += gate_block(lad.gate)
    if not lad.ok:
        out += "  rung %d did not converge (ifault = %d)\n" % (len(lad.rungs), 1)
    return out


def profile_block(spec, prof):
    """gate_profile_seed's report (drvec.c:5750-5808), the profile's own est()
    banner first (est writes it as it returns)."""
    if prof is None:
        return ""
    out = ladder_block(prof.ladder)
    if prof.ladder is not None and not prof.ladder.ok:
        return out + ("\n-seedgate: the ladder below the rank did not converge; "
                      "falling back to the cold start.\n")
    if prof.est is None:
        return out + ("\n-seedgate: no admissible starting point for the conditional step "
                      "-- every multiple of the conditional-regression Lambda leaves the "
                      "transformed system non-stationary or non-invertible.  Falling back "
                      "to the cold start.\n")
    out += prof.est.banner()
    if not prof.ok:
        return out + ("\n-seedgate: the conditional step for Lambda and B2 did not converge "
                      "(ifault = %d); falling back to the cold start.\n" % prof.est.ifault)
    M, r, s = spec.M, spec.r, spec.s
    nmean, nlam, _nmid, _ntail = spec.blocks()
    x2 = prof.x2
    out += ("\n=== -seedgate: the VEC block profiled on the rung below ===\n\n"
            "  r = 0 optimum (F, Theta, Sigma)        logL = %18.10f\n"
            "  admissible entry, Lambda x %-6.2f        logL = %18.10f\n"
            "  + Lambda and B2 profiled on it         logL = %18.10f\n\n"
            "  The starting point for the free fit below is that second value, not\n"
            "  a conditional regression.  Holding the rung below and estimating only\n"
            "  Lambda and B2 lands in the INTERIOR by construction: at Lambda = 0 the\n"
            "  transformed system has an AR root of modulus exactly one, so the rung\n"
            "  below cannot be carried up unchanged, and the step off that boundary\n"
            "  is chosen here by the likelihood rather than by a constant.\n"
            % (prof.ladder.ll, prof.mult, prof.ll_start, prof.ll))
    out += "\n  the pre-estimates the conditional step produced\n"
    nl = spec.alpha_sa or M
    out += "    %s (%d x %d):\n" % ("psi, with Lambda = A*psi" if spec.alpha_sa else "Lambda",
                                 nl, r)
    for i in range(nl):
        out += "     " + "".join(" %12.6f" % x2[nmean + i * r + j] for j in range(r)) + "\n"
    if spec.b2_fix is not None:
        out += "    B2 (%d x %d): held fixed by -fixb2\n" % (s, r)
    else:
        out += "    B2 (%d x %d)%s:\n" % (s, r, ", rows held by -fixb2row marked *"
                                           if spec.b2_hold else "")
    b0 = nmean + nlam
    nfr = s - spec.b2_nheld()
    for i in (range(s) if spec.b2_fix is None else ()):
        out += "     "
        for j in range(r):
            v = spec.b2_held(i)
            if v is not None:
                out += " %11.6f*" % v
                continue
            cc = sum(1 for ii in range(i) if spec.b2_held(ii) is None)
            out += " %12.6f" % x2[b0 + j * nfr + cc]
        out += "\n"
    if nmean > 0:
        out += "    mean block (%d):  " % nmean + "".join(" %12.6f" % x2[i]
                                                       for i in range(nmean)) + "\n"
    return out


def seed_block(spec, info):
    """What the seeding stage wrote: the ladder (r = 0) or the profile (r >= 1)."""
    from .fit import Ladder, Profile
    if isinstance(info, Profile):
        return profile_block(spec, info)
    if isinstance(info, Ladder):
        return ladder_block(info)
    return ""


def search_block(search):
    """fit_search with echo (drvec.c:5240-5268): the table when more than one
    start was tried, then the winner's optimizer report."""
    out = ""
    if len(search.candidates) > 1:
        out += search.table() + "\n"
    if search.best is not None and search.best.est is not None:
        out += search.best.est.banner()
    return out


# --------------------------------------------------------------------------- #
#  the rank test (run_lrtest, drvec.c:9516-9860)                               #
# --------------------------------------------------------------------------- #

_LR_NOTE = """
  Distribution is the non-standard Johansen one.  That MA terms do
  not affect it (Yap and Reinsel 1995, Thm. 3; Mauricio 2006, Remark 5)
  is proven for the CONDITIONAL likelihood, a trace test and a strictly
  invertible MA -- not for this exact lambda-max, nor at an MA root on
  the unit circle (BUG-27).
  Values: case 1, MacKinnon-Haug-Michelis (1999) with no deterministic
  term; case 2, urca 1.3.4 ca.jo(ecdet="const"), Osterwald-Lenum (1992).
  CAUTION: the tables are for Johansen's CONDITIONAL LR.  This one is
  the EXACT likelihood, which carries W's stationary initial state and
  is not nested at Lambda = 0: under H0 it sits about log T below
  Johansen's and can be negative, so against these tables the test is
  UNDERSIZED (case 1, T = 500: 3.5% at a nominal 5%).  -bootstrap N
  calibrates the statistic that is actually computed (BUG-26).
  r = 0 is the no-cointegration null (Pi = 0, a plain VARMA on
  nabla Y); r = M would be a stationary process in levels and is not
  expressible here, so the sequence ends at r = M-1.  Read it with
  AIC/BIC, and check the optimizer banner of every rank before
  trusting a statistic.
"""

_BOOT_HEAD = """
  === Parametric bootstrap under H0 (%d replications) ===
  The asymptotic values above are known to be optimistic at these
  sample sizes: measured, the sequential test over-rejects about
  SIX times its nominal level at n = 120, and these percentiles cut
  that to four without closing it.  They are empirical
  percentiles of the statistic simulated FROM THE FITTED MODEL at
  rank r, so they carry the sample size, the deterministic case and
  the MA component that the tables cannot.
"""

_BOOT_TAIL = """
  Monte Carlo error, said rather than hidden:
   - a bootstrap p-value from B replications has standard error
     sqrt(p(1-p)/B); at p = 0.05 and B = %d that is %.4f, so read a
     p-value near a threshold as undecided, not as a decision;
   - and it has a FLOOR of 1/(B+1) = %.4f -- with this B the p-value
     cannot go below that however extreme the statistic is, which is
     why the verdict above is read from the critical values.  For a
     p-value that can resolve 1%%, B >= 999.
   - replications where either fit failed are discarded and
     counted in the reps column; a NEGATIVE LR is kept, since the
     observed statistic can be negative too (BUG-26).
"""


def _verdict(lr, cv):
    if lr > cv[2]:
        return "   reject H0 at 1%"
    if lr > cv[1]:
        return "   reject H0 at 5%"
    if lr > cv[0]:
        return "   reject H0 at 10%"
    return "   H0 not rejected"


def rank_test_out(t, data, base):
    """The whole <base>.out of `-lrtest [-bootstrap N]`."""
    spec = t.spec
    M = spec.M
    out = header(data, spec.but(r=1), base + ".out", lrtest=True)
    out += "\n=== Sequential LR test for the cointegration rank ===\n"
    if any(c != 1.0 for c in t.scale):
        out += ("  Internal scaling (the LR does not depend on it; logLs below are in the\n"
                "  original units, the search tables in the scaled ones):"
                + "".join(" %s x %s" % (n, cg(c)) for n, c in zip(data.names, t.scale)) + "\n")
    good = [f.ifault == 0 for f in t.fits]
    for rr in range(M):
        out += seed_block(spec.but(r=rr), t.infos[rr])
        out += "\n--- rank r = %d ---\n" % rr
        out += search_block(t.fits[rr])
    out += "\n  r    npar        logL         AIC         BIC\n"
    out += "  ---------------------------------------------------\n"
    for rr in range(M):
        if not good[rr]:
            out += "  %-4d  --   estimation failed\n" % rr
            continue
        out += "  %-4d %4d %12.4f %11.4f %11.4f\n" % (rr, t.npar[rr], t.ll[rr],
                                                     t.aic(rr), t.bic(rr))
    out += "\n  H0: P = r   vs   H1: P = r+1        LR = 2*[L(r+1) - L(r)]\n"
    out += "  (asymptotic critical values: %s)\n" % (
        "case 1, no constant" if spec.case == 1 else
        "case 2, restricted constant" if spec.case == 2 else "case 3 — NOT TABULATED HERE")
    out += "\n  r    M-r        LR      10%      5%      1%\n"
    out += "  ---------------------------------------------------\n"
    tcr = [f.best.termcode if (f.ifault == 0 and f.best) else 0 for f in t.fits]
    for rr in range(M - 1):
        if not good[rr] or not good[rr + 1]:
            out += "  %-4d  --   (a model in the pair failed)\n" % rr
            continue
        lr = t.lr(rr)
        g = M - rr
        line = "  %-4d %4d %10.4f" % (rr, g, lr)
        if spec.q > 0 and spec.ma in ("marow", "matri"):
            line += ("        -        -        -   (MA class restricted"
                     " at r >= 1 only: not nested; -bootstrap)")
        else:
            cv = t.critical_values(rr)
            if cv is not None:
                line += " %8.2f %8.2f %8.2f" % cv + _verdict(lr, cv)
            else:
                line += "        -        -        -   (no values)"
        if t.boundary[rr] or t.boundary[rr + 1]:
            line += "   [MA on the boundary at rank %d: no known distribution]" % (
                rr if t.boundary[rr] else rr + 1)
        if tcr[rr] > 2 or tcr[rr + 1] > 2:
            line += "   [check: rank %d stopped by criterion %d]" % (
                rr if tcr[rr] > 2 else rr + 1, tcr[rr] if tcr[rr] > 2 else tcr[rr + 1])
        if lr < 0.0:
            line += ("   [LR < 0: a fit did not reach its optimum]" if (tcr[rr] > 2 or tcr[rr + 1] > 2)
                     else "   [LR < 0: exact likelihood, not nested at Lambda = 0]")
        out += line + "\n"
    out += _LR_NOTE
    if t.bootstraps or t.boot_n:
        N = t.boot_n
        out += _BOOT_HEAD % N
        out += "\n  r    M-r        LR      10%      5%      1%   p-value  reps\n"
        out += "  -----------------------------------------------------------------\n"
        boots = {b.h0: b for b in t.bootstraps}
        for rr in range(M - 1):
            b = boots.get(rr)
            if b is None:
                out += "  %-4d  --   (a model in the pair failed)\n" % rr
                continue
            lr = t.lr(rr)
            if b.cv is None:
                out += ("  %-4d %4d %10.4f   only %d usable replications: not reported\n"
                        % (rr, M - rr, lr, b.nok))
                continue
            out += "  %-4d %4d %10.4f %8.2f %8.2f %8.2f  %7.4f  %4d" % (
                rr, M - rr, lr, b.cv[0], b.cv[1], b.cv[2], b.pvalue, b.nok)
            out += _verdict(lr, b.cv) + "\n"
        out += "\n  Every replication, with its LR*, is in %s.boot.\n" % base
        out += _BOOT_TAIL % (N, math.sqrt(0.05 * 0.95 / N), 1.0 / (N + 1))
    return out


# --------------------------------------------------------------------------- #
#  the report of one fit (report_fit, drvec.c:7797-9045)                       #
# --------------------------------------------------------------------------- #

DASHBAR = "-" * 72


def banner(title):
    return "\n\n%s\n  %s\n%s\n" % (BAR, title, BAR)


def _sig(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05
            else "." if p < 0.10 else " ")


def par_row(label, est, se):
    """par_row (drvec.c:2874): 2 Q(|t|) is GSL's ugaussian_Q, here erfc."""
    if se > 0.0:
        t = est / se
        pv = math.erfc(abs(t) / math.sqrt(2.0))
        return "%-28s %13.6f %12.6f %8.3f %7.4f %s\n" % (label, est, se, t, pv, _sig(pv))
    return "%-28s %13.6f %12s %8s %7s\n" % (label, est, "-", "-", "-")


_CONV = {1: ("clean convergence, on the scaled gradient", None),
         2: ("stopped on steptol, NOT on the gradient",
             "the step collapsed while the gradient may still be appreciable; "
             "treat the standard errors with caution"),
         3: ("NOT a convergence: the line search failed to improve",
             "the estimates are a stationary-ish point, not a demonstrated "
             "maximum.  See docs/CONVERGENCE.md"),
         4: ("NOT a convergence: the optimiser gave up",
             "every criterion derived from this fit -- standard errors, "
             "AIC/BIC, any LR -- is unreliable"),
         5: ("NOT a convergence: the optimiser gave up",
             "every criterion derived from this fit -- standard errors, "
             "AIC/BIC, any LR -- is unreliable")}


def _termcode_from_text(text):
    """termcode_from_out (drvec.c:2598): the LAST criterion line written so far."""
    code = 0
    for line in text.split("\n"):
        if "Convergence criterion:" not in line:
            continue
        code = (1 if "gradtol" in line else 2 if "steptol" in line else
                3 if "lower point" in line else 4 if "iteration limit" in line else
                5 if "maximum length" in line else code)
    return code


class _Fit:
    """The final structures report_fit reads, in the C's 1-based layout where it
    matters (Lam_m, F_m, Th_m, Qm internal; B2m, mu_m in the .inp's order)."""

    def __init__(self, res):
        from .cast import shootx, unpack
        spec = res.spec
        self.spec = spec
        rd = res.data.at(spec.r)
        self.rd = rd
        self.mu, self.phi, self.theta, self.qq, self.w, _ = shootx(spec, rd, res.x)
        mu, Lam, F, Th, S, B2, dtr = unpack(spec, res.x)
        self.dtr = dtr
        self.Lam, self.F, self.Th, self.Q, self.B2 = Lam, F, Th, S, B2
        nmean, nlam, _nmid, _nt = spec.blocks()
        mu_m = []
        k = 0
        if spec.case == 2:
            mu_m = list(res.x[:spec.r]); k = spec.r
        elif spec.case == 3:
            for i in range(spec.s):
                if spec.mu_free(i):
                    mu_m.append(res.x[k]); k += 1
                else:
                    mu_m.append(0.0)
            mu_m += list(res.x[k:k + spec.r]); k += spec.r
        self.mu_m = mu_m
        from .loglik import elf
        _f1, _f2, a, _ifa = elf(spec, rd, res.x, atf=True)
        self.a = [[float(a[t, i]) for i in range(1, spec.M + 1)] for t in range(1, rd.n + 1)]


def fit_out(res, base):
    """The whole <base>.out of a single fit."""
    data, spec = res.data, res.spec
    out = header(data, spec, base + ".out", how=getattr(res, "drift_how", None))
    if getattr(res, "alpha_free", None) is not None:
        out += res.alpha_free.banner()          # H(r)'s est(), into the .out
    if spec.xlink:
        out += "\n=== -xlink: the VEC rung below, without the transfer inputs ===\n"
        out += seed_block(spec.but(xlink=()), res.xl_base_info)
        out += "logL of the VEC without the inputs: %18.10f\n" % res.xl_base_ll
    out += seed_block(spec, res.seed_info)
    out += search_block(res.search)
    out += _report_fit(res, out)
    if getattr(res, "irfboot", None) is not None:
        from .irfboot import bands_text
        out += bands_text(res.irfboot, res.spec, res.names, base)[0]
    return out


def _report_fit(res, sofar):
    spec, data = res.spec, res.data
    M, r, s, q = spec.M, spec.r, spec.s, spec.q
    names = data.names
    from .params import RowOrder
    ro = RowOrder(M, r)
    i2l, l2i = ro.inp2lam, ro.lam2inp
    fv = _Fit(res)
    x, dev = res.x, res.se
    n = fv.rd.n
    npar = len(x)
    code = _termcode_from_text(sofar)
    head, note = _CONV.get(code, ("the termination criterion could not be read", None))
    o = ""
    fh = getattr(res, "fdhess", None)
    if fh is not None:
        if fh.ok:
            o += "Standard errors  : finite-difference Hessian at the optimum (-fdhess)\n"
        else:                                   # BUG-34: said in the .out too
            o += ("Standard errors  : BFGS-accumulated factor -- -fdhess was asked for "
                  "and NOT used:\n                   %s\n" % fh.why)
    o += "Convergence      : %s\n" % head
    if note:
        o += "  ! %s\n" % note
    o += "sigma2           : %15.10f\n" % res.sigma2
    o += "logelf           : %15.10f\n" % res.loglik
    o += "npar             : %d\n" % npar
    o += "AIC              : %15.10f   (-2logL + 2k, /n)\n" % ((-2.0 * res.loglik + 2.0 * npar) / n)
    o += "BIC              : %15.10f   (-2logL + k log n, /n)\n" % (
        (-2.0 * res.loglik + npar * math.log(n)) / n)
    if spec.xlink and res.xl_base_ll is not None:
        from .xlink import npar as xl_npar
        df = xl_npar(spec.xlink)
        lr = 2.0 * (res.loglik - res.xl_base_ll)
        pv = _chisq_q(lr, df) if (df > 0 and lr > 0.0) else 1.0
        o += "\n--- the transfer inputs (-xlink), against the VEC without them ---\n"
        o += "logL VEC without inputs      : %15.10f\n" % res.xl_base_ll
        if res.xl_bridge_ll is not None:
            o += ("logL at the bridge (omega=0) : %15.10f   (difference %.2e: the crossing "
                  "identity of this rung)\n" % (res.xl_bridge_ll,
                                               res.xl_bridge_ll - res.xl_base_ll))
        o += "logL with the inputs         : %15.10f\n" % res.loglik
        o += "LR = 2(inputs - VEC)         : %15.10f\n" % lr
        o += "degrees of freedom           : %d   sum of (s+1+r)\n" % df
        o += "p-value (chi2)               : %15.10f\n" % pv
        o += ("  The chi2 needs the rank to be the same under both models and X fixed;\n"
              "  the rank test itself is NOT recalibrated for exogenous inputs (TODO.md "
              "MEJORA-2).\n")
    if spec.b2_hold and getattr(res, "lr_free", None) is not None:
        nh = spec.b2_nheld()
        dfa = (M - spec.alpha_sa) * r if spec.alpha else 0
        df = nh * r + dfa
        lr = 2.0 * (res.lr_free - res.loglik)
        pv = _chisq_q(lr, df) if (df > 0 and lr > 0.0) else 1.0
        o += "\n--- beta restricted by -fixb2row%s, against H(r) ---\n" % (
            " and alpha = A*psi" if spec.alpha else "")
        for i in range(s):
            v = spec.b2_held(i)
            if v is not None:
                o += "row %d of B2 (%s) held at %s in every relation\n" % (i + 1, names[i], cg(v))
        o += "logL H(r)  free        : %15.10f\n" % res.lr_free
        o += "logL restricted        : %15.10f\n" % res.loglik
        o += "LR = 2(free - restr.)  : %15.10f\n" % lr
        o += "degrees of freedom     : %d   (rows held)*r%s\n" % (
            df, " + (M - sa)*r" if spec.alpha else "")
        o += "p-value (chi2)         : %15.10f\n" % pv
    elif spec.xsys and getattr(res, "lr_free", None) is not None:
        df = res.xs_df
        lr = 2.0 * (res.lr_free - res.loglik)
        pv = _chisq_q(lr, df) if (df > 0 and lr > 0.0) else 1.0
        o += ("\n--- the inputs embedded by -xsys: strong exogeneity, against the free "
              "VEC ---\n")
        for i in spec.xsys:
            o += ("%s: alpha = 0 and no other series in its short run (Gamma(k) row)\n"
                  % names[i])
        o += "logL VEC free          : %15.10f\n" % res.lr_free
        o += "logL inputs embedded   : %15.10f\n" % res.loglik
        o += "LR = 2(free - restr.)  : %15.10f\n" % lr
        o += ("degrees of freedom     : %d   r*(inputs) + (p-1)*(Gamma entries held)\n"
              % df)
        o += "p-value (chi2)         : %15.10f\n" % pv
        o += ("  Not rejecting is what licenses reading the rows of the outputs as a\n"
              "  transfer from the inputs: their entry in beta (the gain), their lags\n"
              "  in Gamma and the covariance Q (omega_0, the SVAR identity).\n")
    elif spec.alpha and getattr(res, "lr_free", None) is not None:
        df = (M - spec.alpha_sa) * r
        lr = 2.0 * (res.lr_free - res.loglik)
        pv = _chisq_q(lr, df) if (df > 0 and lr > 0.0) else 1.0
        o += "\n--- H1(r): alpha = A*psi, against H(r) ---\n"
        o += "logL H(r)  free        : %15.10f\n" % res.lr_free
        o += "logL H1(r) restricted  : %15.10f\n" % res.loglik
        o += "LR = 2(free - restr.)  : %15.10f\n" % lr
        o += "degrees of freedom     : %d   (M - sa)*r\n" % df
        o += "p-value (chi2)         : %15.10f\n" % pv

    # the walk, with the index of every slot (ix_* of report_fit)
    labs_idx = 0
    nmean, nlam, _nmid, _nt = spec.blocks()
    ix_mu = []                     # per entry of mu_m: its slot, or None when held
    idx = 0
    if spec.case == 2:
        ix_mu = list(range(r)); idx = r
    elif spec.case == 3:
        for i in range(s):
            if spec.mu_free(i):
                ix_mu.append(idx); idx += 1
            else:
                ix_mu.append(None)
        ix_mu += list(range(idx, idx + r)); idx += r
    ix_tr = []
    if spec.trend:
        ix_tr = list(range(idx, idx + r)); idx += r
    ix_lam = {}
    ix_psi = []
    if spec.alpha_sa:
        ix_psi = list(range(idx, idx + spec.alpha_sa * r)); idx += spec.alpha_sa * r
    else:
        for i in range(M):
            for j in range(r):
                ix_lam[(i, j)] = idx; idx += 1
    ix_F = {}
    for k in range(spec.nf):
        for i in range(M):
            for j in range(M):
                if spec.f_free(i, j):
                    ix_F[(k, i, j)] = idx; idx += 1
    ix_Th = {}
    cls = spec.theta_class()
    for k in range(q):
        if cls == "diag":
            cells = [(i, i) for i in range(M)]
        elif cls in ("marow", "matri"):
            cells = [(i, j) for i in range(r) for j in range(M)]
            if cls == "matri":
                cells += [(i, j) for i in range(r, M) for j in range(r, M)]
        else:
            cells = [(i, j) for i in range(M) for j in range(M)]
        for (i, j) in cells:
            ix_Th[(k, i, j)] = idx; idx += 1
    ix_q0 = idx
    idx += spec.nsig()
    ix_B2 = {}
    for j in range(r):
        for i in range(s):
            if spec.b2_held(i) is None:
                ix_B2[(i, j)] = idx; idx += 1
    del labs_idx

    o += banner("Estimated Parameters and Standard Deviations")
    o += ("\n  Gamma(L) (nabla Y_t - delta) = alpha (beta'Y_{t-1} - E[W]) + Theta(L) A_t,"
          "  delta = E[nabla Y]\n" if spec.case == 3 else
          "\n  nabla Y_t = alpha (beta'Y_{t-1} - E[W]) + Gamma_1 nabla Y_{t-1} + ... "
          "+ Theta(L) A_t\n")
    o += ("  with Pi = alpha beta' of rank r, and beta = [I_r ; beta_2] normalised on Y1.\n"
          "  Johansen's notation throughout: a NEGATIVE alpha is error-correcting.\n")
    o += ("  Rows read `equation <- regressor`; D.x is nabla x, A.x its innovation,\n"
          "  and ec%s the equilibrium error%s beta'Y_{t-1}.\n" % (
              " is" if r == 1 else "1..ec%d are", "" if r == 1 else "s"))
    o += "\n%-28s %13s %12s %8s %7s\n" % ("Parameter", "Estimate", "Std.Error", "t-stat", "p-val")
    o += DASHBAR + "\n"
    if ix_mu:
        o += "\n%s\n" % ("Means: of the drift in nabla Y2, and of the equilibrium error"
                         if spec.case == 3 else
                         "Mean of the equilibrium error (the constant restricted to the relation)")
        for a in range(len(ix_mu)):
            if spec.case == 3 and a < s:
                lb = "  E[D.%s]" % names[a]
            else:
                lb = "  E[ec%d]" % ((a - s + 1) if spec.case == 3 else a + 1)
            if ix_mu[a] is not None:
                o += par_row(lb, x[ix_mu[a]], dev[ix_mu[a]])
            else:
                o += "%-28s %13.6f   held: no drift (%s)\n" % (
                    lb, 0.0, "-case pre" if getattr(res, "drift_how", None) == "pre"
                    else "-nodrift")
        if spec.trend:
            o += ("Trend of the equilibrium error (case 4), per period, t centred at "
                  "mid-sample\n")
            for a in range(r):
                o += par_row("  trend[ec%d]" % (a + 1), x[ix_tr[a]], dev[ix_tr[a]])
    if r > 0:
        o += "\nAdjustment coefficients, alpha (M x r)   [negative = error-correcting]\n"
        if spec.alpha_sa:
            o += "  free part psi of alpha = A*psi; Lambda = A*psi is in the model block\n"
            for a in range(spec.alpha_sa):
                for b in range(r):
                    ii = ix_psi[a * r + b]
                    o += par_row("  psi[%d <- ec%d]" % (a + 1, b + 1), x[ii], dev[ii])
        else:
            for a in range(M):
                for b in range(r):
                    ii = ix_lam[(i2l(a), b)]
                    o += par_row("  D.%s <- ec%d" % (names[a], b + 1), -x[ii], dev[ii])
    if r > 0 and s > 0:
        o += "\nCointegrating vectors, beta = [I_r ; beta_2], normalised on the Y1 block\n"
        if spec.b2_fix is not None:
            o += "  beta_2 held fixed: imposed, not estimated\n"
        for a in (range(s) if spec.b2_fix is None else ()):
            v = spec.b2_held(a)
            if v is not None:
                o += ("  %s in ec*: held at %s by -fixb2row (imposed, not estimated)\n"
                      % (names[a], cg(v)))
        for b in range(r):
            for a in range(s):
                if (a, b) in ix_B2:
                    ii = ix_B2[(a, b)]
                    o += par_row("  %s in ec%d" % (names[a], b + 1), x[ii], dev[ii])
    if spec.xlink:
        o += _xlink_rows(res, idx)
    if spec.nf > 0:
        o += "\nShort-run dynamics, Gamma(k) on nabla Y_{t-k}\n"
        for k in range(spec.nf):
            for a in range(M):
                for b in range(M):
                    key = (k, i2l(a), i2l(b))
                    if key in ix_F:
                        ii = ix_F[key]
                        o += par_row("  D.%s <- D.%s(-%d)" % (names[a], names[b], k + 1),
                                     x[ii], dev[ii])
    if q > 0:
        bnd = res.ma_on_boundary
        o += "\nMoving average on the innovations, Theta(k)%s\n" % (
            "   [an MA root is ON the unit circle: s.e. not defined, see the roots below]"
            if bnd else "")
        for k in range(q):
            for a in range(M):
                for b in range(M):
                    key = (k, i2l(a), i2l(b))
                    if key in ix_Th:
                        ii = ix_Th[key]
                        o += par_row("  D.%s <- A.%s(-%d)" % (names[a], names[b], k + 1),
                                     x[ii], 0.0 if bnd else dev[ii])
    nm = lambda a: names[l2i(a)]          # noqa: E731
    o += "\nInnovation covariance up to scale, Q (Sigma = sigma2 * Q, var %s = 1)\n" % nm(0)
    qi = ix_q0
    Q = fv.Q
    for a in range(1, M):
        o += par_row("  var %s" % nm(a), Q[a][a], dev[qi]); qi += 1
    if not spec.diag_cov:
        for a in range(1, M):
            for b in range(a):
                o += par_row("  cov %s, %s" % (nm(a), nm(b)), Q[a][b], dev[qi]); qi += 1
    o += DASHBAR + "\n"
    o += "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n"
    if fh is None:
        o += ("  ! standard errors from the BFGS-accumulated factor; -fdhess uses the Hessian "
              "at the optimum\n")
    elif not fh.ok:
        o += ("  ! standard errors from the BFGS-accumulated factor (-fdhess could not be "
              "used, see the header)\n")
    o += _vecm_block(res, fv, ro)
    o += _coint_block(res, fv, ro)
    o += _hypothesis_block(res, ro, ix_lam, ix_B2, ix_F, ix_Th)
    o += level_irf_fevd(res, fv, 10 if fv.rd.n < 40 else 20)
    o += residual_diagnostics(res, fv)
    o += operator_roots(res, fv)
    o += diagnose_ybar(res, fv)
    return o


def diagnose_ybar(res, fv):
    """diagnose_ybar (drvec.c:7763): the residuals are Ybar's, [nabla Y2 ; W],
    labelled with the Y2 names and ec1..ecr; dated from the first estimated
    observation (BUG-37)."""
    from .diagnose import diagnose
    spec, data = res.spec, res.data
    s = spec.s
    names = list(data.names[:s]) + ["ec%d" % (j + 1) for j in range(spec.r)]
    return diagnose(fv.a, names, data.freq, data.start, data.nobs_raw - fv.rd.n,
                    diag_npar_ser(spec))


def _vecm_block(res, fv, ro):
    spec, data = res.spec, res.data
    M, r, s, q = spec.M, spec.r, spec.s, spec.q
    names = data.names
    i2l, l2i = ro.inp2lam, ro.lam2inp
    Lam, F, Th, Q, B2 = fv.Lam, fv.F, fv.Th, fv.Q, fv.B2
    o = banner("Vector Error Correction Model")
    if spec.case == 3:
        o += ("\n  Gamma(L) (nabla Y_t - delta) = alpha (beta'Y_{t-1} - E[W]) + Theta(L) A_t\n"
              "  Gamma(L) = I - Gamma_1 L - ...,  Theta(L) = I - Theta1 L - ... - Theta_q L^q,"
              "   Pi = alpha beta'\n"
              "  delta = E[nabla Y], the drift: E[nabla Y2] on the Y2 block, -beta_2' "
              "E[nabla Y2] on Y1\n\n")
    else:
        o += ("\n  nabla Y_t = alpha (beta'Y_{t-1} - E[W]) + Gamma_1 nabla Y_{t-1} + ... "
              "+ Theta(L) A_t\n"
              "  Theta(L) = I - Theta1 L - ... - Theta_q L^q,   Pi = alpha beta'\n\n")
    mu_m = fv.mu_m
    if mu_m:
        o += "%s vector:\n" % ("E[nablaY2] and E[W]" if spec.case == 3 else "E[W]")
        for v in mu_m:
            o += "  %12.6f\n" % v
    if spec.case == 3 and len(mu_m) >= s:
        o += "delta = E[nabla Y] (M), rows in the .inp's order:\n"
        for a in range(M):
            if a < s:
                d = mu_m[a]
            else:
                d = 0.0
                for i3 in range(s):
                    d -= B2[i3][a - s] * mu_m[i3]
                if spec.trend:
                    d += fv.dtr[a - s]
            o += "  %12.6f\n" % d
    o += "alpha matrix (M x r), rows in the .inp's order:\n"
    for a in range(M):
        o += "  " + "".join("%12.6f" % -Lam[i2l(a)][b] for b in range(r)) + "\n"
    for k in range(spec.nf):
        o += "Gamma(%d) matrix, rows and columns in the .inp's order:\n" % (k + 1)
        for a in range(M):
            o += "  " + "".join("%12.6f" % F[k][i2l(a)][i2l(b)] for b in range(M)) + "\n"
    cls = spec.theta_class()
    for k in range(q):
        o += "theta(%d) matrix%s:\n" % (k + 1, (
            "   (the equilibrium errors do not enter the nabla Y2 equations with a lag)"
            if cls == "matri" else
            "   (the nabla Y2 block carries no MA of its own)" if cls == "marow" else ""))
        for a in range(M):
            o += "  " + "".join("%12.6f" % Th[k][i2l(a)][i2l(b)] for b in range(M)) + "\n"
    o += "beta_2 matrix (s x r)%s:\n" % (
        "  [rows held by -fixb2row at the values given]" if spec.b2_hold else
        ("  [FIXED at the value given]" if spec.fixb2_given else
         "  [FIXED at its static-OLS estimate: data-chosen, so an LR against the free "
         "model is not valid]") if spec.b2_fix is not None else "")
    for a in range(s):
        o += "  " + "".join("%12.6f" % B2[a][b] for b in range(r)) + "\n"
    o += "beta matrix (M x r), rows in the .inp's order:\n"
    for row in range(M):
        o += "  " + "".join("%12.6f" % (B2[row][c] if row < s else (1.0 if row - s == c else 0.0))
                            for c in range(r)) + "\n"
    if r > 0:
        Pi = [[0.0] * M for _ in range(M)]
        for a in range(M):
            for b in range(M):
                acc = 0.0
                for j in range(r):
                    Bbj = B2[b][j] if b < s else (1.0 if b - s == j else 0.0)
                    acc -= Lam[i2l(a)][j] * Bbj
                Pi[a][b] = acc
        o += "Pi = alpha beta' matrix (M x M), the long run:\n"
        for a in range(M):
            o += "  " + "".join("%12.6f" % Pi[a][b] for b in range(M)) + "\n"
        from ._gsl import eigen_nonsymm
        wr, wi = eigen_nonsymm(Pi)
        o += "  eigenvalues of Pi:"
        for a in range(M):
            o += ("  %.6f" % wr[a]) if abs(wi[a]) < 1.0e-12 else ("  %.6f%+.6fi" % (wr[a], wi[a]))
        o += "\n"
        o += ("  ! Pi has rank r by construction: the %d zero eigenvalue%s prove%s nothing "
              "about r (use -lrtest)\n" % (M - r, "" if M - r == 1 else "s",
                                           "s" if M - r == 1 else ""))

    def qin(a, b):
        return Q[a][b] if a >= b else Q[b][a]
    s2 = res.sigma2
    o += ("Q matrix (lower triangle, rows and columns in the .inp's order; var %s = 1 by "
          "normalisation):\n" % names[l2i(0)])
    for a in range(M):
        o += "  " + "".join("%12.6f" % qin(i2l(a), i2l(b)) for b in range(a + 1)) + "\n"
    o += "Sigma = sigma2 * Q, in the .inp's order:\n"
    for a in range(M):
        o += "  " + "".join("%12.6f" % (s2 * qin(i2l(a), i2l(b))) for b in range(a + 1)) + "\n"
    from .nla import choldcp_det
    Sm = [[s2 * (Q[i][j] if j <= i else Q[j][i]) for j in range(M)] for i in range(M)]
    _L, d1, d2, ifd = choldcp_det(Sm)
    if ifd == 0:
        logdet = math.log(abs(d1)) + d2 * math.log(2.0)
        o += "  |Sigma| = %s      log|Sigma| = %.6f\n" % (cg(math.exp(logdet), 10), logdet)
    else:
        o += "  |Sigma|: not computed (Sigma is not positive definite here)\n"
    Sg = [[s2 * qin(i2l(a), i2l(b)) for b in range(M)] for a in range(M)]
    P = [[1.0 if a == b else 0.0 for b in range(M)] for a in range(M)]
    D = [0.0] * M
    ok = True
    for b in range(M):
        acc = Sg[b][b]
        for k in range(b):
            acc -= P[b][k] * P[b][k] * D[k]
        D[b] = acc
        if D[b] <= 0.0:
            ok = False
            break
        for a in range(b + 1, M):
            t = Sg[a][b]
            for k in range(b):
                t -= P[a][k] * P[b][k] * D[k]
            P[a][b] = t / D[b]
    if ok:
        o += "P matrix (Sigma = P D P', P unit lower triangular):\n"
        for a in range(M):
            o += "  " + "".join("%12.6f" % P[a][b] for b in range(a + 1)) + "\n"
        o += "D vector:\n  " + "".join("%12.6f" % d for d in D) + "\n"
        o += "  own share D_i/Sigma_ii:\n  " + "".join(
            "%11.1f%%" % (100.0 * D[a] / Sg[a][a]) for a in range(M)) + "\n"
        o += "  ! the ordering is the .inp's column order: another order gives another P\n"
    else:
        o += "P, D: not computed (Sigma is not positive definite at the optimum)\n"
    return o


RANKADM_TOL = 0.2
GRANGER_ZERO = 1.0e-6


def granger_smin(Lam, Th, M, r, q):
    """granger_smin (drvec.c:422): the smallest singular value of
    Lambda_perp' Theta(1). Invariant to the orthonormal basis of Lambda_perp,
    so numpy's QR and SVD stand in for GSL's (to rounding)."""
    import numpy as np
    s = M - r
    if r <= 0 or s <= 0:
        return -1.0
    L = np.array([[Lam[i][j] for j in range(r)] for i in range(M)], float)
    Qf, _R = np.linalg.qr(L, mode="complete")
    T1 = np.eye(M)
    for k in range(q):
        T1 = T1 - np.array(Th[k], float)
    Gt = T1.T @ Qf[:, r:]
    sv = np.linalg.svd(Gt, compute_uv=False)
    return float(sv[-1])


def _coint_block(res, fv, ro):
    spec = res.spec
    M, r, s, q = spec.M, spec.r, spec.s, spec.q
    o = banner("Cointegration Diagnostics")
    if r > 0 and q > 0:
        g = granger_smin(fv.Lam, fv.Th, M, r, q)
        if g >= 0.0:
            o += "\nRank condition (Granger): sigma_s(alpha_perp' Theta(1)) = %.3e\n" % g
            if g < RANKADM_TOL:
                o += ("  *** %s the floor %.2f (a convention; -rankadm sets it).\n"
                      "  That matrix is what makes the long-run impact\n"
                      "  C(1) = B_perp (Lambda_perp' Gamma B_perp)^-1 Lambda_perp'\n"
                      "  Theta(1) have rank M-r.  Where it degenerates the FITTED\n"
                      "  model denies the rank it was estimated at -- it says r and\n"
                      "  its parameters leave no stochastic trend.  The estimate is\n"
                      "  then on the edge of the region the model class allows, so\n"
                      "  standard errors and LR statistics do not have their usual\n"
                      "  distributions there.  -rankadm refuses such points; -mawarma\n"
                      "  makes them unreachable by construction.  See\n"
                      "  docs/HOMOLOGATION.md 4h.\n" % (
                          "ZERO to working precision, under" if g < GRANGER_ZERO
                          else "Not zero, but below", RANKADM_TOL))
            else:
                o += "  admissible (tolerance %.2f)\n" % RANKADM_TOL
    if r > 0 and s > 0:
        rd = fv.rd
        n = rd.n
        sdY2 = []
        for a in range(s):
            m1 = 0.0
            for t in range(n):
                m1 += rd.y2[t][a]
            m1 /= n
            v = 0.0
            for t in range(n):
                v += (rd.y2[t][a] - m1) * (rd.y2[t][a] - m1)
            sdY2.append(math.sqrt(v / (n - 1 if n > 1 else 1)))
        o += "\nNormalisation, share of the weight carried by the Y1 block:\n"
        for j in range(r):
            m1 = 0.0
            for t in range(n):
                m1 += rd.dm[t][s + j]
            m1 /= n
            v = 0.0
            for t in range(n):
                v += (rd.dm[t][s + j] - m1) * (rd.dm[t][s + j] - m1)
            w1 = math.sqrt(v / (n - 1 if n > 1 else 1))
            w2 = 0.0
            for a in range(s):
                w2 += abs(fv.B2[a][j]) * sdY2[a]
            share = w1 / (w1 + w2) if w1 + w2 > 0.0 else 1.0
            o += "  relation %d: the Y1 block carries %5.1f%% of the weight" % (j + 1, 100.0 * share)
            o += "   <-- DUBIOUS\n" if share < 0.05 else "\n"
    return o


def wald_sub(x, cov, idx):
    """wald_sub (drvec.c): th' pinv(S) th with S's rank by SVD at 1e-8 of the
    largest singular value. -> (chi2, df) or None."""
    import numpy as np
    k = len(idx)
    if k < 1 or cov is None:
        return None
    th = np.array([x[i] for i in idx], float)
    S = np.array([[cov[i][j] for j in idx] for i in idx], float)
    try:
        U, w, Vt = np.linalg.svd(S)
    except np.linalg.LinAlgError:
        return None
    tol = w.max() * 1.0e-8 if w.size else 0.0
    keep = w > tol
    y = np.where(keep, (U.T @ th) / np.where(keep, w, 1.0), 0.0)
    acc = float(th @ (Vt.T @ y))
    rk = int(keep.sum())
    if not (acc >= 0.0) or rk == 0:
        return None
    return acc, rk


def _chisq_q(x, df):
    from scipy.stats import chi2
    return float(chi2.sf(x, df))


def emit_wald(x, cov, idx, title, h0, reject, accept):
    if len(idx) < 1:
        return ""
    w = wald_sub(x, cov, idx)
    if w is None:
        return "\n%s\n  not computable (the covariance of this block is not usable)\n" % title
    chi2, df = w
    pv = _chisq_q(chi2, df)
    o = "\n%s\n" % title
    if h0:
        o += "  %s\n" % h0
    o += "  Wald chi2(%d) = %.4f, p-value = %.4f\n" % (df, chi2, pv)
    o += "  %s\n" % (reject if pv < 0.05 else accept)
    return o


def _hypothesis_block(res, ro, ix_lam, ix_B2, ix_F, ix_Th):
    """hypothesis_block (drvec.c:6221-6450)."""
    spec, names = res.spec, res.data.names
    M, r, s, q, nf = spec.M, spec.r, spec.s, spec.q, spec.nf
    x, cov = res.x, res.cov
    i2l = ro.inp2lam
    o = banner("Joint Hypothesis Tests (Wald)")
    if r > 0:
        idx = [ix_lam[(i, j)] for i in range(M) for j in range(r) if (i, j) in ix_lam]
        w = wald_sub(x, cov, idx) if idx else None
        if not idx:
            o += ("\nError-correction term as a whole\n"
                  "  alpha is not free (-alpha imposes alpha = A*psi): nothing to test\n")
        elif w is not None:
            o += ("\nError-correction term as a whole\n"
                  "  H0: alpha = 0\n"
                  "  Wald chi2(%d) = %.4f, p-value = %.4f\n"
                  "  ! NOT A TEST: under H0, beta is unidentified (Davies).  Use -lrtest "
                  "-bootstrap\n" % (w[1], w[0], _chisq_q(w[0], w[1])))
        o += ("\nWeak exogeneity, one variable at a time   [H0: row i of alpha = 0, chi2(%d)]\n"
              % r)
        for i in range(M):
            li = i2l(i)
            idx = [ix_lam[(li, j)] for j in range(r) if (li, j) in ix_lam]
            nm = names[i]
            o += emit_wald(x, cov, idx, "%s (%s block):" % (nm, "nabla Y2" if i < s else "Y1"),
                           "H0: this variable does not adjust",
                           "REJECT H0 -> %s adjusts to the disequilibrium." % nm,
                           "Cannot reject H0 -> %s is weakly exogenous for B." % nm)
        idx = [ix_B2[(i, j)] for i in range(s) for j in range(r) if (i, j) in ix_B2]
        if idx:
            o += ("\nExclusion from the cointegrating relations   [H0: row i of beta_2 = 0, "
                  "chi2(%d)]\n  only the %d nabla Y2 variable%s can be excluded: beta = "
                  "[I_r ; beta_2] normalises on Y1\n" % (r, s, "" if s == 1 else "s"))
            for i in range(s):
                nm = names[i]
                o += emit_wald(x, cov, [ix_B2[(i, j)] for j in range(r) if (i, j) in ix_B2],
                               "%s:" % nm,
                               "H0: row %d of beta_2 = 0" % (i + 1),
                               "REJECT H0 -> %s belongs in the long-run relation." % nm,
                               "Cannot reject H0 -> %s can be dropped from it." % nm)
            if s > 1:
                o += emit_wald(x, cov, idx, "All of B2 jointly:",
                               "H0: B2 = 0 (the relation involves only the Y1 block)",
                               "REJECT H0 -> the nabla Y2 block belongs in it.",
                               "Cannot reject H0 -> the relation is inside Y1 alone.")
        elif spec.b2_fix is not None:
            o += ("\nExclusion from the cointegrating relations\n"
                  "  beta_2 is held fixed (-fixb2): imposed, not estimated, so there "
                  "is nothing to test\n")
    else:
        o += ("\n  r = 0: no error-correction term, no Lambda and no B; the short-run "
              "block only\n")
    if nf > 0 or q > 0:
        o += ("\nShort-run dynamics   [Gamma on nabla Y, Theta on the innovations]\n"
              "  ! statements about nabla Y, not about the levels: a variable can drive\n"
              "    another's DIFFERENCES and still be tied to it only through ec\n")
        if nf > 0:
            idx = [ix_F[(nf - 1, i, j)] for i in range(M) for j in range(M)
                   if (nf - 1, i, j) in ix_F]
            o += emit_wald(x, cov, idx,
                           "Joint significance of the last short-run lag, Gamma(%d):" % nf,
                           "H0: Gamma(last) = 0",
                           "REJECT H0 -> the last AR lag is significant.",
                           "Cannot reject H0 -> the last AR lag is not significant.")
        if q > 0:
            idx = [ix_Th[(q - 1, i, j)] for i in range(M) for j in range(M)
                   if (q - 1, i, j) in ix_Th]
            o += emit_wald(x, cov, idx, "Joint significance of the last MA lag, Theta(%d):" % q,
                           "H0: Theta(last) = 0",
                           "REJECT H0 -> the last MA lag is significant.",
                           "Cannot reject H0 -> the last MA lag is not significant.")
        idx = [ix_F[(k, i, j)] for k in range(nf) for i in range(M) for j in range(M)
               if i != j and (k, i, j) in ix_F]
        idx += [ix_Th[(k, i, j)] for k in range(q) for i in range(M) for j in range(M)
                if i != j and (k, i, j) in ix_Th]
        if idx:
            o += emit_wald(x, cov, idx, "All cross effects jointly:",
                           "H0: every off-diagonal coefficient of Gamma and Theta = 0",
                           "REJECT H0 -> the cross structure earns its parameters.",
                           "Cannot reject H0 -> a diagonal short run would do "
                           "(-diagar / -diagma).")
        else:
            o += ("\nNo free cross effects to test (the short run is already diagonal or "
                  "structured).\n")
        for i in range(M):
            li = i2l(i)
            nm = names[i]
            idx = [ix_F[(k, li, j)] for k in range(nf) for j in range(M)
                   if j != li and (k, li, j) in ix_F]
            idx += [ix_Th[(k, li, j)] for k in range(q) for j in range(M)
                    if j != li and (k, li, j) in ix_Th]
            if idx:
                o += emit_wald(x, cov, idx, "D.%s: what the others do to it:" % nm,
                               "H0: no other variable enters the equation of D.%s" % nm,
                               "REJECT H0 -> D.%s is driven by the others." % nm,
                               "Cannot reject H0 -> D.%s is not driven by the others." % nm)
            idx = [ix_F[(k, j, li)] for k in range(nf) for j in range(M)
                   if j != li and (k, j, li) in ix_F]
            idx += [ix_Th[(k, j, li)] for k in range(q) for j in range(M)
                    if j != li and (k, j, li) in ix_Th]
            if idx:
                o += emit_wald(x, cov, idx, "D.%s: what it does to the others:" % nm,
                               "H0: D.%s enters no other equation" % nm,
                               "REJECT H0 -> D.%s drives the others." % nm,
                               "Cannot reject H0 -> D.%s does not drive the others." % nm)
    if getattr(res, "fdhess", None) is None:
        o += ("\n  ! p-values from the BFGS-accumulated covariance; use -fdhess before "
              "quoting them\n")
    return o


# --------------------------------------------------------------------------- #
#  impulse responses and the decomposition, in levels (drvec.c:7211-7486)      #
# --------------------------------------------------------------------------- #

def compute_psi_weights(m, p, q, phi, theta, L):
    """compute_psi_weights (drvec.c:5876), loop for loop. phi[i-1], theta[l-1]
    as lists; -> psi[0..L] (m x m lists)."""
    psi = [[[0.0] * m for _ in range(m)] for _ in range(L + 1)]
    for i in range(m):
        psi[0][i][i] = 1.0
    for l in range(1, L + 1):
        P = psi[l]
        for i in range(1, min(l, p) + 1):
            ph = phi[i - 1]
            prev = psi[l - i]
            for i1 in range(m):
                row = ph[i1]
                for j1 in range(m):
                    acc = 0.0
                    for k in range(m):
                        acc += row[k] * prev[k][j1]
                    P[i1][j1] += acc
        if l <= q:
            th = theta[l - 1]
            for i1 in range(m):
                for j1 in range(m):
                    P[i1][j1] -= th[i1][j1]
    return psi


def level_error_map(Csum, Psi_m, B2, M, r):
    s = M - r
    G = [[0.0] * M for _ in range(M)]
    for i in range(s):
        for j in range(M):
            G[i][j] = Csum[i][j]
    for i in range(r):
        for j in range(M):
            acc = Psi_m[s + i][j]
            for k in range(s):
                acc -= B2[k][i] * Csum[k][j]
            G[s + i][j] = acc
    return G


def ldl_sigma(sigma2, qq, M):
    Sg = [[sigma2 * qq[a][b] for b in range(M)] for a in range(M)]
    P = [[1.0 if a == b else 0.0 for b in range(M)] for a in range(M)]
    D = [0.0] * M
    for b in range(M):
        acc = Sg[b][b]
        for k in range(b):
            acc -= P[b][k] * P[b][k] * D[k]
        D[b] = acc
        if D[b] <= 0.0:
            return None
        for a in range(b + 1, M):
            t = Sg[a][b]
            for k in range(b):
                t -= P[a][k] * P[b][k] * D[k]
            P[a][b] = t / D[b]
    return P, D


def level_irf_R(M, r, p, q, phi, theta, B2, K, P, D):
    psi = compute_psi_weights(M, p, q, phi, theta, K)
    Csum = [[0.0] * M for _ in range(M)]
    R = []
    for k in range(K + 1):
        for i in range(M):
            for j in range(M):
                Csum[i][j] += psi[k][i][j]
        G = level_error_map(Csum, psi[k], B2, M, r)
        Rk = [[0.0] * M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                acc = 0.0
                for l in range(j, M):
                    acc += G[i][l] * P[l][j]
                Rk[i][j] = acc * math.sqrt(D[j])
        R.append(Rk)
    return R


def level_irf_fevd(res, fv, K):
    spec, names = res.spec, res.data.names
    M, r, s, p, q = spec.M, spec.r, spec.s, spec.p, spec.q
    if K < 1 or s < 1:
        return ""
    qq = fv.qq.tolist()
    ld = ldl_sigma(res.sigma2, qq, M)
    if ld is None:
        return "\n(impulse responses not computed: Sigma is not positive definite)\n"
    P, D = ld
    phi, theta, B2 = fv.phi.tolist(), fv.theta.tolist(), fv.B2
    R = level_irf_R(M, r, p, q, phi, theta, B2, K, P, D)
    o = banner("Impulse Response of the Levels")
    o += ("\n  Response of Y_{t+k} to a one-s.d. orthogonalized shock at t.\n"
          "  Shocks are orthogonalized by Sigma = P D P', so the ORDERING is the\n"
          "  .inp's column order: a different order gives different shocks.\n"
          "  ! in a cointegrated system the response does not die out: what it\n"
          "    converges to is the permanent effect, of rank M - r = %d.\n" % s)
    for j in range(M):
        o += "\nShock to %s:\n    k" % names[j] + "".join(" %14s" % nm for nm in names) + "\n"
        for k in range(K + 1):
            o += "%5d" % k + "".join(" %14.6f" % R[k][i][j] for i in range(M)) + "\n"
    KL = 400
    psiL = compute_psi_weights(M, p, q, phi, theta, KL)
    CsL = [[0.0] * M for _ in range(M)]
    Gp = [[0.0] * M for _ in range(M)]
    gain = [[0.0] * M for _ in range(M)]
    wsum = [[0.0] * M for _ in range(M)]
    conv, tail, worst = False, 0.0, 0.0
    sq = [math.sqrt(d) for d in D]
    k = 0
    while k <= KL:
        mx = 0.0
        for i in range(M):
            for j in range(M):
                CsL[i][j] += psiL[k][i][j]
        GL = level_error_map(CsL, psiL[k], B2, M, r)
        for i in range(M):
            for j in range(M):
                acc = 0.0
                for l in range(j, M):
                    acc += GL[i][l] * P[l][j]
                acc *= sq[j]
                d = acc - Gp[i][j]
                gain[i][j] += d
                wsum[i][j] += k * d
                Gp[i][j] = acc
                if abs(d) > mx:
                    mx = abs(d)
        if k > 20 and mx < 1.0e-10:
            conv, tail = True, mx
            break
        tail = mx
        k += 1
    o += banner("Long-Run Gain and Mean Lag, in Levels")
    o += ("\n  gain     = lim_k G_k, the PERMANENT effect of a one-s.d. shock\n"
          "  mean lag = sum k delta_k / sum delta_k, with delta_k the increment of\n"
          "             the response: how long the level takes to get there\n")
    o += "  %s at k = %d (largest increment %.2e)\n" % (
        "converged" if conv else "NOT converged", k, tail)
    o += "\nStructural shock standard deviations (one-s.d. shock size):\n"
    for j in range(M):
        o += "  %-12s : %.6f\n" % (names[j], math.sqrt(D[j]))
    for j in range(M):
        o += "\nShock to %s:\n" % names[j]
        for i in range(M):
            g = gain[i][j]
            if abs(g) > 1.0e-12 and abs(g) > 1.0e-8:
                o += "  %-12s : gain = %10.6f   mean lag = %7.2f periods\n" % (
                    names[i], g, wsum[i][j] / g)
            else:
                o += ("  %-12s : gain = %10.6f   mean lag = undefined (the permanent "
                      "effect is zero)\n" % (names[i], g))
    for j in range(M):
        for kk in range(r):
            acc = 0.0
            for i in range(M):
                bik = B2[i][kk] if i < s else (1.0 if i - s == kk else 0.0)
                acc += bik * gain[i][j]
            if abs(acc) > worst:
                worst = abs(acc)
    o += ("\n  self-check  max |beta' gain| = %.3e\n"
          "  beta'Y_t is stationary, so no shock can move it permanently: this is\n"
          "  zero by construction, and it exercises the weights, the level map and\n"
          "  the orthogonalisation at once.  The gain matrix has rank M - r = %d.\n"
          % (worst, s))
    o += banner("Forecast Error Variance Decomposition, in Levels")
    o += ("\n  Percentage of the h-step level forecast error variance due to each shock.\n"
          "  Same orthogonalization, so the same caveat on the ordering.\n")
    for i in range(M):
        acc = [0.0] * M
        o += "\n%s:\n    h" % names[i] + "".join(" %11s" % nm for nm in names) + "\n"
        for k in range(K + 1):
            tot = 0.0
            for j in range(M):
                acc[j] += R[k][i][j] * R[k][i][j]
                tot += acc[j]
            o += "%5d" % (k + 1) + "".join(
                " %10.1f%%" % (100.0 * acc[j] / tot if tot > 0.0 else 0.0) for j in range(M)) + "\n"
    return o


# --------------------------------------------------------------------------- #
#  the residuals and the roots (drvec.c:1738-2160, lib/diagmv/diagmv.c)        #
# --------------------------------------------------------------------------- #

def diag_k(spec):
    """diag_k (drvec.c): the dynamic coefficients, Lambda, F and Theta, and the
    transfer weights of -xlink."""
    from .xlink import npar as xl_npar
    _nmean, nlam, nmid, _nt = spec.blocks()
    M = spec.M
    return nlam + nmid - ((M if spec.diag_cov else M * (M + 1) // 2) - 1) + xl_npar(spec.xlink)


def _xlink_rows(res, c0):
    """The transfer-function block of the parameter table (drvec.c:8530-8580)."""
    spec, names = res.spec, res.data.names
    innames = res.data.xl.names
    x, dev, cov = res.x, res.se, res.cov
    o = ("\nTransfer-function inputs, on the LEVELS of the output:\n"
         "  nu(B) = omega(B)/delta(B) B^b,  omega(B) = w0 - w1 B - ...,  delta(B) = 1 - "
         "d1 B - ...\n")
    for L in spec.xlink:
        np_k = L.s + 1 + L.r
        ixw = []
        om1, de1 = 0.0, 1.0
        for jj in range(L.s + 1):
            o += par_row("  %s <- %s  w%d" % (names[L.out], innames[L.inp], jj),
                         x[c0 + jj], dev[c0 + jj])
            om1 += x[c0] if jj == 0 else -x[c0 + jj]
            ixw.append(c0 + jj)
        for jj in range(1, L.r + 1):
            o += par_row("  %s <- %s  d%d" % (names[L.out], innames[L.inp], jj),
                         x[c0 + L.s + jj], dev[c0 + L.s + jj])
            de1 -= x[c0 + L.s + jj]
            ixw.append(c0 + L.s + jj)
        g = om1 / de1
        gr = [(1.0 if jj == 0 else -1.0) / de1 for jj in range(L.s + 1)]
        gr += [om1 / (de1 * de1) for _ in range(L.r)]
        gse = 0.0
        if cov is not None:
            for a in range(np_k):
                for b in range(np_k):
                    gse += gr[a] * cov[ixw[a]][ixw[b]] * gr[b]
            gse = math.sqrt(gse) if gse > 0.0 else 0.0
        o += ("  gain nu(1) of %s <- %s: %12.6f  (s.e. %.6f, delta method)  (b, r, s) = "
              "(%d, %d, %d)\n" % (names[L.out], innames[L.inp], g, gse, L.b, L.r, L.s))
        o += emit_wald(x, cov, ixw, "Link %s <- %s, whole:" % (names[L.out], innames[L.inp]),
                       "H0: every omega and delta of the link = 0",
                       "REJECT H0 -> the input enters the output.",
                       "Cannot reject H0 -> the link can be dropped.")
        c0 += np_k
    return o


def diag_npar_ser(spec):
    M, r, s, nf, q = spec.M, spec.r, spec.s, spec.nf, spec.q
    out = []
    for i in range(M):
        row = r + i if i < s else i - s
        w_row = row < r
        n = sum(1 for _k in range(nf) if spec.f_free(row, row))
        if spec.theta_class() == "marow" and not w_row:
            out.append(n)
            continue
        out.append(n + q)
    return out


def _centred(a, n, m):
    x = [[0.0] * m for _ in range(n)]
    for j in range(m):
        mu = 0.0
        for t in range(n):
            mu += a[t][j]
        mu /= n
        for t in range(n):
            x[t][j] = a[t][j] - mu
    return x


def multivariate_diagnostics(a, n, m, freq, k):
    import numpy as np
    from .nla import chisq
    s = (freq + 2) if freq > 1 else 10
    if s > n - 2:
        s = n - 2
    if s < 1:
        s = 1
    x = _centred(a, n, m)
    X = np.array(x)
    C0 = (X.T @ X) / n
    C0i = np.linalg.inv(C0)
    Q = 0.0
    for r in range(1, s + 1):
        Cr = (X[:n - r].T @ X[r:]) / n
        Q += float(np.trace(Cr.T @ C0i @ Cr @ C0i))
    Q *= n
    df = m * m * s - k
    pQ = 1.0 - chisq(Q, df) if df >= 1 else float("nan")
    w, V = np.linalg.eigh(C0)
    JB = sk = ku = pJB = float("nan")
    if np.all(w > 1e-12 * w.max()):
        R = (V / np.sqrt(w)) @ V.T
        v = X @ R.T
        b1 = (v ** 3).sum(axis=0) / n
        b2 = (v ** 4).sum(axis=0) / n - 3.0
        sk = float((b1 ** 2).sum() * (n / 6.0))
        ku = float((b2 ** 2).sum() * (n / 24.0))
        JB = sk + ku
        pJB = 1.0 - chisq(JB, 2 * m)
    o = "\n\n" + BAR + "\n           MULTIVARIATE RESIDUAL DIAGNOSTICS                \n" + BAR + "\n"
    o += "\nHosking's Multivariate Portmanteau Test (lag %d):\n" % s
    if df >= 1:
        o += "  Q(%d) = %.4f, p-value = %.4f\n" % (df, Q, pQ)
        o += "  df = m^2 s - k = %d - %d, k = dynamic coefficients estimated\n" % (m * m * s, k)
        o += ("  *** REJECT H0: residuals are not white noise.\n" if pQ < 0.05 else
              "  Cannot reject H0: residuals appear white noise.\n")
    else:
        o += ("  Q = %.4f; df = m^2 s - k = %d - %d < 1: too few lags for the parameters, "
              "no p-value.\n" % (Q, m * m * s, k))
    o += ("\nMultivariate Jarque-Bera Test (normality; Lutkepohl, symmetric square root of "
          "Sigma):\n")
    if math.isfinite(JB):
        o += "  JB(%d) = %.4f, p-value = %.4f\n" % (2 * m, JB, pJB)
        o += "  skewness %.4f, kurtosis %.4f, each chi2(%d)\n" % (sk, ku, m)
        o += ("  *** REJECT H0: residuals are not normally distributed.\n" if pJB < 0.05 else
              "  Cannot reject H0: residuals appear normal.\n")
    else:
        o += "  not computed: the residual covariance is singular.\n"
    o += BAR + "\n"
    return o


def residual_diagnostics(res, fv):
    spec = res.spec
    a, n, M = fv.a, fv.rd.n, spec.M
    if n < 20 or M < 1:
        return ""
    K = min(n // 4, 12)
    if K < 1:
        return ""
    band = 2.0 / math.sqrt(n)
    mean = []
    for i in range(M):
        sm = 0.0
        for t in range(n):
            sm += a[t][i]
        mean.append(sm / n)
    C = []
    for k in range(K + 1):
        Ck = [[0.0] * M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                sm = 0.0
                for t in range(k, n):
                    sm += (a[t][i] - mean[i]) * (a[t - k][j] - mean[j])
                Ck[i][j] = sm / n
        C.append(Ck)
    o = multivariate_diagnostics(a, n, M, res.data.freq, diag_k(spec))
    o += "\n  residual sd:" + "".join(" %10.6f" % math.sqrt(C[0][i][i]) for i in range(M))
    o += "\n  band = 2/sqrt(n) = %.4f;  * marks |r| > band\n" % band
    o += "\nResidual cross-correlation matrices, R(k)[i][j] = corr(a_i(t), a_j(t-k)):\n"

    def corr(k, i, j):
        return C[k][i][j] / math.sqrt(C[0][i][i] * C[0][j][j])
    for k in range(K + 1):
        o += "  k=%-2d " % k
        for i in range(M):
            if i > 0:
                o += "\n       "
            for j in range(M):
                rr = corr(k, i, j)
                o += "%8.3f%s" % (rr, "*" if abs(rr) > band else " ")
        o += "\n"
    worst, wk, wi, wj, anyx = 0.0, -1, 0, 0, False
    for k in range(1, K + 1):
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                rr = corr(k, i, j)
                if abs(rr) > band:
                    anyx = True
                if abs(rr) > abs(worst):
                    worst, wk, wi, wj = rr, k, i + 1, j + 1
    r0 = 0.0
    for i in range(M):
        for j in range(i):
            rr = corr(0, i, j)
            if abs(rr) > abs(r0):
                r0 = rr
    o += "\n  Contemporaneous correlation, largest |r| at k=0: %+.3f%s\n" % (
        r0, "   ! -diagcov forces Sigma diagonal: an imposed restriction the data do not "
            "support" if (spec.diag_cov and abs(r0) > band) else "   (Sigma carries it)")
    o += "  Cross dynamics left at k >= 1: "
    o += ("none beyond the band\n" if not anyx else
          "YES, largest is equation %d against the innovation of %d at lag %d, r = %+.3f\n"
          % (wi, wj, wk, worst))
    return o


def report_operator_roots(label, A, m, k, quiet=False):
    """-> (text, minmod)."""
    from ._gsl import eigen_nonsymm
    if k <= 0:
        return "", 1.0e12
    mk = m * k
    Cm = [[0.0] * mk for _ in range(mk)]
    for l in range(k):
        for i in range(m):
            for j in range(m):
                Cm[i][j + l * m] = float(A[l][i][j])
    for l in range(1, k):
        for j in range(m):
            Cm[j + l * m][j + (l - 1) * m] = 1.0
    wr, wi = eigen_nonsymm(Cm)
    o = "  %-11s" % label
    minmod = 1.0e12
    for i in range(mk):
        lam = math.sqrt(wr[i] * wr[i] + wi[i] * wi[i])
        if lam <= 1.0e-12:
            o += "  %8s " % "inf"
            continue
        if 1.0 / lam < minmod:
            minmod = 1.0 / lam
        o += "  %8.5f%s" % (1.0 / lam, "*" if 1.0 / lam < 1.0001 else " ")
    return o + "\n", minmod


def operator_roots(res, fv):
    import numpy as np
    spec, names = res.spec, res.data.names
    M, r, s, p, q = spec.M, spec.r, spec.s, spec.p, spec.q
    if p <= 0 and q <= 0:
        return ""
    o = "\nRoots of |phi(B)|=0 and |theta(B)|=0 (moduli; > 1 is stationary/invertible):\n\n"
    t, min_ar = report_operator_roots("AR (Phi)", fv.phi, M, p)
    o += t
    t, min_ma = report_operator_roots("MA (Theta)", fv.theta, M, q)
    o += t
    if min_ar < 1.0001:
        o += ("\n  * A root sits on the unit circle -- an AR root: the system is at\n"
              "    the edge of stationarity, which is where Lambda -> 0 puts it (the\n"
              "    rank-r model degenerating into the rank-(r-1) one).\n")
    if min_ma < 1.0001:
        T = np.eye(M)
        for l in range(q):
            T = T - fv.theta[l].T
        o += ("\n  * A root sits on the unit circle -- an MA root (modulus %.5f).  The\n"
              "    likelihood rose towards a non-invertible MA and the optimiser stopped\n"
              "    at the engine's invertibility gate: this is a CONSTRAINED point, not an\n"
              "    interior maximum.  The standard errors of the MA are not defined along\n"
              "    that direction, and no LR that uses this fit has its usual\n"
              "    distribution (BUG-49).\n" % min_ma)
        _U, sv, Vt = np.linalg.svd(T)
        u = Vt[-1]
        o += ("    The combination the model differences once too often -- the left\n"
              "    null vector u of Theta*(1), smallest singular value %.2e -- has\n"
              "    weights (|u|, on Ybar = [nabla Y2 ; W]):\n" % sv[-1])
        w2 = ww = 0.0
        for i in range(M):
            ui = float(u[i])
            if i < s:
                w2 += ui * ui
                o += "      nabla %-12s %8.4f\n" % (names[i], abs(ui))
            else:
                ww += ui * ui
                o += "      W%-17d %8.4f\n" % (i - s + 1, abs(ui))
        if r == 0 or w2 >= 0.8:
            o += ("    It lies in the nabla Y2 block (share %.2f): that combination of the\n"
                  "    common-trend series looks OVER-DIFFERENCED -- stationary in levels.\n"
                  "    Read it as evidence that the rank is higher than %d, or that the\n"
                  "    series in that block are not I(1).\n" % (w2, r))
        elif ww >= 0.8:
            o += "    It lies in the W block (share %.2f).  %s\n" % (ww, (
                "An AR root is near one as well: an\n"
                "    AR/MA near-cancellation, i.e. a near common factor.  Read it as\n"
                "    evidence that the equilibrium error is barely mean-reverting --\n"
                "    that the rank may be LOWER than the one fitted.") if min_ar < 1.02 else (
                "The equilibrium error's own moving\n"
                "    average is at the boundary: re-examine the MA order of W and the\n"
                "    deterministic case before reading the estimates."))
        else:
            o += ("    It mixes both blocks (nabla Y2 share %.2f, W share %.2f): no single\n"
                  "    reading; re-examine the rank and the orders before reading the\n"
                  "    estimates.\n" % (w2, ww))
    return o

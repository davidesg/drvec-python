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

def header(data, spec, out_name, lrtest=False):
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
    L.append(f"Deterministic    : case {spec.case} -- {det}")
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
    out += "    Lambda (%d x %d):\n" % (M, r)
    for i in range(M):
        out += "     " + "".join(" %12.6f" % x2[nmean + i * r + j] for j in range(r)) + "\n"
    out += "    B2 (%d x %d):\n" % (s, r)
    b0 = nmean + nlam
    for i in range(s):
        out += "     " + "".join(" %12.6f" % x2[b0 + j * s + i] for j in range(r)) + "\n"
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

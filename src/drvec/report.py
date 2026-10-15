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
        mu, Lam, F, Th, S, B2 = unpack(spec, res.x)
        self.Lam, self.F, self.Th, self.Q, self.B2 = Lam, F, Th, S, B2
        nmean, nlam, _nmid, _nt = spec.blocks()
        self.mu_m = list(res.x[:nmean])


def fit_out(res, base):
    """The whole <base>.out of a single fit."""
    data, spec = res.data, res.spec
    out = header(data, spec, base + ".out")
    out += seed_block(spec, res.seed_info)
    out += search_block(res.search)
    out += _report_fit(res, out)
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
    o = "Convergence      : %s\n" % head
    if note:
        o += "  ! %s\n" % note
    o += "sigma2           : %15.10f\n" % res.sigma2
    o += "logelf           : %15.10f\n" % res.loglik
    o += "npar             : %d\n" % npar
    o += "AIC              : %15.10f   (-2logL + 2k, /n)\n" % ((-2.0 * res.loglik + 2.0 * npar) / n)
    o += "BIC              : %15.10f   (-2logL + k log n, /n)\n" % (
        (-2.0 * res.loglik + npar * math.log(n)) / n)

    # the walk, with the index of every slot (ix_* of report_fit)
    labs_idx = 0
    nmean, nlam, _nmid, _nt = spec.blocks()
    ix_mu = list(range(nmean))
    ix_lam = {}
    idx = nmean
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
    if nmean > 0:
        o += "\n%s\n" % ("Means: of the drift in nabla Y2, and of the equilibrium error"
                         if spec.case == 3 else
                         "Mean of the equilibrium error (the constant restricted to the relation)")
        for a in range(nmean):
            if spec.case == 3 and a < s:
                lb = "  E[D.%s]" % names[a]
            else:
                lb = "  E[ec%d]" % ((a - s + 1) if spec.case == 3 else a + 1)
            o += par_row(lb, x[ix_mu[a]], dev[ix_mu[a]])
    if r > 0:
        o += "\nAdjustment coefficients, alpha (M x r)   [negative = error-correcting]\n"
        for a in range(M):
            for b in range(r):
                ii = ix_lam[(i2l(a), b)]
                o += par_row("  D.%s <- ec%d" % (names[a], b + 1), -x[ii], dev[ii])
    if r > 0 and s > 0:
        o += "\nCointegrating vectors, beta = [I_r ; beta_2], normalised on the Y1 block\n"
        for b in range(r):
            for a in range(s):
                ii = ix_B2[(a, b)]
                o += par_row("  %s in ec%d" % (names[a], b + 1), x[ii], dev[ii])
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
    o += ("  ! standard errors from the BFGS-accumulated factor; -fdhess uses the Hessian "
          "at the optimum\n")
    o += _vecm_block(res, fv, ro)
    o += _coint_block(res, fv, ro)
    return o


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
    o += "beta_2 matrix (s x r):\n"
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

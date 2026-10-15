"""Estimation, as the C does it: est(), the admissible-start ladder, the search
over starts, the r = 0 ladder with its entry gate, and the route-(B) profile.

C references (engines/drvec at drvec-v0.12): `est` drvmlest.c:33-145 and
`objcfunc` 149-181; `make_admissible` drvec.c:4919; `est_run` 4967;
`fit_search` 5079; `pack_r0` 5326; `fit_r0_ladder` 5345; `ladder_seed_r0`
5443; `gate_profile_seed` 5508; `gate_contract` 3014.

The order of the fits is the C's, call for call (24 est() calls for case 1, 11
per data set for the rank test), because the winner of the search is decided by
a strict ">" between optima that differ by 1e-10 in logL and by 1e-4 in the
parameters (docs/PORT.md §5): reproducing the optimum is not enough, the
trajectories have to be reproduced too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from drvarma import _qnewt
from drvarma._engine import elf_c

from .cast import shootx
from .loglik import LOG2PI, XITOL, concentrated, elf
from .params import embed_ma, theta_pack
from .start import init_guess
from .stderr import bfgs_covariance

MAXITS, GRADTOL, STEPTOL = 500, 1.0e-5, 1.0e-7          # drvec.c:2898-2899

CRITERION = {1: "norm of scaled gradient <= gradtol",
             2: "scaled distance between last two steps <= steptol",
             3: "last global step failed to locate a lower point",
             4: "iteration limit reached",
             5: "five consecutive steps of maximum length taken"}


class LineSearchRoundoff(RuntimeError):
    """The line search's `disc < 0` branch (lib/optim/lnsrch.c). In the C it is
    `nrerror("ROUNDOFF PROBLEM IN LINE SEARCH")`, which EXITS THE PROGRAM with
    status 1 (nlatools.c:367), whatever was running -- a start of the search, a
    rung of the ladder, a bootstrap replication. A library cannot exit, so it is
    raised and NOT caught anywhere in drvec: it reaches the caller of `fit` or
    `rank_test` exactly as the C's run ends there, with no partial result.
    It does not fire on the milestone's cases."""


# --------------------------------------------------------------------------- #
#  est()                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class EstResult:
    x: list
    ifault: int
    ll: float = float("-inf")
    s2: float = float("nan")
    fk: float = float("nan")
    iters: int = -1
    termcode: int = 0
    dev: np.ndarray = None
    cov: np.ndarray = None

    def banner(self):
        """The text report() writes into the .out (qnewtopt.c:98-120)."""
        tc = self.termcode
        lines = ["", "=" * 61,
                 "  OPTIMIZER %s after %d iterations" % (
                     "CONVERGED" if tc in (1, 2) else "STOPPED", self.iters),
                 "  Objective function = %.12f" % self.fk]
        if tc in CRITERION:
            lines.append("  Convergence criterion: %s" % CRITERION[tc])
        lines += ["=" * 61, "", ""]
        return "\n".join(lines)


def est(spec, rd, x0, hold=None, maxits=MAXITS, gradtol=GRADTOL, steptol=STEPTOL,
        xitol=XITOL):
    """drvmlest.c:est. The objective is the concentrated likelihood normalised
    at the start, (f1/f1_0)^m (f2/f2_0); a refused point is 1.0."""
    x0 = [float(v) for v in x0]
    npar = len(x0)
    m, n = spec.M, rd.n
    f10, f20, _a, ifa = elf(spec, rd, x0, xitol, True, hold)
    if ifa:
        return EstResult(x=x0, ifault=1 if ifa == 6 else ifa)

    def objc(xk1):
        f1, f2, _a, ifa = elf(spec, rd, xk1[1:].tolist(), xitol, False, hold)
        if ifa:
            return 1.0
        f = pow(f1 / f10, m) * (f2 / f20)
        return f if math.isfinite(f) else 1.0

    xk = np.zeros(npar + 1)
    xk[1:] = x0
    try:
        fk, b, k, tc = _qnewt.raxopt(objc, npar, xk, maxits, gradtol, steptol)
    except RuntimeError as e:                      # pragma: no cover (not reached)
        if "ROUNDOFF" in str(e):
            raise LineSearchRoundoff(str(e)) from e
        raise
    x = xk[1:].tolist()
    dev, cov = bfgs_covariance(b, fk, npar, n)
    f1, f2, _a, ifa = elf(spec, rd, x, xitol, True, hold)
    res = EstResult(x=x, ifault=0, fk=float(fk), iters=int(k), termcode=int(tc),
                    dev=dev, cov=cov)
    if ifa:
        # The C returns the ifault of its last (deallocating) cast call, which is
        # only the Sigma check: an elf refusal at the final point would leave a
        # garbage logL. The port refuses the point instead (not reached here).
        res.ifault = 1 if ifa == 6 else ifa
        return res
    res.ll = concentrated(m, n, f1, f2)
    res.s2 = f1 / (n * m)
    return res


def est_run(spec, rd, x0):
    """est() plus est_run's two checks on the final point (drvec.c:5020-5060):
    Sigma passes choldcp, and qq is not singular (det qq / prod qq_ii >= 1e-10)."""
    r = est(spec, rd, x0)
    if r.ifault == 0 and spec.M > 1:
        _mu, _phi, _th, qq, _w, ifq = shootx(spec, rd, r.x)
        if ifq:
            r.ifault = 1
        else:
            m = spec.M
            L = [[float(qq[a, b]) for b in range(m)] for a in range(m)]
            det, pd = 1.0, 1.0
            for a in range(m):
                d = L[a][a]
                for c in range(a):
                    d -= L[a][c] * L[a][c]
                if not (d > 0.0):
                    r.ifault = 1
                    break
                L[a][a] = math.sqrt(d)
                for bb in range(a + 1, m):
                    e = L[bb][a]
                    for c in range(a):
                        e -= L[bb][c] * L[a][c]
                    L[bb][a] = e / L[a][a]
                det *= d
                pd *= float(qq[a, a])
            if r.ifault == 0 and pd > 0.0 and det / pd < 1e-10:
                r.ifault = 1
    return r


# --------------------------------------------------------------------------- #
#  the search                                                                  #
# --------------------------------------------------------------------------- #

_SHRINK = (1.0, 0.8, 0.5, 0.3, 0.1, 0.0)


def make_admissible(spec, rd, x):
    """drvec.c:4919. F and Theta shrunk towards zero, and Lambda halved twice if
    that is not enough, until elf accepts the start. -> (x, step): step 0 when
    accepted as given, -1 when nothing worked (x returned as given)."""
    nmean, nlam, nmid, _ntail = spec.blocks()
    lo, hi = nmean + nlam, nmean + nlam + nmid - spec.nsig()
    x0 = list(x)
    for ps in range(3):
        lamf = (1.0, 0.5, 0.25)[ps]
        for mi in range(6):
            xt = list(x0)
            for i in range(lo, hi):
                xt[i] = _SHRINK[mi] * x0[i]
            for i in range(nmean, nmean + nlam):
                xt[i] = lamf * x0[i]
            _f1, _f2, _a, ifa = elf(spec, rd, xt, XITOL, True)
            if ifa == 0:
                return xt, ps * 6 + mi
    return x0, -1


@dataclass
class Candidate:
    label: str
    ok: bool
    ll: float
    termcode: int
    iters: int
    adm: int
    est: EstResult = None


@dataclass
class SearchResult:
    ifault: int
    x: list
    ll: float
    s2: float
    best: Candidate
    candidates: list

    @property
    def nok(self):
        return sum(c.ok for c in self.candidates)

    def table(self):
        """The search table of the .out (fit_search 5240-5265)."""
        ok = [c.ll for c in self.candidates if c.ok]
        out = ["", "Search: %d starting points, %d converged; logL from %.6f to %.6f.  "
               "The best is `%s'." % (len(self.candidates), self.nok,
                                      min(ok) if ok else 0.0,
                                      self.best.ll if self.best else 0.0,
                                      self.best.label if self.best else "none"),
               "  start        logL            stop  iters  admissible start"]
        for c in self.candidates:
            s = "  %-10s " % c.label
            s += ("%15.6f   " % c.ll) if c.ok else "  (failed)        "
            s += "  %4s  " % ({1: "grad", 2: "step", 3: "tc3", 4: "iter", 5: "maxl"}
                             .get(c.termcode, "  - "))
            s += ("%5d" % c.iters) if c.iters >= 0 else "%5s" % "-"
            s += "  %s" % ("as given" if c.adm == 0 else "shrunk" if c.adm > 0 else "not found")
            out.append(s)
        out.append("  The SPREAD is the diagnostic: on a well-behaved surface every "
                   "start lands in the same place.")
        return "\n".join(out)


def fit_search(spec, rd, seed, njitter=1, log=None):
    """drvec.c:fit_search: the best of seed / cold / johansen / nested.

    `log`, when a list, receives every est() call as (label, spec, start, result),
    for the trajectory tests."""
    if njitter > 1:
        raise NotImplementedError("-multistart is not in milestone 1")
    cands = []
    best = None

    def TRY(label, xs):
        nonlocal best
        xc, adm = make_admissible(spec, rd, list(xs))
        if adm < 0:
            c = Candidate(label, False, 0.0, 0, -1, adm)
        else:
            er = est_run(spec, rd, xc)
            if log is not None:
                log.append((label, spec, list(xc), er))
            c = Candidate(label, er.ifault == 0, er.ll if er.ifault == 0 else 0.0,
                          er.termcode, er.iters, adm, er)
        cands.append(c)
        if c.ok and (best is None or c.ll > best.ll):
            best = c

    seed = [float(v) for v in seed]
    TRY("seed", seed)
    cold = init_guess(spec, rd)
    if any(abs(a - b) > 1e-12 for a, b in zip(cold, seed)):
        TRY("cold", cold)
    if spec.b2_fix is not None and spec.fixb2_given and spec.r > 0:
        # -fixb2 v: the old seed too, Lambda, F and Sigma from the static-OLS W
        # (inconsistent with v, BUG-33, but a search must not end below it)
        TRY("ols-b2", init_guess(spec, rd, consistent=False))
    if spec.r > 0 and spec.b2_fix is None:
        TRY("johansen", init_guess(spec, rd, seedjoh=True))
    cls = spec.theta_class()
    if spec.q > 0 and cls != "q0":
        if cls == "free":
            low = "matri" if spec.r > 0 else "q0"
        elif cls == "matri":
            low = "marow" if spec.r > 0 else "q0"
        else:                                       # marow, diag
            low = "q0"
        lspec = spec.but(q=0, ma="free") if low == "q0" else spec.but(ma=low)
        sub = fit_search(lspec, rd, init_guess(lspec, rd), log=log)
        if sub.ifault == 0:
            TRY("nested", embed_ma(sub.x, lspec, spec))
    if best is None:
        first = cands[0]
        return SearchResult(first.est.ifault if first.est else 6, seed, 0.0, 0.0, None, cands)
    return SearchResult(0, best.est.x, best.ll, best.est.s2, best, cands)


# --------------------------------------------------------------------------- #
#  the ladder below the rank, and its gate                                     #
# --------------------------------------------------------------------------- #

@dataclass
class Gate:
    univariate: list
    sum: float
    joint: float
    fit_ll: float
    p: int
    q: int

    @property
    def gap(self):
        return self.joint - self.sum

    @property
    def verified(self):
        return abs(self.gap) < 1.0e-6


def gate_contract(rs, rd0, x, ll):
    """drvec.c:3014. At r = 0 with F, Theta and Sigma diagonal the likelihood
    factorises: the joint fit must equal the sum of its univariate components,
    both evaluated with the WHOLE xi sequence (xitol < 0). It is not a comparison
    with the .pre models, which this route does not read."""
    M, n, p, q = rs.M, rd0.n, rs.p, rs.q
    mu, phi, theta, qq, w, _ = shootx(rs, rd0, x)
    xt = -abs(XITOL)
    uni = []
    for i in range(M):
        _lg, f1, f2, _a, ifa = elf_c(1, n, p, q, mu[i:i + 1], phi[:, i:i + 1, i:i + 1],
                                     theta[:, i:i + 1, i:i + 1], np.ones((1, 1)),
                                     w[:, i:i + 1], 1.0, xt, False)
        if ifa:
            uni.append(None)
            continue
        uni.append(-0.5 * n * (LOG2PI - math.log(n) + 1.0)
                   - 0.5 * n * (math.log(f1) + math.log(f2)))
    tot = 0.0
    for v in uni:
        if v is not None:
            tot += v
    _lg, f1, f2, _a, ifa = elf_c(M, n, p, q, mu, phi, theta, qq, w, 1.0, xt, False)
    joint = concentrated(M, n, f1, f2) if ifa == 0 else ll
    return Gate(uni, tot, joint, ll, p, q)


@dataclass
class Ladder:
    ok: bool
    F: list = None
    Th: list = None
    S: list = None
    ll: float = 0.0
    rungs: list = field(default_factory=list)     # (diag_ar, diag_ma, diag_cov, logL)
    gate: Gate = None
    cached: bool = False


def pack_r0(rs, F, Th, S, mu):
    """drvec.c:5326: an r = 0 point from full matrices, in rs's walk."""
    M = rs.M
    x = []
    if rs.case == 3:
        x += [mu[i] for i in range(M) if rs.mu_free(i)]
    for k in range(rs.nf):
        x += [float(F[k][i][j]) for i in range(M) for j in range(M) if rs.f_free(i, j)]
    if rs.q > 0:
        x += [float(v) for v in theta_pack(Th, rs.theta_class(), rs.q, M, 0)]
    s11 = float(S[0][0]) if S[0][0] > 1e-24 else 1.0
    x += [float(S[i][i]) / s11 for i in range(1, M)]
    if not rs.diag_cov:
        x += [float(S[i][j]) / s11 for i in range(1, M) for j in range(i)]
    return x


def fit_r0_ladder(spec, data, log=None):
    """drvec.c:5345. Rung 0 (F, Theta, Sigma diagonal; the gate), rung 1 (Sigma
    free), rung 2 (the structure asked for), each from the optimum of the one
    below. Cached per data set, as the C's lad_* cache (keyed on the data)."""
    key = (spec.p, spec.q, spec.case, spec.diag_ar, spec.ma, spec.diag_cov,
           tuple(spec.mu_free(i) for i in range(spec.M)) if spec.case == 3 else None,
           spec.xsys)
    if key in data.ladder_cache:
        c = data.ladder_cache[key]
        return Ladder(True, c.F, c.Th, c.S, c.ll, c.rungs, c.gate, cached=True)
    s0 = spec.but(r=0)
    rd0 = data.at(0)
    dar, dma, dcov = spec.diag_ar, spec.ma == "diag", spec.diag_cov
    rungs = [(True, True, True)]
    if not dcov:
        rungs.append((True, True, False))
    if not dar or not dma:
        rungs.append((dar, dma, dcov))
    out = Ladder(False)
    F = Th = S = mu = None
    for st, (a, b, c) in enumerate(rungs):
        rs = s0.but(diag_ar=a, ma="diag" if b else spec.ma, diag_cov=c)
        x = init_guess(rs, rd0) if st == 0 else pack_r0(rs, F, Th, S, mu)
        res = fit_search(rs, rd0, x, log=log)
        if res.ifault != 0:
            return out
        mu_, phi, theta, qq, _w, _ = shootx(rs, rd0, res.x)
        F = [phi[k].tolist() for k in range(rs.nf)]
        Th = [theta[k].tolist() for k in range(rs.q)]
        S = qq.tolist()
        mu = mu_.tolist()
        out.rungs.append((a, b, c, res.ll))
        if st == 0:
            out.gate = gate_contract(rs, rd0, res.x, res.ll)
        out.ll = res.ll
    out.ok, out.F, out.Th, out.S = True, F, Th, S
    data.ladder_cache[key] = out
    return out


def ladder_seed_r0(spec, data, x, log=None):
    """drvec.c:5443: at r = 0 the ladder's top rung IS the model (BUG-50)."""
    if spec.r != 0 or (spec.diag_ar and spec.ma == "diag" and spec.diag_cov):
        return list(x), None
    mu = [0.0] * spec.M
    if spec.case == 3:
        im = 0
        for i in range(spec.M):
            if spec.mu_free(i):
                mu[i] = x[im]; im += 1
    lad = fit_r0_ladder(spec, data, log)
    if not lad.ok:
        return list(x), lad
    return pack_r0(spec, lad.F, lad.Th, lad.S, mu), lad


# --------------------------------------------------------------------------- #
#  route (B): the VEC block profiled on the rung below                         #
# --------------------------------------------------------------------------- #

_MULT = (1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02,
         0.01, -0.01, 2.0, -2.0, 4.0, -4.0)


@dataclass
class Profile:
    ok: bool
    ladder: Ladder = None
    mult: float = 0.0
    ll_start: float = 0.0
    ll: float = 0.0
    x2: list = None
    est: EstResult = None
    x2_start: list = None


def gate_profile_seed(spec, data, x, log=None):
    """drvec.c:5508. Hold F, Theta and Sigma at the r = 0 optimum (permuted to the
    internal order of rank r, Q renormalised: BUG-32), choose the multiple of the
    conditional-regression Lambda by the likelihood, estimate only the mean,
    Lambda and B2, and return the full start. -> (x, Profile)."""
    r0, M = spec.r, spec.M
    if r0 <= 0:
        return list(x), Profile(False)
    s0 = M - r0
    nmean, nlam, nmid, ntail = spec.blocks()
    nhead = nmean + nlam
    lad = fit_r0_ladder(spec, data, log)
    if not lad.ok:
        return list(x), Profile(False, lad)
    perm = lambda z: s0 + z if z < r0 else z - r0          # noqa: E731
    hold = {
        "F": [[[lad.F[k][perm(a)][perm(b)] for b in range(M)] for a in range(M)]
              for k in range(spec.nf)],
        "Th": [[[lad.Th[k][perm(a)][perm(b)] for b in range(M)] for a in range(M)]
               for k in range(spec.q)],
    }
    s11 = lad.S[perm(0)][perm(0)]
    if s11 <= 1e-24:
        s11 = 1.0
    hold["S"] = [[lad.S[perm(a)][perm(b)] / s11 for b in range(M)] for a in range(M)]
    rd = data.at(r0)
    x2 = [float(v) for v in x[:nhead]] + [float(v) for v in x[nhead + nmid:]]
    lam0 = x2[nmean:nmean + nlam]
    best = None
    for c in _MULT:
        for i in range(nlam):
            x2[nmean + i] = c * lam0[i]
        f1, f2, _a, ifa = elf(spec, rd, x2, XITOL, True, hold)
        if ifa:
            continue
        ll = concentrated(M, rd.n, f1, f2)
        if best is None or ll > best[0]:
            best = (ll, c)
    prof = Profile(False, lad)
    if best is None:
        return list(x), prof
    for i in range(nlam):
        x2[nmean + i] = best[1] * lam0[i]
    prof.mult, prof.ll_start, prof.x2_start = best[1], best[0], list(x2)
    er = est(spec, rd, x2, hold=hold)
    if log is not None:
        log.append(("profile", spec, list(x2), er))
    prof.est = er
    if er.ifault != 0:
        return list(x), prof
    prof.ok, prof.ll, prof.x2 = True, er.ll, er.x
    xf = list(er.x[:nhead])
    for k in range(spec.nf):
        xf += [hold["F"][k][i][j] for i in range(M) for j in range(M) if spec.f_free(i, j)]
    xf += [float(v) for v in theta_pack(hold["Th"], spec.theta_class(), spec.q, M, r0)]
    xf += [hold["S"][i][i] for i in range(1, M)]
    if not spec.diag_cov:
        xf += [hold["S"][i][j] for i in range(1, M) for j in range(i)]
    xf += list(er.x[nhead:])
    assert len(xf) == spec.npar()
    return xf, prof

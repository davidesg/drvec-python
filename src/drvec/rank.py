"""The sequential LR test for the cointegration rank, with its parametric
bootstrap.

C: `run_lrtest` (drvec.c:9516-9960), `bootstrap_rank` (2356-2455),
`simulate_h0` (2226), `fit_ll` (2288), the critical values (161-172).

    LR(r -> r+1) = 2 [L(r+1) - L(r)]          (exact likelihood; may be < 0, BUG-26)

Every rank, observed or simulated, is fitted as the C fits it: the cold start,
the ladder (route (B) for r >= 1), the search. The bootstrap simulates from the
fitted rank-r model in Ybar coordinates and inverts the transformation to levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from drvarma._engine import elf_c

from .cast import shootx
from .fit import fit_search, gate_profile_seed, ladder_seed_r0
from .loglik import XITOL
from .nla import choldcp
from .rng import Rng
from .start import init_guess

# lambda-max critical values, 10% 5% 1%, by M - r = 1..11 (drvec.c:161-172)
CV_NONE = [(2.98, 4.13, 6.94), (9.47, 11.22, 15.09), (15.72, 17.80, 22.25),
           (21.84, 24.16, 29.06), (27.92, 30.44, 35.74), (33.93, 36.63, 42.23),
           (39.91, 42.77, 48.66), (45.89, 48.88, 55.03), (51.85, 54.96, 61.34),
           (57.80, 61.04, 67.64), (63.72, 67.08, 73.89)]
CV_CONST = [(7.52, 9.24, 12.97), (13.75, 15.67, 20.20), (19.77, 22.00, 26.81),
            (25.56, 28.14, 33.24), (31.66, 34.40, 39.79), (37.45, 40.30, 46.82),
            (43.25, 46.45, 51.91), (48.91, 52.00, 57.95), (54.35, 57.42, 63.71),
            (60.25, 63.57, 69.94), (66.02, 69.74, 76.63)]


def fit_rank(spec, data, log=None):
    """One rank as the C fits it (run_lrtest's loop body, fit_ll): init_guess,
    the ladder seed, the search. -> (SearchResult, Profile or Ladder)."""
    rd = data.at(spec.r)
    x = init_guess(spec, rd)
    if spec.r > 0:
        x, info = gate_profile_seed(spec, data, x, log)
    else:
        x, info = ladder_seed_r0(spec, data, x, log)
    return fit_search(spec, rd, x, log=log), info


def ma_min_modulus(theta):
    """Smallest modulus of the roots of |Theta*(B)| = 0 (1/|eigenvalue| of the
    companion matrix, as chekma builds it); inf when there is no MA."""
    theta = np.asarray(theta, float)
    if theta.shape[0] == 0 or not np.any(theta):
        return math.inf
    q, m, _ = theta.shape
    A = np.zeros((m * q, m * q))
    for k in range(q):
        A[:m, k * m:(k + 1) * m] = theta[k]
    for k in range(q - 1):
        A[(k + 1) * m:(k + 2) * m, k * m:(k + 1) * m] = np.eye(m)
    ev = np.abs(np.linalg.eigvals(A))
    ev = ev[ev > 0]
    return float(1.0 / ev.max()) if ev.size else math.inf


def simulate_h0(spec, data, x, rng):
    """drvec.c:2226. One sample of LEVELS (nobs_raw, M) under the rank-r model x,
    or None when sigma2 * qq has no Cholesky factor (`simfail`)."""
    M, r, s, p, q = spec.M, spec.r, spec.s, spec.p, spec.q
    rd = data.at(r)
    n = rd.n
    mu, phi, theta, qq, w, _ = shootx(spec, rd, x)
    _lg, f1, f2, _a, ifa = elf_c(M, n, p, q, mu, phi, theta, qq, w, 1.0, XITOL, True)
    sigma2 = f1 / (n * M)
    B2 = [[0.0] * max(r, 1) for _ in range(s)]
    idx = len(x) - s * r
    for j in range(r):
        for i in range(s):
            B2[i][j] = x[idx]; idx += 1
    burn = 50 + 10 * p
    T = n + burn
    L, ifc = choldcp([[sigma2 * float(qq[i, j]) for j in range(M)] for i in range(M)])
    if ifc:
        return None
    ph = phi.tolist()
    th = theta.tolist()
    mu = mu.tolist()
    wb = [[0.0] * M for _ in range(T)]
    ab = [[0.0] * M for _ in range(T)]
    for t in range(T):
        z = [rng.normal() for _ in range(M)]
        for i in range(M):
            acc = 0.0
            for k in range(i + 1):
                acc += L[i][k] * z[k]
            ab[t][i] = acc
        for i in range(M):
            acc = ab[t][i]
            for j in range(1, p + 1):
                if t - j >= 0:
                    for k in range(M):
                        acc += ph[j - 1][i][k] * (wb[t - j][k] - mu[k])
            for j in range(1, q + 1):
                if t - j >= 0:
                    for k in range(M):
                        acc -= th[j - 1][i][k] * ab[t - j][k]
            wb[t][i] = mu[i] + acc
    out = [[0.0] * M for _ in range(n + 1)]
    out[0] = [float(v) for v in data.raw[0]]
    for t in range(n):
        tb = t + burn
        for i in range(s):
            out[t + 1][i] = out[t][i] + wb[tb][i]
        for j in range(r):
            v = wb[tb][s + j]
            for i in range(s):
                v -= B2[i][j] * out[t + 1][i]
            out[t + 1][s + j] = v
    return np.array(out, float)


@dataclass
class Replication:
    h0: int
    rep: int
    status: str            # ok, simfail, fitfail
    lr: float = float("nan")

    def line(self):
        return (f"{self.h0},{self.rep},{self.status},%.17g" % self.lr if self.status == "ok"
                else f"{self.h0},{self.rep},{self.status},")


@dataclass
class Bootstrap:
    h0: int
    lr: float
    reps: list
    cv: tuple = None
    pvalue: float = None

    @property
    def nok(self):
        return sum(r.status == "ok" for r in self.reps)


def bootstrap_rank(spec_r, data, x, N, lr_obs, rng, progress=None):
    """bootstrap_rank (2356). spec_r is the H0 rank's spec, x its fitted vector."""
    rr = spec_r.r
    reps, stat = [], []
    for b in range(1, N + 1):
        sim = simulate_h0(spec_r, data, x, rng)
        if sim is None:
            reps.append(Replication(rr, b, "simfail"))
            continue
        db = data.with_raw(sim)
        r0, _ = fit_rank(spec_r, db)
        r1, _ = fit_rank(spec_r.but(r=rr + 1), db)
        if r0.ifault == 0 and r1.ifault == 0:
            lr = 2.0 * (r1.ll - r0.ll)
            reps.append(Replication(rr, b, "ok", lr))
            stat.append(lr)
        else:
            reps.append(Replication(rr, b, "fitfail"))
        if progress:
            progress(b, reps[-1])
    out = Bootstrap(rr, lr_obs, reps)
    nok = len(stat)
    if nok >= 10:
        st = sorted(stat)
        idx = [min(max(int(math.ceil(a * nok)), 1), nok) for a in (0.90, 0.95, 0.99)]
        out.cv = tuple(st[i - 1] for i in idx)
        ge = sum(1 for v in st if v >= lr_obs)
        out.pvalue = (ge + 1) / (nok + 1)
    return out


@dataclass
class RankTest:
    spec: object
    names: list
    nobs: int
    fits: list                      # SearchResult per rank 0..M-1
    infos: list                     # Ladder/Profile per rank
    npar: list
    ll: list                        # logL in the original units
    scale: list
    boundary: list
    bootstraps: list = field(default_factory=list)
    boot_n: int = 0                 # -bootstrap N (0: none)
    data: object = None             # the data as read (for the .out's header)

    def lr(self, r):
        return 2.0 * (self.ll[r + 1] - self.ll[r])

    def aic(self, r):
        return (-2.0 * self.ll[r] + 2.0 * self.npar[r]) / self.nobs

    def bic(self, r):
        return (-2.0 * self.ll[r] + self.npar[r] * math.log(self.nobs)) / self.nobs

    def critical_values(self, r):
        g = self.spec.M - r
        if self.spec.case == 3 or not (1 <= g <= 11):
            return None
        return (CV_NONE if self.spec.case == 1 else CV_CONST)[g - 1]

    def verdict(self, r, cv=None):
        cv = cv or self.critical_values(r)
        if cv is None:
            return None
        lr = self.lr(r)
        if lr > cv[2]:
            return "reject H0 at 1%"
        if lr > cv[1]:
            return "reject H0 at 5%"
        if lr > cv[0]:
            return "reject H0 at 10%"
        return "H0 not rejected"

    def out_text(self, base):
        """<base>.out, as the C writes it (report.rank_test_out)."""
        from .report import rank_test_out
        return rank_test_out(self, self.data, base)

    def boot_text(self, version="0.1.0.dev0"):
        """<base>.boot (drvec.c:9775-9785). The first line differs from the C's in
        the program's name only; tools/golden.sh drops it before hashing."""
        N = len(self.bootstraps[0].reps) if self.bootstraps else 0
        lines = [f"# drvec {version} (python) -bootstrap {N}: the rank test, one line "
                 "per replication", "h0,rep,status,lr"]
        for b in self.bootstraps:
            lines += [r.line() for r in b.reps]
        return "\n".join(lines) + "\n"


def rank_test(data, spec, bootstrap=0, boot_path=None, progress=None):
    """run_lrtest. `spec` gives p, q, case and the class; its r is ignored (the
    test walks r = 0..M-1)."""
    M = spec.M
    raw = data.raw
    # BUG-43: a gross mis-scaling (two orders) is undone internally; the LR does
    # not depend on it and the logLs are reported in the original units.
    dsc = [1.0] * M
    T = data.nobs_raw - 1
    for i in range(M):
        d = [float(raw[t, i]) - float(raw[t - 1, i]) for t in range(1, data.nobs_raw)]
        m1 = sum(d) / T
        v = 0.0
        for di in d:
            v += (di - m1) * (di - m1)
        sd = math.sqrt(v / (T - 1 if T > 1 else 1))
        if sd > 0.0 and abs(math.log10(sd)) > 2.0:
            dsc[i] = 10.0 ** (-math.floor(math.log10(sd) + 0.5))
    scaled = any(c != 1.0 for c in dsc)
    ds = data.with_raw(raw * np.array(dsc)) if scaled else data
    fits, infos, npar, ll, bnd = [], [], [], [], []
    for rr in range(M):
        sr = spec.but(r=rr)
        res, info = fit_rank(sr, ds)
        fits.append(res); infos.append(info); npar.append(sr.npar())
        ll.append(res.ll if res.ifault == 0 else float("nan"))
        b = False
        if res.ifault == 0 and spec.q > 0:
            _mu, _phi, theta, _qq, _w, _ = shootx(sr, ds.at(rr), res.x)
            b = ma_min_modulus(theta) < 1.0001
        bnd.append(b)
    nobs = ds.at(0).n
    if scaled:
        jac = sum(math.log(c) for c in dsc) * nobs
        ll = [v + jac for v in ll]
    out = RankTest(spec, data.names, nobs, fits, infos, npar, ll, dsc, bnd,
                   boot_n=bootstrap, data=data)
    if bootstrap > 0:
        rng = Rng()
        for rr in range(M - 1):
            if fits[rr].ifault or fits[rr + 1].ifault:
                continue
            out.bootstraps.append(bootstrap_rank(spec.but(r=rr), ds, fits[rr].x,
                                                 bootstrap, out.lr(rr), rng, progress))
        if boot_path:
            with open(boot_path, "w") as f:
                f.write(out.boot_text())
    return out

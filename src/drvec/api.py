"""The public entry points: `fit` (one rank) and `rank_test` (r = 0..M-1)."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .data import Data, read_pre
from .params import Spec, slot_labels
from .rank import fit_rank, ma_min_modulus
from .rank import rank_test as _rank_test
from .cast import shootx


@dataclass
class Param:
    label: str          # the C's .out wording, in the .inp's names
    kind: str           # mean, alpha, gamma, theta, q, beta
    value: float        # in the reader's notation (alpha = -Lambda)
    se: float


@dataclass
class FitResult:
    spec: Spec
    names: list
    nobs: int
    x: list             # the parameter vector, in drvec's internal layout
    loglik: float
    sigma2: float
    termcode: int
    iters: int
    fk: float
    se: list
    search: object      # fit.SearchResult
    seed_info: object   # fit.Profile (r >= 1) or fit.Ladder (r = 0)
    ma_min_modulus: float
    data: object = None     # the data as read (the .out's header and diagnostics)
    cov: object = None      # the BFGS covariance of the winner (npar x npar)
    drift_how: str = None   # "pre" (-case pre) or "nodrift" when drifts are held
    lr_free: float = None   # -fixb2row, -xsys: logL of the free model
    xs_df: int = None       # -xsys: the parameters the embedding removes
    gz_df: int = None       # -gzero: the parameters the restriction removes (alpha's too)
    xl_base_ll: float = None    # -xlink: logL of the VEC without the inputs
    xl_base_info: object = None  # its ladder and profile (written to the .out)
    xl_bridge_ll: float = None  # logL at that optimum with omega = 0
    irfboot: object = None  # -irfboot: irfboot.IrfBoot
    alpha_free: object = None  # -alpha/-weakex: the est() of H(r) (fit.EstResult)
    fdhess: object = None   # -fdhess: fdhess.FdHess (se and cov are its when .ok)
    forecast_h: int = 0     # -f H
    estwin: int = 0         # -estwin E
    gate: object = None     # r = 0, all diagonal: fit.Gate with the certificate
    provenance: object = None   # results.Provenance: who computed the bootstrap
    _fv: object = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    #  The accessors (STUDY-vec §4.2): the numbers the .out prints, as     #
    #  arrays, from the routines that print them. See drvec.results.       #
    # ------------------------------------------------------------------ #

    def _final(self):
        if self._fv is None:
            from .report import _Fit
            self._fv = _Fit(self)
        return self._fv

    def _ro(self):
        from .params import RowOrder
        return RowOrder(self.spec.M, self.spec.r)

    @property
    def alpha(self):
        """alpha (M x r), the adjustment, in Johansen's sign (negative is
        error-correcting), rows in the .inp's order."""
        fv, i2l = self._final(), self._ro().inp2lam
        M, r = self.spec.M, self.spec.r
        return np.array([[-fv.Lam[i2l(a)][b] for b in range(r)] for a in range(M)],
                        float).reshape(M, r)

    @property
    def beta2(self):
        """B2 (s x r): beta = [B2 ; I_r], normalised on the Y1 block (held rows
        and -fixb2's values included)."""
        fv = self._final()
        s, r = self.spec.s, self.spec.r
        return np.array([[fv.B2[a][b] for b in range(r)] for a in range(s)],
                        float).reshape(s, r)

    @property
    def beta(self):
        """beta (M x r), rows in the .inp's order: B2 on the Y2 block, I_r on Y1."""
        fv = self._final()
        M, s, r = self.spec.M, self.spec.s, self.spec.r
        return np.array([[fv.B2[row][c] if row < s else (1.0 if row - s == c else 0.0)
                          for c in range(r)] for row in range(M)], float).reshape(M, r)

    @property
    def Pi(self):
        """Pi = alpha beta' (M x M), the long run, as the .out computes it."""
        fv, i2l = self._final(), self._ro().inp2lam
        M, s, r = self.spec.M, self.spec.s, self.spec.r
        Pi = np.zeros((M, M))
        B2 = fv.B2
        for a in range(M):
            for b in range(M):
                acc = 0.0
                for j in range(r):
                    Bbj = B2[b][j] if b < s else (1.0 if b - s == j else 0.0)
                    acc -= fv.Lam[i2l(a)][j] * Bbj
                Pi[a, b] = acc
        return Pi

    @property
    def gamma(self):
        """Gamma(1..p-1) (p - 1, M, M), on nabla Y_{t-k}, in the .inp's order."""
        fv, i2l = self._final(), self._ro().inp2lam
        M = self.spec.M
        return np.array([[[fv.F[k][i2l(a)][i2l(b)] for b in range(M)] for a in range(M)]
                         for k in range(self.spec.nf)], float).reshape(self.spec.nf, M, M)

    @property
    def theta(self):
        """Theta(1..q) (q, M, M), the MA on the innovations, in the .inp's order
        (Theta(L) = I - Theta_1 L - ...)."""
        fv, i2l = self._final(), self._ro().inp2lam
        M, q = self.spec.M, self.spec.q
        return np.array([[[fv.Th[k][i2l(a)][i2l(b)] for b in range(M)] for a in range(M)]
                         for k in range(q)], float).reshape(q, M, M)

    def _qin(self, a, b):
        Q = self._final().Q
        return Q[a][b] if a >= b else Q[b][a]

    @property
    def Q(self):
        """Q (M x M, symmetric), the innovation covariance up to scale
        (Sigma = sigma2 Q), in the .inp's order."""
        i2l = self._ro().inp2lam
        M = self.spec.M
        return np.array([[self._qin(i2l(a), i2l(b)) for b in range(M)] for a in range(M)],
                        float)

    @property
    def Sigma(self):
        """Sigma = sigma2 Q (M x M), the innovation covariance, in the .inp's order."""
        i2l = self._ro().inp2lam
        M = self.spec.M
        return np.array([[self.sigma2 * self._qin(i2l(a), i2l(b)) for b in range(M)]
                         for a in range(M)], float)

    @property
    def means(self):
        """The .out's mean vector: E[W] (case 2), or E[nabla Y2] then E[W]
        (case 3; a held drift is 0); empty in case 1."""
        return np.array(self._final().mu_m, float)

    @property
    def delta(self):
        """delta = E[nabla Y] (M), case 3 only (None otherwise), as printed."""
        spec, fv = self.spec, self._final()
        M, s = spec.M, spec.s
        mu_m = fv.mu_m
        if not (spec.case == 3 and len(mu_m) >= s):
            return None
        out = []
        for a in range(M):
            if a < s:
                d = mu_m[a]
            else:
                d = 0.0
                for i3 in range(s):
                    d -= fv.B2[i3][a - s] * mu_m[i3]
                if spec.trend:
                    d += fv.dtr[a - s]
            out.append(d)
        return np.array(out, float)

    @property
    def residual_names(self):
        """The names of the residual columns: Ybar's, the Y2 names then ec1..ecr."""
        s = self.spec.s
        return list(self.names[:s]) + ["ec%d" % (j + 1) for j in range(self.spec.r)]

    @property
    def residuals(self):
        """The innovations A_t of the stationary VARMA (n x M), Ybar's columns
        (residual_names): what the .out's residual diagnostics read."""
        return np.array(self._final().a, float)

    @property
    def stationary(self):
        """The stationary representation on Ybar (drvec.results.Stationary)."""
        from .results import Stationary
        fv = self._final()
        return Stationary(self.residual_names, np.array(fv.mu, float),
                          np.array(fv.phi, float), np.array(fv.theta, float),
                          np.array(fv.qq, float), float(self.sigma2), np.array(fv.w, float))

    def _irf_horizon(self, K):
        return (10 if self._final().rd.n < 40 else 20) if K is None else int(K)

    def _level_numbers(self, K=None):
        from .report import level_irf_numbers
        nb = level_irf_numbers(self, self._final(), self._irf_horizon(K))
        if nb is not None and "error" in nb:
            raise ValueError("the level responses are not defined: " + nb["error"])
        return nb

    def level_irf(self, K=None):
        """The level responses (K + 1, M, M): [k, i, j] = the response of series i
        at lag k to a one-s.d. orthogonalised shock to series j (Sigma = P D P'
        in the .inp's order). K defaults to the .out's (10 when n < 40, else
        20). None when s = 0."""
        nb = self._level_numbers(K)
        return None if nb is None else np.array(nb["R"], float)

    def fevd(self, K=None):
        """The level FEVD in %% (K + 1, M, M): [k, i, j] = share of shock j in the
        (k + 1)-step forecast error variance of series i. None when s = 0."""
        nb = self._level_numbers(K)
        return None if nb is None else np.array(nb["fevd"], float)

    def long_run(self):
        """The long-run gain and mean lag in levels (drvec.results.LongRun)."""
        from .results import LongRun
        nb = self._level_numbers(None)
        if nb is None:
            return None
        ml = np.array([[np.nan if v is None else v for v in row] for row in nb["mean_lag"]],
                      float)
        return LongRun(np.array(nb["gain"], float), ml, np.array(nb["shock_sd"], float),
                       nb["converged"], nb["k_stop"], nb["tail"], nb["beta_gain"])

    def irf_bands(self):
        """The -irfboot bands as arrays (drvec.results.IrfBands), or None without
        -irfboot or with fewer than 20 good replications (the .out prints no
        bands then)."""
        if self.irfboot is None:
            return None
        from .irfboot import band_arrays
        return band_arrays(self.irfboot, self.spec.M)

    def diagnostics(self):
        """Hosking's portmanteau and the multivariate Jarque-Bera of the
        residuals, as the .out prints them (drvec.results.Diagnostics); None when
        n < 20 (the .out has no such section)."""
        from .report import diag_k, multivariate_numbers
        from .results import Diagnostics
        fv = self._final()
        n, M = fv.rd.n, self.spec.M
        if n < 20 or M < 1 or min(n // 4, 12) < 1:
            return None
        d = multivariate_numbers(fv.a, n, M, self.data.freq, diag_k(self.spec))
        return Diagnostics(d["lags"], d["Q"], d["df"], d["pQ"], d["JB"], d["JB_df"],
                           d["pJB"], d["skewness"], d["kurtosis"], d["part_df"])

    def forecasts(self, H=None, conf=0.95):
        """The forecasts in levels as arrays (drvec.results.Forecasts): the
        numbers of <base>.forecast. H defaults to fit(forecast=H)'s; a fit with
        -estwin forecasts from the end of its window, as the C does."""
        from .forecast import forecast_arrays
        from .results import Forecasts
        H = self.forecast_h if H is None else int(H)
        if H < 1:
            raise ValueError("forecasts need a horizon: H >= 1 (or fit(forecast=H))")
        d = forecast_arrays(self, self._final(), H, conf)
        return Forecasts(d["dates"], list(self.names), np.array(d["level"], float),
                         np.array(d["low"], float), np.array(d["high"], float),
                         np.array(d["se"], float), d["conf"], d["z"],
                         np.array(d["model_level"], float), np.array(d["model_se"], float))

    @property
    def restriction_lr(self):
        """The restriction's LR test as the .out prints it
        (drvec.results.RestrictionLR), or None when the fit carries none (a
        free fit, and -fixb2 alone, for which the C prints no LR)."""
        from .report import restriction_lr
        from .results import RestrictionLR
        d = restriction_lr(self)
        return None if d is None else RestrictionLR(**d)

    def forecast_text(self, base):
        """<base>.forecast, as the C writes it (None without -f)."""
        if not self.forecast_h:
            return None
        from .forecast import forecast_vec
        return forecast_vec(self, self._final(), self.forecast_h, base)[1]

    def recursive_text(self):
        """<base>.recursive, the per-origin errors of -estwin (None without it,
        or when the window leaves no room for the horizon)."""
        if not (self.estwin and self.forecast_h):
            return None
        from .forecast import rolling_eval
        rv = rolling_eval(self, self.estwin, self.forecast_h)
        return rv[1] if rv else None

    def irfboot_text(self, base):
        """<base>.irfboot, as the C writes it (None without -irfboot)."""
        if self.irfboot is None:
            return None
        from .irfboot import bands_text
        return bands_text(self.irfboot, self.spec, self.names, base)[1]

    @property
    def npar(self):
        return len(self.x)

    @property
    def aic(self):
        return (-2.0 * self.loglik + 2.0 * self.npar) / self.nobs

    @property
    def bic(self):
        return (-2.0 * self.loglik + self.npar * math.log(self.nobs)) / self.nobs

    @property
    def ma_on_boundary(self):
        """BUG-49: an MA root on the unit circle; the .out prints no s.e. for Theta."""
        return self.ma_min_modulus < 1.0001

    @property
    def ladder(self):
        si = self.seed_info
        return getattr(si, "ladder", si)

    @property
    def params(self):
        labs = slot_labels(self.spec, self.names,
                           self.data.xl.names if (self.data is not None and self.data.xl)
                           else None)
        return [Param(lab, kind, sg * v, se)
                for (lab, kind, sg), v, se in zip(labs, self.x, self.se)]

    def out_text(self, base):
        """<base>.out, as the C writes it (report.fit_out)."""
        from .report import fit_out
        return fit_out(self, base)

    def param(self, label):
        for p in self.params:
            if p.label == label:
                return p
        raise KeyError(label)


def _data(pre_files):
    return pre_files if isinstance(pre_files, Data) else read_pre(pre_files)


ENGINES = ("auto", "python", "c")


def _choose_engine(engine, seed, files, what):
    """-> (CBinary or None, note). engine None reads DRVEC_ENGINE (default
    "auto"). "auto": the C when a matching binary is found and the call can
    go to it, else Python, with the reason in the note. "c": the C or an error."""
    from . import c_engine
    from .rng import SEED
    if engine is None:
        engine = os.environ.get("DRVEC_ENGINE", "").strip() or "auto"
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}, got {engine!r}")
    if engine == "python":
        return None, "engine='python'"
    why = None
    if seed is not None and int(seed) != SEED:
        why = (f"the C's bootstrap generator is fixed at its seed {SEED}; seed={seed} "
               "runs in Python")
    elif not files:
        why = "the data were given as a Data object, not as .pre files the C can read"
    if why is None:
        try:
            return c_engine.find(), f"engine={engine!r}"
        except c_engine.CEngineUnavailable as e:
            why = str(e)
    if engine == "c":
        raise c_engine.CEngineUnavailable(f"{what} cannot run on drvec C: {why}")
    return None, "engine='auto', Python: " + why


def _files_of(pre_files):
    if isinstance(pre_files, Data):
        return None
    if isinstance(pre_files, (str, bytes)) or hasattr(pre_files, "__fspath__"):
        return None
    return [os.fsdecode(f) for f in pre_files]


def load_alpha_A(path, M):
    """load_alpha_A (drvec.c:2665): a first line `M sa`, then M rows of sa
    numbers; lines beginning with '*' or '#' (or empty) are comments. A'A must
    pass choldcp (psi identified). -> A as a tuple of rows."""
    with open(path) as f:
        lines = f.read().splitlines(keepends=True)
    it = iter(lines)
    mm = sa = 0
    got = False
    for line in it:
        if line[:1] in ("*", "#", "\n") or line == "":
            continue
        tok = line.split()
        try:
            mm, sa = int(tok[0]), int(tok[1])
            got = True
            break
        except (ValueError, IndexError):
            continue
    if not got or mm != M or sa < 1 or sa > M:
        raise ValueError(f"{path} must start with 'M sa', with M = {M} and "
                         f"1 <= sa <= {M} (leido {mm} {sa})")
    A = []
    for i in range(M):
        while True:
            line = next(it, None)
            if line is None:
                raise ValueError(f"{path} ends at row {i + 1}")
            if line[:1] not in ("*", "#", "\n"):
                break
        tok = line.split()
        if len(tok) < sa:
            raise ValueError(f"{path}, fila {i + 1}: faltan columnas")
        A.append(tuple(_atof(t) for t in tok[:sa]))
    return tuple(A)


def _atof(t):
    """C's atof: the longest numeric prefix, 0.0 when there is none."""
    import re
    m = re.match(r"\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", t)
    return float(m.group(0)) if m else 0.0


def _alpha_matrix(M, alpha, weakex):
    """-> (A, weakex): -weakex i (build_weakex_A, drvec.c:2727) or -alpha (a
    path, or the matrix), with A's rank checked through A'A (choldcp)."""
    from .nla import choldcp
    if weakex:
        if not 1 <= weakex <= M:
            raise ValueError(f"-weakex {weakex} is outside 1..{M}")
        A = tuple(tuple(1.0 if (i != weakex - 1 and c == (i if i < weakex - 1 else i - 1))
                        else 0.0 for c in range(M - 1)) for i in range(M))
        return A, int(weakex)
    if isinstance(alpha, (str, bytes)) or hasattr(alpha, "__fspath__"):
        A = load_alpha_A(alpha, M)
    else:
        A = tuple(tuple(float(v) for v in row) for row in alpha)
        if len(A) != M or not A[0] or any(len(row) != len(A[0]) for row in A) \
                or len(A[0]) > M:
            raise ValueError(f"A must be M x sa with M = {M} and 1 <= sa <= {M}")
    sa = len(A[0])
    G = [[sum(A[k][i] * A[k][j] for k in range(M)) for j in range(sa)] for i in range(sa)]
    # The C's own check is choldcp's verdict, which only faults on a NEGATIVE
    # pivot: an exactly singular A'A passes, and the fit runs on NaN (psi's
    # seed divides by zero) and exits 0 -- a C defect (bugs/drvec-c-alpha-rank).
    # The port refuses it with the C's message, as the C means to.
    import numpy as np
    if choldcp(G)[1] > 0 or np.linalg.matrix_rank(np.array(A, float)) < sa:
        raise ValueError(f"A does not have rank {sa} (A'A is singular), so psi\n"
                         "       is not identified")
    return A, 0


def _drift_mask(data, case, nodrift):
    """-> (case, drift mask, how): -case pre reads each .pre's mean flag
    (drvec.c:6905-6925); -nodrift i holds series i's drift at 0."""
    M = data.M
    if case == "pre":
        mask = []
        for i in range(M):
            held = not data.pre_mean_free[i]
            if held and abs(data.pre_mu[i]) > 1.0e-12:
                held = False          # a mean held at a non-zero value goes free
            mask.append(not held)
        return 3, (tuple(mask) if not all(mask) else None), "pre"
    if nodrift:
        if case != 3:
            raise ValueError("-nodrift holds a drift of case 3 (-case 3 or -case pre); in "
                             "cases 1 and 2 no series has one.")
        mask = [True] * M
        for i in nodrift:
            if not 1 <= i <= M:
                raise ValueError(f"-nodrift {i}: the system has {M} series")
            mask[i - 1] = False
        return 3, tuple(mask), "nodrift"
    return case, None, None


MSG_GZERO = ("-gzero holds entries of Gamma(k) for a single fit with q = 0\n"
             "       and p >= 2 (at p = 1 there is no Gamma).  It combines with\n"
             "       -alpha/-weakex, not with -xsys, -fixb2row, -fixb2, -xlink,\n"
             "       -diagar, -warma, a moving average, or the modes -lrtest,\n"
             "       -specs, -rungs, -matest, -artest and -eval.")


def fit(pre_files, p, q, r, case=2, ma="free", diag_ar=False, diag_cov=False,
        fixb2row=None, nodrift=None, trend=False, irfboot=0, xsys=None,
        xpre=None, xlink=None, forecast=0, fdhess=False, alpha=None, weakex=None,
        fixb2=None, estwin=0, multistart=0, log=None, seed=None, workers=None,
        engine=None, folder=None, progress=None, gzero=None):
    """Estimate the VARMA-VEC of rank r on the `.pre` files (the C's
    `drvec f1.pre ... p q r -case <case>`), with the C's ladder and search.

    case: 1, 2, 3 or "pre" (case 3 with each drift free or held as its .pre
    says). fixb2row: {row: value} (1-based rows of B2, the nabla Y2 block),
    the C's -fixb2row i v; the free model is fitted too, for the LR. nodrift:
    1-based series whose drift is held at 0 (-nodrift). trend: -trend (case 4).
    irfboot: -irfboot N, bands for the level responses and the decomposition by
    the parametric bootstrap. fdhess: -fdhess, the standard errors from the
    finite-difference Hessian at the optimum. alpha: -alpha FILE (a path, or the
    M x sa matrix A itself), weakex: -weakex i (1-based): alpha = A psi, with the
    LR against the free model. fixb2: -fixb2 v (a number, an a-priori value for
    every entry of B2) or True (-fixb2 alone: B2 held at its static-OLS value).
    forecast: -f H, H steps in levels with their theoretical bands (the .out's
    section and FitResult.forecast_text). estwin: -estwin E, the fit on rows
    1..E and, with -f H, the rolling-origin evaluation (recursive_text).
    multistart: -multistart n, n - 1 jittered starts added to the search.
    gzero: -gzero i j, pairs (i, j) (1-based, the input's order): Gamma_k[i][j]
    = 0 for every k, series j out of equation i's short run; with alpha/weakex
    the conditional model's joint restriction. The free VEC (no mask, alpha
    free) is fitted too, for the LR.

    The -irfboot bootstrap: seed (None: the C's 987654321), workers (the
    process pool for the refits; None: DRVEC_WORKERS or the cores less one;
    1: in this process; the numbers do not depend on it), engine ("auto",
    "python" or "c"; None reads DRVEC_ENGINE, default "auto": drvec C when a
    matching binary is found, see drvec.c_engine), folder (where the C runs and
    leaves its files; None: a temporary folder), progress(b, ok) after each
    replication (Python engine; the C reports at the end). The engine used is
    in FitResult.provenance."""
    if (xpre or xlink) and not (xpre and xlink):
        raise ValueError("-xpre and -xlink go together: the inputs and the\n"
                         "       network of links that says where they enter.")
    if xlink and (forecast or estwin):
        raise ValueError(
            "-xlink needs the .pre route in the levels layout, and a\n"
            "       single fit: it cannot be combined with -differenced, -warma,\n"
            "       -lrtest, -specs, -rungs, -matest, -artest, -eval, -f, -estwin,\n"
            "       -writeinp or -writeres.")
    if trend and (forecast or estwin):
        raise ValueError(
            "-trend (case 4) goes on case 3's drifts (-case 3 or -case pre)\n"
            "       for a single fit: it cannot be combined with -warma, -f,\n"
            "       -estwin, -writeinp, -writeres or the modes -lrtest, -specs,\n"
            "       -rungs, -matest, -artest and -eval (not yet).")
    if irfboot and estwin:
        raise ValueError(
            "-irfboot bootstraps the responses of a single fit in the levels\n"
            "       layout: it cannot be combined with -differenced, -xlink, -trend,\n"
            "       -warma, -estwin, -writeinp, -writeres or the modes -lrtest,\n"
            "       -specs, -rungs, -matest, -artest and -eval (not yet).")
    if xsys and (alpha is not None or weakex):
        raise ValueError(
            "-xsys embeds an input in the system for a single fit with\n"
            "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
            "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
            "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
            "       and -eval.")
    if fixb2row and fixb2 is not None and fixb2 is not False:
        raise ValueError(
            "-fixb2row holds rows of B2 at a single fit.  It cannot be\n"
            "       combined with -fixb2 (which holds every row), -warma, or the\n"
            "       modes -lrtest, -specs, -rungs, -matest, -artest and -eval.")
    if xlink and case == 1:
        raise ValueError("-xlink needs a free mean of W (-case 2 or 3): the input\n"
                         "       keeps its mean, and nu(1) times it goes to E[W].")
    if xlink and (xsys or irfboot):
        raise ValueError(
            "-irfboot bootstraps the responses of a single fit in the levels\n"
            "       layout: it cannot be combined with -differenced, -xlink, -trend,\n"
            "       -warma, -estwin, -writeinp, -writeres or the modes -lrtest,\n"
            "       -specs, -rungs, -matest, -artest and -eval (not yet)." if irfboot else
            "-xsys embeds an input in the system for a single fit with\n"
            "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
            "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
            "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
            "       and -eval.")
    if irfboot and trend:
        raise ValueError(
            "-irfboot bootstraps the responses of a single fit in the levels\n"
            "       layout: it cannot be combined with -differenced, -xlink, -trend,\n"
            "       -warma, -estwin, -writeinp, -writeres or the modes -lrtest,\n"
            "       -specs, -rungs, -matest, -artest and -eval (not yet).")
    if gzero and (xsys or xlink or fixb2row or diag_ar or q > 0 or p < 2
                  or (fixb2 is not None and fixb2 is not False)):
        raise ValueError(MSG_GZERO)
    case0 = case
    data = _data(pre_files)
    gz = ()
    if gzero:
        for i, j in gzero:
            if int(i) == int(j):
                raise ValueError(f"-gzero {i} {j}: the diagonal of Gamma is the series' own "
                                 "model\n       (the ladder's rung), and is never held.")
            if not (1 <= int(i) <= data.M and 1 <= int(j) <= data.M):
                raise ValueError(f"-gzero {i} {j}: the system has {data.M} series")
        gz = tuple(sorted(set((int(i) - 1, int(j) - 1) for i, j in gzero)))
    if estwin:
        # P5.2: the fit on rows 1..E of nabla Y, the rest kept for the
        # rolling evaluation (drvec.c:740-747)
        nfull = data.nobs_raw - 1
        if estwin < 10 or estwin >= nfull:
            raise ValueError(f"-estwin {estwin} is not inside 10..{nfull - 1}")
        data = data.with_raw(data.raw)
        data.estwin = int(estwin)
    links = ()
    if xlink:
        from .xlink import load
        data = data.with_raw(data.raw)
        links, data.xl = load(data, list(xpre) if not isinstance(xpre, str) else [xpre],
                              xlink)
    xs = ()
    if xsys:
        if q > 0 or fixb2row:
            raise ValueError(
                "-xsys embeds an input in the system for a single fit with\n"
                "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
                "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
                "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
                "       and -eval.")
        M = data.M
        if len(set(xsys)) >= M:
            raise ValueError("-xsys declares every series an input: there is nothing "
                             "left to explain.")
        for i in xsys:
            if not 1 <= i <= M:
                raise ValueError(f"-xsys {i}: the system has {M} series")
            if i > M - r:
                raise ValueError(
                    f"-xsys {i}: series {i} is in the Y1 block (the last {r} of\n"
                    "       the input), on which beta is normalised.  An input does\n"
                    "       not adjust to the equilibrium, so it cannot be the\n"
                    "       series a relation is solved for: put it in the nabla Y2\n"
                    f"       block (among the first {M - r}).")
        xs = tuple(sorted(set(i - 1 for i in xsys)))
    case, drift, how = _drift_mask(data, case, nodrift)
    if trend and case != 3:
        raise ValueError("-trend (case 4) goes on case 3's drifts (-case 3 or -case pre)")
    hold = ()
    if fixb2row:
        if r < 1:
            raise ValueError("-fixb2row needs r >= 1: at r = 0 there is no beta to restrict.")
        s = data.M - r
        for i in fixb2row:
            if not 1 <= i <= s:
                raise ValueError(f"-fixb2row {i}: B2 has {s} row{'' if s == 1 else 's'} (the "
                                 f"nabla Y2 block, the first {s} series in the input's order).")
        hold = tuple(sorted((i - 1, float(v)) for i, v in fixb2row.items()))
    A, wx = (), 0
    if alpha is not None or weakex:
        A, wx = _alpha_matrix(data.M, alpha, weakex)
        if r < 1:
            raise ValueError("alpha = A*psi means nothing with r = 0 (there is no "
                             "error-correction term)")
    spec = Spec(data.M, p, q, r, case, ma, diag_ar, diag_cov, drift, trend, hold, xsys=xs,
                alpha=A, weakex=wx, gzero=gz)
    if fixb2 is not None and fixb2 is not False and r > 0:
        if fixb2 is True:
            # -fixb2 alone: B2 at the static-OLS estimate init_guess computes
            # (drvec.c:4253-4262), a data-chosen restriction
            from .start import prelim_b2
            B = prelim_b2(spec, data.at(r))
            bf = tuple(tuple(float(B[i][j]) for j in range(r)) for i in range(spec.s))
            spec = spec.but(b2_fix=bf, fixb2_given=False)
        else:
            v = float(fixb2)
            spec = spec.but(b2_fix=tuple((v,) * r for _ in range(spec.s)), fixb2_given=True)
    if spec.npar() >= data.nobs * data.M:
        raise ValueError(f"the model has {spec.npar()} parameters and the sample "
                         f"{data.nobs * data.M} data points")
    lr_free = None
    alpha_free = None
    if hold:
        # the free model, as the C fits it for the LR: init_guess and the
        # search, no ladder (drvec.c:11351-11380); with -alpha/-weakex the
        # restriction on alpha is off too (the joint test)
        from .fit import fit_search
        from .start import init_guess
        sf = spec.but(b2_hold=(), alpha=(), weakex=0)
        rdf = data.at(r)
        rf = fit_search(sf, rdf, init_guess(sf, rdf))
        lr_free = rf.ll if rf.ifault == 0 else None
    elif spec.alpha and not gz:
        # H(r) for the LR of alpha = A psi: ONE est() from init_guess, no
        # search and no ladder, its banner written to the .out
        # (drvec.c:11415-11439)
        from .fit import est
        from .start import init_guess
        sf = spec.but(alpha=(), weakex=0)
        rdf = data.at(r)
        alpha_free = est(sf, rdf, init_guess(sf, rdf))
        lr_free = alpha_free.ll if alpha_free.ifault == 0 else None
    xs_df = None
    xl_base = None
    if links:
        spec = spec.but(xlink=links)
    if links:
        # the VEC rung below, without the inputs: its ladder and profile are
        # written to the .out, its search is not (drvec.c:11262-11296); its
        # optimum with omega = 0 is the start, and the bridge is evaluated
        from .loglik import loglik
        sb = spec.but(xlink=())
        rb, infob = fit_rank(sb, data)
        xl_base = (rb.ll if rb.ifault == 0 else None, infob)
        x0 = list(rb.x) + [0.0] * (spec.npar() - sb.npar())
        llb, _s2, ifb = loglik(spec, data.at(r), x0)
        xl_bridge = (llb if ifb == 0 else None)
        from .fit import fit_search
        res = fit_search(spec, data.at(r), x0, njitter=multistart if multistart > 1 else 1,
                         log=log)
        info = None
    if xs:
        # the free VEC, as the C fits it for the LR of strong exogeneity:
        # the ladder start and the search (drvec.c:11383-11414)
        sf = spec.but(xsys=())
        rf, _ = fit_rank(sf, data)
        lr_free = rf.ll if rf.ifault == 0 else None
        xs_df = sf.npar() - spec.npar()
    gz_df = None
    if gz:
        # -gzero (with -alpha/-weakex if given): the free VEC, no mask and alpha
        # free, with the ladder start and the search, as -xsys's
        sf = spec.but(gzero=(), alpha=(), weakex=0)
        rf, _ = fit_rank(sf, data)
        lr_free = rf.ll if rf.ifault == 0 else None
        gz_df = sf.npar() - spec.npar()
    if not links:
        res, info = fit_rank(spec, data, log, njitter=multistart if multistart > 1 else 1)
    if res.ifault != 0:
        raise RuntimeError(f"ESTIMATION FAILED: ifault = {res.ifault}")
    rd = data.at(r)
    _mu, _phi, theta, _qq, _w, _ = shootx(spec, rd, res.x)
    e = res.best.est
    out = FitResult(spec, list(data.names), rd.n, list(res.x), res.ll, res.s2,
                    e.termcode, e.iters, e.fk, list(e.dev), res, info,
                    ma_min_modulus(theta), data, e.cov)
    out.drift_how = how
    if r == 0 and spec.diag_ar and spec.ma == "diag" and spec.diag_cov and not links:
        # the diagonal rung as a plain fit: the entry gate with the optimality
        # certificate, the logL AT the start brought in (drvec.c:11571-11595)
        from .fit import gate_contract
        from .loglik import XITOL, concentrated, elf
        out.gate = gate_contract(spec, rd, res.x, res.ll)
        f1, f2, _a, ifs = elf(spec, rd, res.seed, XITOL, False)
        if ifs == 0:
            out.gate.ll_start = concentrated(spec.M, rd.n, f1, f2)
    out.forecast_h = int(forecast or 0)
    out.estwin = int(estwin or 0)
    out.alpha_free = alpha_free
    out.lr_free = lr_free
    out.xs_df = xs_df
    out.gz_df = gz_df
    if links:
        out.xl_base_ll, out.xl_base_info = xl_base
        out.xl_bridge_ll = xl_bridge
    if fdhess:
        # report_fit's first step (drvec.c:7805): the s.e. and the covariance
        # every Wald test reads are replaced when the Hessian is usable
        from .fdhess import exact_hessian_se
        out.fdhess = exact_hessian_se(spec, rd, out.x)
        if out.fdhess.ok:
            out.se = list(out.fdhess.dev)
            out.cov = out.fdhess.cov
    if irfboot > 0:
        K = 10 if rd.n < 40 else 20
        files = _files_of(pre_files)
        binary, note = _choose_engine(engine, seed, files, "-irfboot")
        if binary is not None:
            out.irfboot, out.provenance = _irfboot_c(
                binary, note, files, spec, data, out.x, irfboot, K, folder, progress,
                dict(p=p, q=q, r=r, case=case0, ma=ma, diag_ar=diag_ar, diag_cov=diag_cov, fixb2row=fixb2row,
                     nodrift=nodrift, xsys=xsys, alpha=alpha, weakex=weakex, fixb2=fixb2,
                     multistart=multistart, gzero=gzero))
        else:
            from .irfboot import irf_bootstrap
            from .parallel import resolve_workers
            from .results import python_provenance
            from .rng import Rng
            nw = resolve_workers(workers, irfboot)
            out.irfboot = irf_bootstrap(spec, data, out.x, irfboot, K,
                                        rng=Rng() if seed is None else Rng(seed),
                                        progress=progress, workers=nw)
            out.provenance = python_provenance(nw, seed, note)
    if out.provenance is None:
        from .results import python_provenance
        out.provenance = python_provenance(1, None, "no bootstrap")
    return out


def _irfboot_c(binary, note, files, spec, data, x, N, K, folder, progress, kw):
    """fit(irfboot=N) on drvec C: the bands from its files (drvec.c_engine)."""
    from . import c_engine
    from .irfboot import IrfBoot, _irf_at
    base = "drvecjob"
    with c_engine.Folder(folder) as d:
        argv = c_engine.fit_argv(files, kw.pop("p"), kw.pop("q"), kw.pop("r"), d,
                                 irfboot=N, base=base, **kw)
        job = c_engine.Job(binary, argv, d, base)
        try:
            job.wait()
        finally:
            if job.proc.poll() is None:
                job.proc.kill()
        with open(job.path(".out")) as f:
            otext = f.read()
        ftext = None
        if os.path.exists(job.path(".irfboot")):
            with open(job.path(".irfboot")) as f:
                ftext = f.read()
        nok, nfail, sect, bands = c_engine.read_irfboot(otext, ftext, data.names, K, base)
        prov = c_engine.provenance(binary, d if folder else None, argv, job.elapsed, note)
    if progress:
        progress(N, True)
    bt = IrfBoot(N, K, _irf_at(spec, data, x, K), [], [], nfail, c_nok=nok, c_bands=bands,
                 c_out=sect, c_file=ftext)
    bt.c_base = base
    return bt, prov


def rank_test(pre_files, p, q, case=2, ma="free", bootstrap=0, boot_path=None,
              progress=None, trend=False, xsys=None, xpre=None, xlink=None, multistart=0,
              nodrift=None, seed=None, workers=None, engine=None, folder=None):
    """The sequential LR test for the rank (`-lrtest [-bootstrap N]`).

    case="pre" and nodrift= are the mask on case 3's drifts, as in fit(): every
    rank, observed or simulated, is fitted with it (the C's global mask); the
    asymptotic tables are case 3's and do not apply to the mixed case.

    The C declares the rank test with exogenous inputs (-xsys, -xlink) and with
    -trend as not done yet; so does the port, with its messages.

    The bootstrap: progress(b, replication) after each replication (its .h0 is
    the comparison); seed (None: the C's 987654321); workers (the process pool
    for the refits; None: DRVEC_WORKERS or the cores less one; 1: in this
    process; the numbers do not depend on it); engine ("auto", "python" or "c";
    None reads DRVEC_ENGINE, default "auto": drvec C when a matching binary is
    found, see drvec.c_engine); folder (where the C runs and leaves its files;
    None: a temporary folder). The engine used is in RankTest.provenance."""
    if xlink or xpre:
        raise ValueError(
            "-xlink needs the .pre route in the levels layout, and a\n"
            "       single fit: it cannot be combined with -differenced, -warma,\n"
            "       -lrtest, -specs, -rungs, -matest, -artest, -eval, -f, -estwin,\n"
            "       -writeinp or -writeres.")
    if xsys:
        raise ValueError(
            "-xsys embeds an input in the system for a single fit with\n"
            "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
            "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
            "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
            "       and -eval.")
    if trend:
        raise ValueError(
            "-trend (case 4) goes on case 3's drifts (-case 3 or -case pre)\n"
            "       for a single fit: it cannot be combined with -warma, -f,\n"
            "       -estwin, -writeinp, -writeres or the modes -lrtest, -specs,\n"
            "       -rungs, -matest, -artest and -eval (not yet).")
    data = _data(pre_files)
    case0 = case
    case, drift, how = _drift_mask(data, case, nodrift)
    spec = Spec(data.M, p, q, 0, case, ma, drift=drift)
    nj = multistart if multistart > 1 else 1
    from .results import python_provenance
    if bootstrap <= 0:
        out = _rank_test(data, spec, 0, boot_path, progress, njitter=nj)
        out.drift_how = how
        out.provenance = python_provenance(1, None, "no bootstrap")
        return out
    files = _files_of(pre_files)
    binary, note = _choose_engine(engine, seed, files, "the rank test's bootstrap")
    if binary is None:
        from .parallel import resolve_workers
        nw = resolve_workers(workers, bootstrap)
        out = _rank_test(data, spec, bootstrap, boot_path, progress, njitter=nj,
                         seed=seed, workers=nw)
        out.drift_how = how
        out.provenance = python_provenance(nw, seed, note)
        return out
    from . import c_engine
    from .rank import summarise
    base = "drvecjob"
    with c_engine.Folder(folder) as d:
        argv = c_engine.rank_argv(files, p, q, case=case0, ma=ma, bootstrap=bootstrap,
                                  nodrift=nodrift, multistart=multistart, base=base)
        job = c_engine.Job(binary, argv, d, base)

        def hook(t, ds):
            # the port's observed fits are done (they ran while the C did)
            def line(s):
                r = c_engine.parse_boot_line(s)
                if r is not None and progress:
                    progress(r.rep, r)
            try:
                job.wait(on_boot_line=line)
            finally:
                if job.proc.poll() is None:
                    job.proc.kill()
            reps = c_engine.read_boot(job.path(".boot"))
            want = [rr for rr in range(t.spec.M - 1)
                    if not (t.fits[rr].ifault or t.fits[rr + 1].ifault)]
            if sorted(reps) != want:
                raise c_engine.CEngineError(
                    "drvec C bootstrapped the comparisons %s and the port's fits allow %s: "
                    "the two fits disagree (%s)" % (sorted(reps), want, d))
            bs = []
            for rr in want:
                if len(reps[rr]) != bootstrap:
                    raise c_engine.CEngineError(
                        f"drvec C wrote {len(reps[rr])} replications for H0: r = {rr}, "
                        f"not {bootstrap} ({d})")
                bs.append(summarise(rr, t.lr(rr), reps[rr]))
            return bs
        try:
            out = _rank_test(data, spec, bootstrap, boot_path, None, njitter=nj,
                             boot_hook=hook)
        finally:
            if job.proc.poll() is None:
                job.proc.kill()
                job.proc.wait()
        out.provenance = c_engine.provenance(binary, d if folder else None, argv,
                                             getattr(job, "elapsed", None), note)
    out.drift_how = how
    return out


def rungs(pre_files, p, q, r=0, case=2, ma="free"):
    """-rungs: the ladder below the rank at r = 0 -- rungs 0, 1 and 2 with
    rung 0's gate and the nested chi2 LRs (rungtable.run_rungs). r only names the
    header's rank, as the C's does."""
    from .rungtable import run_rungs
    data = _data(pre_files)
    if not 0 <= r < data.M:
        raise ValueError(f"r={r} must be < M={data.M}")
    return run_rungs(data, Spec(data.M, p, q, r, case, ma))

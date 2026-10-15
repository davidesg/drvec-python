"""The public entry points: `fit` (one rank) and `rank_test` (r = 0..M-1)."""
from __future__ import annotations

import math
from dataclasses import dataclass

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
    lr_free: float = None   # -fixb2row: logL of the free model

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
        labs = slot_labels(self.spec, self.names)
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


def fit(pre_files, p, q, r, case=2, ma="free", diag_ar=False, diag_cov=False,
        fixb2row=None, nodrift=None, trend=False, log=None):
    """Estimate the VARMA-VEC of rank r on the `.pre` files (the C's
    `drvec f1.pre ... p q r -case <case>`), with the C's ladder and search.

    case: 1, 2, 3 or "pre" (case 3 with each drift free or held as its .pre
    says). fixb2row: {row: value} (1-based rows of B2, the nabla Y2 block),
    the C's -fixb2row i v; the free model is fitted too, for the LR. nodrift:
    1-based series whose drift is held at 0 (-nodrift). trend: -trend (case 4)."""
    data = _data(pre_files)
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
    spec = Spec(data.M, p, q, r, case, ma, diag_ar, diag_cov, drift, trend, hold)
    if spec.npar() >= (data.nobs_raw - 1) * data.M:
        raise ValueError(f"the model has {spec.npar()} parameters and the sample "
                         f"{(data.nobs_raw - 1) * data.M} data points")
    lr_free = None
    if hold:
        # the free model, as the C fits it for the LR: init_guess and the
        # search, no ladder (drvec.c:11351-11380)
        from .fit import fit_search
        from .start import init_guess
        sf = spec.but(b2_hold=())
        rdf = data.at(r)
        rf = fit_search(sf, rdf, init_guess(sf, rdf))
        lr_free = rf.ll if rf.ifault == 0 else None
    res, info = fit_rank(spec, data, log)
    if res.ifault != 0:
        raise RuntimeError(f"ESTIMATION FAILED: ifault = {res.ifault}")
    rd = data.at(r)
    _mu, _phi, theta, _qq, _w, _ = shootx(spec, rd, res.x)
    e = res.best.est
    out = FitResult(spec, list(data.names), rd.n, list(res.x), res.ll, res.s2,
                    e.termcode, e.iters, e.fk, list(e.dev), res, info,
                    ma_min_modulus(theta), data, e.cov)
    out.drift_how = how
    out.lr_free = lr_free
    return out


def rank_test(pre_files, p, q, case=2, ma="free", bootstrap=0, boot_path=None,
              progress=None):
    """The sequential LR test for the rank (`-lrtest [-bootstrap N]`)."""
    data = _data(pre_files)
    spec = Spec(data.M, p, q, 0, case, ma)
    return _rank_test(data, spec, bootstrap, boot_path, progress)

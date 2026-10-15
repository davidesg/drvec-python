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

    def param(self, label):
        for p in self.params:
            if p.label == label:
                return p
        raise KeyError(label)


def _data(pre_files):
    return pre_files if isinstance(pre_files, Data) else read_pre(pre_files)


def fit(pre_files, p, q, r, case=2, ma="free", diag_ar=False, diag_cov=False, log=None):
    """Estimate the VARMA-VEC of rank r on the `.pre` files (the C's
    `drvec f1.pre ... p q r -case <case>`), with the C's ladder and search."""
    data = _data(pre_files)
    spec = Spec(data.M, p, q, r, case, ma, diag_ar, diag_cov)
    if spec.npar() >= (data.nobs_raw - 1) * data.M:
        raise ValueError(f"the model has {spec.npar()} parameters and the sample "
                         f"{(data.nobs_raw - 1) * data.M} data points")
    res, info = fit_rank(spec, data, log)
    if res.ifault != 0:
        raise RuntimeError(f"ESTIMATION FAILED: ifault = {res.ifault}")
    rd = data.at(r)
    _mu, _phi, theta, _qq, _w, _ = shootx(spec, rd, res.x)
    e = res.best.est
    return FitResult(spec, list(data.names), rd.n, list(res.x), res.ll, res.s2,
                     e.termcode, e.iters, e.fk, list(e.dev), res, info,
                     ma_min_modulus(theta))


def rank_test(pre_files, p, q, case=2, ma="free", bootstrap=0, boot_path=None,
              progress=None):
    """The sequential LR test for the rank (`-lrtest [-bootstrap N]`)."""
    data = _data(pre_files)
    spec = Spec(data.M, p, q, 0, case, ma)
    return _rank_test(data, spec, bootstrap, boot_path, progress)

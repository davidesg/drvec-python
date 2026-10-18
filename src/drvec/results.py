"""The structured results behind the .out (STUDY-vec §4.2 items 1-6): arrays and
numbers, each computed by the same routine that prints it, so an accessor and
the .out cannot disagree. Nothing here changes a number the port computes.

Conventions:
* matrices of the VEC (alpha, beta, Pi, Gamma, Theta, Q, Sigma) are in the
  .inp's order, the order of the files: the first s = M - r series are the
  nabla Y2 block, the last r the Y1 block -- as the .out prints them;
* alpha is in Johansen's sign (negative is error-correcting), as printed;
* the residuals and the stationary representation are Ybar's, [nabla Y2 ; W],
  labelled as the .out's per-series diagnosis labels them (the Y2 names, then
  ec1..ecr).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stationary:
    """The stationary VARMA(p, q) on Ybar = [nabla Y2 ; W] that the exact
    likelihood is evaluated on (the C's vec_shootx): Ybar_t - mu =
    sum phi_k (Ybar_{t-k} - mu) + A_t - sum theta_k A_{t-k}, Var A = sigma2 Q."""
    names: list
    mu: np.ndarray          # (M,)
    phi: np.ndarray         # (p, M, M)
    theta: np.ndarray       # (q, M, M)
    qq: np.ndarray          # (M, M), Q up to scale, Ybar's order
    sigma2: float
    w: np.ndarray           # (n, M), Ybar itself

    @property
    def Sigma(self):
        return self.sigma2 * self.qq


@dataclass
class LongRun:
    """The .out's "Long-Run Gain and Mean Lag, in Levels": gain[i, j] is the
    permanent effect on series i of a one-s.d. orthogonalised shock to j;
    mean_lag[i, j] is nan where the gain is zero (the .out says "undefined")."""
    gain: np.ndarray
    mean_lag: np.ndarray
    shock_sd: np.ndarray
    converged: bool
    k_stop: int
    tail: float
    beta_gain: float        # max |beta' gain|, the self-check (zero by construction)


@dataclass
class IrfBands:
    """The -irfboot bands as arrays, [k, i, j] = response of series i to a shock
    to j (the FEVD's k is the horizon minus 1, as in the .out). pct maps the
    percentile (2.5, 5, 50, 95, 97.5) to an array of the point's shape.

    From the Python engine they are the replications' own percentiles; from
    the C engine they are read back from its <base>.irfboot, so they carry
    that file's 8 significant digits."""
    K: int
    nok: int
    nfail: int
    irf: np.ndarray
    irf_pct: dict
    fevd: np.ndarray
    fevd_pct: dict


@dataclass
class Diagnostics:
    """The .out's multivariate residual diagnostics. Q's p-value is nan when
    df < 1; the JB numbers are nan when the residual covariance is singular."""
    hosking_lags: int
    hosking_Q: float
    hosking_df: int
    hosking_p: float
    jb: float
    jb_df: int
    jb_p: float
    jb_skewness: float
    jb_kurtosis: float
    jb_part_df: int


@dataclass
class Forecasts:
    """<base>.forecast as arrays, (H, M) in each series' own units."""
    dates: list             # [(period, sub)] of each step
    names: list
    level: np.ndarray
    low: np.ndarray
    high: np.ndarray
    se: np.ndarray
    conf: float
    z: float
    model_level: np.ndarray     # in the model's units (before the back-transform)
    model_se: np.ndarray


@dataclass
class RestrictionLR:
    """The LR the .out prints for a restriction: kind is "alpha" (-alpha,
    -weakex), "fixb2row", "fixb2row+alpha", "xsys" or "xlink". For -xlink the
    test is of the inputs, so ll_free is the fit WITH them and ll_restricted
    the VEC without them; ll_bridge is the logL at omega = 0."""
    kind: str
    ll_free: float
    ll_restricted: float
    lr: float
    df: int
    pvalue: float
    ll_bridge: float = None


@dataclass
class Provenance:
    """Who computed a result. engine "python" or "c"; version is drvec-python's
    or the C binary's; path the package's or the binary's; backend the
    numerical backend (python: drvarma's engine and the eigensolver; c: "c");
    tier "exact" on Linux x86_64 (the oracle's platform), "tolerance" elsewhere.
    workers and seed are the bootstrap's; note says why a choice was made."""
    engine: str
    version: str
    path: str
    backend: str
    platform: str
    tier: str
    workers: int = 1
    seed: int = None
    note: str = ""
    extra: dict = field(default_factory=dict)


def python_provenance(workers=1, seed=None, note=""):
    import os
    import platform as _pf
    import sys
    from . import __version__
    try:
        from drvarma._engine import engine_backend
        eng = engine_backend()
    except Exception:                                   # noqa: BLE001
        eng = "unknown"
    try:
        from ._gsl import backend as gsl_backend
        eig = gsl_backend()
    except Exception:                                   # noqa: BLE001
        eig = "unknown"
    return Provenance("python", __version__, os.path.dirname(os.path.abspath(__file__)),
                      f"drvarma engine {eng}, eigensolver {eig}", platform_string(),
                      platform_tier(), workers, seed, note)


def platform_string():
    import platform as _pf
    import sys
    return f"{sys.platform} {_pf.machine()}"


def platform_tier():
    import platform as _pf
    import sys
    return ("exact" if sys.platform == "linux" and _pf.machine() in ("x86_64", "AMD64")
            else "tolerance")

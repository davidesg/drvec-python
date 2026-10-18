"""The fitted VEC written as a VARMA in levels, for drvarma's forecaster.

The decision (sima's PLAN-vec, decision 5 and A4): a VEC is a VARMA with an
error term, so the suite forecasts it with the VARMA system's code, one
forecaster and not two. This module only WRITES the model; the forecast is
drvarma's (`drvarma.forecast.forecast_given`). drvec's own `-f`
(forecast.forecast_vec) stays: it is the C's, under byte identity, and the
test (tests/test_levels.py) holds the two together.

The VEC is fitted as a stationary VARMA on Ybar = [nabla Y2 ; W], W = Y1 +
B2' Y2 (cast.shootx):

    PhiBar(B) (Ybar_t - mu) = ThetaBar(B) a_t,     a_t ~ N(0, sigma2 Q)

and Ybar_t = T0 Y_t + T1 Y_{t-1}, Y = [Y2 ; Y1] the levels in the .inp's
column order, in the model's units (fue's cast: refactor * BoxCox(y) - det),

    T0 = [ I_s  0 ; B2'  I_r ],     T1 = [ -I_s  0 ; 0  0 ].

Multiplied out and premultiplied by T0^-1 = [ I_s  0 ; -B2'  I_r ] so that
the polynomial starts at I, the levels VARMA of order (p + 1, q) is

    Phi(B) Y_t = c + Theta(B) e_t,     e_t = T0^-1 a_t,
    Phi_1     = T0^-1 (PhiBar_1 T0 - T1)
    Phi_k     = T0^-1 (PhiBar_k T0 + PhiBar_{k-1} T1)      (k = 2..p)
    Phi_{p+1} = T0^-1 PhiBar_p T1
    Theta_k   = T0^-1 ThetaBar_k T0
    c         = T0^-1 PhiBar(1) mu
    Sigma_e   = T0^-1 (sigma2 Q) T0^-T

Phi(1) = T0^-1 PhiBar(1) T(1) has rank r: the unit roots are in Phi(B)
itself, across the series, which is why the ladder's forecast (a
univariate operator per series, after the VARMA) cannot take it and the
forecast goes through `forecast_given`. The drift of case 3 is not of the
form Phi(1) mu (it lies outside Phi(1)'s columns), so it travels as the
constant c. e_t is the levels' one-step error (e_Y2 = a_nablaY2, e_Y1 = a_W -
B2' a_Y2): the psi weights of this Phi and Theta are drvec's level error map
(report.level_error_map) times T0, and Sigma_e undoes the T0, so the band is
the same quantity.

The deterministic part:
- cases 1, 2, 3, -case pre and -nodrift: all in mu, hence in c (a held drift
  is a zero in mu);
- each file's own deterministic variables (steps, trends, impulses, Easter,
  seasonal dummies of fue): kept out of Y by fue's cast and added back over
  the horizon before the Box-Cox is inverted, as drvec's -f does (data.det_path,
  fue's `_build_xi`);
- -trend (case 4) and -xlink are refused, as drvec's -f refuses them: the C
  does not forecast them, so there is nothing to hold the conversion to.

The Box-Cox (a log at lambda = 0) is inverted by fue's `_inv_boxcox`, inside
drvarma, at each series' lambda and refactor.

The fit is read through the public accessors (FitResult.stationary,
.residuals, .beta2) and the deterministic paths through Data.det_path; the
band multiplier is drvec's own (forecast.z_of).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import z_of      # drvec's band multiplier (the C's three constants)


@dataclass
class LevelsVarma:
    """A fitted VEC as a levels VARMA, Phi(B) Y_t = c + Theta(B) e_t, ready
    for `drvarma.forecast.forecast_given`. Arrays in the .inp's column order
    (the Y2 block, then Y1), the levels in the model's units."""
    phi: np.ndarray         # (p + 1, M, M)
    theta: np.ndarray       # (q, M, M)
    sigma: np.ndarray       # (M, M), the covariance of e_t
    const: np.ndarray       # (M,), c
    w: np.ndarray           # (n + 1, M), the levels up to the origin
    a: np.ndarray           # (n + 1, M), e_t (row 0, the first level, has none)
    names: list
    lambdas: list
    refactors: list
    r: int
    origin: int             # the raw observation of the origin (1-based)
    data: object = None     # drvec's Data: the deterministic paths (det_path)

    @property
    def M(self):
        return self.w.shape[1]

    def det(self, H):
        """(M, H): each series' deterministic component at the H dates
        (drvec's bt_det, built over the sample and H steps beyond it)."""
        T = self.data.nobs_raw + H          # bt_alloc's horizon, as -f builds it
        o = self.origin
        return np.array([self.data.det_path(i, T)[o:o + H] for i in range(self.M)],
                        float).reshape(self.M, H)

    def forecast(self, H, conf=0.95, z=None):
        """H steps through drvarma's forecaster. z: the band's multiplier,
        drvec's for `conf` unless given (drvarma's ladder uses 1.96).
        -> drvarma's list of dicts, one per series (w, sd, level, low, high)."""
        from drvarma.forecast import forecast_given
        return forecast_given(self.phi, self.theta, self.sigma, self.w, self.a, H,
                              const=self.const, det=self.det(H), lam=self.lambdas,
                              refactor=self.refactors, z=z_of(conf) if z is None else z)


def to_levels_varma(res):
    """FitResult -> LevelsVarma, at the fit's estimates, from the fit's origin
    (the last observation it saw: with -estwin E, observation E + 1)."""
    spec, data = res.spec, res.data
    if spec.trend:
        raise ValueError("-trend (case 4): drvec does not forecast it (-f refuses it), "
                         "and neither does the levels VARMA")
    if spec.xlink:
        raise ValueError("-xlink: drvec does not forecast it (-f refuses it), "
                         "and neither does the levels VARMA")
    st = res.stationary
    B2 = res.beta2
    resid = res.residuals
    M, r, p, q = spec.M, spec.r, spec.p, spec.q
    s = M - r
    n = resid.shape[0]
    T0 = np.eye(M)
    T0i = np.eye(M)
    for j in range(r):
        for i in range(s):
            T0[s + j, i] = B2[i, j]
            T0i[s + j, i] = -B2[i, j]
    T1 = np.zeros((M, M))
    T1[:s, :s] = -np.eye(s)
    pb = np.asarray(st.phi, float).reshape(p, M, M)
    tb = np.asarray(st.theta, float).reshape(q, M, M)
    phi = np.zeros((p + 1, M, M))
    for k in range(1, p + 2):
        acc = np.zeros((M, M))
        if k <= p:
            acc += pb[k - 1] @ T0
        if k == 1:
            acc -= T1
        else:
            acc += pb[k - 2] @ T1
        phi[k - 1] = T0i @ acc
    theta = np.array([T0i @ tb[k] @ T0 for k in range(q)]).reshape(q, M, M)
    mu = np.asarray(st.mu, float)
    phi1 = np.eye(M) - pb.sum(axis=0)
    const = T0i @ (phi1 @ mu)
    sigma = T0i @ st.Sigma @ T0i.T
    w = np.asarray(data.raw[:n + 1], float)
    a = np.zeros((n + 1, M))
    a[1:] = resid @ T0i.T
    return LevelsVarma(phi, theta, sigma, const, w, a, list(data.names),
                       [float(v) for v in data.lambdas], [float(v) for v in data.refactors],
                       r, n + 1, data)


def forecast_levels(res, H, conf=0.95, z=None):
    """The fitted VEC forecast H steps by drvarma (to_levels_varma + forecast)."""
    return to_levels_varma(res).forecast(H, conf, z)

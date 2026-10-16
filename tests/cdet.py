"""The oracle's deterministic component, for the BUG-40 comparison only.

drvec-py takes each series' deterministic component from fue's cast
(drvec.data.levels_cast: cast_us_py's step [6]). The C drvec builds it its own
way (fue_pre_reader.c build_det_component, drvec.c:6862-6893): per variable,
omega(B) applied to the indicator, then 1/delta(B) by its recursion from zero
initial conditions, NEVER truncated; the variables summed into det; and w =
refactor * BoxCox(y) - det. fue instead convolves the indicator with nu(B),
truncated at 40 lags when delta != 0 (BUG-40, WON'T FIX in the C), and
subtracts one variable at a time.

This module is that arithmetic, on fue's indicators (`_build_indicator`, the
same values the C's gen_detvar gives) and fue's parameters, so that the tests
can (a) measure the difference between the two components, which is the
delta^k tail beyond lag 40 plus rounding, and (b) feed the port the oracle's
w and check that everything downstream reproduces the oracle byte for byte.
It is NOT used by drvec.
"""
import math

import numpy as np

NULAG = 40          # fue's truncation of nu(B) for a rational term


def _indicator(model, itv, T):
    from fue.cast_us import _build_indicator
    ts = model.series
    freq = int(ts.freq) if ts.freq else 1
    return _build_indicator(itv, T, freq, ts.start[1] if ts.start else 1,
                            ts.start[0] if ts.start else None)


def exact_det(model, T):
    """build_det_component's det[1..T] on the file's calendar (0-based list)."""
    det = [0.0] * (T + 1)
    for itv in model.interventions:
        v = _indicator(model, itv, T)
        om, de = list(itv.omega), list(itv.delta)
        num = [0.0] * (T + 1)
        out = [0.0] * (T + 1)
        for t in range(1, T + 1):
            s = 0.0
            for j in range(len(om)):
                if t - j >= 1:
                    s += (om[0] if j == 0 else -om[j]) * v[t - j]
            num[t] = s
        for t in range(1, T + 1):
            s = num[t]
            for j in range(1, len(de) + 1):
                if t - j >= 1:
                    s += de[j - 1] * out[t - j]
            out[t] = s
        for t in range(1, T + 1):
            det[t] += out[t]
    return det[1:]


def bug40_tail(model, T):
    """sum over the rational variables of sum_{k > 40} nu_k I_{t-k}, t = 1..T:
    what fue's truncated nu(B) leaves out of the exact filter (0-based list)."""
    from fue.cast_us import calcnu_py
    tail = np.zeros(T)
    for itv in model.interventions:
        if not itv.delta or T <= NULAG + 1:
            continue
        v = _indicator(model, itv, T)
        nu = calcnu_py(list(itv.omega), list(itv.delta), T)
        for t in range(NULAG + 2, T + 1):
            tail[t - 1] += sum(nu[k] * v[t - k] for k in range(NULAG + 1, t))
    return tail


def boxcox_c(y, lam, refac):
    """drvec.c:6757."""
    if abs(lam) < 1.0e-8:
        return math.log(y) * refac
    if abs(lam - 1.0) < 1.0e-12:
        return y * refac
    return ((y ** lam - 1.0) / lam) * refac


def exact_w(model):
    """The oracle's w: refactor * BoxCox(y) - det, det the exact filter."""
    ts = model.series
    det = exact_det(model, int(ts.nobs)) if model.interventions else [0.0] * int(ts.nobs)
    return np.array([boxcox_c(float(y), float(model.boxlam), float(model.refactor)) - d
                     for y, d in zip(ts.data, det)])


def oracle_det(monkeypatch):
    """Make drvec take the oracle's deterministic component instead of fue's:
    levels_cast's z and det_path over the horizon. Everything else (fue's
    reader, its ARMA expansion and mean for an -xpre input) is untouched."""
    import drvec.data as dd
    import drvec.xlink as dx
    real = dd.levels_cast

    def levels_cast(model):
        phi, theta, mu, _z = real(model)
        return phi, theta, mu, exact_w(model)

    def det_path(model, off, T):
        return exact_det(model, off + T)[off:off + T]

    monkeypatch.setattr(dd, "levels_cast", levels_cast)
    monkeypatch.setattr(dx, "levels_cast", levels_cast)
    monkeypatch.setattr(dd, "det_path", det_path)

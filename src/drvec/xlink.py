"""-xpre X.pre / -xlink FILE: exogenous transfer-function inputs on the LEVELS
of the system's series (drvec.c:520-640, xl_load 7003, xl_compute 592).

    N_t = Y_t - sum_k nu_k(B) X_{k,t},  nu_k(B) = omega_k(B)/delta_k(B) B^b,
    omega(B) = w0 - w1 B - ...,  delta(B) = 1 - d1 B - ...

N is the VEC process; by linearity nabla Y2 receives nu(B) nabla X and W
receives (nu_1 + B2' nu_2)(B) X in levels (the cast "by subtraction"). X's own
model is frozen at its .pre and used only for the presample: observations of X
before the system's sample first, then a backcast. The parameters go last in
x, after B2 (par_blocks' tail).

Each input is read and cast by fue, as the system's series are
(data.levels_cast): w = refactor * BoxCox(z) minus its own deterministic
component on the input's own sample (drvec.c:7041-7051; fue's regressor and
rule, nu(B) truncated at 40 lags for a rational term, as in the C 0.13; the
link's own nu_k(B) above is the exact filter, untruncated, as in the C); inputs at
most I(1) at frequency zero with no seasonal unit roots. The presample the
sample does not cover is backcast with the input's ARMA from its .pre, fue's
factors multiplied out (the phi and theta of fue's cast, `_unscramble` with
the fixed-frequency AR(2)/MA(2)) and its mean's VALUE whatever the flag
(BUG-38): xl_backcast, on the levels (d = 0) or on the differences, cumulated
backwards (d = 1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .data import abs_period, levels_cast, load_model

XL_MAXORD = 12


@dataclass(frozen=True)
class Link:
    out: int        # 0-based series of the system
    inp: int        # 0-based input
    b: int
    r: int
    s: int


@dataclass
class Inputs:
    names: list
    X: np.ndarray           # (nin, xl_len): row t = npre + raw row
    npre: int
    nreal: list
    nbc: list
    d: list


def _parse_net(path, names, innames):
    links = []
    for raw in open(path):
        line = raw.split("#", 1)[0]
        tok = line.split()
        if not tok:
            continue
        if len(tok) < 6 or tok[1] != "<-":
            raise ValueError(f"{path}: a link is `OUTPUT <- INPUT b r s', got: {raw}")
        o, _a, i, b, r, s = tok[:6]
        b, r, s = int(b), int(r), int(s)
        if b < 0 or r < 0 or s < 0 or r > 2 or s > XL_MAXORD:
            raise ValueError(f"{path}: b, r, s must be >= 0, r <= 2 and s <= {XL_MAXORD}")
        if o not in names or i not in innames:
            raise ValueError(f"{path}: `{o} <- {i}': the output must be a series of the "
                             "system\n       and the input the name of an -xpre file.")
        links.append(Link(names.index(o), innames.index(i), b, r, s))
    if not links:
        raise ValueError(f"{path} declares no link")
    return tuple(links)


def backcast(w, ph, th, mu, L):
    """xl_backcast (drvec.c:6965): drtran's backcast_arma. The ARMA in elf's
    sign convention is symmetric under time reversal, so backcasting is
    forecasting the reversed series with the same phi and theta. -> out[0..L-1],
    out[k-1] the value k periods before w[0]."""
    n, p, q = len(w), len(ph), len(th)
    if L <= 0 or n <= 0:
        return []
    u = [0.0] * (n + 1)
    e = [0.0] * (n + 1)
    for t in range(1, n + 1):
        u[t] = w[n - t] - mu
    for t in range(1, n + 1):
        acc = u[t]
        for i in range(1, p + 1):
            if i >= t:
                break
            acc -= ph[i - 1] * u[t - i]
        for j in range(1, q + 1):
            if j >= t:
                break
            acc += th[j - 1] * e[t - j]
        e[t] = acc
    out = [0.0] * (L + 1)
    for k in range(1, L + 1):
        acc = 0.0
        for i in range(1, p + 1):
            ix = k - i
            acc += ph[i - 1] * (out[ix] - mu if ix >= 1 else u[n + ix])
        for j in range(1, q + 1):
            ix = n + k - j
            if ix <= n:
                acc -= th[j - 1] * e[ix]
        out[k] = acc + mu
    return out[1:]


def load(data, xpre, xnet):
    """xl_load: -> (links, Inputs)."""
    from fue.forecast import _nonsop_coefs
    first = abs_period(data.start[0], data.start[1], data.freq)
    nobs_raw = data.nobs_raw
    ws, sts, innames, ds, mus, phs, ths = [], [], [], [], [], [], []
    for f in xpre:
        ts, m = load_model(f, "ERROR: cannot read the input")
        freq = int(ts.freq) if ts.freq else 1
        if freq != data.freq:
            raise ValueError(f"the input {f} has frequency {freq} and the system {data.freq}")
        d, D = int(m.d or 0), int(m.D or 0)
        ornsop = len(_nonsop_coefs(d, D, freq, ifadf=(m.ifadf or None)))
        if D != 0 or d > 1 or ornsop != d:
            raise ValueError(f"the input {f} is not at most I(1) at frequency zero with no\n"
                             f"       seasonal unit roots (d={d}, D={D}).  Its level would "
                             "not be\n       a transfer input W could stay stationary with.")
        lam = float(m.boxlam)
        for z in ts.data:
            if abs(lam - 1.0) > 1.0e-12 and float(z) <= 0.0:
                raise ValueError(f"{f}: Box-Cox with lambda {lam:g} at a value {float(z):g}")
        phi, theta, mu, z = levels_cast(m)
        w = [float(v) for v in z]
        st = abs_period(ts.start[0], ts.start[1], freq)
        if st + len(w) - 1 < first + nobs_raw - 1:
            raise ValueError(f"the input {f} ends before the system's sample does.  An\n"
                             "       exogenous input must be observed over the whole sample.")
        ws.append(w); sts.append(st); innames.append(ts.name or str(f)); ds.append(d)
        mus.append(mu)
        phs.append(phi)
        ths.append(theta)
    links = _parse_net(xnet, list(data.names), innames)
    maxbs = max(L.b + L.s for L in links)
    anyr = any(L.r > 0 for L in links)
    npre = nobs_raw + maxbs if anyr else maxbs
    xlen = npre + nobs_raw
    X = np.zeros((len(xpre), xlen))
    nreal, nbc = [], []
    for i, w in enumerate(ws):
        k0 = 0
        for k in range(1, xlen + 1):
            per = first + (k - npre - 1)
            ix = per - sts[i] + 1
            if 1 <= ix <= len(w):
                X[i, k - 1] = w[ix - 1]
                if not k0:
                    k0 = k
        L = k0 - 1
        nreal.append(npre - L)
        nbc.append(L)
        if L > 0:
            if ds[i] == 0:
                out = backcast(w, phs[i], ths[i], mus[i], L)
                for k in range(1, L + 1):
                    X[i, k0 - k - 1] = out[k - 1]
            else:
                u = [w[t + 1] - w[t] for t in range(len(w) - 1)]
                out = backcast(u, phs[i], ths[i], mus[i], L)
                for k in range(1, L + 1):
                    X[i, k0 - k - 1] = X[i, k0 - k] - out[k - 1]
    return links, Inputs(innames, X, npre, nreal, nbc, ds)


def npar(links):
    return sum(L.s + 1 + L.r for L in links)


def compute(links, inp, xp, nobs_raw, M):
    """xl_compute: T (nobs_raw, M) = sum over links of nu(B) X, or None when a
    denominator is not stable. The convolution keeps the C's order of
    additions, vectorised over the rows."""
    T = np.zeros((nobs_raw, M))
    xlen = inp.X.shape[1]
    c = 0
    for L in links:
        om = [xp[c + j] for j in range(L.s + 1)]
        c += L.s + 1
        de = [0.0] + [xp[c + j] for j in range(L.r)]
        c += L.r
        if (L.r == 1 and abs(de[1]) >= 1.0) or (L.r == 2 and not (
                de[1] + de[2] < 1.0 and de[2] - de[1] < 1.0 and abs(de[2]) < 1.0)):
            return None
        nu = [0.0] * (xlen + 1)
        for t in range(1, xlen + 1):
            acc = 0.0
            lag = t - 1 - L.b
            if 0 <= lag <= L.s:
                acc = om[0] if lag == 0 else -om[lag]
            for j in range(1, L.r + 1):
                if t > j:
                    acc += de[j] * nu[t - j]
            nu[t] = acc
        Xi = inp.X[L.inp]
        kk = inp.npre + np.arange(1, nobs_raw + 1)          # 1-based
        acc = np.zeros(nobs_raw)
        for m in range(1, int(kk.max()) + 1):
            ok = kk >= m
            acc[ok] += nu[m] * Xi[kk[ok] - m]
        T[:, L.out] += acc
    return T


def labels(links, names, innames):
    out = []
    for L in links:
        out += [(f"{names[L.out]} <- {innames[L.inp]}  w{j}", "xlink", 1) for j in range(L.s + 1)]
        out += [(f"{names[L.out]} <- {innames[L.inp]}  d{j}", "xlink", 1)
                for j in range(1, L.r + 1)]
    return out


_ = re

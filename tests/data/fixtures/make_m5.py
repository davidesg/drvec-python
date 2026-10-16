"""Milestone 5's fixtures: copies of the public .pre files of this directory
with their deterministic block, ARMA block, calendar or mean edited.

    python tests/data/fixtures/make_m5.py      (writes them next to this file)

The data are the public ones the suite already carries (Jenkins and Alavi's
muskrat and mink, Lutkepohl's E1); only the model sections are edited, by hand
values chosen to exercise each branch of the C's reader and filter (see
SOURCE). These are SPECIFICATIONS written by hand, not optima: drvec reads only
the deterministic block (and, for an -xpre input, its ARMA and mean) and
re-estimates everything else, so that does not matter here.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def det_block(dets):
    """dets: [(spec, [omega0, omega1, ...], [delta1, ...])]."""
    if not dets:
        return ["0"]
    out = [str(len(dets)), "**"] + [d[0] for d in dets]
    out += ["**", " ".join(str(len(d[1]) - 1) for d in dets) + " "]
    for _, om, _ in dets:
        out += ["**"] + ["%.6f  1" % v for v in om]
    out += ["**", " ".join(str(len(d[2])) for d in dets) + " "]
    for _, _, de in dets:
        if de:
            out += ["**"] + ["%.6f  1" % v for v in de]
    return out


def arma_block(ar=(), ar_s=(), ma=(), ma_s=(), ar_f=(), ma_f=()):
    """Each factor list: [[coef1, coef2, ...], ...]; fixed-frequency: [(f, c2)]."""
    def sec(head, facs):
        if not facs:
            return [head, "0"]
        out = [head, "%d %s" % (len(facs), " ".join(str(len(f)) for f in facs))]
        for f in facs:
            out += ["**"] + ["%.6f  1" % v for v in f]
        return out

    def fsec(head, facs):
        if not facs:
            return [head, "0"]
        out = [head, "%d %s" % (len(facs), " ".join("%.1f" % f for f, _ in facs))]
        for _, c2 in facs:
            out += ["**", "%.6f  1" % c2]
        return out
    return (sec("** Number and orders of regular AR operators:", ar)
            + sec("** Number and orders of annual AR operators:", ar_s)
            + sec("** Number and orders of regular MA operators:", ma)
            + sec("** Number and orders of anual MA operators:", ma_s)
            + fsec("** Number and frequencies of regular AR(2) operators with fixed "
                   "frequency:", ar_f)
            + fsec("** Number and frequencies of regular MA(2) operators with fixed "
                   "frequency:", ma_f))


def edit(src, dest, dets=None, arma=None, sample=None, freq=None, mean=None,
         ifadf=None, d=None):
    L = open(os.path.join(HERE, src)).read().split("\n")
    i_freq = next(i for i, x in enumerate(L) if "requency of time" in x)
    i_nobs = next(i for i, x in enumerate(L) if "starting date" in x)
    i_det = next(i for i, x in enumerate(L) if "deterministic variables" in x)
    i_ar = next(i for i, x in enumerate(L) if "regular AR operators" in x)
    i_mu = next(i for i, x in enumerate(L) if x.startswith("** Mean"))
    i_bc = next(i for i, x in enumerate(L) if "Box-Cox" in x)
    i_if = i_bc + 2
    if freq is not None:
        L[i_freq + 1] = " %d" % freq
    if sample is not None:
        L[i_nobs + 1] = " " + sample
    if mean is not None:
        L[i_mu + 1] = mean
    if ifadf is not None:
        L[i_if + 1] = " " + ifadf
    if d is not None:
        tok = L[i_bc + 1].split()
        tok[1] = str(d)
        L[i_bc + 1] = " ".join(tok)
    ar_part = L[i_ar:i_mu] if arma is None else arma_block(**arma)
    det_part = L[i_det + 1:i_ar] if dets is None else det_block(dets)
    L = L[:i_det + 1] + det_part + ar_part + L[i_mu:]
    open(os.path.join(HERE, dest), "w").write("\n".join(L))


# Annual, mink: a step with delta near one, a rational impulse, a compensated
# impulse with omega of order one.
MINK_DETS = [("step 1900", [0.3], [0.95]),
             ("impulse 1880", [-0.4], [0.5]),
             ("compimp 1870", [0.2, 0.1], [])]
edit("mmpre.mink.pre", "mmdet.mink.pre", dets=MINK_DETS)
edit("mmpre.mink.pre", "mmdet.minkmu.pre", dets=MINK_DETS, mean="0.0 1")
# Annual, muskrat: a ten-year harmonic pair (the series' own cycle), alter, a
# ramp and a trend, the harmonic with a second-order delta.
edit("mmpre.muskrat.pre", "mmdet.muskrat.pre",
     dets=[("cos 0.1", [0.05], [0.5, -0.3]), ("sin 0.1", [-0.08], []),
           ("alter", [0.02], []), ("ramp 1895", [0.01], []), ("trend", [0.001], [])])
# Quarterly, E1: the seasonal harmonics and alter (the shape of a quarterly
# fue model with deterministic seasonality), dated interventions given as
# `sub year', an omega of order one and a step with delta near one.
edit("e1invest.pre", "e1seas.invest.pre",
     dets=[("cos 1", [0.01], []), ("sin 1", [-0.02], []), ("alter", [0.005], []),
           ("impulse 3 1970", [0.05, 0.03], [])])
edit("e1income.pre", "e1seas.income.pre",
     dets=[("step 1 1975", [0.02], [0.9]), ("cos 1", [0.003], [])])
edit("e1cons.pre", "e1seas.cons.pre",
     dets=[("alter", [0.004], []), ("sin 1", [0.006], [])])
# Monthly: muskrat and mink RELABELLED as 62 months from 1990/1 (an artificial
# calendar on public data, for easter and the monthly harmonics).
edit("mmpre.muskrat.pre", "mmmon.muskrat.pre", freq=12, sample="62 1 1990 muskrat",
     ifadf="0 0 0 0 0 0 0",
     dets=[("easter", [0.03], []), ("cos 1", [0.02], []), ("sin 2", [0.01], []),
           ("step 6 1992", [0.1], [0.8])])
edit("mmpre.mink.pre", "mmmon.mink.pre", freq=12, sample="62 1 1990 mink",
     ifadf="0 0 0 0 0 0 0", dets=[("cos 1", [-0.015], []), ("ramp 3 1993", [0.002], [])])
# -xpre inputs with an ARMA model (the backcast of their presample).
edit("xrev.pre", "xrevarma.pre", mean="0.01 1",
     arma=dict(ar=[[0.6]], ma=[[-0.3]]))
edit("xrev.pre", "xrevar0.pre", mean="10.6 0", d=0,
     arma=dict(ar=[[0.5], [-0.2]], ma=[[0.4]]))
edit("xrev.pre", "xrevdet.pre", mean="0.0 0",
     dets=[("step 1890", [0.2], [0.7]), ("impulse 1860", [0.5], [])],
     arma=dict(ar=[[0.3, 0.2]], ma=[[0.5]]))
edit("xrev.pre", "xmon.pre", freq=12, sample="63 12 1989 xmon", ifadf="0 0 0 0 0 0 0",
     mean="10.6 0", d=0, dets=[("cos 1", [0.04], [])],
     arma=dict(ar=[[0.5]], ma_s=[[0.4]], ar_f=[(3, -0.5)], ma_f=[(2, -0.3)]))
# BUG-40: a step with delta near one and 52 periods after it (1860..1911), so
# that fue's nu(B), truncated at 40 lags, and the C's exact omega(B)/delta(B)
# part from lag 41 on: the tail is 0.3 (0.95^41 - 0.95^(m+1)) / 0.05 after m
# periods, 0.29 at the end of the sample.
edit("mmpre.mink.pre", "mmb40.mink.pre", dets=[("step 1860", [0.3], [0.95])])

"""The `.pre` route: one fue model per series, read as the C reads it.

C: `read_pre_inputs` (drvec.c:6777-6965) and `build_y2_levels` (696-765); the
reader itself is `prefile.read`, the C's read_fue_pre transcribed.

What is taken from each file: the series, its calendar, lambda, the rescaling
factor, the mean's value and flag (-case pre) and its DETERMINISTIC VARIABLES.
What is NOT: the univariate ARMA (on this route drvec re-estimates the diagonal
rung itself, docs/PORT.md §4). Each series is carried in w units,
refactor * BoxCox(z) - det, in LEVELS: drvec differences the nabla-Y2 block
itself, rank by rank.

The deterministic component (drvec.c:6862-6893): build_det_component over the
FILE's own sample (its dates: t = 1 is its first observation), the exact
rational filter omega(B)/delta(B), not truncated (BUG-40 closed WON'T FIX,
2026-09-24); the rows before the common sample are skipped after filtering. It
is kept over the common sample AND the forecast horizon for the back-transform
of the levels (bt_det, BUG-36): `Data.det_path`.

The files are lined up by DATE, on the intersection of their samples (the C's
rule, 6810-6840), which is not drvarma's ladder rule (same last date).

Refused here, where the C would carry on: seasonal differences D > 0 (the C
reads them and silently applies none).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def boxcox_w(z, lam, refac):
    """drvec.c:6757. lambda = 1 is z * refac, not (z - 1) * refac."""
    if abs(lam) < 1.0e-8:
        return math.log(z) * refac
    if abs(lam - 1.0) < 1.0e-12:
        return z * refac
    return ((z ** lam - 1.0) / lam) * refac


def abs_period(year, sub, freq):
    """drvec.c:6768: one index on a common calendar."""
    if freq <= 1:
        return int(year)
    return int(year) * freq + (int(sub) - 1)


@dataclass
class RankData:
    """datamat and Y2_levels for one rank (build_y2_levels, levels layout).

    `dm[t][i]` (0-based t = 0..n-1) is nabla Y2_i for i < s and Y1 in levels for
    i >= s; `y2[t][i]` is the Y2 level. Both as lists (for the C-order loops of
    the seeds) and as arrays (for the column-wise W of the cast)."""
    r: int
    n: int
    dm: list
    y2: list
    dm_np: np.ndarray
    y2_np: np.ndarray


@dataclass
class Data:
    """The system's LEVELS in w units, (nobs_raw, M), in the .inp's column
    order: the first s series are the nabla-Y2 block, the last r the Y1 block."""
    raw: np.ndarray
    names: list
    freq: int = 1
    start: tuple = (1, 1)
    files: list = field(default_factory=list)
    refactors: list = field(default_factory=list)
    pre_mean_free: list = field(default_factory=list)   # each .pre frees its mean (Imu)
    pre_mu: list = field(default_factory=list)          # and its value
    xl: object = None       # -xpre/-xlink: xlink.Inputs (the inputs, with presample)
    lambdas: list = field(default_factory=list)          # each series' Box-Cox lambda
    estwin: int = None      # -estwin E: the fit sees rows 1..E of nabla Y (drvec.c:740)
    det_src: list = field(default_factory=list)          # per series: (PreFile, off) or None
    _ranks: dict = field(default_factory=dict, repr=False)
    ladder_cache: dict = field(default_factory=dict, repr=False)

    @property
    def M(self):
        return self.raw.shape[1]

    @property
    def nobs_raw(self):
        return self.raw.shape[0]

    def with_raw(self, raw):
        """The same system on other levels (a bootstrap replication, a rescale):
        a fresh object, so its rank data and its ladder cache start empty."""
        return Data(np.asarray(raw, float), list(self.names), self.freq,
                    self.start, list(self.files), list(self.refactors),
                    list(self.pre_mean_free), list(self.pre_mu),
                    lambdas=list(self.lambdas), det_src=list(self.det_src))

    def det_path(self, i, T):
        """bt_det[i][1..T] (drvec.c:6876-6882): series i's deterministic
        component on the common sample and beyond, in w units, 0-based (row t
        is raw observation t + 1). Zeros when the file declares none."""
        src = self.det_src[i] if i < len(self.det_src) else None
        if src is None:
            return [0.0] * T
        from .prefile import det_component
        pf, off = src
        return det_component(pf, off + T)[off:off + T]

    @property
    def nobs(self):
        """The C's nobs: the rows of nabla Y the fit sees (-estwin trims it)."""
        return self.estwin if self.estwin else self.nobs_raw - 1

    def at(self, r, full=False):
        """build_y2_levels for rank r (cached); full=True ignores -estwin (the
        rolling evaluation filters the whole sample)."""
        key = r if (full is False or not self.estwin) else (r, "full")
        if key in self._ranks:
            return self._ranks[key]
        M = self.M
        s = M - r
        n = self.nobs_raw - 1 if (full or not self.estwin) else self.estwin
        if n < 3:
            raise ValueError("the levels layout needs at least 4 observations")
        raw = self.raw
        dm = [[0.0] * M for _ in range(n)]
        y2 = [[0.0] * s for _ in range(n)]
        for t in range(n):
            for i in range(s):
                dm[t][i] = float(raw[t + 1, i]) - float(raw[t, i])
                y2[t][i] = float(raw[t + 1, i])
            for j in range(r):
                dm[t][s + j] = float(raw[t + 1, s + j])
        rd = RankData(r, n, dm, y2, np.array(dm, float),
                      np.array(y2, float) if s else np.zeros((n, 0)))
        rd.xl = self.xl
        rd.nobs_raw = self.nobs_raw
        self._ranks[key] = rd
        return rd

    def first_estimated(self):
        """(year, sub) of the first observation of the estimation sample
        (BUG-37: one after the raw start, the first is used by the difference)."""
        y, sub = self.start
        # nobs_raw - nobs + 1 (drvec.c:10970): with -estwin the C counts the
        # window's trim at the START, so its date is late by nobs_full - E
        k = abs_period(y, sub, self.freq) + (self.nobs_raw - self.nobs)
        if self.freq <= 1:
            return k, 1
        return k // self.freq, k % self.freq + 1


def read_pre(files):
    """Read M fue files and line them up by date (read_pre_inputs)."""
    from . import prefile
    if len(files) < 2:
        raise ValueError(f"a VEC model needs at least two series, got {len(files)}")
    series = []
    freq0 = None
    first = last = None
    for k, f in enumerate(files):
        try:
            pf = prefile.read(f)
        except ValueError as e:
            raise ValueError(f"{e}\nERROR: cannot read {f}") from None
        if pf.nobs < 4:
            raise ValueError(f"{f} carries {pf.nobs} observations")
        freq = pf.freq
        if freq0 is None:
            freq0 = freq
        elif freq != freq0:
            raise ValueError(f"{f} has frequency {freq} and {files[0]} has {freq0}.  "
                             "A VEC model is one\n       system on one calendar; the "
                             "series cannot be mixed.")
        if pf.nadiff > 0:
            raise NotImplementedError(
                f"{f} declares D = {pf.nadiff}: drvec differences only the nabla-Y2 "
                "block, regularly; the C ignores D in silence, the port refuses it")
        st = abs_period(pf.begyear, pf.begtime, freq)
        en = st + pf.nobs - 1
        first = st if first is None else max(first, st)
        last = en if last is None else min(last, en)
        series.append((f, pf, st))
    if last - first + 1 < 4:
        raise ValueError(f"the .pre files overlap in {last - first + 1} observation(s); "
                         "at least 4 are\n       needed.  They are lined up by DATE, not "
                         "by position.")
    n = last - first + 1
    M = len(series)
    raw = np.zeros((n, M))
    names, refs, mfree, mus, lams, dets = [], [], [], [], [], []
    for i, (f, pf, st) in enumerate(series):
        off = first - st
        refac = pf.refactor if pf.refactor != 0.0 else 1.0
        lam = pf.boxlam
        names.append(pf.name)
        refs.append(refac)
        lams.append(lam)
        mfree.append(pf.imu != 0)
        mus.append(pf.mu)
        if pf.ndet > 0:
            det = prefile.det_component(pf, pf.nobs)
            dets.append((pf, off))
        else:
            det = [0.0] * pf.nobs
            dets.append(None)
        for t in range(n):
            z = pf.data[off + t]
            if abs(lam - 1.0) > 1.0e-12 and z <= 0.0:
                raise ValueError(f"{f} asks for a Box-Cox with lambda = {lam:g} and "
                                 f"observation {off + t + 1} is {z:g}")
            raw[t, i] = boxcox_w(z, lam, refac) - det[off + t]
    if freq0 <= 1:
        start = (first, 1)
    else:
        start = (first // freq0, first % freq0 + 1)
    return Data(raw, names, freq0, start, [str(f) for f in files], refs, mfree, mus,
                lambdas=lams, det_src=dets)

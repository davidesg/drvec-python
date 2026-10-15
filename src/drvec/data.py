"""The `.pre` route: one fue model per series, read with `fue.load()`.

C: `read_pre_inputs` (drvec.c:6777-6965) and `build_y2_levels` (696-765).

What is taken from each file: the series, its calendar, lambda and the
rescaling factor. What is NOT: the univariate ARMA and the mean (on this route
drvec re-estimates the diagonal rung itself, docs/PORT.md §4). Each series is
carried in w units, refactor * BoxCox(z), in LEVELS: drvec differences the
nabla-Y2 block itself, rank by rank.

The files are lined up by DATE, on the intersection of their samples (the C's
rule, 6810-6840), which is not drvarma's ladder rule (same last date).

Refused here, where the C would carry on (docs/PORT.md §1, REVIEW §5.6):
deterministic variables (the C subtracts its own exact filter; fue truncates
at 40 lags, BUG-40: the decision is pending) and seasonal differences D > 0
(the C reads them and silently applies none).
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
                    self.start, list(self.files), list(self.refactors))

    def at(self, r):
        """build_y2_levels for rank r (cached)."""
        if r in self._ranks:
            return self._ranks[r]
        M = self.M
        s = M - r
        n = self.nobs_raw - 1
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
        self._ranks[r] = rd
        return rd

    def first_estimated(self):
        """(year, sub) of the first observation of the estimation sample
        (BUG-37: one after the raw start, the first is used by the difference)."""
        y, sub = self.start
        k = abs_period(y, sub, self.freq) + 1
        if self.freq <= 1:
            return k, 1
        return k // self.freq, k % self.freq + 1


def read_pre(files):
    """Read M fue files and line them up by date (read_pre_inputs)."""
    import fue
    if len(files) < 2:
        raise ValueError(f"a VEC model needs at least two series, got {len(files)}")
    series = []
    freq0 = None
    first = last = None
    for k, f in enumerate(files):
        ts, model = fue.load(str(f))
        if ts.nobs < 4:
            raise ValueError(f"{f} carries {ts.nobs} observations")
        freq = int(ts.freq) if ts.freq else 1
        if freq0 is None:
            freq0 = freq
        elif freq != freq0:
            raise ValueError(f"{f} has frequency {freq} and {files[0]} has {freq0}. "
                             "A VEC model is one system on one calendar; the series "
                             "cannot be mixed.")
        if model.interventions:
            raise NotImplementedError(
                f"{f} declares deterministic variables: not carried yet (the C "
                "subtracts its exact filter, fue truncates it at 40 lags, BUG-40)")
        if int(model.D or 0) > 0:
            raise NotImplementedError(
                f"{f} declares D = {model.D}: drvec differences only the nabla-Y2 "
                "block, regularly; the C ignores D in silence, the port refuses it")
        y, sub = ts.start
        st = abs_period(y, sub, freq)
        en = st + int(ts.nobs) - 1
        first = st if first is None else max(first, st)
        last = en if last is None else min(last, en)
        series.append((f, ts, model, st))
    if last - first + 1 < 4:
        raise ValueError(f"the .pre files overlap in {last - first + 1} observation(s); "
                         "at least 4 are needed.  They are lined up by DATE, not by position.")
    n = last - first + 1
    M = len(series)
    raw = np.zeros((n, M))
    names, refs = [], []
    for i, (f, ts, model, st) in enumerate(series):
        off = first - st
        refac = float(model.refactor) if model.refactor else 1.0
        lam = float(model.boxlam)
        names.append(ts.name or str(f))
        refs.append(refac)
        for t in range(n):
            z = float(ts.data[off + t])
            if abs(lam - 1.0) > 1.0e-12 and z <= 0.0:
                raise ValueError(f"{f} asks for a Box-Cox with lambda = {lam:g} and "
                                 f"observation {off + t + 1} is {z:g}")
            raw[t, i] = boxcox_w(z, lam, refac)
    if freq0 <= 1:
        start = (first, 1)
    else:
        start = (first // freq0, first % freq0 + 1)
    return Data(raw, names, freq0, start, [str(f) for f in files], refs)

"""The `.pre` route: one fue model per series, read and cast by FUE.

C: `read_pre_inputs` (drvec.c:6777-6965) and `build_y2_levels` (696-765).

The reader and the univariate cast are fue's, as in drtran-python: `fue.load()`
reads each file and `fue.cast_us.cast_us_py` builds its series, at the .pre's
own parameters. drvec differs from drtran only in how it EMBEDS the model:
drtran takes fue's stationary w (step [7] of the cast, the non-stationary
operator applied); drvec works in LEVELS (it differences the nabla-Y2 block
itself, rank by rank), so it takes fue's cast of the same model with the
non-stationary operator emptied (`levels_cast`): step [6]'s

    z = refactor * BoxCox(y) - det,   det = sum of nu(B) I_t

(fue's regressor: `_build_indicator` on the file's own calendar and
`calcnu_py`'s impulse response), and nothing of step [7].

What is taken from each file: the series, its calendar, lambda, the rescaling
factor, the mean's value and flag (-case pre) and z. What is NOT: the
univariate ARMA (on this route drvec re-estimates the diagonal rung itself,
docs/PORT.md §4).

BUG-40 (closed WON'T FIX in the C, 2026-09-24): for a rational term (delta !=
0) fue truncates nu(B) at 40 lags, and the C drvec applies the exact filter
omega(B)/delta(B). Where a delta term reaches beyond lag 40 inside the sample
(or the forecast horizon) the port's w is fue's, not the C's: the difference
is the tail sum_{k>40} nu_k I_{t-k} (docs/PORT.md §13).

The deterministic component is kept over the common sample AND the forecast
horizon for the back-transform of the levels (bt_det, BUG-36): `Data.det_path`,
fue's `_build_xi` (the regressor fue's own forecast adds back), as drtran's
`to_level` does.

The files are lined up by DATE, on the intersection of their samples (the C's
rule, 6810-6840), which is not drvarma's ladder rule (same last date).

Refused: what fue's reader refuses, with fue's message (a type it does not
know, a malformed file); a NON-STANDARD deterministic variable (fue reads it as
a data column, the C refuses it by design: it is a transfer input, drtran's
job), with the C's message; `easter` off monthly data (fue would build a null
regressor, the C refuses it); and seasonal differences D > 0 (the C reads them
and silently applies none).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


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
    det_src: list = field(default_factory=list)          # per series: (fue Model, off) or None
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
        model, off = src
        return det_path(model, off, T)

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




NONSTANDARD_MSG = (
    "  Las variables NO ESTÁNDAR no se admiten por diseño: son\n"
    "  una versión rudimentaria de un modelo de transferencia con\n"
    "  input X. Especifica esa relación como transferencia (ω/δ, b),\n"
    "  que es precisamente lo que drtran estima.")


def load_model(path, what="ERROR: cannot read"):
    """fue.load(path) -> (ts, model), its refusals passed on with the file named.
    A non-standard deterministic variable (fue's `custom`) is refused as the C
    refuses it."""
    import fue
    try:
        ts, model = fue.load(str(path))
    except (ValueError, IndexError, KeyError, OSError) as e:
        raise ValueError(f"{e}\n{what} {path}") from None
    freq = int(ts.freq) if ts.freq else 1
    for k, itv in enumerate(model.interventions):
        if itv.type == "custom":
            raise ValueError(f"{path}: variable determinista {k + 1} no reconocida.\n"
                             + NONSTANDARD_MSG)
        if itv.type == "easter" and freq != 12:
            # fue reads it and builds an identically null regressor (its
            # _build_indicator generates easter on monthly data only); the C
            # refuses it. Refused rather than carried as a term that is zero.
            raise ValueError(f"{path}: variable determinista {k + 1} (easter) on data of "
                             f"frequency {freq}: easter is monthly only")
    return ts, model


def levels_cast(model):
    """fue's cast of `model` at its own parameters, with the non-stationary
    operator emptied: -> (phi, theta, mu, z), z[0..nobs-1] = refactor *
    BoxCox(y) - det (cast_us_py's step [6]; step [7] is the identity).

    The model with d = D = 0 and no ifadf factors is built as drtran builds the
    input re-differenced by its output's operator (_alt_est_specs): a copy of
    fue's model with only the operator changed, `build_est_spec`, and the cast
    at `_build_initial_x` (the .pre's values in fue's own parameter order).
    phi and theta are fue's factors multiplied out (`_unscramble`, the
    fixed-frequency AR(2)/MA(2) included), mu the mean's value whatever its
    flag."""
    import copy
    from fue.cast_us import _build_initial_x, build_est_spec, cast_us_py
    alt = copy.deepcopy(model)
    alt.d, alt.D = 0, 0
    alt.ifadf = [0] * len(model.ifadf) if model.ifadf else model.ifadf
    spec = build_est_spec(alt)
    _p, _q, phi, theta, mu, z, ifault = cast_us_py(_build_initial_x(alt), spec)
    if ifault:
        raise ValueError("a fixed-frequency AR(2)/MA(2) factor of the .pre has c2 >= 0: "
                         "fue's cast refuses it")
    return [float(v) for v in phi], [float(v) for v in theta], float(mu), z


def det_path(model, off, T):
    """The deterministic component of fue's `model` on the file's rows off+1 ..
    off+T (its own calendar, t = 1 its first observation), beyond the sample if
    needed: fue's `_build_xi` at the .pre's omegas and deltas (a step or a
    trend goes on). -> list of T floats."""
    from fue.forecast import _build_xi
    nobs = int(model.series.nobs)
    freq = int(model.series.freq) if model.series.freq else 1
    horizon = max(0, off + T - nobs)
    xi = _build_xi(model, nobs, freq, horizon,
                   [list(i.omega) for i in model.interventions],
                   [list(i.delta) for i in model.interventions])
    return [float(v) for v in xi[1 + off:1 + off + T]]


def _check_positive(f, ts, lam):
    if abs(lam - 1.0) > 1.0e-12:
        for t, z in enumerate(ts.data):
            if float(z) <= 0.0:
                raise ValueError(f"{f} asks for a Box-Cox with lambda = {lam:g} and "
                                 f"observation {t + 1} is {float(z):g}")


def read_pre(files):
    """Read M fue files and line them up by date (read_pre_inputs)."""
    if len(files) < 2:
        raise ValueError(f"a VEC model needs at least two series, got {len(files)}")
    series = []
    freq0 = None
    first = last = None
    for k, f in enumerate(files):
        ts, model = load_model(f)
        if ts.nobs < 4:
            raise ValueError(f"{f} carries {ts.nobs} observations")
        freq = int(ts.freq) if ts.freq else 1
        if freq0 is None:
            freq0 = freq
        elif freq != freq0:
            raise ValueError(f"{f} has frequency {freq} and {files[0]} has {freq0}.  "
                             "A VEC model is one\n       system on one calendar; the "
                             "series cannot be mixed.")
        if int(model.D or 0) > 0:
            raise NotImplementedError(
                f"{f} declares D = {model.D}: drvec differences only the nabla-Y2 "
                "block, regularly; the C ignores D in silence, the port refuses it")
        y, sub = ts.start
        st = abs_period(y, sub if freq > 1 else 1, freq)
        en = st + int(ts.nobs) - 1
        first = st if first is None else max(first, st)
        last = en if last is None else min(last, en)
        series.append((f, ts, model, st))
    if last - first + 1 < 4:
        raise ValueError(f"the .pre files overlap in {last - first + 1} observation(s); "
                         "at least 4 are\n       needed.  They are lined up by DATE, not "
                         "by position.")
    n = last - first + 1
    M = len(series)
    raw = np.zeros((n, M))
    names, refs, mfree, mus, lams, dets = [], [], [], [], [], []
    for i, (f, ts, model, st) in enumerate(series):
        off = first - st
        lam = float(model.boxlam)
        _check_positive(f, ts, lam)
        names.append(ts.name or str(f))
        refs.append(float(model.refactor))
        lams.append(lam)
        mfree.append(bool(model.estimate_mu))
        mus.append(float(model.mu0))
        _phi, _theta, _mu, z = levels_cast(model)
        dets.append((model, off) if model.interventions else None)
        raw[:, i] = z[off:off + n]
    if freq0 <= 1:
        start = (first, 1)
    else:
        start = (first // freq0, first % freq0 + 1)
    return Data(raw, names, freq0, start, [str(f) for f in files], refs, mfree, mus,
                lambdas=lams, det_src=dets)

"""The C's reader of a fue `.pre` file, and what drvec does with what it reads.

C: `read_fue_pre`, `gen_detvar`, `build_det_component` (fue_pre_reader.c),
`expand_ar_factors`/`expand_ma_factors`/`DateToObs` (fue_bridge.c),
`pre_ar_order`/`pre_ma_order` (drvec.c:3513), `Easter` (nlatools.c:715).

The reader is transcribed and not delegated to `fue.load()` because the oracle
is the C's reader: its grammar is fgets/fscanf over the file (a coefficient
section is whitespace-agnostic, a comment line is read and dropped), it accepts
the deterministic types gen_detvar knows (impulse, compimp, step, ramp, easter,
trend/time, cos, sin, alter) and refuses any other one, and drvec regenerates
each deterministic variable from its SPEC LINE over a longer horizon when it
forecasts. `Cursor` mirrors the C stream: `fgets` reads through the newline,
`scan_int`/`scan_float` skip leading white space as fscanf does, `skip_ws` is
fscanf's "\n".

THE DETERMINISTIC COMPONENT (BUG-40, closed WON'T FIX, 2026-09-24): drvec keeps
the EXACT rational filter omega(B)/delta(B), not truncated, started at zero at
the file's first observation, in the Box-Jenkins sign convention
(omega(B) = w0 - w1 B - ..., delta(B) = 1 - d1 B - ...). Each variable is
generated on the file's OWN calendar (t = 1 is the file's first observation),
and the caller lines the component up with the system's sample by date.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

PRE_LINE = 4096
_WS = " \t\n\v\f\r"
_INT = re.compile(r"[-+]?\d+")
_FLT = re.compile(r"[-+]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)",
                  re.IGNORECASE)


class Cursor:
    """A C stdio stream over the file's bytes (latin-1, one char per byte)."""

    def __init__(self, text):
        self.s = text
        self.i = 0

    def fgets(self):
        """-> the next line with its newline (at most PRE_LINE - 1 chars), or
        None at the end of the file."""
        if self.i >= len(self.s):
            return None
        j = self.s.find("\n", self.i, self.i + PRE_LINE - 1)
        end = j + 1 if j >= 0 else min(len(self.s), self.i + PRE_LINE - 1)
        line = self.s[self.i:end]
        self.i = end
        return line

    def skip_ws(self):
        while self.i < len(self.s) and self.s[self.i] in _WS:
            self.i += 1

    def _scan(self, rx):
        self.skip_ws()
        m = rx.match(self.s, self.i)
        if not m:
            return None
        self.i = m.end()
        return m.group(0)

    def scan_int(self):
        t = self._scan(_INT)
        return None if t is None else int(t)

    def scan_float(self):
        t = self._scan(_FLT)
        return None if t is None else float(t)


def sscanf(line, fmt):
    """The few sscanf formats the reader uses: a sequence of %d / %lf / %s
    conversions separated by white space. -> the values converted, in order
    (as many as matched: the C's return count is len())."""
    c = Cursor(line)
    out = []
    for conv in fmt.split():
        if conv == "%d":
            v = c.scan_int()
        elif conv == "%lf":
            v = c.scan_float()
        else:                                   # %s (%31s, %4095s, ...)
            c.skip_ws()
            j = c.i
            while j < len(c.s) and c.s[j] not in _WS:
                j += 1
            v = c.s[c.i:j] if j > c.i else None
            width = int(conv[1:-1]) if conv[1:-1].isdigit() else None
            if v is not None and width:
                v = v[:width]
                j = c.i + len(v)
            c.i = j
        if v is None:
            break
        out.append(v)
    return out


@dataclass
class PreFile:
    """What read_fue_pre fills in Tusmodel and Tseries (the fields drvec uses)."""
    path: str
    freq: int = 0
    nobs: int = 0
    begyear: int = 0
    begtime: int = 1
    name: str = ""
    detspec: list = field(default_factory=list)     # the spec lines, as read
    omega: list = field(default_factory=list)       # omega[i][0..Nomega]
    delta: list = field(default_factory=list)       # delta[i][1..Ndelta] (0-based list)
    ar1: list = field(default_factory=list)         # regular AR factors: coef[1..ord]
    ar2: list = field(default_factory=list)         # annual AR factors
    ma1: list = field(default_factory=list)
    ma2: list = field(default_factory=list)
    ar1f: list = field(default_factory=list)        # (frequency, c2) fixed-frequency AR(2)
    ma1f: list = field(default_factory=list)
    mu: float = 0.0
    imu: int = 0
    boxlam: float = 0.0
    nrdiff: int = 0
    nadiff: int = 0
    ifadf: list = field(default_factory=list)
    refactor: float = 1.0
    data: list = field(default_factory=list)

    @property
    def ndet(self):
        return len(self.detspec)

    @property
    def sper(self):
        return self.freq

    @property
    def ornsop(self):
        o = self.nrdiff + self.sper * self.nadiff
        f = self.ifadf
        if self.sper == 12:
            o += sum(w for flag, w in zip(f, (1, 2, 2, 2, 2, 2, 1)) if flag == 1)
        elif self.sper == 4:
            o += sum(w for flag, w in zip(f, (1, 2, 1)) if flag == 1)
        return o


DET_TYPES_MSG = (
    "  Este lector admite: impulse, compimp, step, ramp, easter,\n"
    "  trend, cos, sin, alter.\n"
    "  Las variables NO ESTÁNDAR no se admiten por diseño: son\n"
    "  una versión rudimentaria de un modelo de transferencia con\n"
    "  input X. Especifica esa relación como transferencia (ω/δ, b),\n"
    "  que es precisamente lo que drtran estima.")


def _arma_section(cur):
    """read_arma_section: -> [coef[1..ord] per factor] (0-based lists)."""
    cur.fgets()                                         # header
    num = cur.scan_int()
    if num is None or num <= 0:
        cur.skip_ws()
        return []
    ords = []
    for _ in range(num):
        o = cur.scan_int()
        ords.append(o if o is not None else 0)
    cur.skip_ws()
    out = []
    for o in ords:
        cur.fgets()                                     # "**"
        coefs = []
        for _ in range(o):
            v = cur.scan_float()
            coefs.append(v if v is not None else 0.0)
            if cur.scan_int() is not None:
                cur.skip_ws()
        out.append(coefs)
    return out


def _fixfreq_section(cur):
    """read_fixfreq_section: -> [(frequency, c2)]."""
    cur.fgets()
    num = cur.scan_int()
    if num is None or num <= 0:
        cur.skip_ws()
        return []
    fre = []
    for _ in range(num):
        v = cur.scan_float()
        fre.append(int(v) if v is not None else 0)      # (int) truncation
    cur.skip_ws()
    out = []
    for f in fre:
        cur.fgets()
        c2 = cur.scan_float()
        if cur.scan_int() is not None:
            cur.skip_ws()
        out.append((f, c2 if c2 is not None else 0.0))
    return out


def read(path):
    """read_fue_pre (fue_pre_reader.c:416). Raises ValueError with the C's
    message where the C's reader fails."""
    with open(path, "rb") as fh:
        cur = Cursor(fh.read().decode("latin-1"))
    pf = PreFile(str(path))
    while True:
        line = cur.fgets()
        if line is None:
            raise ValueError(f"{path} no trae el separador de frecuencia; no es un "
                             "fichero del formato fue")
        if "requency" in line:
            break
    line = cur.fgets() or ""
    if "number" in line or "Number" in line:
        pf.freq = 1
    else:
        v = sscanf(line, "%d")
        pf.freq = v[0] if v else 0
    cur.fgets()
    line = cur.fgets() or ""
    v = sscanf(line, "%d %d %d %4095s %511s")
    if len(v) < 3:
        raise ValueError(f"{path}: the sample line needs `nobs period year [name]', and "
                         f"has {len(v)} number(s): \"{line.rstrip(chr(13) + chr(10))[:60]}\"")
    pf.nobs = v[0]
    pf.begyear = v[2]
    pf.begtime = v[1] if pf.freq > 1 else 1
    pf.name = v[3] if len(v) > 3 else ""
    if pf.nobs <= 0:
        raise ValueError(f"{path} declara {pf.nobs} observaciones")

    cur.fgets()
    nd = cur.scan_int()
    if nd is not None:
        cur.skip_ws()
    nd = nd or 0
    if nd > 0:
        cur.fgets()                                     # "**"
        for i in range(nd):
            line = (cur.fgets() or "").split("\r", 1)[0].split("\n", 1)[0]
            pf.detspec.append(line)
            if gen_detvar(line, pf.freq, pf.begyear, pf.begtime, pf.nobs) is None:
                raise ValueError(f"variable determinista {i + 1} (\"{line}\") no "
                                 f"reconocida.\n{DET_TYPES_MSG}")
        cur.fgets()
        nom = []
        for _ in range(nd):
            o = cur.scan_int()
            nom.append(o if o is not None else 0)
        cur.skip_ws()
        for o in nom:
            cur.fgets()
            om = []
            for _ in range(o + 1):
                x = cur.scan_float()
                om.append(x if x is not None else 0.0)
                if cur.scan_int() is not None:
                    cur.skip_ws()
            pf.omega.append(om)
        cur.fgets()
        ndl = []
        for _ in range(nd):
            o = cur.scan_int()
            ndl.append(o if o is not None else 0)
        cur.skip_ws()
        for o in ndl:
            de = []
            if o > 0:
                cur.fgets()
                for _ in range(o):
                    x = cur.scan_float()
                    de.append(x if x is not None else 0.0)
                    if cur.scan_int() is not None:
                        cur.skip_ws()
            pf.delta.append(de)

    pf.ar1 = _arma_section(cur)
    pf.ar2 = _arma_section(cur)
    pf.ma1 = _arma_section(cur)
    pf.ma2 = _arma_section(cur)
    pf.ar1f = _fixfreq_section(cur)
    pf.ma1f = _fixfreq_section(cur)

    cur.fgets()
    line = cur.fgets() or ""
    v = sscanf(line, "%lf %d")
    pf.mu = v[0] if v else 0.0
    pf.imu = v[1] if len(v) >= 2 else 0
    cur.fgets()
    line = cur.fgets() or ""
    v = sscanf(line, "%lf %d %d")
    pf.boxlam = v[0] if len(v) > 0 else 0.0
    pf.nrdiff = v[1] if len(v) > 1 else 0
    pf.nadiff = v[2] if len(v) > 2 else 0
    cur.fgets()
    line = cur.fgets() or ""
    if pf.freq > 1:
        c = Cursor(line)
        for i in range(pf.freq // 2 + 1):
            x = c.scan_int()
            if x is None:
                raise ValueError(f"{path}: the ifadf line needs {pf.freq // 2 + 1} flags "
                                 f"(frequency {pf.freq}), and has {i}")
            pf.ifadf.append(x)
    cur.fgets()
    line = cur.fgets() or ""
    v = sscanf(line, "%lf %lf")
    pf.refactor = v[1] if len(v) >= 2 else 1.0
    if pf.refactor == 0.0:
        pf.refactor = 1.0
    cur.fgets()
    for i in range(1, pf.nobs + 1):
        line = cur.fgets()
        v = sscanf(line, "%lf") if line is not None else []
        if not v:
            raise ValueError(
                f"{path}: observation {i} of {pf.nobs} is "
                + ("missing (the file ends before the sample does)" if line is None
                   else "not a number"))
        pf.data.append(v[0])
    return pf


def date_to_obs(beg_per, beg_sub, per, sub, freq):
    """DateToObs (fue_bridge.c:59)."""
    srest = freq - beg_sub + 1
    if sub == freq:
        return srest + freq * (per - beg_per)
    return srest + freq * (per - beg_per - 1) + sub


def easter(year):
    """Easter (nlatools.c:715): -> (day, month)."""
    a = year % 19
    b = year % 4
    c = year % 7
    d = (19 * a + 24) % 30
    e = (2 * b + 4 * c + 6 * d + 5) % 7
    day = 22 + d + e
    if day <= 31:
        return day, 3
    return day - 31, 4


def _strtod_prefix(s):
    m = _FLT.match(s.lstrip(_WS))
    return float(m.group(0)) if m else None


def gen_detvar(line, freq, begyear, begtime, nobs):
    """gen_detvar (fue_pre_reader.c:197): the variable on t = 1..nobs of a
    calendar starting at (begyear, begtime). -> list v[0..nobs] (v[0] unused),
    or None when the type is not one the C knows."""
    from .diagnose import obs_to_date
    v = [0.0] * (nobs + 1)
    tok = sscanf(line, "%31s %d %d")
    if len(tok) >= 2 and tok[0] in ("impulse", "compimp", "step", "ramp"):
        kind, i1 = tok[0], tok[1]
        i2 = tok[2] if len(tok) > 2 else None
        if freq == 1:
            i2, i1 = i1, 1
        elif i2 is None:
            return None
        obs = date_to_obs(begyear, begtime, i2, i1, freq)
        if kind == "impulse":
            if 1 <= obs <= nobs:
                v[obs] = 1.0
        elif kind == "compimp":
            if 1 <= obs <= nobs:
                v[obs] = 1.0
            if obs >= 0 and obs + 1 <= nobs:
                v[obs + 1] = -1.0
        elif kind == "step":
            if 1 <= obs <= nobs:
                for j in range(obs, nobs + 1):
                    v[j] = 1.0
        else:
            if 1 <= obs <= nobs:
                for j in range(obs, nobs + 1):
                    v[j] = float(j - obs + 1)
        return v
    if line.startswith("easter") and freq == 12:
        for j in range(1, nobs + 1):
            year, month = obs_to_date(begyear, begtime, j, freq)
            eday, emonth = easter(year)
            if emonth == 4 and month == emonth and eday >= 4:
                v[j] = 1.0
            elif emonth == 4 and month == emonth and eday < 4:
                v[j] = 0.5
                if j > 1:
                    v[j - 1] = 0.5
            elif emonth == 3 and month == emonth:
                v[j] = 1.0
        return v
    PI = 3.14159265358979323846
    if line.startswith("cos"):
        r1 = _strtod_prefix(line[3:])
        if r1 is not None:
            for j in range(1, nobs + 1):
                v[j] = math.cos(2.0 * PI * r1 / freq * j)
            return v
    if line.startswith("sin"):
        r1 = _strtod_prefix(line[3:])
        if r1 is not None:
            for j in range(1, nobs + 1):
                v[j] = math.sin(2.0 * PI * r1 / freq * j)
            return v
    if line.startswith("alter"):
        for j in range(1, nobs + 1):
            v[j] = -1.0 if j % 2 else 1.0          # pow(-1.0, j), exact
        return v
    if line.startswith("trend") or line.startswith("time"):
        for j in range(1, nobs + 1):
            v[j] = float(j)
        return v
    return None


def det_component(pf, nobs_ext):
    """build_det_component (fue_pre_reader.c:367): the sum over the file's
    deterministic variables of omega(B)/delta(B) applied to each, on t =
    1..nobs_ext of the FILE's calendar. -> list det[0..nobs_ext-1] (0-based)."""
    det = [0.0] * (nobs_ext + 1)
    for spec, om, de in zip(pf.detspec, pf.omega, pf.delta):
        v = gen_detvar(spec, pf.freq, pf.begyear, pf.begtime, nobs_ext)
        nw, nd = len(om) - 1, len(de)
        num = [0.0] * (nobs_ext + 1)
        out = [0.0] * (nobs_ext + 1)
        for t in range(1, nobs_ext + 1):
            s = 0.0
            for j in range(nw + 1):
                if t - j >= 1:
                    s += (om[0] if j == 0 else -om[j]) * v[t - j]
            num[t] = s
        for t in range(1, nobs_ext + 1):
            s = num[t]
            for j in range(1, nd + 1):
                if t - j >= 1:
                    s += de[j - 1] * out[t - j]
            out[t] = s
        for t in range(1, nobs_ext + 1):
            det[t] += out[t]
    return det[1:]


def ar_order(pf):
    """pre_ar_order (drvec.c:3513)."""
    return (sum(len(c) for c in pf.ar1) + sum(len(c) * pf.sper for c in pf.ar2)
            + 2 * len(pf.ar1f))


def ma_order(pf):
    return (sum(len(c) for c in pf.ma1) + sum(len(c) * pf.sper for c in pf.ma2)
            + 2 * len(pf.ma1f))


def _expand(reg, ann, fix, sper, p):
    """expand_ar_factors / expand_ma_factors (fue_bridge.c:81, 148): the
    factorised operators multiplied out, Phi(B) = 1 - phi_1 B - ...
    -> [phi_1..phi_p]."""
    if p == 0:
        return []
    work = [1.0] + [0.0] * p
    for coef in reg:
        o = len(coef)
        for k in range(p, -1, -1):
            acc = work[k]
            for j in range(1, o + 1):
                if j > k:
                    break
                acc -= coef[j - 1] * work[k - j]
            work[k] = acc
    for coef in ann:
        o = len(coef)
        for k in range(p, -1, -1):
            acc = work[k]
            for j in range(1, o + 1):
                lag = j * sper
                if lag <= k:
                    acc -= coef[j - 1] * work[k - lag]
            work[k] = acc
    for fr, c2 in fix:
        r = math.sqrt(-c2) if c2 < 0.0 else 0.0
        c1 = 2.0 * math.cos(2.0 * math.pi * fr / sper) * r
        for k in range(p, -1, -1):
            acc = work[k]
            if k >= 1:
                acc -= c1 * work[k - 1]
            if k >= 2:
                acc -= c2 * work[k - 2]
            work[k] = acc
    return [-work[k] for k in range(1, p + 1)]


def expand_ar(pf):
    return _expand(pf.ar1, pf.ar2, pf.ar1f, pf.sper, ar_order(pf))


def expand_ma(pf):
    return _expand(pf.ma1, pf.ma2, pf.ma1f, pf.sper, ma_order(pf))

"""-f H (forecast_vec, drvec.c:6456) and -estwin E (rolling_eval, 5990), against
the oracle's .out, <base>.forecast and <base>.recursive (tests/data/oracle,
SOURCE):

  fc201  mink-muskrat 2 0 1 -case 2 -f 8
  fc211  mink-muskrat 2 1 1 -mean -case 2 -f 6        (the MA terms of the recursion)
  fcpre  mink-muskrat (minkmu) 2 0 1 -case pre -f 5    (case 3: the drifts)
  ewin   mink-muskrat 2 0 1 -case 2 -f 4 -estwin 50
  e1fc   E1 2 0 1 -case 2 -f 8 -estwin 70              (quarterly; logs: z = exp(w))

The first line of the .forecast and the .recursive names the program and its
version (the oracle's DRVEC 0.12 or 0.13 against the port's), and is not compared. On the oracle's
platform everything else is byte-identical; elsewhere the numbers follow the
estimates and are compared to 1e-3 relative."""
import os

import pytest

from conftest import E1, EXACT, FIXTURES, PRE, STRICT
from oracle import golden_filter, out_text

import drvec

MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]
CASES = {
    "fc201": (PRE, (2, 0, 1), dict(case=2, forecast=8)),
    "fc211": (PRE, (2, 1, 1), dict(case=2, forecast=6)),
    "fcpre": (MU, (2, 0, 1), dict(case="pre", forecast=5)),
    "ewin": (PRE, (2, 0, 1), dict(case=2, forecast=4, estwin=50)),
    "e1fc": (E1, (2, 0, 1), dict(case=2, forecast=8, estwin=70)),
}
_FITS = {}


def _fit(label):
    if label not in _FITS:
        files, (p, q, r), kw = CASES[label]
        _FITS[label] = drvec.fit(files, p, q, r, **kw)
    return _FITS[label]


def _numbers(line):
    out = []
    for tok in line.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(tok)
    return out


def _close(mine, theirs, rel=1e-3):
    a, b = mine.splitlines(), theirs.splitlines()
    assert len(a) == len(b)
    for x, y in zip(a, b):
        u, v = _numbers(x), _numbers(y)
        assert len(u) == len(v), (x, y)
        for s, t in zip(u, v):
            if isinstance(t, float) and isinstance(s, float):
                assert abs(s - t) <= rel * max(1.0, abs(t)), (x, y)
            else:
                assert s == t, (x, y)


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_fit(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())
        sec = lambda t: t.split("  Forecast\n")[1].split("one-step self-check")[1]  # noqa: E731
        _close(sec(mine).split("\n", 1)[1], sec(theirs).split("\n", 1)[1])


@pytest.mark.parametrize("label", list(CASES))
def test_the_forecast_file(label):
    mine = _fit(label).forecast_text(label).split("\n", 1)[1]
    theirs = out_text(label + ".forecast").split("\n", 1)[1]
    if EXACT:
        assert mine == theirs
    else:
        _close(mine, theirs)


@pytest.mark.parametrize("label", ["ewin", "e1fc"])
def test_the_recursive_file(label):
    mine = _fit(label).recursive_text().split("\n", 1)[1]
    theirs = out_text(label + ".recursive").split("\n", 1)[1]
    if EXACT:
        assert mine == theirs
    else:
        _close(mine, theirs)


def test_the_window():
    """-estwin trims the sample the fit sees, and the C's header dates its
    first observation as if the trim were at the start (drvec.c:10970)."""
    res = _fit("ewin")
    assert res.nobs == 50
    assert "Sample           : 50 observations from 1862, annual (raw 62)" in res.out_text("ewin")
    assert drvec.fit(PRE, 2, 0, 1, case=2, estwin=50).recursive_text() is None
    assert "(-estwin needs a horizon: give -f H)" in drvec.fit(
        PRE, 2, 0, 1, case=2, estwin=50).out_text("x")
    assert "(-estwin 58 leaves no room for 4 steps)" in drvec.fit(
        PRE, 2, 0, 1, case=2, estwin=58, forecast=4).out_text("x")


def test_refusals():
    with pytest.raises(ValueError, match="-estwin 61 is not inside 10..60"):
        drvec.fit(PRE, 2, 0, 1, case=2, estwin=61, forecast=4)
    with pytest.raises(ValueError, match="-trend \\(case 4\\)"):
        drvec.fit(MU, 2, 0, 1, case="pre", trend=True, forecast=4)
    with pytest.raises(ValueError, match="-irfboot bootstraps"):
        drvec.fit(PRE, 2, 0, 1, case=2, irfboot=10, estwin=50)

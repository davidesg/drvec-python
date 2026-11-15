#!/usr/bin/env python3
"""py-pre-mixed-refactors-silent: with .pre files that do not share one
rescaling factor, drvec C warns and the port says nothing.

drvec C, on the .pre route (drvec.c, after reading the files):

    WARNING: the .pre files do not share one refactor. The system is built in
    w = refactor*BoxCox(z) units, as in drtran, so each entry of B2 is its
    z-unit value times refactor_i / refactor_(Y1 block). ...

The port reads each file's factor (data.py) and uses it to take the forecasts
back to levels, but neither `fit` nor the .out it writes mentions that they
differ. The .out is the C's except for that paragraph, and it is the one that
tells the reader that beta is not in the series' units: with Y at 100 and X at
1, beta prints -0.009955 for a relation that is (1, -0.9955), and its
standard errors are no longer comparable (HOMOLOGATION.md, "mixed factors").

Two synthetic I(1) series in logs, cointegrated; Y's .pre with refactor 100,
X's with 1. Exits 1 while the .out of the fit does not mention the factors;
0 when it warns.
"""
import os
import re
import sys
import tempfile
import warnings

import numpy as np

import fue
import drvec

warnings.filterwarnings("ignore")
N = 200


def pre(path, y, factor, name):
    ts = fue.TimeSeries(list(map(float, y)), freq=12, start=(2000, 1), name=name)
    m = fue.Model(ts, d=1, D=0, boxlam=0.0, ar=[[0.1]], ar_free=[[True]],
                  refactor=float(factor))
    m.fit()
    m.write_pre(path)
    return path


def main():
    rng = np.random.default_rng(8)
    c = np.cumsum(rng.normal(0, 0.01, N))
    u = np.zeros(N)
    for t in range(1, N):
        u[t] = 0.7 * u[t - 1] + rng.normal(0, 0.004)
    d = tempfile.mkdtemp()
    files = [pre(os.path.join(d, "Y.pre"), 120 * np.exp(c + u), 100.0, "Y"),
             pre(os.path.join(d, "X.pre"), 100 * np.exp(c), 1.0, "X")]
    r = drvec.fit(files, 1, 0, 1, case="pre")
    out = r.out_text("vec")
    b = re.search(r"Y in ec1\s+(-?[0-9.]+)", out)
    warned = bool(re.search(r"do not share one refactor|refactor", out))
    print(f"beta as printed: {b.group(1) if b else '?'}  (the relation is (1, -1) in the series' units)")
    print(f"the .out mentions the factors: {warned}")
    return 0 if warned else 1


if __name__ == "__main__":
    sys.exit(main())

"""C defect and its port, FIXED: atsw-gui BUG-62 (engines/drvec/docs/BUGS.md).
Found on 2026-10-23 in the shakedown of sima 0.2.0 (Martín Manjón and Treadway,
1997: (GS2, GS30, GS10), 1985-2008, r = 1 normalised on GS10).

What it was: with r >= 1 the residuals drvec diagnoses are those of
(nabla Y2, W), W = Y1 + B2'Y2: the last r columns are the innovations of the
equilibrium errors, not of the Y1 series. The block after the multivariate
diagnostics ("residual sd:" and the matrices R(k)) printed the columns with no
names, and the last one was read as the Y1 series' (a residual sd of 4.41 =
sqrt(beta' Sigma beta) taken for GS10's, whose sd in Sigma is 21.8). BUG-19 had
named them for the per-series diagnosis only.

This repro fits mink-muskrat, 2 1 1 -case 2, with the port and, when one is
found, with drvec C (c_engine.find()), and looks for the line that names the
columns: `the columns: D.muskrat ec1`.

    python3 bugs/drvec-c-residual-labels/repro.py   # exits 1 while the defect is open
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(ROOT, "tests", "data", "fixtures")
FILES = [os.path.join(FIX, "mmpre.muskrat.pre"), os.path.join(FIX, "mmpre.mink.pre")]
WANT = "  the columns: D.muskrat ec1"

import drvec                                                     # noqa: E402
from drvec import c_engine                                       # noqa: E402

open_ = []
port = drvec.fit(FILES, 2, 1, 1, case=2).out_text("r1")
print("port:", "names the columns" if WANT in port else "NO NAMES")
if WANT not in port:
    open_.append("port")
try:
    binary = c_engine.find()
except c_engine.CEngineUnavailable:
    binary = None
if binary:
    with tempfile.TemporaryDirectory() as t:
        subprocess.run([binary.path] + FILES + ["2", "1", "1", "-case", "2", "-name", "r1"],
                       cwd=t, capture_output=True)
        c = open(os.path.join(t, "r1.out"), errors="replace").read()
    print("C   :", "names the columns" if WANT in c else "NO NAMES", f"({binary.path})")
    if WANT not in c:
        open_.append("C")
else:
    print("C   : no drvec C found (c_engine.find())")
sys.exit(1 if open_ else 0)

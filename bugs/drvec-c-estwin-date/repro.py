"""C defect and its port (drvec C 0.13, still open at atsw-gui dda05d8): under
-estwin the header and the diagnosis are dated from the wrong observation,
atsw-gui BUG-66 (engines/drvec/docs/BUGS.md). Found on 2026-10-15 in milestone
3 (tests/test_forecast.py reproduces the line and asserts it).

BUG-37 dates the first estimated observation as nobs_raw - nobs + 1
(drvec.c ~11427, and trans_d = nobs_raw - nobs for the diagnosis, ~7982),
which assumes the trim is at the START (the first difference). -estwin E trims
at the END (nobs = E, ~747): mink-muskrat, annual, raw 62 from 1850,
2 0 1 -case 2 -f 4 -estwin 50 fits 1851..1900, and the .out says
"Sample : 50 observations from 1862" and dates the residuals 1862..1911.

This repro runs the port and, when one is found, drvec C (c_engine.find()),
and reads the header and the first residual's date.

    python3 bugs/drvec-c-estwin-date/repro.py   # exits 1 while open, 2 without drvec C
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(ROOT, "tests", "data", "fixtures")
FILES = [os.path.join(FIX, "mmpre.muskrat.pre"), os.path.join(FIX, "mmpre.mink.pre")]
WANT = "Sample           : 50 observations from 1851, annual (raw 62)"

import drvec                                                     # noqa: E402
from drvec import c_engine                                       # noqa: E402


def dates(text):
    head = next((l for l in text.splitlines() if l.startswith("Sample ")), "")
    first = re.search(r"^ +1 +(\d{4}) +\|", text, re.M)
    return head, first.group(1) if first else "?"


open_ = []
head, first = dates(drvec.fit(FILES, 2, 0, 1, case=2, forecast=4, estwin=50).out_text("ew"))
print("port:", head, "| first residual:", first)
if head != WANT or first != "1851":
    open_.append("port")
try:
    binary = c_engine.find()
except c_engine.CEngineUnavailable:
    binary = None
if not binary:
    print("C   : no drvec C found (c_engine.find(); DRVEC_BIN names one)")
    sys.exit(2)
with tempfile.TemporaryDirectory() as t:
    subprocess.run([binary.path] + FILES + "2 0 1 -case 2 -f 4 -estwin 50 -name ew".split(),
                   cwd=t, capture_output=True)
    head, first = dates(open(os.path.join(t, "ew.out"), errors="replace").read())
print("C   :", head, "| first residual:", first, f"({binary.path})")
if head != WANT or first != "1851":
    open_.append("C")
print("the window is 1851..1900:", "OPEN in " + " and ".join(open_) if open_ else "fixed")
sys.exit(1 if open_ else 0)

"""C defect (drvec C 0.13, still open at atsw-gui dda05d8): -fdhess's standard
errors are not determined to the digits printed, atsw-gui BUG-67
(engines/drvec/docs/BUGS.md; its tests/repro/fdhess_noise.py is this script).
Found on 2026-10-15 in milestone 3 (TOL_FDSE_REL, tests/test_fdhess.py).

    python3 bugs/drvec-c-fdhess-step/repro.py [path/to/drvec]   # exits 1 while open

The perturbations need the Hessian at points the C cannot be given, so they
run through the port's exact_hessian_se, which is exact_hessian_se and
qnewtopt.c's fdhess line for line: its -fdhess .out is byte-identical to the
C's on Linux x86_64 (tests/test_fdhess.py). With a binary, the
script first checks that the port's s.e. are the C's at the optimum.

Two measurements on E1 (2 0 1 -case 2) and mink-muskrat (2 0 1 -case 2):

  ulp   x moved by ~1 ulp (8 draws): the largest relative change of an s.e.
  step  eta = 1e-14 instead of macheps, which multiplies the step by 3.6:
        the same, at the optimum

Open while either moves an s.e. by more than 1e-3 relative, i.e. while the
fourth significant digit of the printed s.e. is not determined.
Exit 2 when drvec cannot be imported. Without a binary only the port is measured.
"""
import os
import re
import subprocess
import sys
import tempfile
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIX = os.path.join(ROOT, "tests", "data", "fixtures")
sys.path.insert(0, os.path.join(ROOT, "src"))
try:
    import drvec
    from drvec import fdhess as FH
except ImportError:
    print("  needs numpy, fue and drvarma (pip install drvec)")
    sys.exit(2)

E1 = [os.path.join(FIX, "e1%s.pre" % n) for n in ("invest", "income", "cons")]
MM = [os.path.join(FIX, "mmpre.%s.pre" % n) for n in ("muskrat", "mink")]
exe = sys.argv[1] if len(sys.argv) > 1 else None
LIMIT = 1e-3


def c_se(files, name):
    """The s.e. column of the C's -fdhess .out, by parameter label."""
    with tempfile.TemporaryDirectory() as d:
        for f in files:
            shutil.copy(f, d)
        subprocess.run([exe] + [os.path.basename(f) for f in files]
                       + "2 0 1 -case 2 -fdhess -name".split() + [name],
                       cwd=d, capture_output=True)
        out = open(os.path.join(d, name + ".out"), errors="replace").read()
    se = {}
    for m in re.finditer(r"^  (\S.*?\S)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+-?\d+\.\d+\s+\d\.\d{4}",
                         out, re.M):
        se.setdefault(m.group(1), float(m.group(3)))
    return se


worst = 0.0
for name, files in (("E1", E1), ("mink-muskrat", MM)):
    res = drvec.fit(files, 2, 0, 1, case=2, fdhess=True)
    rd = res.data.at(1)
    base = np.array(res.se)
    if exe:
        c = c_se(files, "fdh")
        diff = max(abs(c[p.label] - p.se) for p in res.params if p.label in c)
        print("  %-13s port vs C at the optimum: %d s.e. compared, largest difference %.1e"
              % (name, sum(p.label in c for p in res.params), diff))
    ulp = 0.0
    for seed in range(8):
        x = np.array(res.x) * (1 + 4e-16 * np.random.default_rng(seed).standard_normal(len(res.x)))
        ulp = max(ulp, float(np.max(np.abs(FH.exact_hessian_se(res.spec, rd, list(x)).dev - base)
                                    / base)))
    old = FH._qnewt.MACHEPS
    FH._qnewt.MACHEPS = 1e-14
    try:
        st = FH.exact_hessian_se(res.spec, rd, list(res.x)).dev
    finally:
        FH._qnewt.MACHEPS = old
    rel = np.abs(st - base) / base
    k = int(np.argmax(rel))
    print("  %-13s ulp: %.1e   step x3.6: %.1e  (%s: %.6f -> %.6f)"
          % (name, ulp, rel[k], res.params[k].label, base[k], st[k]))
    worst = max(worst, ulp, float(rel[k]))

if worst > LIMIT:
    print("  OPEN: an s.e. moves by %.1e relative (limit %.0e): its fourth digit is not determined"
          % (worst, LIMIT))
    sys.exit(1)
print("  fixed")

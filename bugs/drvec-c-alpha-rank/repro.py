"""C defect (drvec-v0.12, still open in 0.13 at ae95446): -alpha accepts a
matrix A of deficient rank.

load_alpha_A (drvec.c:2665-2717) checks A's rank through choldcp(A'A), but the
modified Cholesky only faults on a NEGATIVE pivot beyond sqrt(eps)*maxoffl; an
exactly singular A'A gives a zero pivot, which it replaces and accepts. Here
A = [1 1; 1 1; 0 0] (rank 1, sa = 2): the C prints "Restriction H1(r): alpha =
A*psi, with A of 3 x 2", seeds psi through a singular LU (drvec.c:4122-4152),
and writes a .out whose logL and psi are nan, then dies with SIGSEGV (exit
-11 from Python's subprocess, on Linux x86_64). Expected: "ERROR: A
does not have rank 2 (A'A is singular), so psi is not identified", exit 1.

    python3 repro.py [path/to/drvec]     # exits 1 while the defect is open
"""
import os
import subprocess
import sys
import tempfile
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
exe = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.local/share/atsw/drvec-ref-0.13/bin/drvec")
with tempfile.TemporaryDirectory() as d:
    for f in ("e1invest.pre", "e1income.pre", "e1cons.pre", "sing.txt"):
        shutil.copy(os.path.join(HERE, f), d)
    p = subprocess.run([exe, "e1invest.pre", "e1income.pre", "e1cons.pre", "2", "0", "1",
                        "-case", "2", "-alpha", "sing.txt", "-name", "sing"],
                       cwd=d, capture_output=True, text=True)
    out = open(os.path.join(d, "sing.out")).read() if os.path.exists(
        os.path.join(d, "sing.out")) else ""
    nan = "nan" in out.split("logelf")[1].split("\n")[0] if "logelf" in out else False
    print("exit", p.returncode, "| logelf nan:", nan)
    if "does not have rank" not in p.stderr:
        print("OPEN: a rank-1 A was accepted (nan fit: %s, exit %d)" % (nan, p.returncode))
        sys.exit(1)
    print("fixed")
    sys.exit(0)

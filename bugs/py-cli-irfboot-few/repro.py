"""Port defect (drvec-python, found 2026-10-24 while porting -gzero, on
8187255): the CLI dies when -irfboot has too few replications for bands.

With -irfboot N and fewer usable replications than the bands need, drvec C
prints "Too few replications for bands (N)." in the .out, writes no
<base>.irfboot and ends with 0. The port writes the same .out, and then
cli._execute writes res.irfboot_text(base) without looking at it: for too
few replications irfboot.bands_text returns no file (None), and
_write(path, None) raises `TypeError: write() argument must be str, not
None`. The run ends with 1, leaves an EMPTY <base>.irfboot behind (opened
before the write failed), prints no "Done", and skips what comes after it
in _execute: the .forecast of -f H and the .recursive of -estwin.

The library is not affected (drvec.fit returns the FitResult; irfboot_text
is None, as documented); only the CLI.

This repro runs the CLI in process on the oracle's mink-muskrat fixtures,
`3 0 1 -case 2 -irfboot 10 -f 4`, on the Python engine (the C engine goes
through the same _execute). Expected, as the C: exit 0, no .irfboot, the
.forecast written, "Done" on stdout.

    python3 repro.py        # exits 1 while the defect is open
"""
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ["DRVEC_ENGINE"] = "python"
from drvec import cli  # noqa: E402

FX = os.path.join(ROOT, "tests", "data", "fixtures")
files = [os.path.join(FX, f"mmpre.{n}.pre") for n in ("muskrat", "mink")]

with tempfile.TemporaryDirectory() as d:
    base = os.path.join(d, "few")
    out, err = io.StringIO(), io.StringIO()
    try:
        rc = cli.run(files + ["3", "0", "1", "-case", "2", "-irfboot", "10", "-f", "4",
                              "-name", base], out=out, err=err)
        why = None
    except Exception as e:                                   # noqa: BLE001
        rc, why = None, f"{type(e).__name__}: {e}"
    irf = base + ".irfboot"
    left = os.path.getsize(irf) if os.path.exists(irf) else None
    fc = os.path.exists(base + ".forecast")
    done = "Done." in out.getvalue()
    toofew = os.path.exists(base + ".out") and "Too few replications for bands" in open(base + ".out").read()

print("exit code:          ", rc if why is None else f"raised {why}")
print(".out says too few:  ", toofew)
print(".irfboot left:      ", "none" if left is None else f"{left} bytes")
print(".forecast written:  ", fc)
print("'Done' on stdout:   ", done)
ok = why is None and rc == 0 and left is None and fc and done and toofew
sys.exit(0 if ok else 1)

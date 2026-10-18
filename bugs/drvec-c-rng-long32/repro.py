"""C defect, FIXED: atsw-gui BUG-61 (engines/drvec/docs/BUGS.md), fixed in drvec
C 0.13 before its tag, atsw-gui commit 7ec4bf7 (branch drvec-bug61-rng32), and
vendored into csrc/ from there. Found on drvec 0.13 at atsw-gui ae95446.

What it was: the bootstrap's generator assumed a 64-bit `unsigned long`, and
on Windows (LLP64: 32 bits, MinGW-w64 and MSVC alike) `-lrtest -bootstrap`
and `-irfboot` never finished. boot_normal ran a 64-bit LCG in `static
unsigned long boot_rng` and took its top 31 bits with `(boot_rng >> 33)`.
With a 32-bit unsigned long the multiplier and increment are truncated and the
shift is by more than the type's width: undefined behaviour, which gcc folds
to 0. Then u1 = 0 on every draw and `do { ... } while (u1 <= 1.0e-12)` spins
for ever. The -multistart jitter had the same 32-bit LCG and shift (it does
not loop: its starts were just all at amp * -1). On Linux and macOS (LP64)
unsigned long is 64 bits and nothing was wrong.

Found by the Windows wheel of drvec-python (PLAN-vec B5): the smoke test's
first bootstrap through the bundled drvec.exe (MinGW-w64 gcc, MSYS2 UCRT64,
-O2) did not end in 25 minutes (publish.yml run 39125179268, cancelled); a
drvec.exe cross-built here with x86_64-w64-mingw32-gcc, run under wine,
printed the fit and stayed at "r = 0 -> 1 ..." until killed.

The fix: uint64_t (<stdint.h>) and ULL constants for both generators. On
LP64 it is the same type: the binary's code section is byte for byte the
old one's, and the oracle's 83 files do not move.

This repro takes boot_normal and its state FROM the vendored drvec.c (csrc/)
and compiles it twice with the host's gcc: as written (LP64 here: the control,
it must return), and with `unsigned long` read as `uint32_t`, which is what
the type is on Windows. The second must return within 10 s. On the fixed
source there is no `unsigned long` left to read, and both return.

    python3 repro.py        # exits 1 while the defect is open; 0 since 7ec4bf7
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "csrc", "engines", "drvec", "src", "drvec.c")

text = open(SRC).read()
state = re.search(r"static [a-z0-9_ ]+ boot_rng = [^;]+;", text).group(0)
fn = re.search(r"static real boot_normal\(void\)\n\{.*?\n\}\n", text, re.S).group(0)

MAIN = """
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef double real;
%s
%s
int main(void) {
    double s = 0.0;
    int i;
    for (i = 0; i < 1000; i++) s += boot_normal();
    printf("%%.6f\\n", s);
    return 0;
}
"""


def run(source, label, d):
    c = os.path.join(d, label + ".c")
    exe = os.path.join(d, label)
    open(c, "w").write(source)
    subprocess.run(["gcc", "-O2", "-w", c, "-o", exe, "-lm"], check=True)
    try:
        cp = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        return cp.stdout.strip()
    except subprocess.TimeoutExpired:
        return None


with tempfile.TemporaryDirectory() as d:
    lp64 = run(MAIN % (state, fn), "lp64", d)
    llp64 = run((MAIN % (state, fn)).replace("unsigned long", "uint32_t"), "llp64", d)

print("unsigned long of 64 bits (Linux, macOS):", lp64 if lp64 else "DOES NOT RETURN")
print("unsigned long of 32 bits (Windows):     ", llp64 if llp64 else "DOES NOT RETURN in 10 s")
if lp64 is None:
    sys.exit("the control did not return: the repro is broken")
sys.exit(1 if llp64 is None else 0)

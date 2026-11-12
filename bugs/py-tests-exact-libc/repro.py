"""Test defect (drvec-python, found 2026-10-30 by the clean-install test of
atsw 2.1.0 in ten distributions, podman, the wheels from PyPI): the suite's
byte-for-byte comparisons with the oracle's .out (test_bundled, test_accessors,
test_m5, test_options...) are turned on by EXACT, and Linux x86_64 does not say
which libc it has, whose libm moves the last digits:

    glibc 2.35 .. 2.41 (Ubuntu 22.04/24.04, Debian 12/13, openSUSE 15.6,
                        Fedora 41)                 436 passed
    glibc 2.34 (Rocky 9)                           4 failed
    glibc 2.28 (AlmaLinux 8, the manylinux_2_28
                floor the wheel promises)          170 failed
    musl (Alpine; pure wheel, no drvec C)          134 failed

The differences are in the 12th significant digit (a residual sum -7.567e-12
for -7.539e-12, a rankboot.boot LR* ...93444172448 for ...93444172562), and on
Rocky 9 in the Pi eigenvalues printed for a rank-1 Pi on three series (the two
zeros come out as -0.000000, or in another order): the bundled binary and the
port agree with the oracle within the tolerance tier, which those platforms
never got to use.

After py-tests-exact-gsl-v3 (EXACT measured on e1fit) AlmaLinux 8 and Alpine
were down to one failure each and Rocky 9 still had its four (e1fit is right
there; e1fdh and e1seas are not). The AlmaLinux 8 and Alpine one,
test_rank_drift::test_ranks_and_logl[cprq], is in the tolerance tier: cprq's
r = 0 winner is trajectory-dependent (the seed start stops on the step
criterion after 2 iterations at logL -45.59, so the cold start wins with
-6.3016 on glibc 2.28, 1.8396 on musl, for the oracle's 1.8879).

FIX (branch fix-tests-exact-libc): the probe that measures EXACT is e1fit,
e1fdh and e1seas; off EXACT, cprq's r = 0 row is checked for its npar only.

The repro runs, in podman, the affected test files of THIS tree against the
PyPI wheels of atsw 2.1.0 in Rocky 9, AlmaLinux 8 and Alpine (about 4 minutes
each). Expected: they pass.

    python3 bugs/py-tests-exact-libc/repro.py   # exits 1 while open, 2 if it cannot run
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DISTROS = {
    "rocky9": ("docker.io/library/rockylinux:9", "dnf -y -q install python3.12 >/dev/null", "python3.12"),
    "alma8": ("docker.io/library/almalinux:8", "dnf -y -q install python3.11 >/dev/null", "python3.11"),
    "alpine": ("docker.io/library/alpine:3.20", "apk add -q python3", "python3"),
}
TESTS = ("tests/test_e1.py tests/test_fdhess.py tests/test_m5.py tests/test_accessors.py "
         "tests/test_rank_drift.py")

SCRIPT = r"""
set -u
{setup} || exit 90
{py} -m venv /v; V=/v/bin/python; $V -m pip install -q --upgrade pip
$V -m pip install -q --no-cache-dir --only-binary fue,drvarma,drvec atsw==2.1.0 pytest || exit 91
mkdir /w && cp -r /src/tests /w/ && cd /w
export DRVEC_TEST_INSTALLED=1 MPLBACKEND=Agg
$V -c "import platform; print('libc', platform.libc_ver())"
$V -m pytest -q -p no:cacheprovider {tests} 2>&1 | grep -E '^(FAILED|ERROR) |passed|failed' | cut -c1-120
"""

if not shutil.which("podman"):
    print("podman not found: this repro needs it")
    sys.exit(2)

open_, broken = [], []
for name, (image, setup, py) in DISTROS.items():
    p = subprocess.run(["podman", "run", "--rm", "-v", f"{ROOT}:/src:ro,Z", image, "sh", "-c",
                        SCRIPT.format(setup=setup, py=py, tests=TESTS)],
                       capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    print(f"== {name}\n{out}")
    if p.returncode or ("passed" not in out and "failed" not in out):
        broken.append(name)
    elif " failed" in out or "ERROR " in out:
        open_.append(name)
if broken:
    print("the container did not get through in: " + ", ".join(broken))
    sys.exit(2)
print("OPEN: the byte comparisons fail in " + ", ".join(open_) if open_
      else "FIXED: the affected tests pass in Rocky 9, AlmaLinux 8 and Alpine")
sys.exit(1 if open_ else 0)

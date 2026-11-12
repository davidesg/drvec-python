"""Test defect, FIXED (drvec-python, branch bug-tests-exact-gsl-v3): EXACT is
now MEASURED, not assumed (tests/conftest.py). On Linux x86_64 the port must
write the oracle's e1fit.out byte for byte, and a drvec C the oracle's
proute.out (conftest.c_exact); where they do not, the suite runs its
tolerance tier, as off Linux x86_64. DRVEC_REQUIRE_EXACT=1 asserts the bytes
instead of measuring them: tests.yml's reference job sets it, on ubuntu-24.04
(the oracle's Ubuntu) instead of ubuntu-latest, so a regression there still
fails. Measured after the fix, in podman: Ubuntu 26.04 with the amd64v3 GSL,
the whole suite 405 passed, 37 skipped (60 failed before); Ubuntu 24.04 with
DRVEC_REQUIRE_EXACT=1, 437 passed, 5 skipped; and on 24.04 the measurement
says EXACT without the flag. This repro: A, B and C pass, exit 0.
py-tests-exact-libc is a different case and stays open: the libc the repro
mocks does not change the bytes measured on the machine that runs it.

Found 2026-11-11 by publish.yml run 43101885358, confirmed on main by run
43114298968. GitHub's ubuntu-latest moved from
Ubuntu 24.04 to 26.04, whose apt installs GSL 2.8 built for x86-64-v3
(`libgsl28_2.8+dfsg-6_amd64v3.deb`: FMA, AVX2). tests/conftest.py turns the
byte-for-byte comparisons on with EXACT = Linux x86_64 (and STRICT = EXACT and
drvarma's GSL), but the bytes are those of the GSL the oracle runs on (2.7.1,
baseline x86-64). Two smoke jobs fail on that runner:

  Smoke sdist          drvec C built from the sdist against the system GSL:
                       test_bundled.py, 17 of 21 oracle files differ
  Smoke ubuntu-latest  the wheel's binary (static GSL 2.7.1) against the port,
  wheel                whose eigensolver is drvarma built from git against the
                       system GSL: test_c_engine.py, rankboot and cprboot

The differences are in the 11th-12th significant digit (a residual sum
-7.567e-12 for -7.596e-12, a ladder logL -1.0494593228 for -1.0494592956, a
rankboot LR* ...61300357 for ...61299788): the same kind as py-tests-exact-libc,
from the GSL build instead of the libc. NOT a defect of the published wheels:
their binary links GSL 2.7.1 statically and the PyPI drvarma wheel brings its
own GSL, and both pass on the same Ubuntu 26.04 (case A below). With apt's
baseline amd64 GSL 2.8 everything passes too: the variant is what moves the
digits, not the version.

The repro runs, in podman, Ubuntu 26.04 with apt's amd64v3 variant and the
tests of THIS tree:
  A  drvec 0.1.0 wheel and drvarma from PyPI         (control: must pass)
  B  drvec built from this tree against the system GSL: test_bundled.py
  C  drvec 0.1.0 wheel, drvarma v0.3.1 built from git against the system GSL:
     test_bundled.py and test_c_engine.py
Expected: B and C pass, or skip the byte comparisons where the GSL is not the
oracle's. Needs podman and a CPU with x86-64-v3 (FMA, AVX2).

    python3 bugs/py-tests-exact-gsl-v3/repro.py   # exits 1 while open, 2 if it cannot run
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IMAGE = "docker.io/library/ubuntu:26.04"

SCRIPT = r"""
set -u
echo 'APT::Architecture-Variants "amd64v3";' > /etc/apt/apt.conf.d/99variant
apt-get update -qq >/dev/null
apt-get install -y -qq python3 python3-venv python3-dev gcc libgsl-dev git >/w.apt 2>&1 || { tail /w.apt; exit 90; }
grep -o 'libgsl28_[^ ]*\.deb' /w.apt | head -1
python3 -m venv /v; V=/v/bin/python; $V -m pip install -q --upgrade pip
cp -r /src /t; cd /t
export DRVEC_TEST_INSTALLED=1 DRVEC_REQUIRE_C=1 DRVARMA_REQUIRE_ENGINE=1
bin() { $V -c "from drvec import c_engine; print(c_engine._bundled())"; }
res() { $V -m pytest -q -p no:cacheprovider "$@" 2>&1 | grep -E '^(FAILED|ERROR) |passed|failed' | cut -c1-120; }
$V -m pip install -q --only-binary fue,drvarma,drvec drvec==0.1.0 pytest || exit 91
echo "== A"; DRVEC_TEST_BIN=$(bin) res tests/test_bundled.py tests/test_c_engine.py
$V -m pip install -q --no-cache-dir --force-reinstall --no-deps /t || exit 92
echo "== B"; DRVEC_TEST_BIN=$(bin) res tests/test_bundled.py
$V -m pip install -q --no-cache-dir --force-reinstall --no-deps --only-binary drvec drvec==0.1.0
$V -m pip install -q --no-cache-dir --force-reinstall --no-deps \
  "drvarma @ git+https://github.com/davidesg/drvarma-python@v0.3.1" || exit 93
echo "== C"; DRVEC_TEST_BIN=$(bin) res tests/test_bundled.py tests/test_c_engine.py
"""

if not shutil.which("podman"):
    print("podman not found: this repro needs it")
    sys.exit(2)
flags = open("/proc/cpuinfo").read() if os.path.exists("/proc/cpuinfo") else ""
if not all(f in flags.split() for f in ("avx2", "fma", "bmi2")):
    print("this CPU is not x86-64-v3: apt would not give the amd64v3 GSL")
    sys.exit(2)

p = subprocess.run(["podman", "run", "--rm", "-v", f"{ROOT}:/src:ro,Z", IMAGE,
                    "bash", "-c", SCRIPT], capture_output=True, text=True)
out = p.stdout + p.stderr
print(out)
if p.returncode or "== C" not in out:
    print(f"the container did not get through (rc {p.returncode})")
    sys.exit(2)
cases = {}
for block in out.split("== ")[1:]:
    name, _, body = block.partition("\n")
    cases[name.strip()] = body
if " failed" in cases.get("A", " failed"):
    print("the control (A) fails: something else is wrong")
    sys.exit(2)
open_ = [c for c in ("B", "C") if " failed" in cases.get(c, "")]
print("OPEN: byte comparisons fail with the amd64v3 GSL in " + ", ".join(open_)
      if open_ else "FIXED: B and C pass")
sys.exit(1 if open_ else 0)

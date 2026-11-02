"""Test defect (drvec-python, found 2026-10-30 by the clean-install test of
atsw 2.1.0 in ten distributions, podman, the wheels from PyPI): tests/conftest.py
sets EXACT = Linux and x86_64, and EXACT is what turns on the byte-for-byte
comparisons with the oracle's .out (test_bundled, test_accessors, test_m5,
test_options...). Byte identity needs the oracle's libm too, and Linux x86_64
does not say which libc it has:

    glibc 2.35 .. 2.41 (Ubuntu 22.04/24.04, Debian 12/13, openSUSE 15.6,
                        Fedora 41)                 436 passed
    glibc 2.34 (Rocky 9)                           4 failed
    glibc 2.28 (AlmaLinux 8, the manylinux_2_28
                floor the wheel promises)          170 failed
    musl (Alpine; pure wheel, no drvec C)          134 failed

The differences are in the 12th significant digit (a residual sum -7.567e-12
for -7.539e-12, a rankboot.boot LR* ...93444172448 for ...93444172562): the
bundled binary runs and agrees with the oracle within tolerance, which those
platforms never get to use, because EXACT says they are the oracle's.

The repro imports the conftest with the libc of each measured platform.
Expected: EXACT only where the bytes were measured to agree.

    DRVEC_TEST_INSTALLED=1 python3 bugs/py-tests-exact-libc/repro.py   # exits 1 while open
"""
import importlib.util
import os
import platform
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
CONFTEST = os.path.join(HERE, "..", "..", "tests", "conftest.py")
sys.path.insert(0, os.path.dirname(CONFTEST))

measured = {("glibc", "2.28"): "differ", ("glibc", "2.34"): "differ",
            ("musl", "1.2.5"): "differ", ("glibc", "2.35"): "agree",
            ("glibc", "2.41"): "agree"}
wrong = []
with mock.patch.object(sys, "platform", "linux"), \
     mock.patch.object(platform, "machine", lambda: "x86_64"):
    for libc, bytes_ in measured.items():
        with mock.patch.object(platform, "libc_ver", lambda *a, **k: libc):
            spec = importlib.util.spec_from_file_location("conftest_probe", CONFTEST)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        ok = bool(m.EXACT) == (bytes_ == "agree")
        print(f"{libc[0]} {libc[1]:5s}: EXACT {m.EXACT!s:5s} bytes {bytes_:6s} {'ok' if ok else 'WRONG'}")
        wrong += [] if ok else [libc]
sys.exit(1 if wrong else 0)

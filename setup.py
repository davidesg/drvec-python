"""drvec's build: the Python package, plus drvec C as package data (PLAN-vec B5).

The metadata is pyproject.toml's. What this file adds is the executable
drvec/_bin/drvec (drvec.exe on Windows), compiled from the vendored sources in
csrc/ by csrc/build_drvec.py, which c_engine finds as the "bundled" binary.

* A wheel WITH the binary is a platform wheel, py3-none-<platform>: one
  executable serves every Python, since it is not an extension module and
  depends on no CPython ABI.
* DRVEC_NO_C=1 builds the pure-Python wheel, py3-none-any: the fallback for the
  platforms without a binary wheel, as in drvarma.
* Without either variable the binary is built when a C compiler and GSL are
  found, and a failure is a WARNING: the package works without it (the
  bootstraps then run in Python).
* DRVEC_REQUIRE_C=1 makes that failure an ERROR. The wheel builds and the CI
  set it: a binary wheel without its binary must not be produced silently.
"""
import os
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                                   # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

HERE = os.path.dirname(os.path.abspath(__file__))


def _flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class BuildPy(build_py):
    """build_py, then drvec C into <build_lib>/drvec/_bin."""

    drvec_binary = None

    def run(self):
        super().run()
        if _flag("DRVEC_NO_C"):
            if _flag("DRVEC_REQUIRE_C"):
                raise SystemExit("DRVEC_NO_C=1 and DRVEC_REQUIRE_C=1 contradict each other")
            print("drvec: DRVEC_NO_C=1, the pure-Python package (no drvec C)", flush=True)
            return
        sys.path.insert(0, os.path.join(HERE, "csrc"))
        try:
            import build_drvec
        finally:
            sys.path.pop(0)
        out = os.path.join(self.build_lib, "drvec", "_bin", build_drvec.EXE)
        try:
            self.drvec_binary = build_drvec.build(out)
        except build_drvec.BuildError as e:
            if os.path.exists(out):
                os.remove(out)
            if _flag("DRVEC_REQUIRE_C"):
                raise SystemExit(f"DRVEC_REQUIRE_C=1 and drvec C did not build:\n{e}")
            print("WARNING: drvec C was not built, the package goes without it "
                  f"(the bootstraps will run in Python):\n{e}", file=sys.stderr, flush=True)


class BdistWheel(bdist_wheel):
    """A platform wheel when the binary is in it, py3-none-<platform>; the pure
    wheel otherwise."""

    def run(self):
        self.run_command("build")
        self.root_is_pure = not self.get_finalized_command("build_py").drvec_binary
        if not self.root_is_pure:
            # into platlib, not the .data/purelib a pure distribution installs to
            self.distribution.has_ext_modules = lambda: True
        super().run()

    def get_tag(self):
        impl, abi, plat = super().get_tag()
        if not self.root_is_pure:
            return "py3", "none", plat
        return impl, abi, plat


setup(cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel})

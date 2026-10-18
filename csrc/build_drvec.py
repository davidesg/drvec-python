"""Build drvec C, the port's oracle, from the vendored sources in csrc/.

The sources are atsw-gui's, copied byte for byte at the commit recorded in
csrc/VENDORED.json (tests/test_vendored.py checks every sha256; to refresh them,
tools/vendor_drvec_c.py). The build is engines/drvec/Makefile's -- the same
eleven translation units, include directories and -O2 -Wall, minus -g3 -- with
GSL linked STATICALLY when its archives are found, so that the executable needs
nothing but the C library.

Used by setup.py, which puts the executable into the wheel as package data,
drvec/_bin/drvec (drvec.exe on Windows), and by hand:

    python csrc/build_drvec.py [OUT]      default src/drvec/_bin/drvec[.exe]

Environment:
  DRVEC_CC           the C compiler (else CC, else gcc, else cc). On Windows
                     it must be a MinGW-w64 gcc (the toolchain ATSW builds drvec
                     with); MSVC is not supported.
  DRVEC_GSL_PREFIX   a GSL install (include/, lib/); else GSL_ROOT, else
                     pkg-config gsl, else gsl-config, else the compiler's paths
  DRVEC_GSL_STATIC   "1": the static archives are required (the wheels);
                     "0": link the shared library; unset: static when found
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = "drvec.exe" if sys.platform == "win32" else "drvec"
C_VERSION = "0.13"          # src/drvec/c_engine.py's C_VERSION

ENGINE = os.path.join("engines", "drvec")
LIB = "lib"
INCLUDES = [os.path.join(ENGINE, "include"), os.path.join(LIB, "optim"),
            os.path.join(LIB, "diagmv"), os.path.join(LIB, "fuedet")]
# engines/drvec/Makefile: DRVEC_SRC ENGINE_SRC SUITE_SRC, then lnsrch, diagmv, fuedet
SOURCES = [os.path.join(ENGINE, "src", f) for f in (
    "drvec.c", "elfvarma.c", "drvmlest.c", "qnewtopt.c", "nlatools.c",
    "fue_pre_reader.c", "fue_bridge.c", "diagnose.c")] + [
    os.path.join(LIB, "optim", "lnsrch.c"), os.path.join(LIB, "diagmv", "diagmv.c"),
    os.path.join(LIB, "fuedet", "fuedet.c")]
# the Makefile silences these two objects' inherited warnings, and only them
EXTRA = {"fue_bridge.c": ["-Wno-free-nonheap-object"],
         "fue_pre_reader.c": ["-Wno-unused-result", "-Wno-misleading-indentation"]}
CFLAGS = ["-O2", "-Wall"]
SYSTEM_LIBDIRS = ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib64",
                  "/usr/lib", "/usr/local/lib", "/opt/homebrew/lib"]


class BuildError(RuntimeError):
    pass


def _log(msg):
    print("build_drvec: " + msg, flush=True)


def compiler():
    for v in (os.environ.get("DRVEC_CC"), os.environ.get("CC")):
        if v and v.strip():
            return shlex.split(v)
    for name in ("gcc", "cc"):
        p = shutil.which(name)
        if p:
            return [p]
    raise BuildError("no C compiler: set DRVEC_CC (on Windows, a MinGW-w64 gcc)")


def _run(cmd, cwd=None):
    env = None
    if os.path.isabs(cmd[0]):
        # a MinGW gcc given by path needs its own bin/ on the PATH (cc1's DLLs)
        env = dict(os.environ)
        env["PATH"] = os.path.dirname(cmd[0]) + os.pathsep + env.get("PATH", "")
    try:
        cp = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    except OSError as e:
        raise BuildError(f"{cmd[0]}: {e}") from None
    if cp.returncode != 0:
        raise BuildError("command failed (%d): %s\n%s%s" % (
            cp.returncode, " ".join(cmd), cp.stdout[-3000:], cp.stderr[-3000:]))
    return cp.stdout


def _pkg_config(*args):
    exe = shutil.which("pkg-config") or shutil.which("pkgconf")
    if not exe:
        return None
    cp = subprocess.run([exe, *args, "gsl"], capture_output=True, text=True)
    return cp.stdout.strip() if cp.returncode == 0 else None


def _archives(libdir):
    a = [os.path.join(libdir, n) for n in ("libgsl.a", "libgslcblas.a")]
    return a if all(os.path.isfile(p) for p in a) else None


def find_gsl():
    """-> (cflags, link arguments, how), how saying static or shared and where."""
    want = os.environ.get("DRVEC_GSL_STATIC", "").strip()
    prefix = (os.environ.get("DRVEC_GSL_PREFIX") or os.environ.get("GSL_ROOT") or "").strip()
    inc, libdirs, shared = [], [], ["-lgsl", "-lgslcblas"]
    if prefix:
        inc = ["-I" + os.path.join(prefix, "include")]
        libdirs = [os.path.join(prefix, "lib")]
        shared = ["-L" + libdirs[0], "-lgsl", "-lgslcblas"]
        where = prefix
    else:
        cf, libs, libdir = _pkg_config("--cflags"), _pkg_config("--libs"), \
            _pkg_config("--variable=libdir")
        if libs is not None:
            inc = shlex.split(cf or "")
            shared = shlex.split(libs)
            libdirs = [libdir] if libdir else []
            where = "pkg-config"
        else:
            gc = shutil.which("gsl-config")
            if gc:
                pre = subprocess.run([gc, "--prefix"], capture_output=True,
                                     text=True).stdout.strip()
                inc = ["-I" + os.path.join(pre, "include")]
                libdirs = [os.path.join(pre, "lib")]
                shared = ["-L" + libdirs[0], "-lgsl", "-lgslcblas"]
                where = "gsl-config"
            else:
                where = "the compiler's default paths"
        # the system's archives too (pkg-config may point at a shared-only GSL)
        libdirs += [d for d in SYSTEM_LIBDIRS if d not in libdirs]
    if want != "0":
        for d in libdirs:
            a = _archives(d)
            if a:
                return inc, a, f"static, {d} ({where})"
        if want == "1":
            raise BuildError("DRVEC_GSL_STATIC=1 and no libgsl.a/libgslcblas.a in "
                             f"{libdirs or '(no library directory)'} ({where})")
    return inc, shared, f"shared ({where})"


def build(out, verbose=True):
    """Compile and link drvec into `out`; check its banner when it can run here.
    Raises BuildError with the reason."""
    cc = compiler()
    inc, link, how = find_gsl()
    win = sys.platform == "win32"
    mac = sys.platform == "darwin"
    if verbose:
        _log(f"{' '.join(cc)}; GSL {how}")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Objects in a short temporary directory, compiled with RELATIVE names from
    # it (a deep build tree under pip's temporary clone passed Windows' 260
    # characters once: drvarma BUG-0024).
    tmp = tempfile.mkdtemp(prefix="dvc")
    try:
        objs = []
        incs = ["-I" + os.path.join(HERE, d) for d in INCLUDES]
        for src in SOURCES:
            name = os.path.basename(src)
            obj = name[:-2] + ".o"
            cmd = cc + CFLAGS + EXTRA.get(name, []) + incs + inc + \
                ["-c", os.path.join(HERE, src), "-o", obj]
            _run(cmd, cwd=tmp)
            objs.append(obj)
        ldflags = []
        if win:
            ldflags = ["-static", "-s"]         # no libgcc or winpthread DLL beside it
        elif not mac:
            ldflags = ["-s"]
        tmp_out = os.path.join(tmp, EXE)
        _run(cc + ldflags + ["-o", EXE] + objs + link + ["-lm"], cwd=tmp)
        shutil.copyfile(tmp_out, out)
        os.chmod(out, 0o755)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    v = banner(out)
    if v is not None and v != C_VERSION:
        raise BuildError(f"{out} says drvec {v}; the vendored sources must be {C_VERSION}")
    if verbose:
        _log(f"{out} built ({os.path.getsize(out)} bytes)" +
             (f", banner drvec {v}" if v else ", not run (cross build)"))
    return out


def banner(path):
    """The version `path --version` prints, or None when it cannot run here."""
    try:
        cp = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = (cp.stdout or "").strip().splitlines()[:1]
    if not first or not first[0].startswith("drvec "):
        raise BuildError(f"{path} --version printed {cp.stdout!r}")
    return first[0].split()[1]


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(HERE), "src", "drvec", "_bin", EXE)
    try:
        build(dest)
    except BuildError as e:
        sys.exit(f"build_drvec: {e}")

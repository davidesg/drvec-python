"""drvec C as the bootstrap engine (PLAN-vec §B).

The C program drvec is the port's oracle: on Linux x86_64 the port is
byte-identical to it. Running it for the bootstraps is therefore not a second
implementation but the same model, run faster (the port is 6-15x slower). Only
the BOOTSTRAPS go to the C -- the rank test's (`-lrtest -bootstrap N`) and the
IRF bands' (`-irfboot N`). The fits, the restrictions and every structure come
from the port's own objects; the C's replications are read back into them:

* rank_test: the port fits the observed ranks (while the C runs, in parallel);
  the C's <base>.boot gives every replication, its status and its LR* to 17
  digits, which round-trip exactly, and the critical values and the p-value are
  computed from them by the port's own code (rank.summarise). The result is the
  same RankTest the Python engine returns, and its .out and .boot the same
  bytes.
* fit(irfboot=N): the port fits the model; the C's <base>.irfboot gives the
  bands (to its 8 significant digits) and its .out the band section, which the
  port's .out then carries verbatim (the replications themselves are in no file
  of the C's).

Finding the binary, in this order:
  1. DRVEC_BIN (a path);
  2. the binary bundled in the wheel, drvec/_bin/drvec (drvec.exe on Windows),
     built from the vendored sources (csrc/) by setup.py, when present (the
     platform wheels; not the pure wheel);
  3. `drvec` on the PATH;
  4. the ATSW install directory (Linux: /usr/lib/atsw/bin, /usr/local/lib/atsw/bin;
     macOS: /Applications/ATSW.app/Contents/Resources/bin and the same under
     ~/Applications; Windows: %LOCALAPPDATA%\\Programs\\ATSW\\bin and
     %ProgramFiles%\\ATSW\\bin).
The first candidate that exists is the one used, and its banner (`drvec
--version`, first line `drvec X.Y`) must name the version the port mirrors,
C_VERSION; otherwise it is refused with the reason.

The C's bootstrap generator is fixed at the port's default seed: a call with
another seed cannot go to the C.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

C_VERSION = "0.13"          # the oracle the port mirrors (atsw-gui ae95446; csrc/ is abe6602: + BUG-61, 62, 63, + -gzero)
_EXE = "drvec.exe" if sys.platform == "win32" else "drvec"


class CEngineUnavailable(RuntimeError):
    """No usable drvec C binary: the message says why."""


class CEngineError(RuntimeError):
    """The C ran and failed, or its products disagree with the port's fits."""


@dataclass
class CBinary:
    path: str
    version: str
    source: str             # DRVEC_BIN, bundled, PATH or ATSW


def _bundled():
    try:
        from importlib.resources import files
        p = files("drvec").joinpath("_bin").joinpath(_EXE)
        if not p.is_file():
            return None
        p = str(p)
        if sys.platform != "win32" and not os.access(p, os.X_OK):
            # an installer that dropped the wheel's executable bit
            try:
                os.chmod(p, os.stat(p).st_mode | 0o111)
            except OSError:
                pass
        return p
    except Exception:                                   # noqa: BLE001
        return None


def _atsw_dirs():
    home = os.path.expanduser("~")
    if sys.platform == "win32":
        out = []
        for env in ("LOCALAPPDATA",):
            if os.environ.get(env):
                out.append(os.path.join(os.environ[env], "Programs", "ATSW", "bin"))
        for env in ("ProgramFiles", "ProgramFiles(x86)"):
            if os.environ.get(env):
                out.append(os.path.join(os.environ[env], "ATSW", "bin"))
        return out
    if sys.platform == "darwin":
        return ["/Applications/ATSW.app/Contents/Resources/bin",
                os.path.join(home, "Applications", "ATSW.app", "Contents", "Resources", "bin")]
    return ["/usr/lib/atsw/bin", "/usr/local/lib/atsw/bin"]


def candidates():
    """[(source, path)] in the search order, only those that exist (DRVEC_BIN
    is listed even when it does not, so that the refusal can say so)."""
    out = []
    env = os.environ.get("DRVEC_BIN", "").strip()
    if env:
        out.append(("DRVEC_BIN", env))
    b = _bundled()
    if b:
        out.append(("bundled", b))
    w = shutil.which("drvec")
    if w:
        out.append(("PATH", w))
    for d in _atsw_dirs():
        p = os.path.join(d, _EXE)
        if os.path.isfile(p):
            out.append(("ATSW", p))
    return out


_CACHE = {}


def banner_version(path):
    """The version in `drvec --version`'s first line (`drvec 0.13`), or raise
    CEngineUnavailable."""
    try:
        st = os.stat(path)
    except OSError as e:
        raise CEngineUnavailable(f"{path}: {e.strerror}") from None
    key = (path, st.st_mtime_ns, st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    try:
        cp = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise CEngineUnavailable(f"{path} does not run: {e}") from None
    first = (cp.stdout or "").strip().splitlines()[:1]
    m = re.match(r"^drvec\s+(\S+)", first[0]) if first else None
    if not m:
        raise CEngineUnavailable(f"{path} is not drvec C: its banner is "
                                 f"`{first[0] if first else ''}'")
    _CACHE[key] = m.group(1)
    return m.group(1)


def find():
    """The binary to use (CBinary), or raise CEngineUnavailable with the
    reason: none found, or the first one found is not drvec C_VERSION."""
    cands = candidates()
    if not cands:
        raise CEngineUnavailable(
            "drvec C not found: set DRVEC_BIN, put `drvec' on the PATH or install "
            "ATSW with it")
    source, path = cands[0]
    if not os.path.isfile(path):
        raise CEngineUnavailable(f"{source} = {path}: no such file")
    v = banner_version(path)
    if v != C_VERSION:
        raise CEngineUnavailable(
            f"{path} ({source}) is drvec C {v}; drvec-python mirrors {C_VERSION}, and "
            "only that version gives the port's numbers")
    return CBinary(path, v, source)


def available():
    """CBinary or None (no reason given; find() gives it)."""
    try:
        return find()
    except CEngineUnavailable:
        return None


# --------------------------------------------------------------------------- #
#  The translation: a port call -> the C's argv (the inverse of cli.parse)     #
# --------------------------------------------------------------------------- #

_MA_FLAG = {"free": None, "diag": "-diagma", "marow": "-marow", "matri": "-matri"}


def _num(v):
    """A float the C's strtod/atof reads back to the same double."""
    return repr(float(v))


def _common(files, p, q, r, case, ma, nodrift, multistart):
    argv = [os.path.abspath(f) for f in files] + [str(int(p)), str(int(q)), str(int(r))]
    argv += ["-case", "pre" if case == "pre" else str(int(case))]
    if ma not in _MA_FLAG:
        raise ValueError(f"ma = {ma!r} has no drvec C flag")
    if _MA_FLAG[ma]:
        argv.append(_MA_FLAG[ma])
    for i in (nodrift or ()):
        argv += ["-nodrift", str(int(i))]
    if multistart and multistart > 1:
        argv += ["-multistart", str(int(multistart))]
    return argv


def rank_argv(files, p, q, case=2, ma="free", bootstrap=0, nodrift=None, multistart=0,
              base="drvecjob"):
    """drvec.rank_test(...) -> the C's arguments (without the program)."""
    argv = _common(files, p, q, 0, case, ma, nodrift, multistart)
    argv += ["-lrtest"]
    if bootstrap:
        argv += ["-bootstrap", str(int(bootstrap))]
    return argv + ["-name", base]


def fit_argv(files, p, q, r, folder, case=2, ma="free", diag_ar=False, diag_cov=False,
             fixb2row=None, nodrift=None, xsys=None, alpha=None, weakex=None, fixb2=None,
             gzero=None,
             multistart=0, irfboot=0, base="drvecjob"):
    """drvec.fit(...) for -irfboot -> the C's arguments. Only what moves the
    fitted model (and so the samples) is passed: -fdhess and -f do not, and
    -trend, -estwin, -xpre/-xlink are refused with -irfboot by both programs.
    An alpha given as a matrix is written to <folder>/<base>.alpha."""
    argv = _common(files, p, q, r, case, ma, nodrift, multistart)
    if diag_ar:
        argv.append("-diagar")
    if diag_cov:
        argv.append("-diagcov")
    for i, v in sorted((fixb2row or {}).items()):
        argv += ["-fixb2row", str(int(i)), _num(v)]
    for i in (xsys or ()):
        argv += ["-xsys", str(int(i))]
    for i, j in (gzero or ()):
        argv += ["-gzero", str(int(i)), str(int(j))]
    if weakex:
        argv += ["-weakex", str(int(weakex))]
    elif alpha is not None:
        if isinstance(alpha, (str, bytes)) or hasattr(alpha, "__fspath__"):
            argv += ["-alpha", os.path.abspath(os.fsdecode(alpha))]
        else:
            A = [list(row) for row in alpha]
            path = os.path.join(folder, base + ".alpha")
            with open(path, "w") as f:
                f.write("%d %d\n" % (len(A), len(A[0])))
                for row in A:
                    f.write(" ".join(_num(v) for v in row) + "\n")
            argv += ["-alpha", path]
    if fixb2 is True:
        argv.append("-fixb2")
    elif fixb2 is not None and fixb2 is not False:
        argv += ["-fixb2", _num(fixb2)]
    if irfboot:
        argv += ["-irfboot", str(int(irfboot))]
    return argv + ["-name", base]


# --------------------------------------------------------------------------- #
#  Running it, and reading its files back                                      #
# --------------------------------------------------------------------------- #

class Job:
    """One run of the C in `folder`, started at once (Popen); wait() returns
    when it is done. stdout and stderr go to <base>.stdout and <base>.stderr."""

    def __init__(self, binary, argv, folder, base):
        self.binary, self.argv, self.folder, self.base = binary, argv, folder, base
        self.t0 = time.time()
        self._out = open(os.path.join(folder, base + ".stdout"), "w")
        self._err = open(os.path.join(folder, base + ".stderr"), "w")
        self.proc = subprocess.Popen([binary.path] + argv, cwd=folder, stdout=self._out,
                                     stderr=self._err, stdin=subprocess.DEVNULL)

    def path(self, ext):
        return os.path.join(self.folder, self.base + ext)

    def wait(self, on_boot_line=None, poll=0.5):
        """Wait for the C; on_boot_line(line) is called for each complete line
        of <base>.boot as it appears (the C writes one per replication)."""
        seen = 0
        buf = ""
        while True:
            rc = self.proc.poll()
            if on_boot_line is not None and os.path.exists(self.path(".boot")):
                # binary, and the offset in bytes: on Windows the C writes
                # CRLF, and text mode's "\n" for "\r\n" put `seen` behind the
                # file (bugs/py-boot-tail-crlf)
                with open(self.path(".boot"), "rb") as f:
                    f.seek(seen)
                    chunk = f.read()
                seen += len(chunk)
                buf += chunk.decode("ascii", "replace")
                *lines, buf = buf.split("\n")
                for line in lines:
                    on_boot_line(line.rstrip("\r"))
            if rc is not None:
                break
            time.sleep(poll)
        self._out.close()
        self._err.close()
        self.elapsed = time.time() - self.t0
        if rc != 0:
            with open(self.path(".stderr")) as f:
                tail = f.read()[-2000:]
            raise CEngineError(f"drvec C exited with {rc} in {self.folder}:\n{tail}")
        return rc


def parse_boot_line(line):
    """`h0,rep,status,lr` -> rank.Replication, or None for a header line."""
    from .rank import Replication
    if not line or not line[0].isdigit():
        return None
    h0, rep, st, lr = line.strip().split(",")
    return Replication(int(h0), int(rep), st, float(lr) if (st == "ok" and lr) else float("nan"))


def read_boot(path):
    """<base>.boot -> {h0: [Replication]} in the file's order."""
    out = {}
    with open(path) as f:
        for line in f:
            r = parse_boot_line(line)
            if r is not None:
                out.setdefault(r.h0, []).append(r)
    return out


BANDS_BANNER = "Bands for the Level Responses, Parametric Bootstrap"


def read_irfboot(out_text, file_text, names, K, c_base):
    """The C's band section of its .out and its <base>.irfboot ->
    (nok, nfail, band section text, results.IrfBands or None)."""
    import numpy as np
    from .report import banner
    from .results import IrfBands
    b = banner(BANDS_BANNER)
    k = out_text.find(b)
    if k < 0:
        raise CEngineError("the C's .out has no band section")
    sect = out_text[k:]
    m = re.search(r"(\d+) replications simulated from the fitted model and REFITTED as the\n"
                  r"  data were \(same rank, restrictions and ladder start\); (\d+) failed", sect)
    if not m:
        raise CEngineError("the C's band section does not say how many replications it kept")
    nok, nfail = int(m.group(1)), int(m.group(2))
    if file_text is None:
        return nok, nfail, sect, None
    M = len(names)
    ix = {nm: i for i, nm in enumerate(names)}
    irf = np.zeros((K + 1, M, M))
    fev = np.zeros((K + 1, M, M))
    keys = (2.5, 5.0, 50.0, 95.0, 97.5)
    ip = {p: np.zeros((K + 1, M, M)) for p in keys}
    fp = {p: np.zeros((K + 1, M, M)) for p in keys}
    for line in file_text.splitlines()[2:]:
        if not line.strip():
            continue
        what, shock, resp, kk, pt, *q = line.split(",")
        j, i, kk = ix[shock], ix[resp], int(kk)
        if what == "irf":
            irf[kk, i, j] = float(pt)
            for p, v in zip(keys, q):
                ip[p][kk, i, j] = float(v)
        else:
            fev[kk - 1, i, j] = float(pt)
            for p, v in zip(keys, q):
                fp[p][kk - 1, i, j] = float(v)
    return nok, nfail, sect, IrfBands(K, nok, nfail, irf, ip, fev, fp)


def provenance(binary, folder, argv, elapsed=None, note=""):
    from .results import Provenance, platform_string, platform_tier
    return Provenance("c", binary.version, binary.path, "c", platform_string(),
                      platform_tier(), 1, None, note,
                      {"source": binary.source, "folder": folder, "argv": list(argv),
                       "elapsed": elapsed})


class Folder:
    """The job's folder: the caller's (kept, created when missing) or a
    temporary one (removed on exit)."""

    def __init__(self, folder=None):
        self.given = folder is not None
        self.path = os.path.abspath(folder) if folder else tempfile.mkdtemp(prefix="drvec-c-")
        os.makedirs(self.path, exist_ok=True)

    def __enter__(self):
        return self.path

    def __exit__(self, *exc):
        if not self.given:
            shutil.rmtree(self.path, ignore_errors=True)
        return False

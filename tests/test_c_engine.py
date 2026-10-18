"""B: drvec C as the bootstrap engine (drvec.c_engine).

* The finder and the translation, with fake binaries: no C needed.
* The parity: on Linux x86_64 the C path and the Python path give the same
  .out and .boot bytes on the suite's rank bootstraps and the same .out and
  .irfboot on the -irfboot case. These skip, with the reason, when no drvec C
  0.13 is found: DRVEC_TEST_BIN, else DRVEC_BIN, else the local oracle build
  ~/.local/share/atsw/drvec-ref-0.13/bin/drvec (atsw-gui ae95446; tests/data/
  oracle/SOURCE). CI builds it from atsw-gui at that commit (tests.yml).

The one known difference is the C's, BUG-60 (TODO.md): with a drift held, the
C keys its ladder cache so that replication 16 of cprboot starts from
replication 15's ladder. The C path then gives the oracle's .boot, the Python
path its own, and the two differ in that one line; the .out is the same.
"""
import os
import stat
import sys

import pytest

from conftest import E1, EXACT, FIXTURES, PRE, STRICT
from oracle import golden_filter, out_text

import drvec
from drvec import c_engine
from drvec.cli import parse

MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]
MD = [os.path.join(FIXTURES, f) for f in ("mmdet.muskrat.pre", "mmdet.mink.pre")]
_REAL_BUNDLED = c_engine._bundled
LOCAL_ORACLE = os.path.expanduser("~/.local/share/atsw/drvec-ref-0.13/bin/drvec")


def _the_c():
    if os.environ.get("DRVEC_TEST_BIN"):
        return os.environ["DRVEC_TEST_BIN"]      # CI built it: never a silent skip
    for p in (os.environ.get("DRVEC_TEST_BIN"), os.environ.get("DRVEC_BIN"), LOCAL_ORACLE):
        if p and os.path.isfile(p):
            return p
    return None


C_BIN = _the_c()
needs_c = pytest.mark.skipif(C_BIN is None, reason=(
    "no drvec C 0.13: set DRVEC_TEST_BIN or DRVEC_BIN, or build the oracle in "
    "~/.local/share/atsw/drvec-ref-0.13 (tests/data/oracle/SOURCE)"))
posix = pytest.mark.skipif(sys.platform == "win32", reason="fake binaries are shell scripts")


@pytest.fixture
def the_c(monkeypatch):
    monkeypatch.setenv("DRVEC_BIN", C_BIN)
    return C_BIN


# --------------------------------------------------------------------------- #
#  The finder                                                                  #
# --------------------------------------------------------------------------- #

def _fake(path, version="0.13"):
    with open(path, "w") as f:
        f.write("#!/bin/sh\necho 'drvec %s'\necho 'VEC model EML estimation'\n" % version)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def clean(monkeypatch, tmp_path):
    """No binary anywhere: DRVEC_BIN unset, no bundled one, an empty PATH and
    no ATSW install."""
    monkeypatch.delenv("DRVEC_BIN", raising=False)
    monkeypatch.setattr(c_engine, "_bundled", lambda: None)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    monkeypatch.setattr(c_engine, "_atsw_dirs", lambda: [str(tmp_path / "atsw")])
    c_engine._CACHE.clear()
    return tmp_path


def test_none_found(clean):
    with pytest.raises(c_engine.CEngineUnavailable, match="not found"):
        c_engine.find()
    assert c_engine.available() is None


@posix
def test_the_search_order(clean, monkeypatch):
    d = clean
    for sub in ("env", "bundled", "path", "atsw"):
        (d / sub).mkdir()
    atsw = _fake(d / "atsw" / "drvec")
    assert c_engine.find().source == "ATSW" and c_engine.find().path == atsw
    path = _fake(d / "path" / "drvec")
    monkeypatch.setenv("PATH", str(d / "path"))
    assert c_engine.find().source == "PATH" and c_engine.find().path == path
    bundled = _fake(d / "bundled" / "drvec")
    monkeypatch.setattr(c_engine, "_bundled", lambda: bundled)
    assert c_engine.find().source == "bundled" and c_engine.find().path == bundled
    env = _fake(d / "env" / "drvec")
    monkeypatch.setenv("DRVEC_BIN", env)
    b = c_engine.find()
    assert (b.source, b.path, b.version) == ("DRVEC_BIN", env, "0.13")


@posix
def test_the_bundled_binary_in_the_package(clean, monkeypatch, tmp_path):
    """The wheel will carry drvec/_bin/drvec (PLAN-vec §B5): the finder reads it
    through importlib.resources, so a package directory holding one finds it."""
    pkg = tmp_path / "pkg" / "drvec"
    (pkg / "_bin").mkdir(parents=True)
    _fake(pkg / "_bin" / "drvec")

    class _Files:
        def __init__(self, p):
            self.p = p

        def joinpath(self, name):
            return _Files(self.p / name)

        def is_file(self):
            return self.p.is_file()

        def __str__(self):
            return str(self.p)
    import importlib.resources
    monkeypatch.setattr(importlib.resources, "files", lambda name: _Files(pkg))
    monkeypatch.setattr(c_engine, "_bundled", _REAL_BUNDLED)
    b = c_engine.find()
    assert b.source == "bundled" and b.path == str(pkg / "_bin" / "drvec")


@posix
def test_the_wrong_version_is_refused(clean, monkeypatch):
    monkeypatch.setenv("DRVEC_BIN", _fake(clean / "drvec", "0.12"))
    with pytest.raises(c_engine.CEngineUnavailable, match="0.12.*mirrors 0.13"):
        c_engine.find()
    # engine="auto" goes to Python and says why; engine="c" refuses
    t = drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, engine="auto", workers=1)
    assert t.provenance.engine == "python" and "0.12" in t.provenance.note
    with pytest.raises(c_engine.CEngineUnavailable, match="0.12"):
        drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, engine="c")


def test_a_missing_drvec_bin_is_refused(clean, monkeypatch):
    monkeypatch.setenv("DRVEC_BIN", str(clean / "no-such-drvec"))
    with pytest.raises(c_engine.CEngineUnavailable, match="no such file"):
        c_engine.find()


@posix
def test_not_drvec_is_refused(clean, monkeypatch):
    p = clean / "other"
    p.write_text("#!/bin/sh\necho 'something else 1.0'\n")
    p.chmod(0o755)
    monkeypatch.setenv("DRVEC_BIN", str(p))
    with pytest.raises(c_engine.CEngineUnavailable, match="not drvec C"):
        c_engine.find()


def test_engine_argument_validated():
    with pytest.raises(ValueError, match="engine must be"):
        drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, engine="fortran")


# --------------------------------------------------------------------------- #
#  The translation: the inverse of cli.parse                                   #
# --------------------------------------------------------------------------- #

def test_rank_argv_parses_back(tmp_path):
    argv = c_engine.rank_argv(MU, 2, 1, case="pre", ma="matri", bootstrap=99,
                              nodrift=[1], multistart=3, base="job")
    o = parse(argv)
    assert o["files"] == [os.path.abspath(f) for f in MU]
    assert (o["p"], o["q"], o["case"], o["ma"], o["lrtest"], o["bootstrap"],
            o["nodrift"], o["multistart"], o["name"]) == (
        2, 1, "pre", "matri", True, 99, [1], 3, "job")


def test_fit_argv_parses_back(tmp_path):
    argv = c_engine.fit_argv(E1, 2, 0, 1, str(tmp_path), case=2, diag_cov=True,
                             fixb2row={1: 0.1}, weakex=3, multistart=2, irfboot=50,
                             base="job")
    o = parse(argv)
    assert (o["r"], o["case"], o["diag_cov"], o["fixb2row"], o["weakex"], o["multistart"],
            o["irfboot"]) == (1, 2, True, {1: 0.1}, 3, 2, 50)
    A = [[1.0, 0.0], [1.0 / 3.0, 1.0], [0.0, 0.0]]
    argv = c_engine.fit_argv(E1, 2, 0, 1, str(tmp_path), alpha=A, fixb2=0.25, irfboot=30,
                             base="job")
    o = parse(argv)
    assert o["fixb2"] == 0.25 and o["irfboot"] == 30
    from drvec.api import load_alpha_A
    assert load_alpha_A(o["alpha"], 3) == tuple(tuple(r) for r in A)   # bit for bit
    assert parse(c_engine.fit_argv(PRE, 2, 0, 1, str(tmp_path), fixb2=True,
                                   irfboot=30))["fixb2"] is True


# --------------------------------------------------------------------------- #
#  The parity                                                                  #
# --------------------------------------------------------------------------- #

RANKS = {
    "rankboot": (PRE, dict(case=2, bootstrap=20)),
    "cprboot": (MU, dict(case="pre", bootstrap=20)),
    "dboot": (MD, dict(case=2, bootstrap=10)),
    "e1boot": (E1, dict(case=2, bootstrap=20)),
}


def _tail(text):
    return text.split("\n", 1)[1]


@needs_c
@pytest.mark.parametrize("label", [pytest.param(k, marks=pytest.mark.slow) if k == "e1boot"
                                   else k for k in RANKS])
def test_rank_c_and_python_are_the_same_bytes(label, the_c, tmp_path):
    files, kw = RANKS[label]
    seen = []
    tc = drvec.rank_test(files, 2, 0, engine="c", folder=str(tmp_path),
                         progress=lambda b, r: seen.append((r.h0, b, r.status)), **kw)
    tp = drvec.rank_test(files, 2, 0, engine="python", **kw)
    assert tc.provenance.engine == "c" and tc.provenance.version == "0.13"
    assert tc.provenance.path == the_c and tc.provenance.extra["folder"] == str(tmp_path)
    assert tp.provenance.engine == "python"
    assert len(seen) == kw["bootstrap"] * len(tc.bootstraps)
    # the C's own files are in the folder
    c_out = open(tmp_path / "drvecjob.out").read()
    c_boot = open(tmp_path / "drvecjob.boot").read()
    assert _tail(tc.boot_text()) == _tail(c_boot)
    if not EXACT:
        # another platform's C: the same decisions, its own bits
        assert [r.status for b in tc.bootstraps for r in b.reps] == \
            [r.status for b in tp.bootstraps for r in b.reps]
        return
    assert golden_filter(tc.out_text(label)) == golden_filter(c_out.replace(
        "drvecjob.", label + "."))
    if STRICT:
        assert tc.out_text(label) == tp.out_text(label)
    if label == "cprboot":
        # BUG-60: the C path is the oracle's .boot; the port's differs in one line
        assert _tail(tc.boot_text()) == _tail(out_text("cprboot.boot"))
        a, b = _tail(tc.boot_text()).splitlines(), _tail(tp.boot_text()).splitlines()
        assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [16]
        assert a[16].startswith("0,16,ok,")
    else:
        assert tc.boot_text() == tp.boot_text()


@needs_c
def test_irfboot_c_and_python_are_the_same_bytes(the_c, tmp_path):
    MM = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "mmpre.mink.pre")]
    rc = drvec.fit(MM, 2, 0, 1, case=2, irfboot=40, engine="c")
    rp = drvec.fit(MM, 2, 0, 1, case=2, irfboot=40, engine="python")
    assert rc.provenance.engine == "c" and rp.provenance.engine == "python"
    assert rc.irfboot.nok == rp.irfboot.nok == 40
    bc, bp = rc.irf_bands(), rp.irf_bands()
    assert abs(bc.irf_pct[95.0] - bp.irf_pct[95.0]).max() < 1e-7
    if not EXACT:
        return
    assert rc.out_text("irfboot") == rp.out_text("irfboot")
    assert rc.irfboot_text("irfboot") == rp.irfboot_text("irfboot")
    assert golden_filter(rc.out_text("irfboot")) == golden_filter(out_text("irfboot.out"))


@needs_c
def test_auto_takes_the_c(the_c):
    t = drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, engine="auto")
    assert t.provenance.engine == "c"
    # without a bootstrap there is nothing for the C to do
    assert drvec.rank_test(MD, 2, 0, case=2, engine="c").provenance.engine == "python"


@needs_c
def test_the_refusals_are_the_same(the_c):
    with pytest.raises(ValueError, match="-trend"):
        drvec.rank_test(MU, 2, 0, case="pre", bootstrap=10, trend=True, engine="c")
    with pytest.raises(ValueError, match="-irfboot bootstraps"):
        drvec.fit(MU, 2, 0, 1, case="pre", trend=True, irfboot=10, engine="c")

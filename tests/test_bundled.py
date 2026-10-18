"""The drvec C that comes with the package (PLAN-vec B5): drvec/_bin/drvec,
built from the vendored sources (csrc/, tests/test_vendored.py) by setup.py.

* It is there when it must be: with DRVEC_REQUIRE_C=1 (the platform wheels'
  smoke tests, and the CI jobs that build it) a missing binary FAILS; without
  it these tests skip when the package carries none (the pure wheel).
* A small bootstrap goes through it: c_engine finds the bundled binary when
  DRVEC_BIN is unset, and the result's provenance says so.
* On Linux x86_64, the oracle's platform, it IS the oracle: run on the oracle's
  commands (tests/data/oracle/SOURCE), its .out, .boot, .forecast, .recursive
  and .irfboot equal the oracle's files byte for byte, after golden.sh's filter
  (the version banners and /tmp paths) -- the binary built here and the one
  that wrote the goldens are the same program.
"""
import os
import re
import shutil
import subprocess

import pytest

from conftest import E1, EXACT, FIXTURES, PRE
from oracle import golden_filter, out_text

import drvec
from drvec import c_engine

REQUIRED = os.environ.get("DRVEC_REQUIRE_C", "").strip() == "1"
BUNDLED = c_engine._bundled()

needs_bundled = pytest.mark.skipif(BUNDLED is None, reason=(
    "this drvec carries no binary (the pure wheel, or a build without a C "
    "compiler or GSL: see csrc/build_drvec.py)"))


def test_the_binary_is_there_when_required():
    if not REQUIRED:
        pytest.skip("DRVEC_REQUIRE_C is not set")
    assert BUNDLED is not None, "DRVEC_REQUIRE_C=1 and drvec/_bin has no drvec"
    assert os.access(BUNDLED, os.X_OK)


@needs_bundled
def test_its_banner():
    assert c_engine.banner_version(BUNDLED) == c_engine.C_VERSION


@needs_bundled
def test_a_bootstrap_through_the_bundled_binary(monkeypatch, tmp_path):
    """The smoke test of every platform wheel: one small rank bootstrap on the
    C path, found as the bundled binary."""
    monkeypatch.delenv("DRVEC_BIN", raising=False)
    c_engine._CACHE.clear()
    b = c_engine.find()
    assert (b.source, b.path, b.version) == ("bundled", BUNDLED, c_engine.C_VERSION)
    t = drvec.rank_test(PRE, 2, 0, case=2, bootstrap=10, engine="c", folder=str(tmp_path))
    pv = t.provenance
    print(f"\nbundled drvec C: {pv.path} ({pv.extra['source']}, {pv.version}, "
          f"{pv.platform}, tier {pv.tier}, {pv.extra['elapsed']:.2f} s)")
    assert pv.engine == "c" and pv.extra["source"] == "bundled" and pv.path == BUNDLED
    assert len(t.bootstraps) >= 1 and all(len(bs.reps) == 10 for bs in t.bootstraps)
    # engine="auto" takes it too
    monkeypatch.setenv("DRVEC_ENGINE", "auto")
    a = drvec.rank_test(PRE, 2, 0, case=2, bootstrap=10)
    assert a.provenance.engine == "c" and a.provenance.extra["source"] == "bundled"
    assert a.boot_text() == t.boot_text()


# --------------------------------------------------------------------------- #
#  The oracle, byte for byte, on its platform                                  #
# --------------------------------------------------------------------------- #

MM = ["mmpre.muskrat.pre", "mmpre.mink.pre"]
MU = ["mmpre.muskrat.pre", "minkmu.pre"]
MD = ["mmdet.muskrat.pre", "mmdet.mink.pre"]
MON = ["mmmon.muskrat.pre", "mmmon.mink.pre"]
E1F = ["e1invest.pre", "e1income.pre", "e1cons.pre"]
# label: (arguments, the files besides the .out); tests/data/oracle/SOURCE
ORACLE = {
    "proute": (MM + "2 1 1 -mean -case 2".split(), ()),
    "rankboot": (MM + "2 0 1 -case 2 -lrtest -bootstrap 20".split(), (".boot",)),
    "cprboot": (MU + "2 0 1 -case pre -lrtest -bootstrap 20".split(), (".boot",)),
    "dboot": (MD + "2 0 1 -case 2 -lrtest -bootstrap 10".split(), (".boot",)),
    "e1boot": (E1F + "2 0 1 -case 2 -lrtest -bootstrap 20".split(), (".boot",)),
    "irfboot": (MM + "2 0 1 -case 2 -irfboot 40".split(), (".irfboot",)),
    "e1fc": (E1F + "2 0 1 -case 2 -f 8 -estwin 70".split(), (".forecast", ".recursive")),
    "dfcmon": (MON + "2 0 1 -case 2 -f 12".split(), (".forecast",)),
    "fcpre": (MU + "2 0 1 -case pre -f 5".split(), (".forecast",)),
    "fdh211": (MM + "2 1 1 -mean -case 2 -fdhess".split(), ()),
    "xsys": (MM + "3 0 1 -case 2 -xsys 1".split(), ()),
    "e1alpha": (E1F + "2 0 1 -case 2 -alpha e1alpha.txt".split(), ()),
    "wxrow": (E1F + "2 0 1 -case 2 -weakex 3 -fixb2row 1 0".split(), ()),
    "trend": (MU + "2 0 1 -case pre -trend".split(), ()),
    "xarma": (MM + "2 0 1 -case 2 -xpre xrevarma.pre -xlink xnet2.txt".split(), ()),
    "e1ndrank": (E1F + "2 0 1 -case 3 -nodrift 2 -lrtest".split(), ()),
    "dgatepq": (MU + "2 1 0 -case pre -diagar -diagma -diagcov".split(), ()),
    "b40rank": (["mmpre.muskrat.pre", "mmb40.mink.pre"] + "2 0 1 -case 2 -lrtest".split(), ()),
    "rungs3": (MU + "2 0 0 -case 3 -rungs".split(), ()),
}
_VERSION = re.compile(r"(?i)\bdrvec 0\.1[23]\b")


def _norm(text):
    """golden.sh's filter, and the version in the banners of the files it does
    not filter (milestones 1-4's goldens still say 0.12: SOURCE)."""
    return _VERSION.sub("DRVEC <v>", golden_filter(text))


@needs_bundled
@pytest.mark.skipif(not EXACT, reason="byte identity with the oracle is Linux x86_64's")
@pytest.mark.parametrize("label", [pytest.param(k, marks=pytest.mark.slow) if k == "e1boot"
                                   else k for k in ORACLE])
def test_the_bundled_binary_is_the_oracle(label, tmp_path):
    args, extra = ORACLE[label]
    for f in os.listdir(FIXTURES):
        if os.path.isfile(os.path.join(FIXTURES, f)):
            shutil.copy(os.path.join(FIXTURES, f), tmp_path / f)
    cp = subprocess.run([BUNDLED] + args + ["-name", label], cwd=tmp_path,
                        capture_output=True, text=True, timeout=600)
    assert cp.returncode == 0, cp.stderr[-2000:]
    for ext in (".out",) + extra:
        mine = open(tmp_path / (label + ext)).read()
        assert _norm(mine) == _norm(out_text(label + ext)), label + ext

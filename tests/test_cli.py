"""drvec-py, the command line of the .pre route (cli.py), end to end: run as a
process in a scratch directory, its products against the oracle's
(tests/data/oracle, SOURCE). The C's argument syntax, its checks and its exit
codes: 0 done, 1 a malformed line, 2 a refusal."""
import io
import os
import shutil
import subprocess
import sys

import pytest

from conftest import FIXTURES, STRICT
from oracle import golden_filter, out_text

from drvec.cli import run

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
MM = ["mmpre.muskrat.pre", "mmpre.mink.pre"]


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    for f in MM + ["minkmu.pre"]:
        shutil.copy(os.path.join(FIXTURES, f), tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cli(args, cwd):
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "drvec.cli"] + args, cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_end_to_end_forecast(scratch):
    """drvec mmpre.muskrat.pre mmpre.mink.pre 2 0 1 -case 2 -f 8 -name fc201:
    fc201.out and fc201.forecast, as the oracle's."""
    p = _cli(MM + ["2", "0", "1", "-case", "2", "-f", "8", "-name", "fc201"], scratch)
    assert p.returncode == 0, p.stderr
    assert "Done. Output written to fc201.out" in p.stdout
    mine = golden_filter((scratch / "fc201.out").read_text())
    theirs = golden_filter(out_text("fc201.out"))
    fmine = (scratch / "fc201.forecast").read_text().split("\n", 1)[1]
    ftheirs = out_text("fc201.forecast").split("\n", 1)[1]
    if STRICT:
        assert mine == theirs and fmine == ftheirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())
        assert len(fmine.splitlines()) == len(ftheirs.splitlines())


def test_rank_test_with_its_boot_file(scratch):
    """-lrtest -bootstrap N writes <base>.out and <base>.boot (here N = 3, a
    smoke test: the full rankboot is homologated in test_rank.py)."""
    err = io.StringIO()
    rc = run(MM + ["2", "0", "1", "-case", "2", "-lrtest", "-bootstrap", "3", "-name", "rb"],
             out=io.StringIO(), err=err)
    assert rc == 0, err.getvalue()
    assert (scratch / "rb.out").exists()
    assert len((scratch / "rb.boot").read_text().splitlines()) == 2 + 3


def test_default_name_is_the_stems(scratch):
    rc = run(MM + ["2", "0", "1", "-case", "2"], out=io.StringIO(), err=io.StringIO())
    assert rc == 0
    assert (scratch / "mmpre.muskrat_mmpre.mink.out").exists()


def _err(args):
    out, err = io.StringIO(), io.StringIO()
    rc = run(args, out=out, err=err)
    return rc, out.getvalue(), err.getvalue()


def test_usage_and_version():
    rc, out, _ = _err(["-h"])
    assert rc == 0 and "drvec-py s1.pre s2.pre [... sM.pre] p q r [options]" in out
    assert "-fdhess" in out and "-specs*" in out and "-multistart n " in out
    rc, out, _ = _err(["--version"])
    assert rc == 0 and out.startswith("drvec-py ")
    rc, _, err = _err(["a.pre"])
    assert rc == 1 and "drvec-py s1.pre" in err


@pytest.mark.parametrize("args,code,msg", [
    (MM + ["x", "0", "1"], 1, "p (the AR order) must be an integer >= 1, got `x'"),
    (MM + ["2", "-1", "1"], 1, "q (the MA order) must be an integer >= 0"),
    (MM + ["2", "0", "1", "-fdhes"], 1, "did you mean `-fdhess'?"),
    (MM + ["2", "0", "1", "-f", "0"], 1, "-f needs an integer >= 1, got `0'"),
    (MM + ["2", "0", "1", "-case", "4"], 1, "or pre (case 4 is -trend)"),
    (MM + ["2", "0", "1", "-fixb2row", "1"], 1, "-fixb2row needs i v"),
    (MM + ["2", "0", "1", "oops"], 1, "unexpected argument `oops'"),
    (["a.inp", "2", "0", "1"], 2, "the .inp route"),
    (MM + ["2", "0", "1", "-case", "1", "-mean"], 2, "-case 1 has no deterministic term"),
    (MM + ["2", "0", "1", "-bootstrap", "5"], 2, "-bootstrap calibrates -lrtest"),
    (MM + ["2", "0", "1", "-specs"], 2, "-specs is not ported to drvec-py yet"),
    (MM + ["2", "0", "1", "-rungs"], 2, "-rungs is not ported"),
    (MM + ["2", "0", "1", "-lrtest", "-f", "4"], 2, "-f and -estwin act on a single fit"),
    (MM + ["2", "0", "1", "-interv", "x"], 1, "-interv does nothing on the .pre route"),
    (MM + ["2", "0", "1", "-differenced"], 1, "-differenced does not apply"),
])
def test_the_command_line_checks(args, code, msg):
    rc, _out, err = _err(args)
    assert rc == code and msg in err, err


def test_a_refused_combination_exits_2(scratch):
    rc, _out, err = _err(MM + ["2", "0", "1", "-case", "2", "-fixb2row", "1", "0",
                               "-fixb2", "0"])
    assert rc == 2 and "combined with -fixb2" in err
    rc, _out, err = _err(MM + ["2", "0", "1", "-case", "2", "-weakex", "3"])
    assert rc == 2 and "-weakex 3 is outside 1..2" in err

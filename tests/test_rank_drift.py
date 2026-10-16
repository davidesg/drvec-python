"""Milestone 4: the rank test with a mask on case 3's drifts (`-lrtest` with
`-case pre` or `-nodrift i`, with and without `-bootstrap N`), against the
oracle's cprank, cprboot (+ .boot), cprq, e1cprank and e1ndrank
(tests/data/oracle/SOURCE, milestone 4).

Every rank, observed or simulated, is fitted with the mask (the C's global
nodrift[]): the ladder, the profile, the search and each replication. The
asymptotic tables are case 3's, which the C does not tabulate ("no values"):
the decision is the bootstrap's."""
import io
import os

import pytest

from conftest import E1, EXACT, FIXTURES, STRICT, TOL_LL, TOL_LR, fragile
from oracle import boot_rows, golden_filter, out_text

import drvec

MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]

CASES = {
    "cprank": (MU, (2, 0), dict(case="pre")),
    "cprq": (MU, (2, 1), dict(case="pre")),
    "e1cprank": (E1, (2, 0), dict(case="pre")),
    "e1ndrank": (E1, (2, 0), dict(case=3, nodrift=[2])),
}
_RT = {}


def _rt(label):
    if label not in _RT:
        files, (p, q), kw = CASES[label]
        _RT[label] = drvec.rank_test(files, p, q, **kw)
    return _RT[label]


@pytest.fixture(scope="module")
def cprboot(tmp_path_factory):
    path = tmp_path_factory.mktemp("cpb") / "cprboot.boot"
    t = drvec.rank_test(MU, 2, 0, case="pre", bootstrap=20, boot_path=str(path))
    return t, path


def _table_rows(text):
    """{r: (npar, logL)} from the .out's rank table."""
    body = text.split("  r    npar        logL         AIC         BIC")[1]
    rows = {}
    for line in body.splitlines()[2:]:
        f = line.split()
        if len(f) != 5:
            break
        rows[int(f[0])] = (int(f[1]), float(f[2]))
    return rows


@pytest.mark.parametrize("label", list(CASES))
def test_ranks_and_logl(label):
    t = _rt(label)
    rows = _table_rows(out_text(label + ".out"))
    assert sorted(rows) == list(range(t.spec.M))
    # cprq (q = 1) is ill-defined at r = 1: the C's own four starts end at four
    # points (logL -1.50 to 6.48) and the best stops on the step, not the
    # gradient. Off the oracle's platform it stops ~1e-3 further along the
    # ridge (CI run 38912495044, macOS and Windows), as E1 does (TODO.md):
    # there the rank table's shape and the logL to 1e-2, not the point.
    tol = 5.01e-5 + (0 if EXACT else 1e-2 if label == "cprq" else TOL_LL)
    for r, (npar, ll) in rows.items():
        assert t.npar[r] == npar
        assert abs(t.ll[r] - ll) <= tol
    # case 3's tables are not tabulated: no asymptotic verdict, everywhere
    assert t.critical_values(0) is None and t.verdict(0) is None


def test_the_mask_is_the_pre_files():
    """-case pre: the muskrat .pre has no mean, so its drift is held at 0; the
    E1 .pre files all estimate theirs, so -case pre there IS case 3 (no mask)
    and the header does not say "drifts as in the .pre files"."""
    t = _rt("cprank")
    assert t.spec.case == 3 and t.spec.drift == (False, True) and t.drift_how == "pre"
    assert t.npar[0] == 7          # case 3 with both drifts free would be 8
    e = _rt("e1cprank")
    assert e.spec.case == 3 and e.spec.drift is None
    assert "(drifts as" not in e.out_text("e1cprank").split("\n")[6]
    n = _rt("e1ndrank")
    assert n.spec.drift == (True, False, True) and n.drift_how == "nodrift"


@pytest.mark.parametrize("label", list(CASES))
def test_the_out_file(label):
    mine = golden_filter(_rt(label).out_text(label))
    theirs = golden_filter(out_text(label + ".out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


def test_bootstrap_decisions(cprboot):
    """The p-value, the percentiles and the replication statuses: decisions,
    equal everywhere."""
    t, _ = cprboot
    b = t.bootstraps[0]
    line = "  %-4d %4d %10.4f %8.2f %8.2f %8.2f  %7.4f  %4d" % (
        0, 2, t.lr(0), b.cv[0], b.cv[1], b.cv[2], b.pvalue, b.nok)
    assert line in out_text("cprboot.out")
    assert [r.status for r in b.reps] == [s for _h, _r, s, _v in boot_rows("cprboot.boot")]


def test_bootstrap_out_file(cprboot):
    t, _ = cprboot
    mine = golden_filter(t.out_text("cprboot"))
    theirs = golden_filter(out_text("cprboot.out"))
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())


# The replication the C fits from the PREVIOUS one's ladder (C defect,
# bugs/drvec-c-ladder-key): with a drift held, ladder_key() is ~1e12 and
# rounds the data away, so replications 15 and 16 share a key. The port fits
# every replication's own ladder; on rep 16 the two land 8e-11 apart.
STALE = {16}


def test_boot_file_replication_by_replication(cprboot):
    _t, path = cprboot
    c = boot_rows("cprboot.boot")
    mine = open(path).read().splitlines()[2:]
    assert len(mine) == len(c) == 20
    differ = set()
    for line, (h0, rep, st, lr) in zip(mine, c):
        a, b, s, v = line.split(",")
        assert (int(a), int(b), s) == (h0, rep, st)
        assert abs(float(v) - lr) < (1e-9 if EXACT else TOL_LR)
        if line != "%d,%d,%s,%.17g" % (h0, rep, st, lr):
            differ.add(rep)
    if STRICT:
        assert differ == STALE


@fragile
def test_the_c_reuses_the_previous_ladder_on_rep_16():
    """Handing replication 15's ladder to replication 16, as the C's one-slot
    cache does on a key collision, gives the C's LR* bit for bit."""
    from drvec.api import _data, _drift_mask
    from drvec.params import Spec
    from drvec.rank import fit_rank, simulate_h0
    from drvec.rng import Rng
    data = _data(MU)
    case, drift, _ = _drift_mask(data, "pre", None)
    spec = Spec(2, 2, 0, 0, case, drift=drift)
    x0 = fit_rank(spec, data)[0].x
    rng = Rng()
    sims = [simulate_h0(spec, data, x0, rng) for _ in range(16)]
    d15, d16 = data.with_raw(sims[14]), data.with_raw(sims[15])
    fit_rank(spec, d15)
    d16.ladder_cache.update(d15.ladder_cache)
    lr = 2.0 * (fit_rank(spec.but(r=1), d16)[0].ll - fit_rank(spec, d16)[0].ll)
    assert lr == [v for _h, r, _s, v in boot_rows("cprboot.boot") if r == 16][0]


def test_cli_warns_and_runs(tmp_path, monkeypatch):
    from drvec.cli import run
    monkeypatch.chdir(tmp_path)
    err = io.StringIO()
    rc = run(MU + ["2", "0", "0", "-case", "pre", "-lrtest", "-name", "cp"],
             out=io.StringIO(), err=err)
    assert rc == 0, err.getvalue()
    assert "read the bootstrap (-bootstrap N), not the table" in err.getvalue()
    mine = golden_filter((tmp_path / "cp.out").read_text()).replace("cp.out", "cprank.out")
    if STRICT:
        assert mine == golden_filter(out_text("cprank.out"))


@pytest.mark.parametrize("flag,name", [(["-diagar"], "-diagar"), (["-diagcov"], "-diagcov"),
                                       (["-weakex", "1"], "-weakex"),
                                       (["-fixb2", "0"], "-fixb2")])
def test_the_other_rank_test_refusals_stay(flag, name):
    from drvec.cli import run
    err = io.StringIO()
    rc = run(MU + ["2", "0", "1", "-case", "pre", "-lrtest"] + flag,
             out=io.StringIO(), err=err)
    assert rc == 2 and ("-lrtest with %s" % name) in err.getvalue(), err.getvalue()

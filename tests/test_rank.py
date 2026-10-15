"""(d) The rank test with the parametric bootstrap (`2 0 1 -case 2 -lrtest
-bootstrap 20`) against rankboot.out and rankboot.boot, replication by
replication."""

import pytest

from conftest import PRE
from oracle import boot_rows, est_pairs, out_text

import drvec
from drvec.rank import simulate_h0, fit_rank
from drvec.rng import Rng
from drvec.start import init_guess

OUT = out_text("rankboot.out")


@pytest.fixture(scope="module")
def test_result(tmp_path_factory):
    path = tmp_path_factory.mktemp("rb") / "rankboot.boot"
    res = drvec.rank_test(PRE, 2, 0, case=2, bootstrap=20, boot_path=str(path))
    return res, path


def test_ranks_and_lr(test_result):
    t, _ = test_result
    assert t.ll[0] == -12.890025517197202 and t.ll[1] == -8.681458025727963
    assert t.npar == [6, 10]
    rows = ["  %-4d %4d %12.4f %11.4f %11.4f" % (r, t.npar[r], t.ll[r], t.aic(r), t.bic(r))
            for r in range(2)]
    for row in rows:
        assert row in OUT
    assert "  0       2     8.4171    13.75    15.67    20.20   H0 not rejected" in OUT
    assert "%10.4f" % t.lr(0) == "    8.4171"
    assert t.critical_values(0) == (13.75, 15.67, 20.20)
    assert t.verdict(0) == "H0 not rejected"


def test_bootstrap_table(test_result):
    t, _ = test_result
    b = t.bootstraps[0]
    line = "  %-4d %4d %10.4f %8.2f %8.2f %8.2f  %7.4f  %4d" % (
        0, 2, t.lr(0), b.cv[0], b.cv[1], b.cv[2], b.pvalue, b.nok)
    assert line + "   H0 not rejected" in OUT


def test_boot_file_replication_by_replication(test_result):
    t, path = test_result
    c = boot_rows()
    mine = [l for l in open(path).read().splitlines()[2:]]
    assert len(mine) == len(c) == 20
    for line, (h0, rep, st, lr) in zip(mine, c):
        a, b, s, v = line.split(",")
        assert (int(a), int(b), s) == (h0, rep, st)
        # 18 of 20 are byte-identical; reps 6 and 13 differ by ~1e-13 through
        # the Johansen start (scipy vs GSL), which wins there (TODO.md)
        assert abs(float(v) - lr) < 1e-9
    identical = sum(l == ("%d,%d,%s,%.17g" % (h0, rep, st, lr))
                    for l, (h0, rep, st, lr) in zip(mine, c))
    assert identical >= 18


def test_the_first_replication_data_is_the_cs():
    """The H0 model, the generator and simulate_h0: the rung-0 start of every
    one of the first five replications is the C's, bit for bit."""
    data = drvec.read_pre(PRE)
    spec = drvec.Spec(2, 2, 0, 0, 2)
    x0 = est_pairs("rankboot")[5][4]           # the r = 0 winner (the seed)
    pairs = est_pairs("rankboot")
    rng = Rng()
    for b in range(5):
        db = data.with_raw(simulate_h0(spec, data, x0, rng))
        rs = spec.but(ma="diag", diag_ar=True, diag_cov=True)
        assert init_guess(rs, db.at(0)) == pairs[11 + 11 * b][0]


def test_the_observed_est_calls_are_the_cs():
    log = []
    data = drvec.read_pre(PRE)
    for r in (0, 1):
        fit_rank(drvec.Spec(2, 2, 0, r, 2), data, log)
    c = est_pairs("rankboot")[:11]
    assert len(log) == 11
    for k, ((label, _sp, start, er), (cs, cll, _s, _i, cx)) in enumerate(zip(log, c)):
        if label == "johansen":
            assert abs(er.ll - cll) < 1e-12
            continue
        assert (start, er.x, er.ll) == (cs, cx, cll), k

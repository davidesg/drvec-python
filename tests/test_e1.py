"""The three-series case (Lutkepohl's E1, tests/data/fixtures/e1-inp/SOURCE)
against the oracle: the fit `2 1 1 -mean -case 2` and the rank test with its
bootstrap `2 0 1 -case 2 -lrtest -bootstrap 20`.

On this case the four starts of the fit end between 760.5 and 773.1 (all tc-3
stops on a flat surface) and the winner is `nested`: the search is only
reproduced start by start if Johansen's eigenproblem is GSL's (drvec._gsl);
with scipy's the `johansen` start ends at 760.07 instead of 760.54 (it does not
win) and one bootstrap LR* moves by 3.5e-5."""
import os

import pytest

from conftest import E1, EXACT, TOL_EST, TOL_LL, TOL_LR, TOL_SE
from oracle import ORACLE, golden_filter, out_text, param_table

import drvec
from drvec._gsl import backend

GSL = backend() == "gsl"
FIT_OUT = out_text("e1fit.out")
BOOT_OUT = out_text("e1boot.out")


@pytest.fixture(scope="module")
def fit():
    return drvec.fit(E1, 2, 1, 1, case=2)


def test_the_fit(fit):
    assert fit.search.best.label == "nested"
    assert fit.npar == 29
    if EXACT:
        assert "%.10f" % fit.loglik == "773.1127716394"
    assert abs(fit.loglik - 773.1127716394) < (1e-6 if EXACT else TOL_LL)
    table = param_table(FIT_OUT)
    assert len(table) == 29
    for p in fit.params:
        est, se = table[p.label]
        assert abs(p.value - est) <= (5.01e-7 if EXACT else TOL_EST), p.label
        assert abs(p.se - se) <= (5.01e-7 if EXACT else TOL_SE), p.label


def test_the_out_file(fit):
    """e1fit.out, byte for byte after golden.sh's filter (with GSL: the losing
    starts' rows of the search table need GSL's Johansen seed)."""
    mine = golden_filter(fit.out_text("e1fit"))
    if EXACT and GSL:
        assert mine == golden_filter(FIT_OUT)
    else:
        assert len(mine.splitlines()) == len(golden_filter(FIT_OUT).splitlines())


def test_the_search_table(fit):
    block = FIT_OUT.split("\nSearch:")[1].split("The SPREAD")[0]
    mine = fit.search.table().split("\nSearch:")[1].split("The SPREAD")[0]
    if EXACT and GSL:
        assert mine == block
    else:
        # the losing starts are tc-3 stops on a flat surface; the winner holds
        assert mine.splitlines()[0].endswith("The best is `nested'.")


@pytest.fixture(scope="module")
def ranks():
    return drvec.rank_test(E1, 2, 0, case=2)


def test_the_rank_test(ranks):
    t = ranks
    assert t.npar == [14, 20, 24]
    for r, ll in enumerate((721.1988, 750.5247, 753.5354)):
        assert abs(t.ll[r] - ll) < 5.01e-5
    if EXACT:
        for r in range(3):
            row = "  %-4d %4d %12.4f %11.4f %11.4f" % (r, t.npar[r], t.ll[r], t.aic(r), t.bic(r))
            assert row in BOOT_OUT
    assert [t.verdict(r) for r in (0, 1)] == ["reject H0 at 1%", "H0 not rejected"]
    assert "%10.4f" % t.lr(0) == "   58.6518" and "%10.4f" % t.lr(1) == "    6.0213"


@pytest.mark.slow
def test_the_bootstrap_replication_by_replication(tmp_path):
    path = tmp_path / "e1boot.boot"
    t = drvec.rank_test(E1, 2, 0, case=2, bootstrap=20, boot_path=str(path))
    mine = path.read_text().splitlines()[1:]
    c = open(os.path.join(ORACLE, "e1boot.boot")).read().splitlines()[1:]
    assert len(mine) == len(c) == 41
    if EXACT and GSL:
        assert mine == c
    else:
        for a, b in zip(mine[1:], c[1:]):
            a, b = a.split(","), b.split(",")
            assert a[:3] == b[:3]
            assert abs(float(a[3]) - float(b[3])) < (1e-4 if not GSL else TOL_LR)
    if EXACT and GSL:
        assert golden_filter(t.out_text("e1boot")) == golden_filter(BOOT_OUT)
    # the decisions: percentiles to the printed digits, p-values, verdicts
    for b, (pv, cv) in zip(t.bootstraps, ((0.0476, (17.03, 19.03, 20.24)),
                                          (0.4286, (10.15, 10.60, 20.99)))):
        assert "%.4f" % b.pvalue == "%.4f" % pv and b.nok == 20
        assert tuple(float("%.2f" % v) for v in b.cv) == cv

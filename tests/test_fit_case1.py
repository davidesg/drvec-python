"""(c) The full fit of case 1 (`2 1 1 -mean -case 2` on the .pre route) against
tests/data/oracle/proute.out: logL to 1e-6, every estimate and s.e. to 1e-4
(and, as it is, to the printed digits), the ladder, the gate, the profile and
the search -- and the 24 est() calls in the C's order."""
import pytest

from conftest import PRE
from oracle import est_pairs, number_after, out_text, param_table

import drvec
from drvec._gsl import backend

GSL = backend() == "gsl"

OUT = out_text("proute.out")


@pytest.fixture(scope="module")
def fit():
    log = []
    res = drvec.fit(PRE, 2, 1, 1, case=2, log=log)
    return res, log


def test_the_est_calls_are_the_cs(fit):
    _res, log = fit
    c = est_pairs("proute")
    assert len(log) == len(c) == 24
    for k, ((label, spec, start, er), (cstart, cll, _s2, _i, cx)) in enumerate(zip(log, c)):
        if label == "johansen" and not GSL:
            # scipy's eigh where the C calls GSL's gensymmv: the canonical B2 is
            # 2 ulps off, so these starts and fits are near, not equal
            assert max(abs(a - b) for a, b in zip(start, cstart)) < 1e-12, k
            assert abs(er.ll - cll) < 1e-6, k
            continue
        assert start == cstart, (k, label)
        assert er.x == cx, (k, label)
        assert er.ll == cll, (k, label)


def test_logl_and_fit_statistics(fit):
    res, _ = fit
    assert abs(res.loglik - number_after(OUT, "logelf           :")) < 1e-6
    assert res.loglik == 6.4786201604443221                 # the C's, full precision
    assert abs(res.sigma2 - number_after(OUT, "sigma2           :")) < 1e-10
    assert abs(res.aic - number_after(OUT, "AIC              :")) < 1e-10
    assert abs(res.bic - number_after(OUT, "BIC              :")) < 1e-10
    assert res.npar == 14
    assert (res.termcode, res.iters) == (2, 66)
    assert "%.12f" % res.fk == "0.607104141583"


def test_every_estimate_and_se(fit):
    res, _ = fit
    table = param_table(OUT)
    assert len(table) == 14
    assert res.ma_on_boundary            # so the .out prints no s.e. for Theta
    for p in res.params:
        est, se = table[p.label]
        assert abs(p.value - est) < 1e-4, p.label
        assert abs(p.value - est) <= 5.01e-7, p.label        # the printed digits
        if p.kind == "theta":
            assert se is None
        else:
            assert abs(p.se - se) < 1e-4, p.label
            assert abs(p.se - se) <= 5.01e-7, p.label


def test_the_ladder_and_the_gate(fit):
    res, _ = fit
    lad = res.ladder
    printed = [number_after(OUT, f"rung {k}  ({lab})   logL =") for k, lab in
               ((0, "F diag, Theta diag, Sigma diag"), (1, "F diag, Theta diag, Sigma free"),
                (2, "F free, Theta free, Sigma free"))]
    for (_a, _b, _c, ll), pr in zip(lad.rungs, printed):
        assert "%.10f" % ll == "%.10f" % pr
    g = lad.gate
    assert "%.10f" % g.univariate[0] == "-20.0579540523"
    assert "%.10f" % g.univariate[1] == "-14.5698646497"
    assert "%.10f" % g.sum == "-34.6278187020"
    assert "%.10f" % g.joint == "-34.6278187020"
    assert "%.3e" % g.gap == "-7.567e-12"
    assert g.verified


def test_the_profile(fit):
    res, _ = fit
    pr = res.seed_info
    assert pr.ok and pr.mult == 0.05
    assert "%.10f" % pr.ladder.ll == "-1.0494593228"
    assert "%.10f" % pr.ll_start == "-0.6964067884"
    assert "%.10f" % pr.ll == "-0.1533711816"
    assert (pr.est.iters, "%.12f" % pr.est.fk) == (20, "0.982353116537")
    assert ["%.6f" % v for v in pr.x2] == ["6.631570", "0.019488", "0.001811", "-0.313737"]


def test_the_search_table(fit):
    res, _ = fit
    table = res.search.table()
    block = OUT.split("\nSearch:")[1].split("The SPREAD")[0]
    assert table.split("\nSearch:")[1].split("The SPREAD")[0] == block

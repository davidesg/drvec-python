"""-rungs: the ladder below the rank, emitted (run_rungs, drvec.c:9047), against
the oracle (tests/data/oracle, SOURCE):

  rungs    mink-muskrat 2 1 1 -case 2 -rungs
  e1rungs  E1 2 1 1 -case 2 -rungs
  rungs3   mink-muskrat (minkmu) 2 0 0 -case 3 -rungs     (the drifts at r = 0)

Rung 0 is the factorisation gate's; the LRs are ordinary nested chi2."""
import os

import pytest

from conftest import E1, EXACT, FIXTURES, PRE, STRICT, TOL_LL
from oracle import golden_filter, out_text

import drvec

MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]
CASES = {
    "rungs": (PRE, (2, 1, 1), 2),
    "e1rungs": (E1, (2, 1, 1), 2),
    "rungs3": (MU, (2, 0, 0), 3),
}


@pytest.mark.parametrize("label", list(CASES))
def test_the_rungs(label):
    files, (p, q, r), case = CASES[label]
    t = drvec.rungs(files, p, q, r, case=case)
    out = out_text(label + ".out")
    mine = golden_filter(t.out_text(label))
    theirs = golden_filter(out)
    if label == "e1rungs" and not EXACT:
        # E1's rungs are criterion-3 stops on a flat surface (TODO.md, "E1 is
        # ill-defined"), and its gate is BORDERLINE in the C itself: joint - sum
        # = -4.120e-06 against a tolerance of 1e-6, NOT VERIFIED. CI run
        # 38753032614: macOS lands on the other side (VERIFIED) and Windows
        # moves rung 2 by 5.8e-3. Off the oracle's platform what must hold is
        # that the verdict printed is the one its own gap gives, and the rungs.
        line = mine.split("crossing identity (joint - sum) =")[1].split("\n")[0]
        gap, tol = float(line.split()[0]), float(line.split("tolerance")[1].split(")")[0])
        assert t.ladder.gate.verified == (abs(gap) <= tol)
        assert ("NOT VERIFIED" in line) == (not t.ladder.gate.verified)
        assert sum(l.startswith(("  0  F", "  1  Sigma", "  2  F")) for l in mine.splitlines()) == 3
        return
    assert t.ladder.gate.verified == ("VERIFIED" in theirs.split("crossing identity")[1].split("\n")[0].replace("NOT VERIFIED", ""))
    rows = [l for l in theirs.splitlines() if l.startswith(("  0  F", "  1  Sigma", "  2  F"))]
    assert len(rows) == 3
    for k, row in enumerate(rows):
        npar, ll = int(row[30:35]), float(row[35:48])
        assert npar == t.npar[k]
        assert abs(t.ll[k] - ll) < 5.01e-5 + (0 if EXACT else TOL_LL)
    if STRICT:
        assert mine == theirs
    else:
        assert len(mine.splitlines()) == len(theirs.splitlines())

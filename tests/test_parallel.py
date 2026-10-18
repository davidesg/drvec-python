"""A2 and A3: the parallel bootstrap is bit-identical to the sequential one, and
seed= leaves the default stream alone.

The samples are drawn in sequence, in the generator's order, and refitted in
a process pool (drvec.parallel): workers=1 (the sequential path, in this
process) and workers=4 must give the same .out and .boot (and .irfboot) byte
for byte, on every bootstrap case of the suite. That they also equal the
oracle's is each case's own test (test_rank, test_rank_drift, test_m5,
test_e1, test_options), which runs on the default workers."""
import os

import pytest

from conftest import E1, FIXTURES, PRE
from oracle import out_text

import drvec
from drvec.rng import SEED

MU = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "minkmu.pre")]
MD = [os.path.join(FIXTURES, f) for f in ("mmdet.muskrat.pre", "mmdet.mink.pre")]

RANKS = {
    "rankboot": (PRE, dict(case=2, bootstrap=20)),
    "cprboot": (MU, dict(case="pre", bootstrap=20)),
    "dboot": (MD, dict(case=2, bootstrap=10)),
    "e1boot": (E1, dict(case=2, bootstrap=20)),
}
SLOW = {"e1boot"}


def _rank(label, workers, tmp_path, **extra):
    files, kw = RANKS[label]
    calls = []
    t = drvec.rank_test(files, 2, 0, workers=workers, engine="python",
                        progress=lambda b, rep: calls.append((rep.h0, b, rep.status)),
                        **{**kw, **extra})
    return t, t.out_text(label), t.boot_text(), calls


@pytest.mark.parametrize("label", [pytest.param(k, marks=pytest.mark.slow) if k in SLOW
                                   else k for k in RANKS])
def test_rank_workers_1_and_4_are_the_same_bytes(label, tmp_path):
    t1, out1, boot1, calls1 = _rank(label, 1, tmp_path)
    t4, out4, boot4, calls4 = _rank(label, 4, tmp_path)
    assert out1 == out4
    assert boot1 == boot4
    # progress: once per replication, in order, failed ones included
    N = RANKS[label][1]["bootstrap"]
    assert calls1 == calls4 and len(calls1) == N * len(t1.bootstraps)
    assert t1.provenance.workers == 1 and t4.provenance.workers == 4
    assert t4.provenance.engine == "python" and t4.provenance.seed is None


def test_irfboot_workers_1_and_4_are_the_same_bytes():
    MM = [os.path.join(FIXTURES, f) for f in ("mmpre.muskrat.pre", "mmpre.mink.pre")]
    seen = {1: [], 4: []}
    res = {w: drvec.fit(MM, 2, 0, 1, case=2, irfboot=40, workers=w, engine="python",
                        progress=lambda b, ok, w=w: seen[w].append((b, ok)))
           for w in (1, 4)}
    assert res[1].out_text("irfboot") == res[4].out_text("irfboot")
    assert res[1].irfboot_text("irfboot") == res[4].irfboot_text("irfboot")
    assert seen[1] == seen[4] == [(b, True) for b in range(1, 41)]


def test_seed_default_is_the_cs(tmp_path):
    a = _rank("rankboot", 2, tmp_path)
    b = _rank("rankboot", 2, tmp_path, seed=SEED)
    assert a[2] == b[2] and a[1] == b[1]
    assert b[0].provenance.seed == SEED


def test_another_seed_another_stream(tmp_path):
    t, out, boot, _ = _rank("dboot", 2, tmp_path, seed=12345)
    t0, out0, boot0, _ = _rank("dboot", 2, tmp_path)
    assert boot != boot0
    assert [b.lr for b in t.bootstraps] == [b.lr for b in t0.bootstraps]   # observed LR
    assert t.provenance.seed == 12345
    # the C's generator is fixed at its seed: such a call cannot go to the C
    with pytest.raises(drvec.CEngineUnavailable, match="seed"):
        drvec.rank_test(MD, 2, 0, case=2, bootstrap=10, seed=12345, engine="c")


def test_workers_validated():
    with pytest.raises(ValueError, match="workers"):
        drvec.rank_test(PRE, 2, 0, case=2, bootstrap=10, workers=0, engine="python")


def test_default_workers(monkeypatch):
    from drvec.parallel import default_workers, resolve_workers
    monkeypatch.setenv("DRVEC_WORKERS", "3")
    assert default_workers() == 3 and resolve_workers(None, 2) == 2
    monkeypatch.delenv("DRVEC_WORKERS")
    assert default_workers() == max(1, (os.cpu_count() or 1) - 1)
    monkeypatch.setenv("DRVEC_WORKERS", "x")
    with pytest.raises(ValueError):
        default_workers()


def test_the_oracle_still(tmp_path):
    """rankboot at workers=4 against the oracle's .boot, replication by
    replication (its full byte test is test_rank's)."""
    from conftest import EXACT
    from oracle import boot_rows
    t, _out, boot, _ = _rank("rankboot", 4, tmp_path)
    mine = [(r.h0, r.rep, r.status, r.lr if r.status == "ok" else None)
            for b in t.bootstraps for r in b.reps]
    theirs = boot_rows("rankboot.boot")
    if EXACT:
        assert mine == theirs
    assert out_text("rankboot.out")

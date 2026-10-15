"""(e) The bootstrap generator against the C's boot_normal, compiled verbatim
(data/golden/rng_reference.c)."""
import os

from conftest import same
from oracle import GOLDEN

from drvec.rng import Rng


def test_first_draws_and_state():
    lines = open(os.path.join(GOLDEN, "rng_first.txt")).read().split("\n")
    draws = [float(v) for v in lines[:12]]
    state = int(lines[12].split()[1])
    r = Rng()
    # the state is integer arithmetic, exact everywhere; the draws go through
    # libm's log/cos/sin, which Windows' rounds differently in the last ulp
    for a, b in zip([r.normal() for _ in range(12)], draws):
        assert same(a, b, 1e-14)
    assert r.state == state

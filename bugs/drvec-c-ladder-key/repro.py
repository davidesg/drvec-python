"""C defect (drvec-v0.12): the r = 0 ladder's cache key loses the data when a
drift is held, so a bootstrap replication can reuse the previous one's ladder.

fit_r0_ladder (drvec.c:5345) caches its optimum under ladder_key()
(drvec.c:5310-5323), a double: p*1e6 + q*1e4 + case*1e3 + ... + the drift mask
as 1e12 * 2^(i-1) for each held drift + sum_t,i rawmat[t][i] * 1e-7 (t%97 + i).
With a held drift (-nodrift i, or -case pre on a .pre without a mean) the key
is ~1e12, where a double's spacing is 1.2e-4: the data's terms (~1e-6 each)
are mostly rounded away, and two different data sets get the same key. In
`-lrtest -bootstrap` the replications differ only in their data, so a
replication whose key equals the previous one's starts its rung-2 search from
the PREVIOUS replication's ladder instead of its own (rung 0 and rung 1 are not
fitted, the gate not checked). Here (`mmpre.muskrat.pre minkmu.pre 2 0 1 -case
pre -lrtest -bootstrap 20`, the muskrat .pre has no mean) replications 15 and 16
share the key 1000002003000.062, and the C's LR* of replication 16 is
5.4210743829648322 where fitting its own ladder gives 5.4210743828848535.
The size is small here (8e-11, both land on the same optimum) because the
stale seed is close; nothing bounds it in general. Without a held drift (cases
1, 2, 3 with every drift free) the key is ~1e6 and keeps the data (spacing
2.3e-10): the goldens rankboot and e1boot are not affected. Expected: the key
(or the cache) compares the data itself.

This repro runs the C, then the port (../../src, which fits every
replication's own ladder) on replication 16, and the port with the C's stale
cache emulated (rep 15's ladder handed to rep 16), which gives the C's value
bit for bit.

    python3 repro.py [path/to/drvec]     # exits 1 while the defect is open
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
exe = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.local/share/atsw/drvec-ref/bin/drvec")
FILES = ("mmpre.muskrat.pre", "minkmu.pre")

with tempfile.TemporaryDirectory() as d:
    for f in FILES:
        shutil.copy(os.path.join(HERE, f), d)
    subprocess.run([exe, *FILES, "2", "0", "1", "-case", "pre", "-lrtest", "-bootstrap",
                    "20", "-name", "lk"], cwd=d, capture_output=True, check=True)
    c16 = float([l for l in open(os.path.join(d, "lk.boot")) if l.startswith("0,16,")][0]
                .split(",")[3])

from drvec.api import _data, _drift_mask          # noqa: E402
from drvec.params import Spec                     # noqa: E402
from drvec.rank import fit_rank, simulate_h0      # noqa: E402
from drvec.rng import Rng                         # noqa: E402

data = _data([os.path.join(HERE, f) for f in FILES])
case, drift, _how = _drift_mask(data, "pre", None)
spec = Spec(data.M, 2, 0, 0, case, "free", drift=drift)


def c_key(raw):
    """ladder_key(), in the C's order of additions (p = 2, q = 0, case 3)."""
    k = 2 * 1e6 + 0 * 1e4 + 3 * 1e3 + 0 + 0 + 0 + raw.shape[0] * 1e-3 + 0 * 0.5
    for i in range(data.M):
        if not spec.mu_free(i):
            k += 1e12 * float(1 << i)
    for t in range(1, raw.shape[0] + 1):
        for i in range(1, data.M + 1):
            k += float(raw[t - 1, i - 1]) * (1e-7 * (t % 97 + i))
    return k


x0 = fit_rank(spec, data)[0].x
rng = Rng()
sims = [simulate_h0(spec, data, x0, rng) for _ in range(16)]
k15, k16 = c_key(sims[14]), c_key(sims[15])
d15, d16 = data.with_raw(sims[14]), data.with_raw(sims[15])


def lr(db):
    return 2.0 * (fit_rank(spec.but(r=1), db)[0].ll - fit_rank(spec, db)[0].ll)


own = lr(d16)
fit_rank(spec, d15)                    # rep 15's ladder, as the C's cache holds it
d16s = data.with_raw(sims[15])
d16s.ladder_cache.update(d15.ladder_cache)
stale = lr(d16s)
print("C's key, rep 15: %r  rep 16: %r  (equal: %s)" % (k15, k16, k15 == k16))
print("rep 16 LR*: C %.17g | own ladder %.17g | rep 15's ladder %.17g" % (c16, own, stale))
if c16 != own:
    print("OPEN: replication 16 was fitted from replication 15's ladder"
          + (" (reproduced bit for bit)" if c16 == stale else ""))
    sys.exit(1)
print("fixed")
sys.exit(0)

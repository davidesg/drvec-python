"""The parallel bootstrap: the samples drawn in sequence, the refits in a pool.

Why it is bit-identical to the sequential path. The bootstrap's generator
(rng.Rng, the C's boot_normal) is consumed ONLY by simulate_h0: a refit draws
no random number -- fit_rank is init_guess, the ladder or the profile, and
fit_search, whose only generator is -multistart's local one, and a
replication is fitted with one start (njitter = 1), as the C's fit_ll is. Nor
does a refit change anything the next sample is drawn from: the sample comes
from the OBSERVED fit and the observed data, and each replication is fitted
on a fresh Data (Data.with_raw), with its own rank data and ladder cache. So
drawing the B samples first, in the generator's order, and fitting them
anywhere, in any order, gives the same numbers; the results are put back in
replication order before anything reads them.

The pool is a ProcessPoolExecutor in the platform's default start method (fork
on Linux before Python 3.14, spawn on macOS and Windows): the base data are
sent once, to each worker's initializer, and every task carries only its
sample. Under spawn the caller's script must be importable without running
the analysis (the usual `if __name__ == "__main__":` guard).
"""
from __future__ import annotations

import os

_BASE = None


def default_workers():
    """DRVEC_WORKERS when set (an integer >= 1), else the cores less one
    (at least 1): one core is left to the caller and the system."""
    env = os.environ.get("DRVEC_WORKERS", "").strip()
    if env:
        try:
            v = int(env)
        except ValueError:
            raise ValueError(f"DRVEC_WORKERS must be an integer >= 1, got `{env}'") from None
        if v < 1:
            raise ValueError(f"DRVEC_WORKERS must be an integer >= 1, got `{env}'")
        return v
    n = os.cpu_count() or 1
    return max(1, n - 1)


def resolve_workers(workers, ntasks):
    """workers=None -> default_workers(); never more than the tasks."""
    if workers is None:
        workers = default_workers()
    workers = int(workers)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return max(1, min(workers, ntasks))


def _init(base):
    global _BASE
    _BASE = base


def _call(job):
    fn, args = job
    spec, sim = args[0], args[1]
    return fn(spec, _BASE, sim, *args[2:])


def map_refits(fn, base, tasks, workers):
    """Yield fn(spec, base, sim, *rest) for each task (spec, sim, *rest), or
    None for a task that is None (a sample that could not be drawn), IN THE
    TASKS' ORDER, computed in a pool of `workers` processes. fn must be a
    module-level function (it is pickled by name)."""
    from concurrent.futures import ProcessPoolExecutor
    live = [(fn, t) for t in tasks if t is not None]
    if not live:
        for _t in tasks:
            yield None
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(live)), initializer=_init,
                             initargs=(_strip(base),)) as ex:
        it = ex.map(_call, live, chunksize=1)
        for t in tasks:
            yield None if t is None else next(it)


def _strip(data):
    """The base data for the workers: a fresh copy (Data.with_raw keeps what a
    replication needs and drops the caches)."""
    return data.with_raw(data.raw)

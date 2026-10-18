#!/usr/bin/env python3
"""Time the port's rank bootstrap against the C: the input of option (b), a cast
in C.

    python tools/time_bootstrap.py [B] [--case mm|e1] [--oracle PATH-TO-drvec]
    python tools/time_bootstrap.py [B] [--case mm|e1] --engines [--workers 1,2,4]

Cases: `mm`, muskrat and mink (M = 2, n = 61), and `e1`, Lutkepohl's E1
(M = 3, n = 91; tests/data/fixtures/e1-inp/SOURCE). The run is `2 0 1 -case 2
-lrtest -bootstrap B`, plus the plain fit `2 1 1 -mean -case 2` for e1.

Reports, per likelihood evaluation, the cast in Python and drvarma's compiled
`elf` (alone, and the extra cost of its Python wrapper); over the whole run, the
number of evaluations and the split of the wall time into the cast, the
compiled likelihood with its wrapper, and the rest (drvarma's optimiser, which
is pure Python, the starts, the simulation); the C's wall time for the same run
when the oracle binary is found (default ~/.local/share/atsw/drvec-ref-0.13/bin/drvec),
and the extrapolation to the article's bootstraps.

--engines measures the engines instead (PLAN-vec A2 and B): the whole
`rank_test(..., bootstrap=B)` call on the Python engine with each number of
workers (default 1, 2, 3, 4; the numbers are the same, so the .boot is checked
against workers=1's), and on drvec C through drvec.c_engine (the oracle given,
or DRVEC_BIN), with the speed-up of each against workers=1.
"""
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import drvec                                   # noqa: E402
import drvec.cast as C                         # noqa: E402
import drvec.fit                               # noqa: E402,F401
import drvec.loglik as L                       # noqa: E402
import drvec.rank                              # noqa: E402,F401
from drvec.start import init_guess             # noqa: E402

FIT, RK = sys.modules["drvec.fit"], sys.modules["drvec.rank"]   # (drvec.fit is the function)

FIX = os.path.join(ROOT, "tests", "data", "fixtures")
CASES = {
    "mm": [os.path.join(FIX, "mmpre.muskrat.pre"), os.path.join(FIX, "mmpre.mink.pre")],
    "e1": [os.path.join(FIX, f"e1{n}.pre") for n in ("invest", "income", "cons")],
}
ARTICLE_C_MIN = {"B = 999": 13.0, "B = 499 (a)": 89.0, "B = 499 (b)": 23.0}


def per_call(f, n=3000):
    f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


class Meter:
    """Wraps the cast and the compiled likelihood where the port calls them, and
    adds up their time and calls."""

    def __init__(self):
        self.t_cast = self.t_elf = 0.0
        self.n_elf = 0
        self.saved = []

    def install(self):
        real_shootx, real_elf = C.shootx, L.elf_c

        def shootx(*a, **k):
            t = time.perf_counter()
            try:
                return real_shootx(*a, **k)
            finally:
                self.t_cast += time.perf_counter() - t

        def elf_c(*a, **k):
            t = time.perf_counter()
            try:
                return real_elf(*a, **k)
            finally:
                self.t_elf += time.perf_counter() - t
                self.n_elf += 1
        for mod in (L, FIT, RK):
            self.saved += [(mod, "shootx", mod.shootx), (mod, "elf_c", mod.elf_c)]
            mod.shootx, mod.elf_c = shootx, elf_c

    def uninstall(self):
        for mod, name, f in reversed(self.saved):
            setattr(mod, name, f)


def run_c(oracle, files, args):
    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        subprocess.run([oracle, *files, *args, "-name", os.path.join(d, "x")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return time.perf_counter() - t


def measured(label, fn, oracle, files, c_args):
    m = Meter()
    m.install()
    t = time.perf_counter()
    fn()
    wall = time.perf_counter() - t
    m.uninstall()
    rest = wall - m.t_cast - m.t_elf
    print(f"{label}: {wall:7.2f} s, {m.n_elf} likelihood evaluations "
          f"({wall / max(m.n_elf, 1) * 1e6:.0f} us each, everything included)")
    print(f"    cast (Python) {m.t_cast:7.2f} s = {100 * m.t_cast / wall:4.1f}% | "
          f"compiled elf + wrapper {m.t_elf:7.2f} s = {100 * m.t_elf / wall:4.1f}% | "
          f"the rest (optimiser, starts, simulation) {rest:7.2f} s = {100 * rest / wall:4.1f}%")
    c_wall = None
    if os.path.exists(oracle):
        c_wall = run_c(oracle, files, c_args)
        print(f"    the C, same run: {c_wall:7.2f} s -> ratio {wall / c_wall:.1f}x; "
              f"with a cast that cost nothing it would be {(wall - m.t_cast) / c_wall:.1f}x")
    return wall, c_wall


def main():
    args = sys.argv[1:]
    B = int(args[0]) if args and args[0].isdigit() else 20
    case = args[args.index("--case") + 1] if "--case" in args else "mm"
    oracle = os.path.expanduser("~/.local/share/atsw/drvec-ref-0.13/bin/drvec")
    if "--oracle" in args:
        oracle = args[args.index("--oracle") + 1]
    files = CASES[case]
    if "--engines" in args:
        ws = [1, 2, 3, 4]
        if "--workers" in args:
            ws = [int(v) for v in args[args.index("--workers") + 1].split(",")]
        return engines(files, B, ws, oracle)
    data = drvec.read_pre(files)
    M = data.M
    from drvarma._engine import _load_c, elf_c

    print(f"case {case}: M = {M}, n = {data.nobs_raw - 1}")
    for (p, q, r) in ((2, 0, 1), (2, 1, 1), (2, 0, 0)):
        spec = drvec.Spec(M, p, q, r, 2)
        rd = data.at(r)
        x = init_guess(spec, rd)
        t_cast = per_call(lambda: C.shootx(spec, rd, x))
        mu, phi, th, qq, w, _ = C.shootx(spec, rd, x)
        t_wrap = per_call(lambda: elf_c(M, rd.n, p, q, mu, phi, th, qq, w, 1.0, 1e-3, False))
        ffi, lib = _load_c()
        f1, f2, lg = ffi.new("double *"), ffi.new("double *"), ffi.new("double *")
        bufs = [ffi.from_buffer("double[]", a.ravel()) if a.size else ffi.NULL
                for a in (mu, phi, th, qq, w)]
        t_raw = per_call(lambda: lib.drvarma_elf(M, rd.n, p, q, *bufs, 1.0, 1e-3, 0,
                                                 ffi.NULL, f1, f2, lg))
        print(f"  one evaluation, p={p} q={q} r={r}: cast {t_cast * 1e6:6.1f} us | "
              f"compiled elf {t_raw * 1e6:6.1f} us (+{(t_wrap - t_raw) * 1e6:4.1f} us "
              f"of wrapper) | the cast is {100 * t_cast / (t_cast + t_wrap):.0f}% of it")

    if case == "e1":
        measured("  plain fit 2 1 1 -mean -case 2", lambda: drvec.fit(files, 2, 1, 1, case=2),
                 oracle, files, ["2", "1", "1", "-mean", "-case", "2"])
    wall, c_wall = measured(
        f"  rank test, B = {B}", lambda: drvec.rank_test(files, 2, 0, case=2, bootstrap=B,
                                                         engine="python", workers=1),
        oracle, files, ["2", "0", "1", "-case", "2", "-lrtest", "-bootstrap", str(B)])
    if c_wall:
        ratio = wall / c_wall
        ext = ", ".join(f"{k}: {v:.0f} min -> ~{v * ratio / 60:.1f} h"
                        for k, v in ARTICLE_C_MIN.items())
        print(f"  extrapolated at this ratio: {ext}")


def engines(files, B, ws, oracle):
    print(f"rank_test(2, 0, case=2, bootstrap={B}) on {len(files)} series, "
          f"{os.cpu_count()} cores")
    base = None
    ref = None
    for w in ws:
        t = time.perf_counter()
        rt = drvec.rank_test(files, 2, 0, case=2, bootstrap=B, engine="python", workers=w)
        wall = time.perf_counter() - t
        boot = rt.boot_text()
        if ref is None:
            ref, base = boot, wall
        print(f"  python, workers={w}: {wall:8.2f} s   speed-up {base / wall:5.2f}x"
              f"   .boot {'identical' if boot == ref else 'DIFFERENT'}")
    if os.path.exists(oracle) and not os.environ.get("DRVEC_BIN"):
        os.environ["DRVEC_BIN"] = oracle
    try:
        t = time.perf_counter()
        rt = drvec.rank_test(files, 2, 0, case=2, bootstrap=B, engine="c")
        wall = time.perf_counter() - t
        boot = rt.boot_text()
        print(f"  C ({rt.provenance.path}): {wall:8.2f} s   speed-up {base / wall:5.2f}x"
              f"   .boot {'identical' if boot == ref else 'DIFFERENT'}"
              f"   (the C alone {rt.provenance.extra['elapsed']:.2f} s)")
    except drvec.CEngineUnavailable as e:
        print(f"  C: not run ({e})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Time the port's rank bootstrap against the C: the input of option (b), a cast
in C.

    python tools/time_bootstrap.py [B] [--oracle PATH-TO-drvec]

Reports the cost of one likelihood evaluation (the cast plus drvarma's compiled
elf), the number of evaluations and the wall time of `2 0 1 -case 2 -lrtest
-bootstrap B` on the milestone's fixtures, the C's wall time for the same run
when the oracle binary is given (default: ~/.local/share/atsw/drvec-ref/bin/drvec),
and the extrapolation to the article's bootstraps.
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
import drvec.loglik as L                       # noqa: E402

FIX = os.path.join(ROOT, "tests", "data", "fixtures")
PRE = [os.path.join(FIX, "mmpre.muskrat.pre"), os.path.join(FIX, "mmpre.mink.pre")]
ARTICLE_C_MIN = {"B = 999": 13.0, "B = 499 (a)": 89.0, "B = 499 (b)": 23.0}


def main():
    args = sys.argv[1:]
    B = int(args[0]) if args and args[0].isdigit() else 20
    oracle = os.path.expanduser("~/.local/share/atsw/drvec-ref/bin/drvec")
    if "--oracle" in args:
        oracle = args[args.index("--oracle") + 1]

    # one evaluation, alone
    data = drvec.read_pre(PRE)
    spec = drvec.Spec(2, 2, 0, 1, 2)
    x = [7.3225275397972043, 0.34000361149816843, 0.24773414781346859, 0.0782379161934907,
         0.19445070796192246, -0.63837779995823107, 0.31984700500018431, 1.1455668255369984,
         0.29045592480521099, -0.26442913821741265]
    rd = data.at(1)
    n_rep = 3000
    t = time.perf_counter()
    for _ in range(n_rep):
        L.elf(spec, rd, x)
    per_eval = (time.perf_counter() - t) / n_rep
    from drvarma._engine import elf_c
    mu, phi, theta, qq, w, _ = L.shootx(spec, rd, x)
    t = time.perf_counter()
    for _ in range(n_rep):
        elf_c(2, rd.n, 2, 0, mu, phi, theta, qq, w, 1.0, 1e-3, False)
    per_elf = (time.perf_counter() - t) / n_rep

    # the whole run, counting evaluations
    count = [0]
    real = L.elf_c

    def counted(*a, **k):
        count[0] += 1
        return real(*a, **k)
    L.elf_c = counted
    t = time.perf_counter()
    drvec.rank_test(PRE, 2, 0, case=2, bootstrap=B)
    wall = time.perf_counter() - t
    L.elf_c = real

    print(f"one likelihood evaluation : {per_eval * 1e6:8.1f} us  "
          f"(drvarma's compiled elf alone {per_elf * 1e6:.1f} us; the rest is the cast in Python)")
    print(f"rank test, B = {B:<4d}      : {wall:8.2f} s, {count[0]} evaluations "
          f"({wall / count[0] * 1e6:.0f} us each, everything included)")

    c_wall = None
    if os.path.exists(oracle):
        with tempfile.TemporaryDirectory() as d:
            t = time.perf_counter()
            subprocess.run([oracle, *PRE, "2", "0", "1", "-case", "2", "-lrtest",
                            "-bootstrap", str(B), "-name", os.path.join(d, "rb")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            c_wall = time.perf_counter() - t
        print(f"the C, same run           : {c_wall:8.2f} s   -> ratio {wall / c_wall:.1f}x")
    else:
        print("the C, same run           : (oracle binary not found; 0.54 s measured on "
              "2026-10-15)")
        c_wall = 0.54
    ratio = wall / c_wall
    ext = ", ".join(f"{k}: {v:.0f} min -> ~{v * ratio / 60:.1f} h" for k, v in ARTICLE_C_MIN.items())
    print(f"extrapolated at the same ratio (an assumption: on larger systems the "
          f"compiled elf weighs more, the Python cast grows like M^3): {ext}")


if __name__ == "__main__":
    main()

# Golden numbers captured from the oracle

All of them come from the C program at atsw-gui tag `drvec-v0.12`
(`engines/drvec`), built on Linux with gcc and the system GSL, run on the two
fixtures exactly as `tests/data/oracle/SOURCE` says.

## `proute.est.txt`, `rankboot.est.txt` — every est() call, at full precision

A copy of the tag (`git archive drvec-v0.12 engines/drvec lib`) with the two
hooks of `est-dump.patch` added to `src/drvmlest.c:est`: when the environment
variable `DRVEC_DUMPX` names a file, each call appends

    START npar=<k> x= <the starting vector, %.17g>
    EST nobjc=<cumulative objcfunc calls> npar=<k> logelf=<logL> sigma2=<..> ifault=<..> x= <the final vector>

(the EST line is written after the final likelihood evaluation, so its ifault is
that evaluation's). With the hooks the build's `proute.out`, `rankboot.out` and
`rankboot.boot` are byte-identical to the oracle's in `../oracle` (apart from the
Output File path). The vectors are in drvec's internal layout (docs/PORT.md §2).
`specs.json` says which model each pair belongs to.

The calls appear in the C's order: 24 for `proute` (the r = 0 ladder, the
profile, the r = 1 search with its nested chain) and 11 + 20 x 11 for `rankboot`
(the observed data, then each replication).

## `rng_first.txt`, `rng_reference.c` — the bootstrap generator

`rng_reference.c` is `boot_normal` copied verbatim from drvec.c:2203-2223 with a
`main` that prints its first 12 draws (`%.17g`) and the state after them;
`rng_first.txt` is its output (gcc -O2, x86-64 Linux, 64-bit `unsigned long`).

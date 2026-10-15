# drvec-python — TODO

## Milestone 1 — the cast, one fit, one bootstrap timed

- [x] Evaluate the exact log-likelihood at the oracle's estimates (`.pre`
      route, muskrat and mink, p = 2, q = 1, r = 1, `-mean -case 2`): equal to the
      C's at every est() optimum (tests/test_loglik.py).
- [x] The full fit of that case, homologated against `tests/data/oracle/proute.out`:
      the 24 est() calls in the C's order, logL, every estimate and s.e. to the
      printed digits, the ladder, the gate, the profile, the search table
      (tests/test_fit_case1.py, tests/test_trajectory.py).
- [x] The rank test with the parametric bootstrap (`-lrtest -bootstrap 20`),
      replication by replication against `rankboot.boot` (tests/test_rank.py), and
      its time against the C's (`tools/time_bootstrap.py`): 5.7 s against 0.48 s,
      ~12x; at that ratio the article's bootstraps (13, 89, 23 min in C) would take
      ~2.6, ~17.5 and ~4.5 h. Decision input for option (b), a cast in C.

## Milestone 2 — the measurement, the .out, the remaining PRE_CASES (2026-10-15)

- [x] A three-series case (Lutkepohl's E1 through fue) and the option-(b)
      measurement below (tools/time_bootstrap.py --case e1, tests/test_e1.py).
- [x] Johansen's seed through GSL's gensymmv (ctypes) when libgsl is present:
      with scipy E1's `johansen` start and one LR* moved beyond tolerance.
- [x] The tests follow the suite's platform convention: exact on Linux x86_64,
      a tolerance tier elsewhere (tests/conftest.py).
- [x] The .out writer (report.py, diagnose.py): proute.out, rankboot.out,
      e1fit.out, e1boot.out and the .boot files byte-identical after golden.sh's
      filter (docs/PORT.md §8).
- [x] Every PRE_CASE of golden.sh, .out byte-identical: fixb2row, nodrift,
      casepre, trend, xsys, xlink, irfboot (+ irfboot.irfboot), rankboot
      (tests/test_options.py, tests/test_rank.py).
- [x] The gaps the C declares, refused with its message: -xsys with q > 0, the rank
      test with -xsys, -xlink or -trend, -xlink with -f (and -irfboot with -trend,
      -xlink with case 1).

## Next

- [ ] The CLI (`drvec f1.pre f2.pre p q r [options]`), writing <base>.out/.boot/.irfboot.
- [ ] -f (forecast) and -estwin, -fdhess, -alpha/-weakex (the alpha = A psi
      machinery is in place for -xsys), -fixb2, -multistart, -specs, -rungs.
- [ ] -xlink: the backcast of an input WITH an ARMA model (xl_backcast with
      expand_ar_factors/expand_ma_factors; refused today), inputs with
      deterministic variables.
- [ ] The goldens on the `.inp` route (base2, case1, ... in golden.sh's CASES) are
      out of the port's scope (the .pre route only).

## Option (b): the measurement (milestone 2, 2026-10-15)

`python tools/time_bootstrap.py 20 --case e1` (and `--case mm`), on Lutkepohl's E1
(M = 3, n = 91, tests/data/fixtures/e1-inp/SOURCE) and on muskrat-mink (M = 2):

| run | Python | C | ratio | cast | compiled elf + wrapper | the rest |
|---|---|---|---|---|---|---|
| E1 fit `2 1 1 -mean -case 2` (72 526 evals) | 13.7 s | 2.25 s | 6.1x | 50% | 24% | 26% |
| E1 `-lrtest -bootstrap 20` (681 256 evals) | 112 s | 12.2 s | 9.2x | 52% | 19% | 28% |
| mm `-lrtest -bootstrap 20` (51 437 evals) | 5.8 s | 0.50 s | 11.6x | 52% | 21% | 27% |

One evaluation on E1 (p = 2, r = 1): the cast 82-103 us in Python, drvarma's compiled
elf 17-22 us plus 7-9 us of its Python wrapper: the cast is ~75% of an evaluation.
"The rest" is mostly drvarma's optimiser (`_qnewt`: bfgsfac, jacrot, cdgrad's loop),
which is pure Python, plus the starts and the simulation.

What this says:
- The ratio does NOT grow with M from 2 to 3 (11.6x -> 9.2x): the compiled elf grows
  faster with M and n than the cast does.
- A compiled cast that cost nothing would leave the bootstrap at 4.4x the C (E1) and
  5.6x (mm), because a quarter of the time is drvarma's Python optimiser. Option (b)
  alone halves the time; to get near the C it would need the optimiser compiled too
  (that is drvarma's, not drvec's).
- The article's bootstraps at the E1 ratio: 13 min -> ~2.0 h, 89 min -> ~13.6 h,
  23 min -> ~3.5 h (at the mm ratio ~2.5, ~17, ~4.5 h).
- Tried and dropped: an exactness-preserving faster cast in pure Python (the
  structural zeros of Cbar, Cinv and Hbar skipped, which is bit-identical while
  every value is finite; and a numpy version with matrix_multiply's order of
  additions, also bit-identical). Both were bit-identical to the transcription on
  410 random layouts, and neither moved the end-to-end time (the list version is
  ~75 us, the structural one ~72 us, the numpy one ~150 us: numpy's per-call cost
  dominates 3x3 products). The pure-Python floor is close; the transcription stays.
- Cheap, not taken: calling `drvarma_elf` directly instead of `elf_c` saves its
  7-9 us of wrapper (~5% of an evaluation), at the price of drvarma's private API.

## Open risks (docs/PORT.md)

- Johansen's seed (decision D2, revised in milestone 2): GSL's gensymmv through
  ctypes when libgsl is found (`drvec._gsl`), scipy's `eigh` otherwise. The risk the
  milestone-1 note foresaw was seen on E1: with scipy the `johansen` start ends at
  760.07 instead of 760.54 and one bootstrap LR* moves by 3.5e-5 (beyond 1e-6 in
  logL). With GSL everything is byte-identical. Without libgsl (Windows, macOS
  without Homebrew's gsl) a near-tie can still flip a winner.
- Off Linux x86_64 the compiled likelihood rounds differently and the tests run on
  the tolerance tier of tests/conftest.py (first CI run 38736500814).
- The case-1 optimum is on the MA invertibility wall (BUG-49) and depends on the
  bit-exact trajectory of one start: any change to the cast's arithmetic (numpy
  products, a different Cholesky) must re-run tests/test_fit_case1.py.
- The tolerance tier off Linux x86_64 was measured on milestone 1's tests only (CI run
  38736500814); the milestone-2 tests (E1, the options, the .out files: line counts
  off-platform) use the same tier unmeasured. E1's fit has four tc-3 stops within
  13 logL units: off the oracle's platform its winner (`nested`) is asserted and
  may be the first decision to move.
- The .out's Wald and Granger blocks, Hosking's C0 inverse and Jarque-Bera's
  S^(-1/2) are numpy where the C uses svdcp/GSL: byte-identical on every case here,
  but a printed fourth decimal could round the other way elsewhere (PORT.md §8).
- `nrerror` (the line search's ROUNDOFF) is an exception, not an exit (D3).
- The `.pre` reader: deterministic variables and D > 0 are refused; alignment is
  the C's (date intersection), not drvarma's ladder rule.
- C defect candidate, not filed: the bootstrap generator's `unsigned long`
  (drvec.c:2203) is 32-bit on Windows, so the C's bootstrap stream (and its
  `.boot` goldens) are platform-dependent. The port computes the 64-bit stream.
- The suite was run against drvarma 0.2.0 and fue 0.1.17.dev0 installed as
  EDITABLE dev trees; re-run against the released wheels.

## The port's first version (scope decided 2026-10-11)

Cases 1-4, the `.pre` route with `-case pre`/`-nodrift`, the rank test,
the impulse responses and variance decomposition with `-irfboot`, `-f`,
`-fixb2row`, `-xpre`/`-xlink` and `-xsys`, `-alpha`, `-weakex`, `-fixb2`,
`-fdhess`. The gaps the C declares are ported as refusals, with the C's
message.

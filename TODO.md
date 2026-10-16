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

## Milestone 3 — the rest of the first version's scope (2026-10-15)

- [x] `-fdhess`: the finite-difference Hessian s.e. (fdh201, e1fdh; fdh211 on the
      MA wall keeps the BFGS ones and says why). Path-independent but NOT
      bit-stable: an ulp in x moves an s.e. by up to 4.5e-3 relative; off the
      oracle's platform the relative tier TOL_FDSE_REL = 2e-2 (tests/conftest.py).
- [x] `-alpha FILE`, `-weakex i`, `-fixb2 [v]` with their LRs, alone and with
      `-fixb2row` (the joint LR): weakex1/2, e1alpha, fixb2v, fixb2s, wxfix, wxrow.
- [x] `-f H` and `-estwin E`: the .out's sections, `<base>.forecast` and
      `<base>.recursive` (fc201, fc211, fcpre, ewin, e1fc).
- [x] The CLI `drvec-py` ([project.scripts]), end to end against fc201.
- [x] `-multistart n` (ms201, ms211, msrank) and `-rungs` (rungs, e1rungs, rungs3).
- Every case above is byte-identical to the oracle on Linux x86_64 with GSL
  (tests/data/oracle/SOURCE, milestone 3).

## Milestone 4 — what the article's scripts use (2026-10-16)

- [x] `-lrtest` with `-case pre` and with `-nodrift i` (it fell out: the same
      mask), with and without `-bootstrap N`: every rank and every replication
      fitted with the mask, the C's warning on stderr that case 3's tables do
      not apply to the mixed case. cprank, cprboot (+ .boot), cprq (q = 1),
      e1cprank (the E1 .pre files all free their mean: -case pre is case 3
      there), e1ndrank (three series, a drift held). Every .out
      byte-identical; cprboot.boot in 19 of 20 replications, the 20th is the C
      defect below (tests/test_rank_drift.py). Bootstrap time: 7.5 s against
      the C's 0.51 s (~15x).
- [x] The entry gate of a plain fit at the diagonal rung (r = 0 with -diagar
      -diagma -diagcov), with its optimality certificate (the logL at the start
      brought in): dgate, dgate2, dgateq, dgatepq, e1gate byte-identical
      (tests/test_gate.py). On E1 the C's own gate is NOT VERIFIED
      (-4.120e-06, the rung-0 fit stops on criterion 3), as in e1rungs.

## Next

- [ ] `-specs`: it fits -warma and -mawarma, neither of which is ported (the
      -warma branch of vec_shootx and init_guess, the -mawarma MA class).
- [ ] The rank test with `-alpha`/`-weakex`/`-fixb2`, `-diagar`/`-diagcov` (the C
      has them; drvec-py refuses them, named). `-rungs` with `-case pre`/`-nodrift`
      is still refused too (not tried).
- [ ] `-m 2` (the xi sequence not truncated), `-matest`, `-artest`, `-eval`,
      `-seed*`, `-noladder`, `-seedjoh`, `-seedb2`, `-rankadm`, `-writeinp`,
      `-writeres`, `-warma`/`-mawarma`: refused by drvec-py, named.
- [ ] `-irfboot` with `-fixb2` (no value): the C re-reads B2_fixed from each
      replication's static OLS (init_guess's side effect); not checked.
- [ ] -xlink: the backcast of an input WITH an ARMA model (xl_backcast with
      expand_ar_factors/expand_ma_factors; refused today), inputs with
      deterministic variables.
- [ ] The goldens on the `.inp` route (base2, case1, ... in golden.sh's CASES) are
      out of the port's scope (the .pre route only).
- [ ] Run the suite once off Linux x86_64 and measure the milestone-3 tiers
      (TOL_FDSE_REL above all, and the 1e-3 relative of the forecasts).

## C defects found in milestone 3 (not patched; for the lead)

- [ ] **-alpha accepts an A of deficient rank** (`bugs/drvec-c-alpha-rank`,
      `repro.py` exits 1 while open). load_alpha_A checks the rank with
      choldcp(A'A), which only faults on a negative pivot: A = [1 1; 1 1; 0 0]
      passes, psi's seed divides by zero, the .out carries logL = nan and the
      program dies with SIGSEGV. The port refuses it with the C's own message
      (a rank check added, docs/PORT.md §11).
- [ ] **-estwin dates its header wrong** (cosmetic): `Sample : 50 observations
      from 1862` for a fit on 1851..1900 -- ObsToDate(nobs_raw - nobs + 1)
      assumes the trim is the first difference; the per-series diagnosis is
      offset the same way (nobs_raw - nobs). The port reproduces it.
- [ ] **-fdhess's step is below the objective's noise**: eta = macheps assumes an
      objective exact to an ulp, and AS 311's is ~1e-14 relative (E1, |Sigma| ~
      1e-11), so the printed s.e. carry ~0.1-0.5 % of numerical noise (fourth
      significant digit). Not a crash; a matter for the article's quoted s.e.

## C defects found in milestone 4 (not patched; for the lead)

- [ ] **The r = 0 ladder's cache key loses the data when a drift is held**
      (`bugs/drvec-c-ladder-key`, `repro.py` exits 1 while open). ladder_key()
      (drvec.c:5310) is a double: p*1e6 + ... + 1e12 * 2^(i-1) per held drift +
      sum rawmat * 1e-7 (t%97 + i). With a drift held (-nodrift, -case pre) it is
      ~1e12, where the spacing is 1.2e-4, and the data's terms (~1e-6) are mostly
      rounded away: two bootstrap replications can share a key, and the second
      starts from the FIRST one's ladder (rungs 0 and 1 not fitted, its gate not
      checked). On cprboot replications 15 and 16 share 1000002003000.062; the
      C's LR* of rep 16 is 5.4210743829648322, its own ladder gives
      5.4210743828848535 (8e-11; the stale seed was close; nothing bounds it).
      Handing rep 15's ladder to rep 16 in the port reproduces the C's value bit
      for bit. **Port difference**: the port keys its cache on the data object
      (each replication its own ladder), so cprboot.boot differs from the
      oracle's in that one line; the .out (p-value, percentiles) is identical.
      Without a held drift the key is ~1e6 and keeps the data (rankboot, e1boot
      unaffected). To byte-match the C instead, the port would have to emulate
      the key's rounding: a decision for the lead.

## E1 is ill-defined: a ridge, not a maximum (2026-10-15)

- [ ] **Study the E1 fit `2 1 1 -mean -case 2`.** Every start stops on criterion
      3 and the C's .out says "NOT a convergence ... not a demonstrated maximum".
      CI run 38744569511 shows what that means: the same `nested` start ends at
      773.1128 on Linux x86_64 (the C's), 773.4646 on macOS -- ABOVE the C -- and
      772.3638 on Windows. The compiled likelihood rounds differently there, and
      the optimiser stops elsewhere on the ridge. It is not Johansen's
      eigensolver: on Linux x86_64 scipy gives 773.1128 too. To find out: does
      the ridge keep rising (restart from macOS's point), is an MA root near
      the wall (min modulus 1.005 at the C's point), and what the analyst gets
      told -- sima's `study_estimation` does this for drvarma. The C would
      behave the same way built on macOS or Windows.
      **The gate too** (`-rungs`, e1rungs): the C's own entry gate on E1 is
      NOT VERIFIED by a hair (joint - sum = -4.120e-06, tolerance 1e-6), and on
      macOS the port lands on VERIFIED (CI run 38753032614). A gate decision
      that rounding can flip says the rung-0 fit is not converged to the
      gate's tolerance: part of the same study.
- [x] Tests: off Linux x86_64 E1 checks the declaration (termcode 3 and the
      .out's warning), not the point; the BFGS s.e. of case 1 are compared only
      on the oracle's platform (they follow the trajectory; `-fdhess`, ported in
      milestone 3, is path-independent but has a noise floor of ~1e-3 relative).

## Johansen's eigensolver: GSL everywhere through drvarma (decided 2026-10-15)

- [ ] Today drvec uses GSL through ctypes when libgsl is installed, scipy
      otherwise. Decided with the user: the next drvarma release exposes GSL's
      `gsl_eigen_gensymmv` (+ `_sort`) from its compiled engine, which already
      links GSL on every wheel platform; drvec then calls it and drops `_gsl.py`'s
      system lookup. A secondary fix: it removes one source of platform
      difference, not the E1 ridge above.

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
- Milestone 3's tiers are reasoned, not measured off-platform: -fdhess s.e. on
  TOL_FDSE_REL = 2e-2 (4x the measured ulp-level floor), forecasts and rolling
  errors to 1e-3 relative, and line counts for the .out. ms201's winner
  (`jitter 1`) is a 1e-10 near-tie, asserted on the oracle's platform only.
- -multistart's jitters use the 64-bit LCG; on Windows the C's are 32-bit (D4).
- The suite was run against drvarma 0.2.0 and fue 0.1.17.dev0 installed as
  EDITABLE dev trees; re-run against the released wheels.

## The port's first version (scope decided 2026-10-11)

Cases 1-4, the `.pre` route with `-case pre`/`-nodrift`, the rank test,
the impulse responses and variance decomposition with `-irfboot`, `-f`,
`-fixb2row`, `-xpre`/`-xlink` and `-xsys`, `-alpha`, `-weakex`, `-fixb2`,
`-fdhess`. The gaps the C declares are ported as refusals, with the C's
message. All of it is in place after milestone 3, with the CLI, `-estwin`,
`-multistart` and `-rungs`; `-specs` waits for -warma/-mawarma.

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
      (Since 2026-10-15 from drvarma's engine, not ctypes: see below.)
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

## The article's runs: acceptance test (2026-10-16)

The 211 invocations of the article's revision-suite scripts that carry no
-bootstrap and no -irfboot, C against drvec-py, on copies of the
article's inputs (private: nothing of them in this repo).
- [x] **Deterministic variables in a .pre** (milestone 5, below): 47 of the
      article's 50 .pre files declare them (seasonal cos/sin, alter, step,
      impulse), none with a delta: BUG-40's tail does not reach them; with
      fue <= 0.1.17 several variables per file made fue's w differ from the
      C's by ulps (gone with fue 0.1.18, below).
- [x] **-xpre/-xlink with an input that has an ARMA model** (xl_backcast): 10 of
      the 211 (milestone 5).
- With the deterministic block removed from every .pre (both programs on the
  same edited files): 201 accepted, every .out byte-identical except one line
  in 159 of them, a C defect (below); 10 refused (the -xlink item above). Wall
  time, sum over jobs: C 278 s, Python 3153 s (~11x).
- Rerun after milestone 5's first version (the C's reader transcribed): 211
  accepted, 42 byte-identical, 169 differing only in the C's truncated "Input
  Data File" line (below). That version was undone (milestone 5).
- **Rerun with fue's cast (2026-10-16), on the REAL inputs, C against
  drvec-py, 6 jobs in parallel: 211 accepted, 0 refused; 0 byte-identical, 2
  differing only in the "Input Data File" line, 0 because of BUG-40 (no input
  of these runs has a delta term), 209 other.** The 209 are all the
  deterministic component's rounding: each run has a file with 2 to 15
  variables, which fue subtracts one at a time and the C sums first,
  |w_fue - w_C| <= 1.1e-13. Rerun with the C's component (tests/cdet.py, since
  removed), the same 209 give 42
  byte-identical and 167 with the "Input Data File" line only, the previous
  acceptance exactly. On fue's w, 183 of the 209 agree in every decision and
  in logL to 1e-6; 26 do not: 13 winning starts change among starts tied to
  the printed digits, 4 fits change their criterion-3 banner, 4 rank tests
  their criterion notes (all on "(no values)" rows of case 3 / -case pre),
  and ill-conditioned fits move, all at r = M - 1 of three-series systems:
  one fit (-case 3, r = 2, a criterion-3 stop
  in both) by 20.0 in logL, five rank tests by 1.25 to 71.9 at a rank whose
  fit fails in both programs (LR < 0), three by 0.008 to 0.26. No
  tabulated rank-test verdict changes. Wall time 641 s; sum over jobs C
  270 s, Python 3427 s (~13x, under the parallel load).
- **Rerun with drvec C 0.13 and fue 0.1.18 (2026-10-17), the same 211 on the
  real inputs, 3 jobs in parallel: 211 accepted, 0 refused; 42
  byte-identical, 169 differing only in the C's truncated "Input Data File"
  line, 0 other.** Every decision agrees, the 26 of the previous rerun
  included. C 0.13 against C 0.12: all 211 byte-identical but the version
  banner (no input has a delta term, and with none the two rules give the
  same bits); no tabulated rank-test verdict changes between them or
  against the port (63 jobs carry a rank table). Wall time 725 s; sum over
  jobs C 146 s, Python 1979 s (~13.5x).
- [x] **Byte identity with the C on multi-variable files** (for the lead,
  2026-10-16): done by the suite's one deterministic routine (lib/fuedet;
  drvec C 0.13 and fue 0.1.18), with no change of drvec's own. drvec still
  uses fue's private names (cast_us.build_est_spec, cast_us_py,
  _build_initial_x, forecast._build_xi, _nonsop_coefs), as drtran-python
  does; a public "levels cast" would let both ports drop them.
- [ ] **C defect: "Input Data File : the .pre fil"** (atsw-gui BUG-59) -- drvec.c ~10370 sizes
      inputf as strlen(argv[1]) + 8, and ~10905 writes "the .pre files" into it
      with that size: a first file name of 6 characters or fewer truncates the
      line. Cosmetic; the port prints it whole. Repro: `drvec c.pre e1income.pre
      1 0 0 -case 2 -lrtest` with c.pre a copy of e1cons.pre.
- cprq (2 1 r -case pre) is ill-defined at r = 1 as E1 is: the C's four starts
  end at logL -1.50 to 6.48, the best on the step criterion; off Linux x86_64 it
  stops ~1e-3 elsewhere (CI run 38912495044). The test compares the logL to 1e-2
  there; e1gate's to 1e-4.

## Milestone 5 — what blocks the article (2026-10-16)

- [x] Deterministic variables in a .pre, every type fue's reader accepts
      (impulse, compimp, step, ramp, easter, trend, cos, sin, alter): the
      .pre is read with fue.load() and its series built by fue's own cast
      at the .pre's parameters, the non-stationary operator emptied
      (data.levels_cast: cast_us_py's step [6], z = refactor * BoxCox(y) -
      det), as drtran-python does; the component over the horizon for -f and
      -estwin is fue's _build_xi; the same for an -xpre input. fue is a
      runtime dependency again. (The first version transcribed the C's
      reader, drvec.prefile; removed by the lead's decision: every port
      reads through fue's cast.)
- [x] -xpre/-xlink inputs with an ARMA model: xl_backcast with fue's factors
      multiplied out (the phi, theta and mean of the same cast), d = 0 and
      d = 1.
- [x] Refused as fue refuses it (an unknown word asks for a data column;
      `time` is one); a non-standard variable WITH its column, with the C's
      message; easter off monthly data (fue would carry a null regressor).
- [x] Fixtures from the public data (tests/data/fixtures/make_m5.py, SOURCE)
      and 23 oracle cases, two of them for BUG-40 (b40, b40rank: a step with
      delta 0.95, 52 periods on). The port's w is fue's cast bit for bit
      (tests/test_m5.py, docs/PORT.md §13).
- [x] **BUG-40 closed by agreeing with fue (2026-10-17).** The deterministic
      component follows fue's rule everywhere (nu(B) truncated at 40 lags,
      the variables summed, one subtraction), one routine for the suite:
      atsw-gui lib/fuedet (ae95446), called by the oracle drvec C 0.13 and
      vendored by fue 0.1.18 (csrc/fuedet.c, behind cast_us_py's step [6] and
      _build_xi). Milestone 5's oracle files regenerated with 0.13 (24 of 30
      changed, all through the component); every one of the 23 cases is
      byte-identical on Linux x86_64 with GSL, b40rank's verdict ("reject H0
      at 10%", LR(0) 13.7873) included. The BYTE/FUE/BUG40 split and
      tests/cdet.py (the C 0.12's exact component, kept to prove it was the
      only difference) removed. The transfer functions of -xlink inputs keep
      the exact, untruncated filter, as in the C.
- No C defect found in the deterministic or backcast paths.

## Next

- [x] **The CLI dies when -irfboot has too few replications for bands**
      (port defect, 2026-10-24, fixed the same day; `bugs/py-cli-irfboot-few`, `repro.py` exits 1
      while open). The C prints "Too few replications for bands (N)." and
      writes no .irfboot; the port's `cli._execute` writes
      `res.irfboot_text(base)`, which is None then: TypeError, exit 1, an empty
      .irfboot left behind, and the .forecast/.recursive that follow never
      written. The library is fine. Fix: write the .irfboot (and say so) only
      when the text is not None, as the C does.
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
- [ ] The goldens on the `.inp` route (base2, case1, ... in golden.sh's CASES) are
      out of the port's scope (the .pre route only).
- [ ] Run the suite once off Linux x86_64 and measure the milestone-3 tiers
      (TOL_FDSE_REL above all, and the 1e-3 relative of the forecasts).
- [x] **After the release: name the residual columns in the .out** (done 2026-10-24:
      BUG-62, atsw-gui e9304c6 and branch bug62-residual-labels 95b6c6c; found in the
      sima 0.2.0 shakedown, 2026-10-22, Martín Manjón and Treadway's (R0T, R0)).
      The residual diagnostics read Ybar's innovations, [nabla Y2 ; W] = (R0T,
      ec1), but "residual sd" and the "Residual cross-correlation matrices" print
      no names, so they read as (nabla R0T, nabla R0): sd 8.82 and r(0) = 0.024
      look inconsistent with Sigma's 0.88 correlation, and are right for (R0T,
      ec1). Not a defect in the numbers. A presentation change in BOTH the C
      (atsw-gui engines/drvec, residual diagnostics) and report.py's
      `residual_diagnostics`, together, since the .out is byte-identical.

## C defects found in milestone 3 (not patched; for the lead)

- [ ] **-alpha accepts an A of deficient rank** (atsw-gui BUG-58) (`bugs/drvec-c-alpha-rank`,
      `repro.py` exits 1 while open). load_alpha_A checks the rank with
      choldcp(A'A), which only faults on a negative pivot: A = [1 1; 1 1; 0 0]
      passes, psi's seed divides by zero, the .out carries logL = nan and the
      program dies with SIGSEGV. The port refuses it with the C's own message
      (a rank check added, docs/PORT.md §11).
- [ ] **-estwin dates its header wrong** (atsw-gui BUG-66) (`bugs/drvec-c-estwin-date`,
      `repro.py` exits 1 while open): `Sample : 50 observations
      from 1862` for a fit on 1851..1900 -- ObsToDate(nobs_raw - nobs + 1)
      assumes the trim is the first difference; the per-series diagnosis is
      offset the same way (nobs_raw - nobs). The port reproduces it.
- [ ] **-fdhess's step is below the objective's noise** (atsw-gui BUG-67)
      (`bugs/drvec-c-fdhess-step`, `repro.py` exits 1 while open): eta = macheps assumes an
      objective exact to an ulp, and AS 311's is ~1e-14 relative (E1, |Sigma| ~
      1e-11), so the printed s.e. carry ~0.1-0.5 % of numerical noise (fourth
      significant digit). Not a crash; a matter for the article's quoted s.e.
      Measured 2026-10-30: a larger eta is not the fix -- the forward
      differences' O(h) bias moves an s.e. by 1.9 % (mink-muskrat) and 5.1 %
      (E1) when the step is multiplied by 3.6 (eta = 1e-14).

## C defects found in milestone 4 (not patched; for the lead)

- [ ] **The r = 0 ladder's cache key loses the data when a drift is held** (atsw-gui BUG-60)
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

- [x] Today drvec uses GSL through ctypes when libgsl is installed, scipy
      otherwise. Decided with the user: the next drvarma release exposes GSL's
      `gsl_eigen_gensymmv` (+ `_sort`) from its compiled engine, which already
      links GSL on every wheel platform; drvec then calls it and drops `_gsl.py`'s
      system lookup. A secondary fix: it removes one source of platform
      difference, not the E1 ridge above.
      DONE (unreleased): drvarma 0.3.0 exports `gensymmv` and, beside it,
      `eigen_nonsymm` -- the other half of the ctypes lookup, the eigenvalues
      of Pi and of the operator roots that the reports print in GSL's order.
      `_gsl.py` calls them, keeps scipy/numpy when drvarma's engine is
      unavailable, and no longer looks for a system libgsl; the CI installs no
      libgsl. Bit identity with the ctypes path: every eigenproblem of the full
      suite (`--runslow`) solved both ways and compared with `==`, before the
      lookup was removed (CHANGELOG). Off Linux x86_64 the CI must now show the
      GSL backend and, where the tolerance tier allowed scipy's seed before,
      whether anything tightens: to be measured on the first off-platform run.

## What sima 0.2.0 needs: accessors, parallel bootstrap, the C engine (2026-10-18)

PLAN-vec (sima-python, branch estudio-vec) phases A1-A5 and B.

- [x] A4, the VEC in levels (`drvec.levels`, `to_levels_varma`), forecast by
      drvarma's `forecast_given` (drvarma gsl-gensymmv 7822189); built from
      the public accessors. Agrees with drvec's `-f` to 1e-12 relative in the
      model's units on the ten golden `.forecast` cases and four corners
      (tests/test_levels.py); the oracle's `.forecast` rebuilt from drvarma's
      numbers byte-identical on Linux x86_64. drvec's `-f` is one computation
      (`forecast_arrays`; `forecast_vec` prints from it).

- [x] A1, the accessors (STUDY-vec §4.2 items 1-6): docs/API.md. Every one
      equals the number the .out prints on every golden fit
      (tests/test_accessors.py: 44 fits, 9 forecast files, the .irfboot).
- [x] A2, the parallel bootstrap: `workers=` on rank_test and
      fit(irfboot=). Verified before relying on it: the only consumer of the
      bootstrap generator is rank.simulate_h0 (grep: no Rng, no random in the
      fit path; -multistart's generator is local to fit_search and a
      replication is fitted with one start), and each replication is fitted
      on a fresh Data. workers=1 and workers=4 give the same .out, .boot and
      .irfboot on rankboot, cprboot, dboot, e1boot (slow) and irfboot
      (tests/test_parallel.py); the oracle tests run on the default workers.
      Default: DRVEC_WORKERS, else cores - 1.
- [x] A3, `seed=`: the C's 987654321 by default; another seed always runs in
      Python (the C's generator is fixed).
- [x] B, drvec.c_engine: the finder (DRVEC_BIN, drvec/_bin bundled for B5,
      PATH, ATSW), the 0.13 banner check, the argv (the CLI parser's inverse),
      the run in a folder, the .boot/.irfboot read back into RankTest and
      FitResult; engine="auto"|"python"|"c" (DRVEC_ENGINE); provenance on
      every result. C path == Python path byte for byte on rankboot, dboot,
      e1boot and irfboot; cprboot differs in replication 16 only, the C's
      BUG-60 (the C path gives the oracle's .boot). CI builds the C from
      atsw-gui ae95446 on ubuntu-latest (run 39118715613: the C tests ran,
      all 8 jobs green).
- [x] B5, drvec C inside the wheels (decided 2026-10-18): csrc/ (atsw-gui
      ae95446, 20 files, sha256 in csrc/VENDORED.json, tests/test_vendored.py;
      tools/vendor_drvec_c.py refreshes), setup.py builds drvec/_bin/drvec
      with GSL static (csrc/build_drvec.py), py3-none-<platform> wheels by
      cibuildwheel on 4 platforms (Windows since BUG-61: below) + the pure
      wheel + the sdist (publish.yml),
      DRVEC_REQUIRE_C. The manylinux x86_64 wheel's binary writes the oracle's
      files byte for byte on 19 commands (tests/test_bundled.py); tests.yml
      builds the vendored C instead of atsw-gui's checkout.
- [x] B5 follow-ups (2026-10-18): the Windows platform wheel is published
      again. bugs/drvec-c-rng-long32 is fixed in atsw-gui as BUG-61
      (7ec4bf7, PR #60, not merged yet); csrc/ re-vendored from it; the
      artifact is wheels-windows-latest, windows-latest is back in
      smoke_wheels, c_engine.find no longer refuses drvec C on Windows, and
      tests.yml builds the C on Windows too. tests/data/oracle/SOURCE now says
      the GSL the oracle RUNS on: Ubuntu 24.04's 2.7.1 (libgsl.so.27), not
      "the system GSL 2.8", and records the BUG-61 oracle build.
- [ ] Intel macOS has no binary wheel (the pure one; as drvarma).
- **The speed-ups** (`tools/time_bootstrap.py B --case mm|e1 --engines`, on
  this machine: an i5-6300U, 2 cores / 4 threads, with another job's tests
  running at the same time, load 2-5; every .boot identical to workers=1's):

  | run | workers=1 | 2 | 3 | 4 | drvec C |
  |---|---|---|---|---|---|
  | mm, `2 0 -case 2 -lrtest -bootstrap 99` | 28.9 s | 15.5 s (1.86x) | 14.3 s (2.02x) | 14.0 s (2.07x) | 2.4 s (12.1x) |
  | E1, `2 0 -case 2 -lrtest -bootstrap 20` | 121-138 s | 106 s (1.14x) | 83-119 s (1.16-1.46x) | 69-86 s (1.60-1.75x) | 13.3-16.9 s (8.2-9.1x) |

  The pool scales with the PHYSICAL cores: two here, so about 2x on mm. E1
  with B = 20 has only 40 refits of very unequal cost (some starts run long
  on its ridge), so the tail dominates; with B = 499 the pool's balance
  improves. STUDY-vec's "3-4x on this machine's 4 cores" assumed 4 real
  cores. The C stays 8-12x faster than one Python core, so `engine="auto"`
  takes it when present (PLAN-vec decision E2).
- [x] (done 2026-10-24: atsw-gui BUG-63, PR #86; csrc/ from abe6602) The C engine reports progress as the C's stdio buffer flushes the .boot
      (about every 4 KB, ~130 replications); -irfboot through the C reports
      only at the end. A `setvbuf` line-buffered .boot in the C would give
      per-replication progress: a C change, for atsw-gui.
- [x] `-fixb2 v` prints no LR in the C's .out (nor does the port), but
      `drvec-py -h` said the restrictions came "each with its LR against the
      free model" (the port's own help; the C's usage does not say it).
      Fixed in the help text; `restriction_lr` is None there, as the .out.

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

- Johansen's seed (decision D2, revised in milestone 2 and on 2026-10-15): GSL's
  gensymmv from drvarma's compiled engine (`drvec._gsl`), scipy's `eigh` when that
  engine is unavailable. The risk the milestone-1 note foresaw was seen on E1:
  with scipy the `johansen` start ends at 760.07 instead of 760.54 and one
  bootstrap LR* moves by 3.5e-5 (beyond 1e-6 in logL). With GSL everything is
  byte-identical. Without drvarma's engine (before 2026-10-15: without a system
  libgsl, i.e. Windows and macOS without Homebrew's gsl) a near-tie can still
  flip a winner.
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
- The `.pre` files go through fue's reader and cast (milestone 5): D > 0 is
  refused (the C ignores it); alignment is the C's (date intersection), not
  drvarma's ladder rule. The deterministic component is the C's bit for bit
  only with fue >= 0.1.18 and drvec C >= 0.13 (lib/fuedet): with fue 0.1.17
  any file with two or more variables, or a rational one, moves w by ulps
  (and by BUG-40's tail past lag 40), and 16 of the 22 cases of
  tests/test_m5.py::test_products fail (measured).
- [x] **C defect, filed 2026-10-18 and FIXED the same day as atsw-gui BUG-61
  (7ec4bf7; `bugs/drvec-c-rng-long32`, `repro.py` exits 0 since):** the
  bootstrap generator's `unsigned long`
  (drvec.c:2203) is 32 bits on Windows, and `boot_rng >> 33` is then a shift
  past the type's width: gcc folds it to 0, u1 = 0 on every draw, and
  boot_normal's rejection loop never ends. drvec C 0.13 built for Windows
  (MinGW-w64; MSVC would be the same) hangs on its first bootstrap
  replication (found by the Windows wheel's smoke test, B5). The fix belongs
  in atsw-gui (uint64_t for both LCGs; on LP64 nothing moves). It was fixed
  there and re-vendored; the Windows wheel is published again. The port
  computes the 64-bit stream on every platform, and now so does the C.
- Milestone 3's tiers are reasoned, not measured off-platform: -fdhess s.e. on
  TOL_FDSE_REL = 2e-2 (4x the measured ulp-level floor), forecasts and rolling
  errors to 1e-3 relative, and line counts for the .out. ms201's winner
  (`jitter 1`) is a 1e-10 near-tie, asserted on the oracle's platform only.
- -multistart's jitters use the 64-bit LCG; on Windows the C's were 32-bit
  (D4) until BUG-61, and are the same 64-bit stream since.
- The suite was run (2026-10-17) against drvarma 0.2.0 installed as an
  EDITABLE dev tree and fue 0.1.18 built from its branch fue-det-unico (not
  yet released): re-run against the released wheels.

## The port's first version (scope decided 2026-10-11)

Cases 1-4, the `.pre` route with `-case pre`/`-nodrift`, the rank test,
the impulse responses and variance decomposition with `-irfboot`, `-f`,
`-fixb2row`, `-xpre`/`-xlink` and `-xsys`, `-alpha`, `-weakex`, `-fixb2`,
`-fdhess`. The gaps the C declares are ported as refusals, with the C's
message. All of it is in place after milestone 3, with the CLI, `-estwin`,
`-multistart` and `-rungs`; `-specs` waits for -warma/-mawarma.

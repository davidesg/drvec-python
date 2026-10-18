# Changelog

## 0.1.0 — unreleased

- `drvec.to_levels_varma(fit)` (PLAN-vec A4): the fitted VEC as a VARMA in
  levels, forecast by drvarma's `forecast_given` (drvarma >= 0.3.0); it
  agrees with drvec's own `-f` to 1e-12 relative on every golden `.forecast`
  (docs/API.md). drvec's forecaster stays, under byte identity, and is now
  one computation: `forecast_arrays` computes, `forecast_vec` prints the
  `.out` section and the `.forecast` from its numbers.
- **drvec C inside the wheels** (PLAN-vec B5). The C sources of drvec 0.13
  and the `lib/` pieces it links (optim, diagmv, fuedet) are vendored in
  `csrc/`, byte for byte from atsw-gui 7ec4bf7 (0.13 with BUG-61 fixed;
  ae95446 until 2026-10-18): `csrc/VENDORED.json` records
  the commit and each file's sha256, `tests/test_vendored.py` checks them,
  and `tools/vendor_drvec_c.py --repo --commit` refreshes the copy and the
  manifest together. `setup.py` builds the executable (`csrc/build_drvec.py`:
  the Makefile's units and flags, GSL static) into the package as
  `drvec/_bin/drvec` (`drvec.exe`), which `c_engine` finds after
  `DRVEC_BIN`. The wheels: `py3-none-<platform>` for Linux x86_64 and
  aarch64 (manylinux, GSL 2.7.1 from source), macOS arm64 (the same) and
  Windows AMD64 (MinGW-w64 and MSYS2's GSL, `-static`), plus the pure
  `py3-none-any` wheel (`DRVEC_NO_C=1`) as the fallback and the sdist, which
  builds the C when it finds a compiler and GSL. `DRVEC_REQUIRE_C=1` turns
  a failed C build into an error (the wheels and the CI set it). On Linux
  x86_64 the wheel's binary writes the oracle's files byte for byte
  (`tests/test_bundled.py`, nineteen oracle commands); every wheel's smoke
  test runs a bootstrap through it and the C-path tests. The CI no longer
  builds the C from atsw-gui: it builds the vendored copy, on Windows too.
- **drvec C on Windows** (2026-10-18). drvec C 0.13 as vendored from ae95446
  never finished a bootstrap on Windows: its two generators held their
  64-bit state in an `unsigned long`, 32 bits there (`bugs/drvec-c-rng-long32`,
  found by the Windows wheel's smoke test). Fixed in the C as atsw-gui
  BUG-61 (7ec4bf7, `uint64_t`), before drvec-v0.13's tag, and re-vendored. On
  Linux x86_64 the fixed oracle writes ae95446's 83 files byte for byte
  (`tests/data/oracle/SOURCE`). The Windows wheel is published again with its
  smoke job, `c_engine.find()` no longer refuses drvec C on Windows (the
  banner cannot tell the fixed build from the old one, so a hand-built
  `DRVEC_BIN` from before the fix would still hang there), and tests.yml
  builds the C on Windows (MinGW-w64, MSYS2's GSL).
- The package: Python on drvarma's exact likelihood and fue's reader.
  Its oracle is drvec C 0.13 (atsw-gui ae95446; milestones 1-4 were
  homologated against drvec-v0.12, which 0.13 reproduces on them).
- Milestone 1: the VEC cast (`vec_shootx`), the cold and Johansen starts, `est`
  on drvarma's optimiser with the BFGS standard errors, the C's r = 0 ladder with
  its entry gate, the route-(B) profile and the four-start search, the sequential
  rank test with the parametric bootstrap and its `.boot` file. Public API:
  `drvec.fit(pre_files, p, q, r, case=2)` and `drvec.rank_test(pre_files, p, q,
  case=2, bootstrap=B)`. Homologated against the oracle's two milestone runs
  (bit-identical except Johansen's eigenproblem, docs/PORT.md).
- `tools/time_bootstrap.py`: the bootstrap's cost against the C's, split into
  the cast, the compiled likelihood and the rest; `--case e1` (M = 3).
- Milestone 2: the `.out` writer (`FitResult.out_text`, `RankTest.out_text`),
  byte-identical to the oracle's on every PRE_CASE of its golden set;
  `fit(..., fixb2row=, nodrift=, case="pre", trend=, xsys=, xpre=, xlink=,
  irfboot=)`; Johansen's seed through GSL (ctypes) when libgsl is present
  (from drvarma's engine since, below);
  the tests exact on Linux x86_64 and on a tolerance tier elsewhere.
- Milestone 3: `fit(..., fdhess=True)` (the finite-difference Hessian s.e.),
  `alpha=`, `weakex=`, `fixb2=` (restrictions with their LR against the free
  model), `forecast=H` and `estwin=E` (`FitResult.forecast_text`,
  `recursive_text`), `multistart=n` (also in `rank_test`), `drvec.rungs()`;
  the command line `drvec-py` (the C's .pre-route syntax and products). Every
  oracle case byte-identical on Linux x86_64. C defect found: -alpha accepts a
  rank-deficient A (bugs/drvec-c-alpha-rank).
- Milestone 4: `rank_test(..., case="pre")` and `rank_test(..., nodrift=[i])`
  (also with `bootstrap=N`; the CLI's `-lrtest -case pre`/`-nodrift`, with the
  C's warning on the mixed case); the entry gate with its optimality
  certificate in a plain fit at the diagonal rung (`FitResult.gate`). Every
  oracle case byte-identical on Linux x86_64 but one bootstrap replication of
  cprboot.boot, where the C reuses the previous replication's ladder: C defect
  bugs/drvec-c-ladder-key.
- Milestone 5: deterministic variables in a `.pre` (impulse, compimp, step,
  ramp, easter, trend, cos, sin, alter): each file's component subtracted from
  w and added back to the forecasts in levels; `-xpre` inputs with an ARMA
  model (the presample backcast, xl_backcast) and with deterministic
  variables. The `.pre` files are read and cast by fue (`fue.load`,
  `fue.cast_us.cast_us_py` with the non-stationary operator emptied:
  `drvec.data.levels_cast`), as in drtran-python; fue is a runtime dependency.
  (A first version transcribed the C's reader, `drvec.prefile`; it was
  removed.) The component follows fue's rule (a rational term truncated at
  40 lags, the variables summed and subtracted once), and every oracle case
  is byte-identical on Linux x86_64 with GSL, the BUG-40 case included (b40,
  b40rank: a step with delta 0.95, 52 periods on).
- The oracle is drvec C 0.13 (atsw-gui ae95446; the tag drvec-v0.13 comes
  with the release) and fue is required at >= 0.1.18: both build a .pre's
  deterministic component with the suite's one routine (atsw-gui
  lib/fuedet). BUG-40 -- drvec C 0.12 applied the exact filter where fue
  truncates at 40 lags, and fue <= 0.1.17 subtracted one variable at a time
  -- is closed by agreeing with fue. Milestone 5's oracle files regenerated
  (24 of 30; those of milestones 1-4, without deterministic variables, are
  byte-identical under 0.13 and were kept); the BYTE/FUE split of
  tests/test_m5.py and tests/cdet.py, the C's old exact component, removed.
- Johansen's seed and the printed eigenvalues come from GSL through
  drvarma's compiled engine (`drvarma._engine.gensymmv` and `eigen_nonsymm`,
  drvarma >= 0.3.0), on every platform drvarma ships a binary wheel for;
  `drvec._gsl` no longer looks for a system libgsl through ctypes, and the CI
  installs none. Without drvarma's engine, scipy's `eigh` and numpy's
  `eigvals` as before (decided 2026-10-15, TODO.md). Verified bit-identical
  to the ctypes path on Linux x86_64 before removing it: the full suite with
  `--runslow`, with every call solved both ways and compared with `==` --
  271 generalized eigenproblems (190 with M = 2, 81 with M = 3) and 117
  nonsymmetric ones (n = 2, 3, 4, 6), all equal; the suite is unchanged,
  byte-identical to the oracle.
- Packaging for PyPI (not published; the suite is released together): the
  name `drvec`, the metadata (description, keywords, classifiers for Python
  3.10-3.13, URLs, `license-files`), `MANIFEST.in` (the sdist carries the
  docs and the test suite with its public data; not TODO.md, tools/, bugs/),
  a README for the PyPI page with absolute links, and `publish.yml`: on a
  `v*` tag, the platform wheels (above), the pure wheel and the sdist,
  `twine check`, a check that nothing private ships, every one
  smoke-tested on Python 3.10 and 3.13, then trusted publishing. The tests can run against the installed package
  (`DRVEC_TEST_INSTALLED=1`; the CI does), and `tests.yml` has a manual run
  that builds fue and drvarma from git refs, to measure off the oracle's
  platform before they are released.
- **What sima 0.2.0 needs from the engine** (PLAN-vec A1-A3, B; none of it
  changes a number: the suite is byte-identical to the oracle as before).
  - The accessors (A1): `FitResult.alpha`, `beta2`, `beta`, `Pi`, `gamma`,
    `theta`, `Q`, `Sigma`, `means`, `delta`, `residuals` (with
    `residual_names`), `stationary` (the VARMA on Ybar); `level_irf(K)`,
    `fevd(K)`, `long_run()` (gain, mean lag); `irf_bands()`;
    `diagnostics()` (Hosking's Q and the multivariate JB with their df and
    p-values); `forecasts(H)` (point, bands, s.e. and dates); and
    `restriction_lr` (the LR with its df and p-value). Each comes from the
    routine that prints it (`report.level_irf_numbers`,
    `multivariate_numbers`, `restriction_lr`; `forecast.forecast_arrays`,
    from which `forecast_vec` prints the files), and `tests/test_accessors.py`
    checks every one against the printed number on every golden fit.
  - The parallel bootstrap (A2): `rank_test(..., workers=)` and
    `fit(..., irfboot=N, workers=)`. The samples are drawn in sequence, in the
    C's order, and refitted in a process pool (`drvec.parallel`); a refit
    draws no random numbers (checked: the only consumer of the generator is
    `simulate_h0`), so `workers=1` and `workers=4` give the same `.out`,
    `.boot` and `.irfboot` byte for byte (`tests/test_parallel.py`). Default
    `DRVEC_WORKERS`, else the cores less one. The `-irfboot` bootstrap gains
    `progress(b, ok)`; the rank test's `progress` is now also called for a
    replication whose sample could not be drawn (simfail).
  - `seed=` on both bootstraps (A3), the C's 987654321 by default.
  - drvec C as the bootstrap engine (B): `drvec.c_engine` finds the binary
    (`DRVEC_BIN`, a binary bundled at `drvec/_bin/drvec`, `drvec` on the PATH,
    the ATSW install directory), refuses any version but 0.13, translates the
    call into the C's argv (the inverse of the CLI's parser), runs it in a
    folder and reads its `.boot` and `.irfboot` back into the same
    `RankTest`/`FitResult`. `engine="auto" | "python" | "c"` on both
    bootstraps (`DRVEC_ENGINE`; "auto" takes the C when it matches). Every
    result carries `provenance` (engine, version, path, backend, platform
    tier, workers, seed). On Linux x86_64 the two engines give the same bytes
    (`tests/test_c_engine.py`), except the C's BUG-60 replication of cprboot.
  - `docs/API.md`; `tools/time_bootstrap.py --engines` measures the engines.

# Changelog

## 0.1.0 — unreleased

- The package: pure Python on drvarma's exact likelihood and fue's reader.
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
  `v*` tag, sdist + pure wheel, `twine check`, a check that nothing private
  ships, both smoke-tested on Python 3.10-3.13 from PyPI's dependencies, then
  trusted publishing. The tests can run against the installed package
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
    `multivariate_numbers`, `restriction_lr`; `forecast.forecast_arrays`
    repeats `forecast_vec`'s arithmetic), and `tests/test_accessors.py`
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
    (`tests/test_c_engine.py`; CI builds the C from atsw-gui ae95446 on
    ubuntu-latest), except the C's BUG-60 replication of cprboot.
  - `docs/API.md`; `tools/time_bootstrap.py --engines` measures the engines.

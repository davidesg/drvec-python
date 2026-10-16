# Changelog

## 0.1.0 — unreleased

- The package: pure Python on drvarma's exact likelihood and fue's reader.
  Its oracle is drvec-v0.12 (atsw-gui).
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
  irfboot=)`; Johansen's seed through GSL (ctypes) when libgsl is present;
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
  removed.) Consequence, BUG-40 (WON'T FIX in the C): fue truncates a rational
  term at 40 lags and the C does not, and the two sum the variables in a
  different order, so only the oracle cases whose files carry no variable or
  a single one without delta stay byte-identical (4 of 23); the others are
  checked against the oracle fed the C's component (byte-identical) and, on
  fue's, in sections, npar, verdicts and logL. A new case at BUG-40's size
  (b40, b40rank: a step with delta 0.95) moves the logL by 0.47 and flips the
  rank test's verdict at 10%.

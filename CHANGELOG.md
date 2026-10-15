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

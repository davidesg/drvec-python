# drvec — the Python API

Everything below is importable from `drvec`. The functions mirror the C
program's `.pre` route (`drvec f1.pre ... fM.pre p q r [options]`); the
reports they write are the C's, byte for byte on Linux x86_64. What this page
adds to the docstrings is the map: which call, which result, which number.

Contents: [fit](#fit) · [rank_test](#rank_test) · [the accessors](#the-accessors-of-a-fitresult)
· [the bootstrap: workers, seed, engine](#the-bootstrap-workers-seed-and-engine)
· [drvec C as the engine](#drvec-c-as-the-bootstrap-engine) · [provenance](#provenance)

## `fit`

```python
drvec.fit(pre_files, p, q, r, case=2, ma="free", diag_ar=False, diag_cov=False,
          fixb2row=None, nodrift=None, trend=False, irfboot=0, xsys=None,
          xpre=None, xlink=None, forecast=0, fdhess=False, alpha=None,
          weakex=None, fixb2=None, estwin=0, multistart=0,
          seed=None, workers=None, engine=None, folder=None, progress=None)
    -> FitResult
```

One VARMA-VEC of rank `r`, with the C's ladder start and search. The options
are the C's (`-case`, `-fixb2row i v`, `-nodrift i`, `-trend`, `-irfboot N`,
`-xsys i`, `-gzero i j` (`gzero=[(i, j), ...]`), `-xpre/-xlink`, `-f H`, `-fdhess`, `-alpha FILE`, `-weakex i`,
`-fixb2 [v]`, `-estwin E`, `-multistart n`); the combinations the C refuses
are refused with its words. `seed`, `workers`, `engine`, `folder` and
`progress` act on the `-irfboot` bootstrap only (below).

`FitResult` keeps the C's scalars (`loglik`, `sigma2`, `npar`, `aic`, `bic`,
`nobs`, `termcode`), the labelled parameters (`params`, `param(label)`: label,
kind, value, s.e.), and writes the C's files: `out_text(base)`,
`forecast_text(base)`, `recursive_text()`, `irfboot_text(base)`.

## `rank_test`

```python
drvec.rank_test(pre_files, p, q, case=2, ma="free", bootstrap=0, boot_path=None,
                progress=None, multistart=0, nodrift=None,
                seed=None, workers=None, engine=None, folder=None)
    -> RankTest
```

The sequential LR test for the rank (`-lrtest [-bootstrap N]`). `RankTest`
has `ll[r]`, `npar[r]`, `lr(r)`, `critical_values(r)` (the asymptotic table,
cases 1 and 2), `verdict(r)`, and `bootstraps`, one per comparison r -> r + 1
with `cv` (the 90/95/99 % percentiles of LR*), `pvalue`, `nok` and `reps`
(every replication: `h0`, `rep`, `status` ok/simfail/fitfail, `lr`).
`out_text(base)` and `boot_text()` are the C's `.out` and `.boot`.
`progress(b, replication)` is called after each replication, failed ones
included; `replication.h0` is the comparison.

## The accessors of a `FitResult`

The numbers the `.out` prints, as arrays (STUDY-vec §4.2, items 1-6). Each is
computed by the routine that prints it, and the suite checks, on every golden
case, that it equals the printed number (`tests/test_accessors.py`). None of
them changes a number.

Conventions: the VEC's matrices are in the **.inp's order** (the order of the
files: the first `s = M - r` series are the nabla Y2 block, the last `r` the Y1
block), as the `.out` prints them; `alpha` is in **Johansen's sign** (negative
is error-correcting). The residuals and the stationary representation are
**Ybar's**, `[nabla Y2 ; W]`, labelled by `residual_names` (the Y2 names, then
`ec1..ecr`).

| accessor | shape / type | the `.out`'s |
|---|---|---|
| `alpha` | (M, r) | "alpha matrix" |
| `beta2` | (s, r) | "beta_2 matrix" (held rows included) |
| `beta` | (M, r) | "beta matrix", `[B2 ; I_r]` |
| `Pi` | (M, M) | "Pi = alpha beta'" |
| `gamma` | (p - 1, M, M) | "Gamma(k) matrix" |
| `theta` | (q, M, M) | "theta(k) matrix" |
| `Q`, `Sigma` | (M, M), symmetric | "Q matrix", "Sigma = sigma2 * Q" |
| `means` | (k,) | "E[W] vector" / "E[nablaY2] and E[W] vector" |
| `delta` | (M,) or None | "delta = E[nabla Y]" (case 3) |
| `residuals`, `residual_names` | (n, M), list | what the residual diagnostics read |
| `stationary` | `Stationary` | the VARMA on Ybar: `mu`, `phi`, `theta`, `qq`, `sigma2`, `Sigma`, `w` |
| `level_irf(K=None)` | (K + 1, M, M) | "Impulse Response of the Levels": `[k, i, j]` = response of i to a one-s.d. shock to j |
| `fevd(K=None)` | (K + 1, M, M), % | "Forecast Error Variance Decomposition": `[k, i, j]` at horizon k + 1 |
| `long_run()` | `LongRun` | "Long-Run Gain and Mean Lag": `gain`, `mean_lag` (nan where undefined), `shock_sd`, `converged`, `k_stop`, `tail`, `beta_gain` |
| `irf_bands()` | `IrfBands` or None | `-irfboot`: `irf`, `irf_pct[p]`, `fevd`, `fevd_pct[p]` for p in 2.5, 5, 50, 95, 97.5; `nok`, `nfail` |
| `diagnostics()` | `Diagnostics` or None | Hosking's Q (`hosking_Q`, `hosking_df`, `hosking_p`, `hosking_lags`) and the multivariate JB (`jb`, `jb_df`, `jb_p`, `jb_skewness`, `jb_kurtosis`) |
| `forecasts(H=None, conf=0.95)` | `Forecasts` | `<base>.forecast`: `dates`, `level`, `low`, `high`, `se` (H, M) in each series' units; `model_level`, `model_se` before the back-transform |
| `restriction_lr` | `RestrictionLR` or None | the LR block: `kind` (alpha, fixb2row, fixb2row+alpha, xsys, xlink), `ll_free`, `ll_restricted`, `lr`, `df`, `pvalue` |

`K` defaults to the `.out`'s horizon (10 when n < 40, else 20). `-fixb2`
alone carries no LR: the C prints none. The bands from the C engine carry the
8 significant digits of its `.irfboot` file.

`forecasts()` and the `.forecast`/`.out` writers are one computation:
`forecast.forecast_arrays` computes the numbers and `forecast_vec` prints the
C's files from them, so an array and the file cannot disagree.

## The VEC in levels, forecast by drvarma (`to_levels_varma`)

`drvec.to_levels_varma(fit)` writes the fitted VEC as a VARMA in levels,
`Phi(B) Y_t = c + Theta(B) e_t`, of order (p + 1, q), with `Phi(1)` of rank r
(PLAN-vec A4: a VEC is a VARMA in levels; the suite forecasts it with the
VARMA system's code). It returns a `LevelsVarma` (`phi`, `theta`, `sigma`,
`const`, `w`, `a`, `names`, `lambdas`, `refactors`, `r`, `origin`) built from
the public accessors (`stationary`, `residuals`, `beta2`), and
`LevelsVarma.forecast(H, conf=0.95, z=None)` forecasts it with drvarma's
`forecast.forecast_given` (drvarma >= 0.3.0), each file's deterministic path
added back and the Box-Cox inverted. The band's multiplier is drvec's
(1.959964 at 95%) unless `z` is given; drvarma's ladder uses 1.96.

drvec's own forecaster (`fit(forecast=H)`, the C's `-f`) stays: it is under
byte identity with the oracle. The two agree to 1e-12 relative in the model's
units on every golden case with a `.forecast` (`tests/test_levels.py`), and
the oracle's `.forecast` rebuilt from drvarma's numbers is byte-identical on
Linux x86_64. `-trend` (case 4) and `-xlink` are refused, as `-f` refuses
them.

## The bootstrap: workers, seed and engine

Both bootstraps -- the rank test's and `fit(irfboot=N)`'s -- take:

- **`workers`**: the size of the process pool that refits the replications.
  `None` (the default) is `DRVEC_WORKERS` when set, else the cores less one;
  `1` runs in the calling process, as before. **The numbers do not depend on
  it.** The B samples are drawn first, in sequence, so the generator is
  consumed in the C's order; the refits draw no random numbers (verified in
  `rank.py`, `irfboot.py` and the fit path: the only generator is
  `simulate_h0`'s), and each is fitted on fresh data. `workers=1` and
  `workers=4` give the same `.out`, `.boot` and `.irfboot`, byte for byte
  (`tests/test_parallel.py`). In the rank test the samples of every comparison are
  drawn first and all the refits share one pool. Under spawn (macOS, Windows) the calling script
  needs the usual `if __name__ == "__main__":` guard.
- **`seed`**: the generator's seed. `None` is the C's (987654321), so parity
  with the C is untouched; another seed is a Monte Carlo check of the same
  bootstrap and always runs in Python.
- **`engine`**: `"auto"`, `"python"` or `"c"`; `None` reads `DRVEC_ENGINE`
  and defaults to `"auto"`: drvec C when a matching binary is found, else
  the parallel Python. `"c"` raises `CEngineUnavailable` with the reason when
  it cannot run. Only the bootstrap goes to the C; without one, everything is
  Python.
- **`folder`**: where the C runs and leaves its files (`drvecjob.out`,
  `.boot`, `.irfboot`, `.stdout`, `.stderr`); `None` is a temporary folder,
  removed afterwards.

## drvec C as the bootstrap engine

`drvec.c_engine` runs the C program for the bootstraps and reads its files
back into the same `RankTest` and `FitResult` the Python engine returns.

- **The binary**, found in this order: `DRVEC_BIN`; the binary bundled in the
  wheel at `drvec/_bin/drvec` (`drvec.exe`; see below); `drvec` on the
  `PATH`; the ATSW install directory (Linux `/usr/lib/atsw/bin`,
  `/usr/local/lib/atsw/bin`; macOS `ATSW.app/Contents/Resources/bin` in
  `/Applications` or `~/Applications`; Windows
  `%LOCALAPPDATA%\Programs\ATSW\bin`, `%ProgramFiles%\ATSW\bin`). The first
  one found is used, and its banner (`drvec --version`) must say **0.13**, the
  version the port mirrors; otherwise it is refused, with the reason.
  `c_engine.find()` returns it (path, version, where it was found) or raises.
- **The translation**: `c_engine.rank_argv(...)` and `c_engine.fit_argv(...)`
  turn a call into the C's arguments, the inverse of `drvec-py`'s parser.
  The refusals happen before, in the port, with the same words.
- **The rank test**: the port fits the observed ranks while the C runs; the
  C's `.boot` gives every replication (LR* to 17 digits, which round-trip
  exactly), and the percentiles and p-value are computed by the port's code.
  `progress` is called as the C writes the `.boot` (in its buffer's steps).
- **The IRF bands**: the port fits the model; the C's `.irfboot` and its
  `.out`'s band section are carried verbatim; `irf_bands()` reads the file.
- **Parity**: on Linux x86_64 the two engines give the same `.out` and `.boot`
  bytes on the suite's rank bootstraps and the same `.out` and `.irfboot` on
  its `-irfboot` case (`tests/test_c_engine.py`). One exception is the C's
  defect BUG-60: with a drift held, one replication of `cprboot` differs (the
  C engine gives the oracle's `.boot`, the port its own); the `.out` is the
  same.

### The bundled binary

The platform wheels carry drvec C 0.13 as package data, `drvec/_bin/drvec`
(`drvec.exe` on Windows): Linux x86_64 and aarch64 (manylinux_2_28), macOS
arm64 (14.0) and Windows AMD64. On Windows that needs atsw-gui's BUG-61 fix
(`bugs/drvec-c-rng-long32`): built from ae95446, the bootstrap never ends
there. The banner says 0.13 either way, so `c_engine.find()` cannot refuse
the old build; the bundled one is the fixed one. It is built from `csrc/`,
atsw-gui 7ec4bf7's
sources byte for byte (`csrc/VENDORED.json`, `tests/test_vendored.py`), by
`csrc/build_drvec.py`: engines/drvec/Makefile's eleven units with `-O2
-Wall`, GSL linked statically, so it needs only the C library. The wheel is
tagged `py3-none-<platform>`: an executable has no CPython ABI. `provenance`
says `extra["source"] == "bundled"` when it ran.

- The pure wheel (`py3-none-any`) has none: the C is then found only through
  `DRVEC_BIN`, the PATH or ATSW, else the bootstraps run in Python.
- An sdist install builds it when a C compiler and GSL are found, and warns
  and goes without it otherwise. Variables: `DRVEC_NO_C=1` (do not build),
  `DRVEC_REQUIRE_C=1` (a failed build is an error), `DRVEC_CC`,
  `DRVEC_GSL_PREFIX` (else `GSL_ROOT`, pkg-config, gsl-config),
  `DRVEC_GSL_STATIC` (`1` requires the static archives, `0` links the shared
  library). On Windows the compiler must be a MinGW-w64 gcc.
- From a checkout, `python csrc/build_drvec.py` puts it at
  `src/drvec/_bin/drvec`, where the source tree finds it.
- On Linux x86_64 it writes the oracle's `.out`, `.boot`, `.forecast`,
  `.recursive` and `.irfboot` byte for byte on nineteen of the oracle's
  commands (`tests/test_bundled.py`); off it, it is that platform's C, with
  the tolerance tier.

## Provenance

Every `FitResult` and `RankTest` carries `provenance`, a `Provenance` with
`engine` (python or c), `version` (drvec-python's, or the C binary's),
`path`, `backend` (drvarma's engine and the eigensolver; or "c"), `platform`,
`tier` ("exact" on Linux x86_64, "tolerance" elsewhere), `workers`, `seed` and
`note` (why that engine: e.g. "engine='auto', Python: drvec C not found ...");
for the C, `extra` holds where the binary was found, the folder, the argv and
the C's wall time.

# The port: what drvec-python reproduces of the C, and how

Oracle: atsw-gui tag `drvec-v0.12`, directory `engines/drvec` (and `lib/optim`).
File:line references below are to that tag; `drvec.c` means
`engines/drvec/src/drvec.c`. The homologation target is the oracle's output on
the two milestone runs (tests/data/oracle):

```
drvec mmpre.muskrat.pre mmpre.mink.pre 2 1 1 -mean -case 2 -name proute
drvec mmpre.muskrat.pre mmpre.mink.pre 2 0 1 -case 2 -lrtest -bootstrap 20 -name rankboot
```

## 1. The result, in one paragraph

On these runs the port is not "within tolerance" of the C: it is **the same
computation**. The cast plus drvarma's compiled `elf` gives the C's logL at every
one of its optima with difference 0.0; the cold start, the 24 est() calls of case 1
(in the C's order, each with the C's start and the C's final vector), the ladder,
the entry gate (gap -7.567e-12), the route-(B) profile, the search table, the BFGS
standard errors, the bootstrap generator and the simulated data are bit-identical.
Johansen's canonical seed is too when libgsl is on the system (§6, decision D2);
with scipy's eigensolver instead it is a few ulps off, which moves the `johansen`
fits by up to 9.5e-8 in logL and two of the twenty LR* of the bootstrap by ~1e-13
here, and by much more on E1 (§6). All of this holds on Linux x86_64, the oracle's
platform; elsewhere the compiled likelihood rounds differently (tests/conftest.py).

## 2. The model, the vector and the two row orders

```
F(L) nabla Y_t = -Lambda (B'Y_{t-1} - E[W]) + Theta(L) A_t,   A_t ~ N(0, sigma2 Q)
B = [I_r ; B2],  W_t = Y1_t + B2'Y2_t,  Ybar_t = [nabla Y2_t ; W_t]
```

Johansen's alpha = -Lambda (printed so), beta_2 = B2, Gamma_k = F_k. The command
line's `p q r`: p is the AR order on Ybar (F_1..F_{p-1} are free), q the MA order,
r the rank; the first s = M - r series are the nabla-Y2 block, the last r the Y1
block (muskrat is Y2, mink Y1). `-mean` only means "case 2 unless a case is given"
(drvec.c:10566-10576). The default MA class is FREE (drvec.c:317, 10577-10597;
`docs/ARCHITECTURE.md` §4 of the C is stale on this).

**Two row orders** (the BUG-17 family). Lambda's rows, F's, Theta's and Sigma's
rows and columns are INTERNAL, [Y1 ; Y2] (so Q[1][1] = 1 is mink's variance); the
data, names, mu, B2 and w are in the .inp order [Y2 ; Y1]. `inp2lam(i) = i <= s ?
r + i : i - s` (2841), `lam2inp` (2850); it depends on r. In the port:
`params.RowOrder`, and `params.slot_labels` names every slot in the .inp's names.

**The vector** (`par_blocks` 881, walked by `vec_shootx` 4439-4583): mean (case 2:
E[W], r values) | Lambda row-major | F_k row-major | Theta_k in the class pattern
(free, diag, marow = rows 1..r, matri = rows 1..r plus the lower-right block) | Q:
Q[i][i] for i >= 2 then Q[i][j], j < i | B2 column-major. No transformation of any
block; Q[1][1] = 1 fixed (the scale is concentrated). In the port: `params.py`,
the one walker every module uses.

## 3. The cast (`vec_shootx`, drvec.c:4303-4790) — `cast.py`

```
Cbar = [0 I_s ; I_r B2']   Cinv = [-B2' I_r ; I_s 0]   Hbar = [0 0 ; 0 I_r]   LamBar = [0, Lambda]
PhiBar_1 = Cinv Hbar - LamBar + F_1 Cinv;  PhiBar_k = F_k Cinv - F_{k-1} Cinv Hbar;  PhiBar_p = -F_{p-1} Cinv Hbar
phi*_k = Cbar PhiBar_k,  theta*_k = Cbar Theta_k Cinv,  qq = Cbar Sigma Cbar',  mu = [0 ; E[W]]
w_t = [nabla Y2_t ; Y1_t + B2'Y2_t]
```

Sigma failing `choldcp` (the modified Cholesky, nlatools.c:124) refuses the point.
The products keep `matrix_multiply`'s loop order (nlatools.c:1068) and W is built
column by column in the C's order of additions: that is what makes the
trajectories bit-identical, at ~90-130 us per evaluation.

## 4. The likelihood — `loglik.py`

drvarma's `elf_c` (AS 311, the same engine as the C's elfvarma.c) with sigma2 = 1,
xitol = +1e-3 (truncated; -1e-3 only in the gate), and the C's concentrated logL
(drvmlest.c:139-142) with **LOG2PI = 1.837877066** (truncated, not log 2 pi).

## 5. The fitting order — `fit.py`, `rank.py`

* `est()` (drvmlest.c:33-145): objective (f1/f1_0)^m (f2/f2_0), 1.0 when refused;
  drvarma's `_qnewt.raxopt` with 500 iterations, gradtol 1e-5, steptol 1e-7
  (drvec.c:2898); the covariance from the BFGS factor (`stderr.py`); `est_run`'s
  checks on the final point (drvec.c:5020-5060).
* `make_admissible` (4919): F and Theta shrunk, Lambda halved, until elf accepts.
* `fit_search` (5079): seed, cold (`init_guess`), johansen (r >= 1), nested (the
  class below fitted by its own search from `init_guess`, embedded); the winner by
  strictly greater logL.
* `fit_r0_ladder` (5345): rung 0 (F, Theta, Sigma diagonal; the gate after it),
  rung 1 (Sigma free), rung 2 (the requested structure), each from the one below;
  cached per data set. `ladder_seed_r0` (5443) at r = 0; `gate_profile_seed`
  (5508) at r >= 1: the rung-2 optimum permuted to the internal order (BUG-32),
  Lambda scaled by the best admissible of 18 multipliers, then only mean, Lambda and
  B2 estimated; the result is the seed.
* The gate (`gate_contract`, 3014): the rung-0 fit against the sum of its own
  univariate components, both untruncated, |gap| < 1e-6. It does NOT compare with
  the `.pre` models: this route does not read their ARMA (§7).
* The rank test (`run_lrtest`, 9516): every rank r = 0..M-1 fitted as above,
  LR = 2[L(r+1) - L(r)], Osterwald-Lenum / MHM tables (161-172). The bootstrap
  (`bootstrap_rank` 2356, `simulate_h0` 2226): H0 = the fitted rank-r model in Ybar
  coordinates, burn-in 50 + 10p, the levels rebuilt from the observed first row,
  both ranks refitted exactly as the observed data; quantiles `stat[ceil(a nok)]`,
  p = (#{LR* >= LR} + 1)/(nok + 1). The generator (`rng.py`, 2203-2223): 64-bit
  LCG with the PCG constants, Box-Muller with a cached spare, seed 987654321,
  global for the whole run.

Case 1 makes 24 est() calls (the ladder 11, the profile 1, the r = 1 search 12 with
its nested chain); the rank test 11 per data set. On case 1 the four starts end
between -1.50 and 6.48 and the winner (`cold`, 66 iterations, steptol) sits on the
MA invertibility wall (root modulus 0.99995, BUG-49): only a trajectory-faithful
port lands there. On case 2 different starts end 1e-10 apart in logL and 1e-4 apart
in the parameters: the estimates are only reproduced if the same start wins.

## 6. Decisions of milestone 1

* **D1. drvec's own ladder.** The rungs, the profile and the four-start search are
  the C's; drvarma's `Ladder` class is not used (its rung 0 is the `.pre` models,
  drvec's is re-estimated). Reused from drvarma: `elf_c`, `_qnewt.raxopt`,
  `_qnewt.cholsol`, all matched line by line against qnewtopt.c and lib/optim/lnsrch.c.
* **D2. Johansen's eigenproblem** -- revised in milestone 2: GSL's own
  `gsl_eigen_gensymmv` through ctypes when libgsl is on the system (`_gsl.py`,
  `DRVEC_EIGEN=auto|gsl|scipy`), scipy's `eigh` otherwise. Why: on the three-series
  case (Lutkepohl's E1) scipy's few ulps moved the `johansen` start's optimum by 0.47
  in logL and one bootstrap LR* by 3.5e-5; with GSL the case is byte-identical
  (the search table, all 40 replications). The milestone-1 measurement with scipy: Measured: the
  canonical B2 is 0.41947218341803055 against the C's 0.4194721834180301; on case 1
  the `johansen` start ends 9.5e-8 away in logL (a tc-3 stop) and does not win; on
  case 2 it ends 6e-14 away; in the bootstrap it wins in some replications and two
  LR* (reps 6 and 13) differ by 1.1e-13 and 1.7e-13. No winner changes, no
  tolerance is approached; `rankboot.boot` is identical in 18 of 20 lines
  (tests/test_johansen.py, tests/test_rank.py).
* **D3. `nrerror`.** The line search's `disc < 0` branch exits the C program
  (`nrerror`, nlatools.c:367). drvarma's `_qnewt` raises RuntimeError; `fit.est`
  turns it into `drvec.LineSearchRoundoff` and nothing in drvec catches it, so it
  ends `fit()`/`rank_test()` with no partial result, which is what the C's exit
  does to its run. What cannot be mirrored is the process exit itself (status 1,
  the .out left truncated): a library raises instead. It does not fire on the
  milestone's cases.
* **D4. The bootstrap stream** is the 64-bit one (the oracle's). On Windows the C's
  `unsigned long` is 32-bit and its stream differs (TODO.md).

## 7. The `.pre` route — `data.py`

Each file is read with the C's own reader, transcribed (`prefile.py`:
read_fue_pre's fgets/fscanf grammar, gen_detvar, build_det_component; since
milestone 5, §13; before it, `fue.load()`). The series is taken in w units,
refactor * BoxCox(z) (`lambda = 1` is z * refactor) MINUS its deterministic
component, in levels; the files are lined up by DATE on the intersection of
their samples (6810-6840; drvarma's ladder requires a common last date
instead). Not read: the ARMA (but for an -xpre input, §13). Refused, where the
C carries on: D > 0 (the C ignores it).

## 8. The .out writer — `report.py`, `diagnose.py`

The aim is byte identity with the oracle after tools/golden.sh's filter (the
`Program : DRVEC` line and the `.boot` banner dropped, `/tmp` paths removed). On
Linux x86_64 with libgsl, **every line** of the four milestone outputs is
byte-identical: `proute.out` (805 lines), `rankboot.out`, `e1fit.out` (1290),
`e1boot.out`, and the `.boot` files `rankboot.boot` (20 of 20 replications) and
`e1boot.boot` (40 of 40). The first line names the port (`DRVEC <version>
(python)`), which is the line the filter drops.

What it takes, beyond the numbers being bit-identical:

* the C's own `chisq()` (nlatools.c:772-885, Wilson-Hilferty above 30 df) for the
  p-values of Hosking's portmanteau and Jarque-Bera: scipy's exact tail prints
  0.7452 where the C prints 0.7456 (`nla.chisq`);
* GSL's `gsl_eigen_nonsymm` for the eigenvalues of Pi and the operator roots,
  because the ORDER in which they are printed is GSL's (`_gsl.eigen_nonsymm`);
* choldcp's determinant in mantissa and exponent for `|Sigma|` (`nla.choldcp_det`);
* the transcription of diagnose.c's ASCII plot, histogram, ACF/PACF and CCF,
  with `round_local` and the C's string surgery.

Where the port computes the same quantity another way and only the printed
digits are compared (all byte-identical on the four outputs): the Wald tests (a
pseudo-inverse by numpy's SVD, where the C uses svdcp), their p-values (scipy's
chi2 where the C calls `gsl_cdf_chisq_Q`), `par_row`'s p-values (erfc for GSL's
ugaussian_Q), Granger's rank condition (numpy's QR and SVD; the smallest singular
value does not depend on the basis of Lambda_perp), the null vector of the MA
boundary block (numpy's SVD; printed in absolute value), Hosking's C0 inverse and
Jarque-Bera's S^(-1/2) (numpy). A p-value or a fourth decimal could round the
other way on another data set; none does on these four.

Without libgsl (scipy's eigh for Johansen, numpy's eigvals for Pi and the
roots): the losing `johansen` row of a search table can differ, and the order
of printed eigenvalues may differ. The `.boot` replications where the Johansen
start wins move in the last digits (2 of 20 on rankboot, ~1e-13; on E1 up to
3.5e-5): the tests then compare the LR* numerically and the line count of the
.out files.

The same holds for every remaining PRE_CASE of golden.sh: `fixb2row.out`,
`nodrift.out`, `casepre.out`, `trend.out`, `xsys.out`, `xlink.out`,
`irfboot.out` and `irfboot.irfboot` are byte-identical (tests/test_options.py).

Milestone 3 adds the blocks of `-fdhess`, `-alpha`/`-weakex`, `-fixb2`, `-f`,
`-estwin`, `-multistart` and `-rungs` (§11). Not written yet: `-specs` and the
`-warma` blocks.

## 9. The options of milestone 2

* `-fixb2row i v` (`Spec.b2_hold`): the held rows are out of the vector and in
  the cast; the cold start sets them BEFORE W is built (BUG-33); the free model
  is fitted for the LR as the C does (cold start and search, no ladder).
* `-nodrift i`, `-case pre` (`Spec.drift`): a mask on case 3's drifts by series;
  `-case pre` reads each .pre's mean flag (the flag of its mean line), a mean held at
  a non-zero value goes free; the ladder is keyed on the mask.
* `-trend` (`Spec.trend`): r slopes after the means; W detrended on centred t in
  the cold start, `w - dtr (t - (n+1)/2)` in the cast.
* `-xsys i` (`Spec.xsys`): alpha = A psi (A the identity without the inputs'
  columns) and the inputs' rows of Gamma held at zero through `Spec.f_free`; psi
  seeded by projecting the conditional regression; the free VEC fitted with its
  ladder for the LR. q > 0 refused, as the C does.
* `-xpre/-xlink` (`xlink.py`): see its docstring. The VEC without the inputs is
  fitted first and its optimum with omega = 0 starts the search (the bridge).
* `-irfboot N` (`irfboot.py`): the rank bootstrap's generator from its seed, each
  replication refitted with the ladder start.

## 10. Not yet ported

`-specs` (it fits `-warma` and `-mawarma`, neither ported), `-warma`/`-mawarma`,
`-m 2`, `-matest`/`-artest`, `-eval`, the `-seed*` family, `-noladder`, `-rankadm`,
`-writeinp`/`-writeres`; the rank test with `-alpha`/`-weakex`/`-fixb2` or the
diagonal flags (with `-nodrift`/`-case pre` since milestone 4, §12); the .inp
route. drvec-py refuses each of them
by name (exit 2). The -lrtest internal rescaling is ported but not exercised (no
fixture triggers it).

## 11. Milestone 3

* **`-fdhess`** (`fdhess.py`; exact_hessian_se and fdh_obj, drvec.c:1877-2007).
  The objective normalised at the optimum, drvarma's `_qnewt.fdhess` (qnewtopt.c:404
  line for line) with eta = macheps, choldcp, cov = 2 f H^-1 / n. A refused step
  (the cast's Sigma, or AS 311) counts as 1e10 and as a rejection; any rejection
  keeps the BFGS s.e. ("the optimum is on the boundary"), and so does an
  indefinite Hessian; the .out and the stderr warning say which (BUG-34). The
  eigenvalues of the warning are numpy's, where the C uses eigenqr (they are
  only counted). **The s.e. are path-independent but not bit-stable**: AS 311's
  objective carries ~1e-14 relative noise (an ulp in x moves g by ~3e-14 on
  E1), the step is macheps^(1/3) max(|x|, 1) ~ 6e-6, and the second difference
  divides by its square: an s.e. moves by up to 4.5e-3 relative (E1) and
  1.3e-3 (mink-muskrat) for an ulp in x. Off the oracle's platform the compiled
  likelihood rounds differently, the same kind of perturbation, so the tests'
  tier is relative, TOL_FDSE_REL = 2e-2 (tests/test_fdhess.py, test_noise_floor).
* **`-alpha FILE`, `-weakex i`** (`Spec.alpha`, `api.load_alpha_A`): alpha = A psi
  through the machinery -xsys already had (A in the .inp's rows, permuted with
  lam2inp where it meets Lambda; the psi seed projected, drvec.c:4122). H(r) for
  the LR is ONE `est()` from init_guess, no search, no ladder, and its optimiser
  banner goes into the .out right after the header (drvec.c:11415-11439); df =
  (M - sa) r. With `-fixb2row` the free model drops both restrictions (init_guess
  and the search) and the LR is the joint one, df = (rows held) r + (M - sa) r.
  **Port difference**: the C checks A's rank with choldcp(A'A), which accepts an
  exactly singular A'A (a zero pivot is replaced, not refused) and then fits NaN
  and crashes (bugs/drvec-c-alpha-rank); the port adds numpy's matrix_rank and
  refuses with the C's own message.
* **`-fixb2 [v]`** (`Spec.b2_fix`, `fixb2_given`): B2 held whole. With v the seed
  is built from W = Y1 + v'Y2 (BUG-33) and the search adds the `ols-b2` start
  (init_guess with the static-OLS W, `consistent=False`, drvec.c:5143); without
  a value B2 is held at its static-OLS estimate (B2_fixed, which init_guess sets
  as a side effect in the C). No `johansen` start under -fixb2 (5151). No free
  model is fitted: the C prints no LR for -fixb2.
* **`-f H`** (`forecast.py`; forecast_core 5935, forecast_vec 6456): the recursion
  on Ybar from the origin with the shocks after it at zero, the levels through
  B2, the bands through the psi weights and level_error_map, the one-step
  self-check, and the back-transform to each series' units (bt_level/bt_jac;
  the det path is each file's deterministic component since milestone 5, §13).
  The `.forecast` file's first line names the port and its version.
* **`-estwin E`** (`Data.estwin`, rolling_eval 5990): the fit on rows 1..E of
  nabla Y; the rolling evaluation filters the WHOLE sample with those parameters
  and rolls the origin over E..n-H; `<base>.recursive` (or -C FILE). The C's
  header dates the first observation as nobs_raw - nobs + 1 (1862 for a window
  on 1851..1900), and the per-series diagnosis is offset the same way; the port
  does the same.
* **`-multistart n`** (fit_search's step 5, 5202): n - 1 jitters of the caller's
  seed from the C's LCG, 64-bit (as `unsigned long` on Linux). In the rank test
  every observed rank gets them, the bootstrap replications do not (BUG-25).
* **`-rungs`** (`rungtable.py`, run_rungs 9047): the three rungs at r = 0 with
  the gate, which is `fit_r0_ladder` with every rung forced.
* **The CLI** (`cli.py`, `drvec-py`): parse_cli and validate_cli's checks and
  messages for the .pre route, the refusals in the C's order and words, the
  products named as the C names them. The terminal output is a summary, not
  the C's progress log.

Every milestone-3 case (tests/data/oracle/SOURCE) is byte-identical to the
oracle on Linux x86_64 with GSL: the .out, and the .forecast/.recursive but for
their first line.

## 12. Milestone 4

* **`-lrtest` with `-case pre` / `-nodrift i`** (`api.rank_test(case="pre",
  nodrift=)`). The C's mask is global (`nodrift[]`, drvec.c:779-798), so every
  rank of the test and every bootstrap replication is fitted with it: in the
  port the rank test's Spec carries `drift`, which fit_rank, simulate_h0 and
  the replications already honoured. The header says "(drifts as in the .pre
  files)" / "(some drifts held by -nodrift)" only when a drift is held (the
  E1 .pre files all free their mean: -case pre is plain case 3 there); the
  critical values are case 3's, "NOT TABULATED HERE"; the CLI writes the C's
  warning (drvec.c:10701) on stderr. -nodrift was the same mask and came for
  free.
* **Port difference, the ladder cache** (C defect, bugs/drvec-c-ladder-key):
  the C caches the r = 0 ladder in one slot keyed by a double that, with a
  drift held (+1e12 per held drift), rounds the data's contribution away; a
  bootstrap replication whose key equals the previous one's reuses that
  replication's ladder. The port keys the cache on the data object (one per
  replication, `Data.with_raw`), so it fits every replication's own ladder:
  cprboot.boot differs in replication 16 by 8e-11, and the .out is identical.
  Emulating the C's key would byte-match it (the repro and
  tests/test_rank_drift.py show the stale ladder reproduces the C's LR* bit for
  bit); not done, a decision for the lead.
* **The entry gate of a plain fit** (`FitResult.gate`, report._coint_block).
  At r = 0 with -diagar -diagma -diagcov, report_fit calls gate_contract after
  the Granger block (drvec.c:8903); main() has evaluated the logL at the
  start fit_search receives (drvec.c:11571-11595, truncated xi, concentrated),
  so the gate adds the optimality certificate: that logL, the fit's, their gap,
  and AN OPTIMUM / A SPECIFICATION at 1e-3. On this route the start is
  init_guess's (the ladder is the model itself at the diagonal rung and is
  not run, ladder_seed_r0), so the verdict is always A SPECIFICATION. The
  "largest coefficient movement" line needs load_seed_pre's univariate seeds
  (the .inp route's) and is never written here. In the rank test and -rungs
  the gate has no certificate, as in the C (gate_have_start is only set by
  the plain fit).

Every milestone-4 case (tests/data/oracle/SOURCE) is byte-identical to the
oracle on Linux x86_64 with GSL, but for cprboot.boot's replication 16.

## 13. Milestone 5

* **Deterministic variables in a .pre** (`prefile.py`, `data.py`). BUG-40 is
  closed WON'T FIX (atsw-gui engines/drvec/docs/BUGS.md, 2026-09-24): drvec
  keeps the EXACT rational filter omega(B)/delta(B) and the port does what the
  C does. read_pre_inputs (drvec.c:6862-6893) calls build_det_component
  (fue_pre_reader.c:367) on each FILE's own sample: every variable is
  regenerated from its spec line by gen_detvar on the file's calendar (t = 1 is
  its first observation; impulse, compimp, step, ramp take `year` on annual
  data and `sub year` otherwise; cos/sin r are cos(2 pi r t / freq); alter is
  (-1)^t; trend/time is t; easter is monthly only), filtered by omega(B) =
  w0 - w1 B - ... and 1/delta(B) = 1/(1 - d1 B - ...) from zero initial
  conditions at the file's start, never truncated, and summed. The rows before
  the common sample are skipped AFTER filtering, so the date, not the
  position, places each term. w - det is what everything downstream sees (the
  ladder, the fits, the rank test and its bootstrap, -xsys).
  An unknown type is refused with the reader's message (the C exits 1).
* **Forecasts in levels** (`forecast.py`): bt_det, the same component on the
  common sample and over the horizon (built over off + nobs_raw + H, drvec.c:
  6876-6882: a step or a trend goes on), is added back before the Box-Cox
  inverse in the .out's table, `<base>.forecast` and the rolling evaluation of
  -estwin (`Data.det_path`); the delta-method s.e. at the same point.
* **-xpre inputs** (`xlink.py`): w - det of the input on its own sample
  (drvec.c:7041-7051), and the presample backcast with its ARMA (xl_backcast,
  drvec.c:6965, drtran's backcast_arma): the factors multiplied out as
  expand_ar_factors/expand_ma_factors do (regular, annual at multiples of the
  period, fixed-frequency AR(2)/MA(2) with c1 = 2 cos(2 pi f / s) sqrt(-c2)),
  the mean's value whatever its flag, on the levels (d = 0) or on the
  differences and cumulated backwards (d = 1). An input with seasonal
  differencing or ifadf factors is refused, as the C refuses it.
* The reader is the C's, not fue's: they agree on every fixture (series,
  calendar, lambda, mean flag, omegas and deltas: tests/test_m5.py), and where
  they could differ the C's wins (a `time` variable, an empty name, an
  annual dated term written `sub year`).

Every milestone-5 case (tests/data/oracle/SOURCE) is byte-identical to the
oracle on Linux x86_64 with GSL: the .out, the .boot, and the .forecast and
.recursive but for their first line.

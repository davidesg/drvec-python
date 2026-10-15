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
The one exception is Johansen's canonical seed (§6, decision D2): it is a few ulps
off, which moves the `johansen` fits by up to 9.5e-8 in logL and two of the twenty
LR* of the bootstrap by ~1e-13.

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
* **D2. Johansen's eigenproblem with scipy**, not GSL through ctypes. Measured: the
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

Each file is read with `fue.load()`; the series is taken in w units, refactor *
BoxCox(z) (`lambda = 1` is z * refactor), in levels; the files are lined up by
DATE on the intersection of their samples (6810-6840; drvarma's ladder requires a
common last date instead). Not read: the ARMA, the mean. Refused, where the C
carries on: deterministic variables (BUG-40 pending) and D > 0 (the C ignores it).

## 8. Not yet ported

The .out writer beyond the numbers the tests read, the CLI, `-fdhess`, the
diagnostics, IRF/FEVD, forecasting, every option but `-case 1|2|3` and the MA
classes the search needs, `-multistart`, the -lrtest internal rescaling is ported
but not exercised (no fixture triggers it).

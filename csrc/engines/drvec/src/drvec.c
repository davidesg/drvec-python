/*****************************************************************************/
/*  drvec.c — Exact ML estimation of VEC models (Mauricio 2006).              */
/*                                                                             */
/*  Implements the transformation described in:                               */
/*    Mauricio, J.A. (2006), "Exact maximum likelihood estimation of          */
/*    partially nonstationary vector ARMA models",                           */
/*    Computational Statistics & Data Analysis, 50, 3644-3662.               */
/*                                                                             */
/*  The stationary representation Ȳ_t = (∇Y_{2t}', W_t')' follows a          */
/*  standard VARMA(p,q), estimated by Mauricio's EML engine.                 */
/*  The VEC parameters are recovered after estimation.                        */
/*                                                                             */
/*  Usage:  drvec file p q r [-mean] [-case 1|2|3] [-diagar] [-diagma]       */
/*                          [-diagcov] [-m 1|2] [-differenced] [-fixb2]      */
/*                          [-lrtest] [-rungs]                               */
/*                                                                             */
/*    file   : data file name (without .inp extension)                        */
/*    p      : AR order of the stationary VARMA on Ȳ_t                        */
/*    q      : MA order                                                       */
/*    r      : cointegration rank (0 <= r < M; ignored with -lrtest)          */
/*                                                                             */
/*  Copyright (C) 1995-2026  J.A. Mauricio, A.B. Treadway & D.E. Guerrero.   */
/*  GPL v2 or later.                                                           */
/*****************************************************************************/

/*****************************************************************************/
/*  WHERE THINGS ARE IN THIS FILE                                            */
/*                                                                           */
/*  Eight thousand lines is a lot to read from the top, and until 2026-08-23  */
/*  that is what finding anything here required: main() was 2320 lines and    */
/*  carried the argument parsing, five whole modes, the multi-start and the   */
/*  entire report at one indentation.  P8 cut it into the phases it always    */
/*  had; this map says where they went.  main() is now 468 lines and reads    */
/*  as the sequence it is.                                                   */
/*                                                                           */
/*  THE MODEL AND ITS PARAMETER VECTOR                                       */
/*    par_blocks, calc_nparametrs   the layout of x[], in ONE place           */
/*    init_guess                    where the fit starts, from the data      */
/*    vec_shootx                    x[] -> the VARMA the engine estimates    */
/*    warma_inverse                 and back, for the -warma coordinates     */
/*    build_y2_levels, build_ybar   the data, differenced and transformed    */
/*                                                                           */
/*  WHAT THE FIT IS CHECKED AGAINST                                          */
/*    granger_smin                  the rank condition (Theorem 3)           */
/*    operator_roots                the AR/MA roots and the boundary         */
/*    residual_diagnostics          Hosking, Jarque-Bera, the R(k)           */
/*    gate_contract                 the ladder's two contracts               */
/*    exact_hessian_se              standard errors at the optimum (-fdhess) */
/*                                                                           */
/*  INFERENCE                                                                */
/*    wald_sub, emit_wald,          the hypotheses a VEC answers, by default */
/*      hypothesis_block                                                     */
/*    bootstrap_rank, bootstrap_ma  the distributions that are simulated     */
/*      simulate_h0, fit_ll           because they are not chi2             */
/*    load_alpha_A, build_weakex_A  alpha = A*psi, and weak exogeneity       */
/*                                                                           */
/*  FORECASTING (P5)                                                         */
/*    compute_psi_weights           the MA(inf) weights -- drvarma's, copied */
/*    level_error_map               innovations -> level error, ONE map      */
/*    forecast_core, forecast_vec   the recursion, and <base>.forecast       */
/*    rolling_eval                  out of sample, and <base>.recursive      */
/*                                                                           */
/*  THE BRIDGE TO THE SUITE (F2, P9)                                         */
/*    read_inp_input                the .inp route                            */
/*    read_pre_inputs               the .pre route: one fue model per series */
/*    load_seed_pre, pre_univariate what a .pre can and cannot seed          */
/*    write_component_inps,         what drvec writes for fue                */
/*      write_resid_inps, write_inp_series                                   */
/*    subtract_interventions        the deterministic terms, on the .inp route*/
/*                                                                           */
/*  THE COMMAND LINE (P1)                                                    */
/*    OPT_TABLE, validate_cli       what exists, checked before anything runs*/
/*    parse_cli                     and what it sets                          */
/*    usage, usage_option_list      the same table, printed                  */
/*                                                                           */
/*  THE MODES -- each estimates, prints and exits, so each is a function     */
/*    run_rungs        -rungs      the ladder below the rank                 */
/*    run_specs        -specs      the specification ladder                  */
/*    run_lrtest       -lrtest     the sequential rank test                  */
/*    run_ma_ar_test   -matest/-artest   the two bootstrapped comparisons    */
/*    run_eval         -eval       the likelihood at the starting point      */
/*    fit_search       the best of several starts; -multistart adds more    */
/*                                                                           */
/*  AND THE REPORT                                                           */
/*    report_fit                    everything the .out says about a fit     */
/*                                                                           */
/*  WHAT IS STILL GLOBAL, and why it is not an oversight: the option flags    */
/*  (what the user asked for), the data (rawmat, datamat, Y2_levels, and the  */
/*  calendar), and the three file names.  Every one of them is read by more   */
/*  than one of the functions above and written by exactly one.  Threading    */
/*  them through six signatures would say less than declaring them once, and  */
/*  would be a large diff whose only effect is a larger diff.                 */
/*****************************************************************************/

#include "main.h"
#include "fue_pre_reader.h"   /* .pre reader, copied from drtran (see F2.1) */
#include "fue_bridge.h"       /* expansion of the .pre's factors                */
#include "diagmv.h"           /* Hosking and Jarque-Bera, shared (lib/diagmv)   */
#include <gsl/gsl_cdf.h>      /* chi2 p-value of the LR of H1(r) against H(r) */
#include <gsl/gsl_eigen.h>    /* symmetric generalised eigenvalue problem */
#include <gsl/gsl_linalg.h>   /* QR and SVD for the Granger rank condition       */
#include <gsl/gsl_errno.h>     /* P12: errors returned, not abort()ed        */
#include <stdarg.h>           /* bad_cli: the usage message takes a format      */
#include <errno.h>            /* strtol/strtod: ERANGE                         */
#include <stdint.h>           /* uint64_t: the generators' state (BUG-61)      */
#ifdef _WIN32                 /* P12: the search silences the optimiser's      */
#include <io.h>               /* per-iteration console trace while it tries    */
#define dup  _dup             /* its candidate starts                          */
#define dup2 _dup2
#define fileno _fileno
#else
#include <unistd.h>
#endif

/*  THE VERSION.  The number lives here; the reasoning behind it lives in
 *  docs/VERSIONS.md, which is also where the release policy and the rule for
 *  moving this number are written.  A source file is the wrong place for an
 *  argument that a reader needs before running the program, and the right
 *  place for the single definition the binary is built from.                 */
#ifndef DRVEC_VERSION
#define DRVEC_VERSION "0.13"
#endif

real macheps;
FILE *outputv;
int quiet_mode = 0;

real **datamat;     /* data as the estimation needs it: ∇Y_{2t} (cols 1..s)
                       and Y_{1t} in levels (cols s+1..M).  Derived from
                       rawmat by build_y2_levels().                     */
int  nser, nobs;

/* Levels of Y_{2t}, aligned row-for-row with datamat.  W_t = Y_{1t} + B2'Y_{2t}
   needs them, and they do NOT depend on any parameter, so they are built once
   here instead of being rebuilt on every likelihood evaluation.
   By default they are the true levels read from the .inp; with -differenced
   they are reconstructed by cumulating from an arbitrary zero origin, which is
   the legacy behaviour and is wrong for case 1 (see build_y2_levels).         */
real **Y2_levels = NULL;
int    global_estwin = 0;   /* -estwin E: fit on 1..E and roll the origin (P5.2) */
int    nobs_full     = 0;   /* observations available, before trimming             */
int  global_levels = 1;   /* default: .inp carries every series in LEVELS.
                             -differenced selects the legacy layout, where
                             cols 1..s arrive already differenced.            */

/* Data exactly as read from the .inp.  datamat and Y2_levels are derived from
   it, and the derivation depends on r (the column split is s = M - r), so with
   -lrtest they are rebuilt for every candidate rank.                          */
real **rawmat = NULL;
int  nobs_raw = 0;

/* Asymptotic critical values for the sequential (lambda-max type) rank test,
   indexed by the number of common trends M-r = 1..11, at 10%, 5% and 1%.
   Non-standard Johansen distribution; MA terms do not affect it (Yap and
   Reinsel, 1995, Theorem 3), as noted in Mauricio (2006), Remark 5.
   Case 1 has NO deterministic term at all, so its table is Johansen's model
   with no constant: MacKinnon, Haug and Michelis (1999), the source Mauricio
   (2006, section 4) computes his p-values with (statsmodels' c_sja(n, -1)).
   The table this replaced came from urca's ca.jo(ecdet="none"), which still
   fits an unrestricted intercept, and made case 1 badly undersized (BUG-24).
   Case 2 (restricted constant): urca 1.3.4, ca.jo(ecdet="const",
   type="eigen"), Osterwald-Lenum (1992).                                     */
#define LR_MAXTRENDS 11
static const real lr_cval_none[LR_MAXTRENDS][3] = {   /* case 1: no constant  */
    {   2.98,   4.13,   6.94}, {   9.47,  11.22,  15.09}, {  15.72,  17.80,  22.25},
    {  21.84,  24.16,  29.06}, {  27.92,  30.44,  35.74}, {  33.93,  36.63,  42.23},
    {  39.91,  42.77,  48.66}, {  45.89,  48.88,  55.03}, {  51.85,  54.96,  61.34},
    {  57.80,  61.04,  67.64}, {  63.72,  67.08,  73.89}
};
static const real lr_cval_const[LR_MAXTRENDS][3] = {  /* case 2: restricted c */
    {  7.52,   9.24,  12.97}, { 13.75,  15.67,  20.20}, { 19.77,  22.00,  26.81},
    { 25.56,  28.14,  33.24}, { 31.66,  34.40,  39.79}, { 37.45,  40.30,  46.82},
    { 43.25,  46.45,  51.91}, { 48.91,  52.00,  57.95}, { 54.35,  57.42,  63.71},
    { 60.25,  63.57,  69.94}, { 66.02,  69.74,  76.63}
};

/* --- .inp metadata -------------------------------------------------------- */
int   data_freq      = 1;
int   data_start_sub = 1;
int   data_start_year = 1;
char **series_names  = NULL;

/* --- transformation spec -------------------------------------------------- */
real trans_lambda = 1.0;
real trans_scale  = 1.0;
int  trans_d = 0, trans_D = 0;

/* --- model configuration -------------------------------------------------- */
int global_p, global_q, global_r;    /* r = cointegration rank */
int global_include_mean = 0;
int global_diag_ar = 0;
int global_diag_ma = 0;
int global_diag_cov = 0;
int met = 1;          /* both EXACT ML (AS 311): 1 truncates the xi sequence at
                         1e-3, 2 does not (xitol < 0).  BUG-47: this said
                         "2 = approximate", and the report followed it.      */
int global_case = 1;  /* deterministic case (Mauricio Remark 6) */
static int case_given = 0;   /* -case was on the command line (BUG-45) */
int global_lrtest = 0; /* if 1, perform sequential LR test for rank */
int global_rungs  = 0; /* if 1, report the ladder's rungs 0-2 and their LRs   */

/*  -seedgate — ROUTE (B) OF THE PLAN, behind an option and NOT the default.
 *
 *  The bridge the ladder uses everywhere else -- take the optimum of the rung
 *  below and start there -- does not reach the r = 1 rung: with Lambda = 0 the
 *  transformed system has an AR root of modulus exactly 1 and the likelihood is
 *  not defined there (docs/VEC_EMBEDDING_PLAN.md 3).  (B) crosses without
 *  choosing any constant: F, Theta and Sigma are held at the r = 0 optimum and
 *  ONLY Lambda and B2 are estimated, so the step off the boundary is the
 *  likelihood's to choose.  Everything is released afterwards.
 *
 *  prof_hold is the conditional mode: while it is set, the parameter vector
 *  carries the mean, Lambda and B2, and the F, Theta and Sigma blocks are read
 *  from hold_*, not from x.  Same pattern as -fixb2 and -alpha: the restriction
 *  lives in the cast and the optimiser never learns about it.                */
int global_seedgate = 0;
/*  P12: the ladder -- gate, rungs below the rank, route (B) -- is the default
 *  start for r >= 1; -noladder turns it off (the cold conditional-regression
 *  start of before), and -seedgate, which used to switch it on, is now the
 *  default and is accepted as such.                                         */
int global_noladder = 0;

/*  -seedb2 v — start B2 at v and estimate it FREE.  This is not -fixb2, which
 *  holds it: here it moves.  It exists as a MEASURING INSTRUMENT, so that the
 *  question of what the fit depends on -- where B2 starts, or where the
 *  optimiser stops -- can be asked without recompiling for each value.  That a
 *  question about the starting point could only be answered by recompiling is,
 *  by itself, a reason for the option to exist.                              */
int  global_seedb2 = 0;
real global_seedb2_value = 0.0;

/*  -seedjoh — seed B2 with the canonical reduced-rank solution instead of the
 *  static OLS one.  See canonical_b2.                                        */
int  global_seedjoh = 0;
static int canon_used = 0;      /* 1 = the canonical solution really went in  */

/*  -mawarma — THE MA IS NOT FREE: IT IS INHERITED.
 *
 *  Corollary 2 of the BVECM paper (WARMA-VEC equivalence with an MA) says that
 *  if the process admits a WARMA representation
 *
 *      Phi(B) w_t = Theta(B) a_t,     Delta z2_t = gamma w_{t-1} + ... + eta_t
 *
 *  -- that is, the MA lives in the cointegrated block and the differenced block
 *  is white noise -- then the error of the VEC representation is
 *
 *      eps_t = [ beta' eta_t + Theta(B) a_t ;  eta_t ].
 *
 *  Regrouping on A_t = [a_t + beta' eta_t ; eta_t], which is an invertible
 *  transformation of the noise, that is eps_t = A_t - Theta1 A_{t-1} with
 *
 *      Theta = [ Theta11   Theta11 B2' ]        (B2 = -beta; ESTUDIO_BVECM 2.1)
 *              [    0           0      ]
 *
 *  that is: THE LAST s ROWS ARE ZERO and the upper-right block is NOT free, it
 *  is determined by the left one and by B2.  With M = 2 and r = 1 that leaves
 *  ONE moving-average parameter where the free model carries FOUR.
 *
 *  And this is not a modelling preference, it is measured.  Simulating the
 *  paper's own WARMA DGP (theta = 0.5, beta = 0.5, n = 1000) and fitting with a
 *  free Theta, drvec returns entries of 3.26, 7.50 or -2.44 where the truth is
 *  [[0.5, -0.25],[0,0]], and parks on the invertibility boundary.  B2 comes out
 *  right in every replication -- it is superconsistent -- but the (Lambda,
 *  Theta) block is practically unidentified when Theta is free.
 *  See docs/HOMOLOGATION.md 4g.                                              */
int global_mawarma = 0;

/*  -matri — THE COMPROMISE, and the MINIMAL of the three restrictions.
 *
 *  Corollary 2 imposes two things at once: that the lower-left block of Theta be
 *  ZERO -- the innovations of the cointegrated block do not enter the
 *  differenced equations with a lag -- and that the upper-right one be
 *  Theta11 B2', i.e. that the differenced block have NO MA OF ITS OWN.  The
 *  bootstrap of 4i says the data reject the second in eight of eleven cases.
 *  The first has never been tested on its own, and it is where the free fit
 *  goes out of control: the (2,1) entries estimated freely come to -0.75,
 *  -1.32, 1.51 and even 5.09 over the eight pairs, and (2,2) lands between 1.0
 *  and 2.0, which is what puts the root on the circle.
 *
 *  -matri leaves Theta BLOCK-TRIANGULAR: [T11  T12 ; 0  T22], with T22 free.
 *  It costs q*s*r parameters against the free model -- exactly one with M = 2
 *  and r = 1 -- and keeps of the corollary precisely the part the structure
 *  implies and the data do not contradict.                                   */
int global_matri = 0;

/*  -marow — THE MIDDLE RUNG, and the one the measurement points at.
 *
 *  Theta = [T11  T12 ; 0  0]: the DIFFERENCED block carries no moving average
 *  of its own, but the cross block T12 stays free instead of being determined
 *  by B2.
 *
 *  Why there and not somewhere else: -matri, which zeroes only the lower-LEFT
 *  block and leaves T22 free, does NOT remove the pathology -- G stays between
 *  0.02 and 0.12 and the MA root at 1.000 in eight of eleven cases.  What
 *  removes it is Theta(1) having the identity in its lower block, and that is
 *  what zeroing T22 gives: with T22 free the optimiser drives it to unity,
 *  which is a (1-B) sitting on the already differenced block.  So of the two
 *  restrictions corollary 2 imposes at once -- T21 = 0 with T22 = 0, and T12
 *  determined -- the one that sustains admissibility is the first and the one
 *  the data reject is the second.  This rung separates them.                 */
int global_marow = 0;

/*  -mafree -- THE FREE Theta, Mauricio (2006)'s model, and THE DEFAULT again
 *  since 2026-09-23 (it was also the default until 2026-08-20).
 *
 *  Between those dates the default was -marow, on Corollary 6.3 of
 *  docs/DEMOSTRACIONES.md: "with the lower s rows of every Theta_k zero,
 *  chekma on Theta IS the admissibility condition, so the point of P \ C is
 *  not reachable".  The study of 2026-09-23 found that false in both
 *  directions (a -marow point with Theta_1 = [[1, .7], [0, 0]] passes chekma
 *  and is inadmissible; Theta_1 = [[3, .4], [0, 0]] is rejected and is
 *  admissible): the gate checks invertibility, admissibility is a condition
 *  at z = 1.  What stays true is what HOMOLOGATION.md 4q/4r measured -- in
 *  short samples the free Theta is hard to estimate from a single start --
 *  and that is what the search of P12 answers, by starting the free fit from
 *  the optimum of every class it contains.  The flag is kept so that an
 *  explicit request is recorded as one.  See docs/ESTUDIO_MAROW_2026-09-23.md
 *  and BUGS.md BUG-48. */
int global_mafree = 0;

/*  THE STRUCTURED CLASSES COLLAPSE AT r = 0, and that has to be said in one
 *  place only.  With r = 0 there is no W block: the model is a VARMA on
 *  nabla Y and the "r x r block" is 0 x 0, so zeroing the lower s = M rows
 *  would zero Theta ENTIRELY.  That is what SPECIFICATION_PLAN.md 9 already
 *  said -- the restricted classes are defined with respect to the r/s partition
 *  and collapse at r = 0 -- and it is also where the two contracts of the
 *  ladder live (Theorem 9).  So at r = 0 the moving average is free, whoever
 *  asks for what.  -lrtest walks r = 0..M-1, so this is consulted with the r of
 *  the fit in hand and not with the one on the command line.                 */
static int ma_struct_on(void)  { return global_r > 0 && (global_marow || global_mawarma); }
static int marow_on(void)      { return global_r > 0 && global_marow; }
static int mawarma_on(void)    { return global_r > 0 && global_mawarma; }

/*  -warma — THE CLASS OF THE THEOREMS, PARAMETERISED WHERE IT IS STATED.
 *
 *  So far every restriction has been written on Theta in VEC coordinates, and
 *  there the same restriction couples Theta with B2 and has to be rebuilt at
 *  every evaluation.  In the coordinates of the transformed system,
 *  Ybar = [nabla Y2 ; W], which are the ones Definition 3 of the BVECM uses and
 *  the ones Phillips uses, the class is a ZERO PATTERN:
 *
 *      Phi*_k = [ 0   Psi_k ]      Theta*_k = [ 0    0    ]
 *               [ 0   Phi_k ]                 [ 0  Th_k   ]
 *
 *  that is: nothing depends on lags of nabla Y2 -- everything enters through W
 *  -- and the differenced block carries no moving averages.  And B2 enters ONLY
 *  THROUGH THE DATA, when W is formed by subtraction, like the input of a
 *  transfer function; it touches no parameter.  That is what the legacy shootx
 *  does and the measured reason why its surface is better conditioned
 *  (docs/ESTUDIO_BVECM_vs_DRVEC.md 3.1).
 *
 *  The parameter vector REUSES the same slots -- mean, one M x r block, then
 *  (p-1) M x r blocks, q r x r blocks, Sigma, and B2 in the tail -- so as not
 *  to touch the bootstrap, the profiling or B2's tail, all of which depend on
 *  that layout.  What changes is that they are read as coefficients of W_{t-k}
 *  and not as Lambda and F.                                                  */
int global_warma = 0;

/*  -rankadm — THE CONDITION THAT MAKES THE RANK BE THE ONE CLAIMED.
 *
 *  For a VEC with moving-average errors to represent an I(1) process with
 *  cointegration rank EXACTLY r, the matrix
 *
 *      G = Lambda_perp' Theta(1) B_perp        (s x s,  Theta(1) = I - sum Theta_k)
 *
 *  must be non-singular: it is the one that appears in Granger's
 *  representation, C(1) = B_perp (Lambda_perp' Gamma B_perp)^-1 Lambda_perp'
 *  Theta(1).  If G degenerates, C(1) loses rank and the FITTED MODEL DENIES ITS
 *  OWN RANK: it says r, and its parameters imply that no stochastic trend is
 *  left.
 *
 *  Mauricio (2006) assumes partial non-stationarity OF THE TRUE PROCESS
 *  (section 2) and refers to Yap and Reinsel (1995) for the identification
 *  conditions, but the ESTIMATION imposes none: the set where G degenerates
 *  lies inside the region the program admits, because the engine only checks
 *  that the roots are not INSIDE the circle and this pathology lives exactly ON
 *  it, on the permitted edge.  Measured on the eight pairs with a free Theta:
 *  |Theta(1)| between -7e-5 and 4e-4, and the direction in which Theta(1) is
 *  singular aligned with Lambda_perp between 0.946 and 0.9996.  So the free
 *  optimum is THERE, not near it.  See docs/HOMOLOGATION.md 4h.
 *
 *  sigma_min(G) is REPORTED always, like the roots.  -rankadm also IMPOSES it,
 *  rejecting the point the way a non-positive-definite Sigma is rejected, so
 *  that the optimiser cannot enter.                                          */
int  global_rankadm = 0;
/*  THE FLOOR, AND WHAT IS LEFT OF ITS CALIBRATION.  0.2 was set in the empty
 *  gap between the degenerate fits (0.016-0.133) and the admissible ones
 *  (0.52-1.00) of HOMOLOGATION.md 4h/4j -- but that G was the wrong statistic,
 *  sigma_min(Lambda_perp' Theta(1) B_perp) (BUG-46).  Re-measured with
 *  Theorem 3's sigma_s(Lambda_perp' Theta(1)) on the eight pairs, 2 1 1
 *  -case 2 (2026-09-24): -mafree 0.025-0.31, -matri 0.10-0.30, -marow
 *  0.035-0.97, -mawarma 0.93-1.13.  THERE IS NO GAP ANY MORE, and the BUG-46
 *  counterexample -- a correct rank -- has G = 0.141.  So 0.2 stays as a
 *  CONVENTION, said to be one: below it the report warns that the fit is
 *  NEAR the rank-deficient set; only G < GRANGER_ZERO is a denial of the
 *  rank.  -rankadm changes it.                                               */
real global_rankadm_tol = 0.2;
int  global_matest = 0;      /* -matest N: bootstrap of the inherited vs free MA */
int  global_specs  = 0;      /* -specs: the specification ladder               */
int  global_artest = 0;      /* -artest N: bootstrap of Gamma_i = m_i alpha'   */
static real granger_sv = -1.0;
/*  Below this, G is zero to working precision and the fit DENIES its rank;
 *  between it and the floor, the fit is only NEAR that set (BUG-46).        */
#define GRANGER_ZERO 1.0e-6   /* sigma_min(G) at the last evaluation */

/*  granger_smin -- the rank condition of Theorem 3, or -1 if it does not apply:
 *
 *      sigma_s( Lambda_perp' Theta(1) ),   an s x M matrix, s = M - r,
 *
 *  its smallest (s-th) singular value.  The rank is exactly r iff it is
 *  non-zero, and that is left-coprimeness of the levels VARMA at z = 1:
 *  rank(Lambda_perp' Theta(1)) = s  <=>  rank[Phi(1) Theta(1)] = M, since
 *  Phi(1) = Lambda B' with B' of full row rank (STUDY_M3.md s11, checked on
 *  20 000 draws).  What it measures is a unit MA root cancelling a unit AR
 *  root in a direction that must stay integrated.
 *
 *  It used to be sigma_min(Lambda_perp' Theta(1) B_perp), an s x s product:
 *  sufficient, not necessary.  Multiplying by B_perp can only shrink the
 *  singular values, so it denied correct ranks (the counterexample of BUG-46),
 *  and under -mawarma / -warma, where Theta(1) B_perp = B_perp identically, it
 *  did not depend on Theta at all.  Lambda_perp is orthonormalised (QR), so
 *  the scale of the statistic is Theta(1)'s.                                 */
static real granger_smin(real **Lam, real **B2, real ***Th, int M, int r, int q)
{
    int s = M - r, i, j, k;
    gsl_matrix *L, *Q, *T1, *Gt, *V;
    gsl_vector *tau, *sv, *wk;
    real out = -1.0;

    (void) B2;   /* B_perp no longer enters: Theorem 3 does not need it */
    if (r <= 0 || s <= 0) return -1.0;

    /*  Lambda_perp: the last s columns of the Q of Lambda's QR.             */
    L   = gsl_matrix_alloc(M, r);
    Q   = gsl_matrix_alloc(M, M);
    tau = gsl_vector_alloc(r < M ? r : M);
    for (i = 0; i < M; i++)
        for (j = 0; j < r; j++) gsl_matrix_set(L, i, j, Lam[i+1][j+1]);
    {
        gsl_matrix *R = gsl_matrix_alloc(M, r);
        gsl_linalg_QR_decomp(L, tau);
        gsl_linalg_QR_unpack(L, tau, Q, R);
        gsl_matrix_free(R);
    }

    /*  Theta(1) = I - sum_k Theta_k                                          */
    T1 = gsl_matrix_alloc(M, M);
    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++) {
            real acc = (i == j) ? 1.0 : 0.0;
            for (k = 1; k <= q; k++) acc -= Th[k][i+1][j+1];
            gsl_matrix_set(T1, i, j, acc);
        }

    /*  Gt = (Lambda_perp' Theta(1))' = Theta(1)' Lambda_perp, M x s: GSL's
     *  SVD wants rows >= columns, and the singular values are the same.     */
    Gt = gsl_matrix_alloc(M, s);
    for (i = 0; i < M; i++)
        for (j = 0; j < s; j++) {
            real acc = 0.0;
            for (int a = 0; a < M; a++)
                acc += gsl_matrix_get(T1, a, i) * gsl_matrix_get(Q, a, r+j);
            gsl_matrix_set(Gt, i, j, acc);
        }

    V  = gsl_matrix_alloc(s, s);
    sv = gsl_vector_alloc(s);
    wk = gsl_vector_alloc(s);
    if (gsl_linalg_SV_decomp(Gt, V, sv, wk) == 0) out = gsl_vector_get(sv, s-1);
    gsl_vector_free(wk); gsl_vector_free(sv); gsl_matrix_free(V);
    gsl_matrix_free(Gt); gsl_matrix_free(T1);
    gsl_vector_free(tau); gsl_matrix_free(Q); gsl_matrix_free(L);
    return out;
}

static int    prof_hold = 0;
static real ***hold_F = NULL, ***hold_Th = NULL;
static real  **hold_S = NULL;
static int     hold_nf = 0, hold_q = 0, hold_M = 0;
static real    gate_seed_ll0 = 0.0, gate_seed_ll1 = 0.0;  /* r=0 y condicional */
static real    gate_seed_lam = 0.0;       /* the admissible multiple of Lambda */
static real    gate_seed_ll_start = 0.0;  /* logL at that starting point       */
static int     gate_seed_ok  = 0;

/* -fixb2: hold B2 at the static-OLS value computed by init_guess instead of
   estimating it.  This is the restricted model the literature tests against
   the free one (Mauricio 2006, Table 5: B = [1,0]'; BVECM Table 1: beta = 1),
   and it is also the natural first leg of a fix-then-relax warm start.       */
int  global_fixb2 = 0;
int  global_fixb2_given = 0;
/*  P12: seed the rest of the vector from W = Y1 + v'Y2 under -fixb2 v
 *  (BUG-33).  The search also tries the old seed (0), and keeps the best. */
int  seed_fixb2_consistent = 1;        /* 1 = a value was supplied on the line */
real global_fixb2_value = 0.0;      /* that value, applied to every entry   */
static real **B2_fixed = NULL;      /* (s x r), owned here */
static int   b2f_s = 0, b2f_r = 0;  /* dims of the current allocation */

/*  -fixb2row i v -- hold ONE row of B2 (variable i of the nabla Y2 block, in
 *  the .inp's order) at v in every relation, and estimate the other rows.
 *  -fixb2 is all-or-nothing; a hypothesis on one coefficient of beta (e.g.
 *  beta_USA = -1 in a price system, i.e. row USA = 0 once the system is
 *  written in the gap) needs the rest of beta free, and the LR against the
 *  free model is then a valid chi2 with (rows held)*r degrees of freedom
 *  (beta is superconsistent; Johansen 1995, Ch. 7).  Unlike the Wald on a
 *  normalised coefficient, the LR does not depend on the normalisation
 *  (TODO.md, MEJORA-1).  Held entries live in B2_fixed, the rest in x[],
 *  column-major over the FREE entries only: that is the one convention all
 *  four walks of the vector share (par_blocks, init_guess, vec_shootx and
 *  the printer).                                                           */
#define B2ROW_MAX 64
static int  b2row_on = 0;                 /* any row held by -fixb2row     */
static int  b2row_fix[B2ROW_MAX + 1];     /* 1 = row i held                */
static real b2row_val[B2ROW_MAX + 1];     /* its value                     */

/*  Row i of B2 is held (by -fixb2, all rows, or by -fixb2row).            */
static int b2_held(int i)
{
    if (global_fixb2) return 1;
    return b2row_on && i >= 1 && i <= B2ROW_MAX && b2row_fix[i];
}

/*  -xpre X.pre / -xlink FILE -- EXOGENOUS TRANSFER-FUNCTION INPUTS (MEJORA-2).
 *
 *  The hybrid of the suite's two embeddings: drtran adds a filter, drvec the
 *  error-correction matrix; here the VEC system receives drtran's filters.
 *  The school's doctrine (drtran-python/docs/LEVEL_TRANSFER_PLAN.md): the
 *  transfer relates LEVELS, the differencing belongs to the noise.  So
 *
 *      N_t = Y_t - sum_k nu_k(B) X_{k,t},   nu_k(B) = omega_k(B)/delta_k(B) B^b
 *
 *  with Y in w = refactor*BoxCox(z) units minus its deterministics (as every
 *  .pre series here) and X in the same units built the same way (drtran's
 *  apply_univariate_model: Box-Cox, ITS deterministics subtracted, so an
 *  intervention of X is not transmitted), and N is the VEC process: Ybar is built from N exactly
 *  as from Y.  By linearity nabla Y2 receives nu(B) nabla X and W receives
 *  (nu_1 + B2'nu_2)(B) X in levels.  The cast is drtran's "by subtraction";
 *  X's own model is FROZEN at its .pre (TASTE, Munoz 2.6): it is used only
 *  for the presample.  omega = 0 reproduces the VEC rung bit for bit, and
 *  unlike Lambda = 0 it is an interior point (VEC_EMBEDDING_PLAN.md 3), so
 *  the ladder's usual bridge works: the VEC optimum with omega = 0 is the
 *  start, and the LR against the VEC is the test.
 *
 *  WHY BY SUBTRACTION, AND SO WHY A BACKCAST.  drtran's dispatch
 *  (links_need_subtracting, BUG-8): the embedded cast, which needs no
 *  backcast because the exact likelihood integrates the presample out,
 *  holds only when input and output share their differencing operator AND
 *  the input is a series of the system with its own row.  Here X is
 *  frozen and outside the system, and an I(0) input of an I(1) output
 *  crosses operators, so the cast is by subtraction and the presample of X
 *  must be built: real observations of X before the system's sample first
 *  (drtran's w_head); what is still missing is backcast with X's ARMA (on
 *  nabla X when d = 1, cumulated back to levels), as drtran's
 *  build_pre_sample does.  A zero would put a spurious step nu(1)*level at
 *  t = 1.  With b = r = s = 0 nothing is needed and both casts coincide
 *  (drtran-note, Proposition Exactness iii).  An I(1) input is allowed:
 *  in levels its gain enters W, i.e. the cointegrating relation.  Its
 *  embedded alternative (X as a weakly exogenous series of the VEC) is
 *  TODO.md MEJORA-2.  omega(B) follows Box-Jenkins: omega_0 - omega_1 B
 *  - ...; delta(B) = 1 - delta_1 B - ... (drtran compute_irf).  The block
 *  goes LAST in x[], after B2, inside par_blocks' tail, so every routine that
 *  moves the tail as a block (the ladder, the MA classes, the profiling)
 *  carries it unchanged.                                                    */
#define XL_MAXIN  8
#define XL_MAXLNK 16
#define XL_MAXORD 12
static int   xl_on = 0, xl_nin = 0, xl_nlnk = 0;
static char *xl_netfile = NULL;
static char *xl_infile[XL_MAXIN + 1];
static char *xl_inname[XL_MAXIN + 1];
static int   xl_d[XL_MAXIN + 1];          /* X's regular differences (0 or 1) */
static int   xl_nreal[XL_MAXIN + 1];      /* presample values that are data   */
static int   xl_nbc[XL_MAXIN + 1];        /* presample values backcast        */
struct xl_link { int out, in, b, r, s; };
static struct xl_link xl_lnk[XL_MAXLNK + 1];
static int   xl_npre = 0, xl_len = 0;     /* presample length; total length   */
static real **xl_X = NULL;                /* [in][1..xl_len]; row t = xl_npre+t */
static real **xl_T = NULL;                /* [row 1..nobs_raw][col]: sum nu X  */
static real  xl_base_ll = 0.0, xl_bridge_ll = 0.0;
static int   xl_base_ok = 0, xl_bridge_ok = 0;

/*  Number of transfer parameters: (s + 1) omegas and r deltas per link.   */
static int xl_npar(void)
{
    int k, n = 0;
    if (!xl_on) return 0;
    for (k = 1; k <= xl_nlnk; k++) n += xl_lnk[k].s + 1 + xl_lnk[k].r;
    return n;
}

/*  xl_compute -- xl_T = sum over links of nu(B) X, in levels, on the rows of
 *  rawmat, from the parameters xp[1..xl_npar()].  Returns 1 if a denominator
 *  is not stable (the filter would explode), 0 otherwise.                  */
static int xl_compute(const real *xp)
{
    int k, j, t, m, c = 1, row, col;
    real *nu = vector(1, xl_len);
    for (row = 1; row <= nobs_raw; row++)
        for (col = 1; col <= nser; col++) xl_T[row][col] = 0.0;
    for (k = 1; k <= xl_nlnk; k++) {
        struct xl_link *L = &xl_lnk[k];
        real om[XL_MAXORD + 1], de[XL_MAXORD + 1];
        for (j = 0; j <= L->s; j++) om[j] = xp[c++];
        for (j = 1; j <= L->r; j++) de[j] = xp[c++];
        if ((L->r == 1 && fabs(de[1]) >= 1.0)
            || (L->r == 2 && !(de[1] + de[2] < 1.0 && de[2] - de[1] < 1.0
                               && fabs(de[2]) < 1.0))) {
            free_vector(nu, 1, xl_len);
            return 1;
        }
        for (t = 1; t <= xl_len; t++) {          /* nu[t] weighs lag t-1 */
            real sum = 0.0;
            int lag = t - 1 - L->b;
            if (lag >= 0 && lag <= L->s) sum = (lag == 0) ? om[0] : -om[lag];
            for (j = 1; j <= L->r; j++) if (t > j) sum += de[j] * nu[t - j];
            nu[t] = sum;
        }
        for (row = 1; row <= nobs_raw; row++) {
            int kk = xl_npre + row;
            real acc = 0.0;
            for (m = 1; m <= kk; m++) acc += nu[m] * xl_X[L->in][kk - m + 1];
            xl_T[row][L->out] += acc;
        }
    }
    free_vector(nu, 1, xl_len);
    return 0;
}

/*  Number of B2 rows held among the s of the current rank.                */
static int b2_nheld(int s)
{
    int i, n = 0;
    for (i = 1; i <= s; i++) n += b2_held(i);
    return n;
}

/*  F3 — linear restrictions on the adjustment coefficients.
 *
 *  Johansen and Swensen (2024, JTSA 45:248-268) define H1(r): alpha = A*psi with
 *  A a known M x sa matrix of rank sa, against H(r) with alpha free.  Weak
 *  exogeneity is the particular case in which A selects rows, so no ad-hoc test
 *  is needed: the general class is implemented and that one falls out of it.
 *
 *  And here drvec is well placed, better than the legacy: **Lambda is IN its
 *  parameter vector**, so imposing alpha = A*psi is replacing M*r free entries
 *  by sa*r and computing Lambda = A*psi inside the cast -- the same kind of
 *  change as -fixb2 -- and its covariance comes straight out of the Hessian.
 *  In BVECM coordinates alpha is DERIVED, which is why drv_project needed the
 *  delta method with an SVD pseudo-inverse (LEGACY_NOTES.md 5) for the same
 *  thing.
 *
 *  Degrees of freedom of the LR against H(r): (M - sa) * r, explicit in the
 *  paper.                                                                    */
static int    global_alpha = 0;      /* -alpha <file> or -weakex <i>           */
static real **alpha_A      = NULL;   /* (M x sa), Johansen and Swensen's A     */
static int    alpha_sa     = 0;
static char  *alpha_file   = NULL;
static int    alpha_weakex = 0;      /* i > 0: ecuacion declarada exogena debil */

/* --- Mauricio transformation matrices (Mauricio 2006, eq. 11-13) --------- */
/*   Cbar = [ 0_{s x r}    I_s      ]                                      */
/*         [    I_r       B2'       ]                                      */
/*   Hbar = [ 0_{s x s}   0_{s x r} ]                                      */
/*         [ 0_{r x s}      I_r     ]                                      */
/* where s = M - r                                                         */

/* --- Local prototypes ----------------------------------------------------- */
static void vec_shootx(real *x, struct Tvarma *armax,
                       int *ifaultx, int firstx, int lastx);
static int  calc_nparametrs(void);
static void init_guess(real *x, int npar);
static void build_y2_levels(void);
static void banner(const char *title);
static int  inp2lam(int i);
struct search_out;
static void free_ladder_cache(void);
static int  fit_search(real *x, int npar, real *dev, real **cov, real *ll,
                       real *s2, int njitter, int echo, struct search_out *so);

/*****************************************************************************/
/*  build_y2_levels — fill Y2_levels, once, before any estimation            */
/*                                                                           */
/*  default       : rawmat cols 1..s hold Y_{2t} in LEVELS.  The first row is  */
/*                  consumed to form the differences, so datamat has one row   */
/*                  fewer: col i = nabla Y_{2t}, col s+j = Y_{1t}, and         */
/*                  Y2_levels holds the matching true levels.                  */
/*  -differenced  : legacy layout, rawmat cols 1..s already hold nabla Y_{2t}. */
/*                  The levels are then unknown and get cumulated from zero, so */
/*                  W_t is off by B2'c for an unknown c.  E[W] absorbs that in  */
/*                  cases 2 and 3 (verified: identical log-likelihood), but NOT */
/*                  in case 1, where E[W] = 0 leaves nothing to absorb it.      */
/*****************************************************************************/
/*  The dimensions datamat and Y2_levels were allocated with.  They used to be
 *  static locals inside build_y2_levels; they move up to file scope so that
 *  free_case_data() can release them at the end.                             */
static int alloc_nobs = 0, alloc_s = 0;

static void build_y2_levels(void)
{
    int M = nser, r = global_r, s = M - r;
    int t, i, j;

    if (datamat)   free_matrix(datamat,   1, alloc_nobs, 1, M);
    if (Y2_levels) free_matrix(Y2_levels, 1, alloc_nobs, 1, alloc_s);

    nobs      = global_levels ? nobs_raw - 1 : nobs_raw;
    datamat   = matrix(1, nobs, 1, M);
    Y2_levels = matrix(1, nobs, 1, s);
    alloc_nobs = nobs;
    alloc_s    = s;
    nobs_full  = nobs;

    if (global_levels) {
        if (nobs < 3) {
            fprintf(stderr, "ERROR: the levels layout needs at least 4 observations\n");
            exit(1);
        }
        for (t = 1; t <= nobs; t++) {
            for (i = 1; i <= s; i++) {
                datamat[t][i]   = rawmat[t+1][i] - rawmat[t][i];
                Y2_levels[t][i] = rawmat[t+1][i];
            }
            for (j = 1; j <= r; j++)
                datamat[t][s + j] = rawmat[t+1][s + j];
        }
    } else {
        for (t = 1; t <= nobs; t++)
            for (i = 1; i <= M; i++)
                datamat[t][i] = rawmat[t][i];
        for (i = 1; i <= s; i++) Y2_levels[1][i] = 0.0;
        for (t = 2; t <= nobs; t++)
            for (i = 1; i <= s; i++)
                Y2_levels[t][i] = Y2_levels[t-1][i] + datamat[t][i];
    }

    /*  P5.2 — THE ESTIMATION WINDOW.  With -estwin E the fit is on 1..E and the
     *  evaluation on what comes after, which is the only way for the
     *  measurement to be OUT OF SAMPLE: the parameters cannot have seen the
     *  datum they are scored against.  The matrices are filled whole and only
     *  `nobs` is trimmed, so the evaluation has the rest to hand and the frees
     *  keep using alloc_nobs, which is the real size.                        */
    if (global_estwin > 0) {
        if (global_estwin < 10 || global_estwin >= nobs_full) {
            fprintf(stderr, "drvec: -estwin %d is not inside 10..%d\n",
                    global_estwin, nobs_full - 1);
            exit(1);
        }
        nobs = global_estwin;
    }
}

/*****************************************************************************/
/*  calc_nparametrs — number of free parameters in the VEC model             */
/*                                                                           */
/*  Parameter vector x[] layout (Mauricio 2006, Remark 1 and 6):            */
/*    1. Mean E[Ȳ_t] (Remark 6):                                            */
/*         case 1: none                                                      */
/*         case 2: E[W_j], j=1..r                          (r params)        */
/*         case 3: E[∇Y₂_i], i=1..s; E[W_j], j=1..r        (M params)        */
/*    2. Λ  (M×r) adjustment matrix                     (M·r params)        */
/*    3. F_i, i=1..p-1, each M×M                     ((p-1)·M² params)      */
/*    4. Θ_i, i=1..q, each M×M                         (q·M² params)         */
/*    5. Σ lower triangle                            (M(M+1)/2 params)       */
/*    6. B₂ (s×r) cointegration matrix                 (s·r params)          */
/*****************************************************************************/
/*  -nodrift i / -case pre -- A DRIFT PER SERIES (MEJORA-3).
 *
 *  Cases 1-3 treat the nabla Y2 block as a block: no drift (case 2) or a
 *  drift in every series (case 3).  A system of prices with drift and a
 *  ratio without (ln G/Y, ln CPI_USA, ln CPI_EC) is neither, and in case 3
 *  the diagonal rung cannot reproduce the univariate model of the ratio
 *  (G/Y's M1, with no mean, gives -222.91 there for its -223.42).
 *  -nodrift i holds E[nabla Y2_i] = 0 inside case 3; -case pre reads it from
 *  the .pre files -- the drift is free iff the univariate model frees its
 *  mean (Imu = 1) -- so the ladder's bottom rung IS the univariates.  E[W] is
 *  free, and the drift of the Y1 block stays -beta_2' E[nabla Y2] (W has
 *  none).  The series is named by its position in the input; at r = 0 every
 *  series is in the nabla Y2 block and the mask reads the same.  Every walk
 *  of the mean goes through y2mu_free().                                 */
#define ND_MAX 64
static int case_pre = 0;                  /* -case pre                      */
/*  -trend -- CASE 4 (Johansen 1995, ch. 6): a linear trend RESTRICTED to the
 *  cointegrating space, E[W_t] = E[W] + d (t - tc), on top of case 3 (or
 *  -case pre).  The equilibrium may drift (convergence, Balassa-Samuelson)
 *  while nabla Y2 keeps its drifts; the Y1 block's drift becomes
 *  d - beta_2' E[nabla Y2], free of the Y2 block's.  In the cast it is a
 *  deterministic term of W, subtracted from the data exactly as -xlink's
 *  transfer is: W_t - d (t - tc), with tc the middle of the sample so that
 *  E[W] stays the level at the centre and does not trade off with d.  The r
 *  slopes go after E[W] in the mean block, so every walk that moves the
 *  head carries them.  At r = 0 there is no W and no trend.               */
static int global_trend = 0;
static int nodrift_on = 0;
static int nodrift[ND_MAX + 1];           /* 1 = E[nabla Y2_i] held at 0    */

/*  The drift of the i-th series of the nabla Y2 block is a parameter.     */
static int y2mu_free(int i)
{
    if (global_case != 3) return 0;
    return !(nodrift_on && i >= 1 && i <= ND_MAX && nodrift[i]);
}

static int y2mu_count(int s)
{
    int i, n = 0;
    for (i = 1; i <= s; i++) n += y2mu_free(i);
    return n;
}

/*  -xsys i -- AN EXOGENOUS INPUT EMBEDDED IN THE SYSTEM (MEJORA-2, phase 2).
 *
 *  drtran's dispatch (links_need_subtracting, BUG-8 there): when input and
 *  output share their differencing operator the transfer goes INSIDE the
 *  model, the input is a series of the system with its own row, and the
 *  exact likelihood integrates its presample out -- no backcast, nothing
 *  subtracted.  In a VEC that is the case of an I(1) input of I(1) outputs,
 *  and "inside" means: X is one more series of the VEC whose equation is its
 *  own model, i.e. it is STRONGLY exogenous --
 *
 *      alpha_X = 0            weak exogeneity: X does not adjust to the
 *                             equilibrium error (Johansen 1992), and
 *      Gamma_k[X][j] = 0      no feedback: no other series enters the
 *        (j != X)             short run of X (Granger non-causality),
 *
 *  while the rows of Y keep everything: the entry of X in beta (the gain,
 *  X in the long run), Gamma_k[Y][X] (the lags of the transfer) and
 *  Sigma[Y][X] (omega_0: the SVAR identity, drtran-note Prop. 3).  The
 *  diagonal rung of the ladder is X's univariate .pre, as for any series.
 *  This is the conditional VEC of Johansen (1992) / Harbo et al. (1998)
 *  with the marginal model of X written out, so the likelihood is the JOINT
 *  one and the LR against the free VEC is the test of strong exogeneity.
 *
 *  The F (Gamma) mask is walked through f_free(), which every walk of the
 *  vector uses (par_blocks, init_guess, vec_shootx, search_blocks, pack_r0,
 *  the profiling and the printer): one predicate, so they cannot disagree.
 *  F is in the INTERNAL order [Y1 ; Y2] (see inp2lam), so the mask is read
 *  through lam2inp with the r of the moment -- the ladder fits at r = 0.
 *  q = 0 only for now: Theta's walks carry the MA classes, and the mask has
 *  not been taken through them (TODO.md MEJORA-2).                        */
#define XS_MAX 16
static int xs_on = 0;                     /* any series declared by -xsys   */
static int xs_inp[XS_MAX + 1];            /* 1 = series i (.inp order)      */
static int xs_nx = 0;
static int xs_df = 0;                     /* parameters the mask removes    */
static int lam2inp(int a);

static int xs_is_int(int a)               /* internal row a is an input     */
{
    int i;
    if (!xs_on) return 0;
    i = lam2inp(a);
    return i >= 1 && i <= XS_MAX && xs_inp[i];
}

/*  -gzero i j -- NO FEEDBACK IN THE SHORT RUN, entry by entry.
 *
 *  Gamma_k[i][j] = 0 for every k = 1..p-1: series j (position in the input)
 *  does not enter the short run of equation i.  Lambda and F are the VEC's
 *  own parameters (vec_shootx builds Ybar's VARMA from them with Cbar and
 *  Cinv), so a zero in F IS a zero in Gamma, whatever the r/s partition --
 *  unlike -xsys, it may name a series of the Y1 block.  With -weakex/-alpha
 *  on top it is the conditional model's restriction (Johansen 1992): the
 *  inputs weakly exogenous AND not fed back by the outputs in the short run,
 *  which is what licenses reading the system as a transfer network.  The
 *  free VEC (no mask, no alpha = A psi) is fitted too and the LR reported,
 *  with df = (M - sa) r + (p - 1) (entries held).  Same walk as -xsys's:
 *  f_free() below, so no walk of x[] can disagree.  The diagonal is never
 *  held (the ladder's rung is the univariate model).  q = 0, as -xsys.     */
#define GZ_MAX 16
static int gz_on = 0;
static int gz[GZ_MAX + 1][GZ_MAX + 1];    /* 1 = Gamma_k[i][j] held at 0    */
static int gz_n = 0;                      /* entries held, per lag          */
static int gz_df = 0;                     /* parameters the restriction removes */

static int gz_held_int(int a, int b)      /* internal (a, b) is held        */
{
    int i, j;
    if (!gz_on) return 0;
    i = lam2inp(a); j = lam2inp(b);
    return i >= 1 && i <= GZ_MAX && j >= 1 && j <= GZ_MAX && gz[i][j];
}

/*  Entry (i, j) of each F_k (internal order) is a parameter.              */
static int f_free(int i, int j)
{
    if (global_diag_ar) return i == j;
    return i == j || (!xs_is_int(i) && !gz_held_int(i, j));
}

static int f_nfree(void)
{
    int i, j, n = 0;
    for (i = 1; i <= nser; i++)
        for (j = 1; j <= nser; j++) n += f_free(i, j);
    return n;
}

/*  par_blocks — the parameter vector, split into the three stretches the
 *  profiling needs to separate, in ONE place only.
 *
 *    head    the mean and Lambda      what the conditional step estimates
 *    middle  F, Theta and Sigma       what the conditional step holds
 *    tail    B2                       what the conditional step estimates
 *
 *  Head and tail are contiguous at the two ends of the vector, which is what
 *  makes the conditional mode cheap: removing the middle stretch reorders
 *  nothing.  It is computed here and not at each site because this program
 *  already has FOUR walks of the same vector -- calc_nparametrs, init_guess,
 *  vec_shootx and the printer -- and adding a fifth loose counting rule is
 *  exactly how the §4.1 bug was opened.                                      */
static void par_blocks(int *nmean, int *nlam, int *nmid, int *ntail)
{
    int M = nser, r = global_r, s = M - r;
    int p = global_p, q = global_q;
    int nf = (p > 1) ? p - 1 : 0;

    /* 1. Mean E[Ȳ_t] */
    *nmean = (global_case == 2 ? r : (global_case == 3 ? y2mu_count(s) + r : 0))
           + (global_trend ? r : 0);

    /* 2. Lambda (M x r), or psi (sa x r) with alpha = A*psi */
    *nlam  = (global_alpha ? alpha_sa : M) * r;

    /*  -warma: the middle blocks are (p-1) M x r matrices -- the coefficients
     *  of W_{t-k} -- and q r x r moving-average matrices.                    */
    if (global_warma) {
        *nmid = nf * M * r + q * r * r
              + (global_diag_cov ? M : M * (M + 1) / 2) - 1;
        *ntail = global_fixb2 ? 0 : s * r;
        return;
    }

    /* 3. F_i (M x M, i=1..p-1)   4. Theta_j (M x M, j=1..q)
       5. Sigma (lower triangle), minus the redundant scale.
          The engine calls elf with sigma2 = 1 and concentrates the scale, so
          the objective is exactly invariant to rescaling this block
          (f1 -> f1/c, f2 -> c^m f2).  Carrying the whole triangle would leave
          a direction the likelihood cannot see: a flat ridge that makes the
          line search fail and the Hessian singular.  Sigma[1][1] is held at 1
          and the scale is reported through sigma2 (so that Sigma[1][1] =
          sigma2 exactly).                                                    */
    *nmid  = nf * f_nfree()
           + q  * (mawarma_on() ? r * r
                  : (marow_on() ? r * M
                  : (global_matri  ? M * M - s * r
                                   : (global_diag_ma ? M : M * M))))
           + (global_diag_cov ? M : M * (M + 1) / 2) - 1;

    /* 6. B_2 (s x r): the entries -fixb2 / -fixb2row do not hold */
    *ntail = (s - b2_nheld(s)) * r + xl_npar();
}

static int calc_nparametrs(void)
{
    int nmean, nlam, nmid, ntail;
    par_blocks(&nmean, &nlam, &nmid, &ntail);
    return nmean + nlam + (prof_hold ? 0 : nmid) + ntail;
}

/*****************************************************************************/
/*  warma_inverse — BACK TO VEC COORDINATES.                                 */
/*                                                                           */
/*  -warma estimates the transformed system, which is where the class of  */
/*  the theorems is a zero pattern and where B2 touches no parameter.  But*/
/*  what a user needs to read is Lambda, F, Pi and the rank condition, so */
/*  the transformation is INVERTED ONCE AT THE END -- which is the dual   */
/*  direction ESTUDIO_BVECM_vs_DRVEC.md 1 describes, and what the legacy  */
/*  analisis_BEC does.                                                    */
/*                                                                           */
/*  THE EQUATIONS.  From PhiBar_k = Cinv Phi*_k and from recursion (16):  */
/*                                                                           */
/*      F_1     = (PhiBar_1 - E) Cbar + Pi          with E = Cinv Hbar       */
/*      F_i     = (PhiBar_i + F_{i-1} E) Cbar       (i = 2 .. p-1)           */
/*      0       = PhiBar_p + F_{p-1} E              (the one that is left)*/
/*                                                                           */
/*  and Pi = LamBar Cbar = Lambda B', so the last equation determines     */
/*  Lambda.  With p = 2 and M = 2 it comes out in closed form -- checked  */
/*  Lambda = -(PhiBar_2)_{:,s+1..M} - ((PhiBar_1 - E) Cbar)_{:,1..r}.  Here  */
/*  with sympy --: the general case is solved by least squares on a system*/
/*  AFFINE in Lambda, which avoids a case analysis by p and also gives the*/
/*  RESIDUAL: if the point were not in the image of the map, the residual */
/*  would say so instead of the program publishing an invented Lambda.    */
/*                                                                           */
/*  Theta_j = Cinv Theta*_j Cbar and Sigma = Cinv Sigma* Cinv', which are */
/*  the same relations of theorem 1 read backwards.                       */
/*****************************************************************************/
static void wi_forward(real ***PhB, real **Cbar, real **E, real **Lam,
                       int M, int r, int p, real ***F, real **R)
{
    int s = M - r, nf = (p > 1) ? p - 1 : 0;
    int i, j, k;
    real **T1 = matrix(1, M, 1, M), **T2 = matrix(1, M, 1, M);

    /*  LamBar = [0, Lambda], y Pi = LamBar Cbar = Lambda B'.                */
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) T1[i][j] = 0.0;
    for (i = 1; i <= M; i++)
        for (j = 1; j <= r; j++) T1[i][s + j] = Lam[i][j];
    matrix_multiply(T1, Cbar, T2, M, M, M);          /* T2 = Pi */

    if (nf >= 1) {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) T1[i][j] = PhB[1][i][j] - E[i][j];
        {
            real **T3 = matrix(1, M, 1, M);
            matrix_multiply(T1, Cbar, T3, M, M, M);
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) F[1][i][j] = T3[i][j] + T2[i][j];
            free_matrix(T3, 1, M, 1, M);
        }
        for (k = 2; k <= nf; k++) {
            real **T3 = matrix(1, M, 1, M), **T4 = matrix(1, M, 1, M);
            matrix_multiply(F[k-1], E, T3, M, M, M);
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) T3[i][j] += PhB[k][i][j];
            matrix_multiply(T3, Cbar, T4, M, M, M);
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) F[k][i][j] = T4[i][j];
            free_matrix(T4, 1, M, 1, M);
            free_matrix(T3, 1, M, 1, M);
        }
        /*  The residual of the equation that is left: PhiBar_p + F_{p-1} E.  */
        matrix_multiply(F[nf], E, T1, M, M, M);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) R[i][j] = PhB[p][i][j] + T1[i][j];
    } else {
        /*  p = 1: the only equation is PhiBar_1 = E - LamBar, and its residual.*/
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) R[i][j] = PhB[1][i][j] - E[i][j];
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++) R[i][s + j] += Lam[i][j];
    }
    free_matrix(T2, 1, M, 1, M);
    free_matrix(T1, 1, M, 1, M);
}

static real warma_inverse(struct Tvarma *v, real **B2, real **Lam, real ***F,
                          real ***Th, real **Sg)
{
    int M = nser, r = global_r, s = M - r, p = global_p, q = global_q;
    int nf = (p > 1) ? p - 1 : 0, nl = M * r;
    int i, j, k, a, b, c;
    real **Cbar = matrix(1, M, 1, M), **Cinv = matrix(1, M, 1, M);
    real **Hbar = matrix(1, M, 1, M), **E = matrix(1, M, 1, M);
    real ***PhB = tensor(1, p, 1, M, 1, M);
    real **R0 = matrix(1, M, 1, M), **R1 = matrix(1, M, 1, M);
    real **J  = matrix(1, M * M, 1, (nl > 0 ? nl : 1));
    real **N  = matrix(1, (nl > 0 ? nl : 1), 1, (nl > 0 ? nl : 1));
    real  *rhs = vector(1, (nl > 0 ? nl : 1));
    int   *ind = ivector(1, (nl > 0 ? nl : 1));
    real  res = 0.0;

    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) {
            Cbar[i][j] = 0.0; Cinv[i][j] = 0.0; Hbar[i][j] = 0.0;
        }
    for (i = 1; i <= s; i++) Cbar[i][r + i] = 1.0;
    for (j = 1; j <= r; j++) Cbar[s + j][j] = 1.0;
    for (j = 1; j <= r; j++)
        for (i = 1; i <= s; i++) Cbar[s + j][r + i] = B2[i][j];
    for (i = 1; i <= r; i++)
        for (j = 1; j <= s; j++) Cinv[i][j] = -B2[j][i];
    for (i = 1; i <= r; i++) Cinv[i][s + i] = 1.0;
    for (i = 1; i <= s; i++) Cinv[r + i][i] = 1.0;
    for (i = 1; i <= r; i++) Hbar[s + i][s + i] = 1.0;
    matrix_multiply(Cinv, Hbar, E, M, M, M);

    for (k = 1; k <= p; k++) matrix_multiply(Cinv, v->phi[k], PhB[k], M, M, M);
    for (k = 1; k <= q; k++) {
        real **T = matrix(1, M, 1, M);
        matrix_multiply(Cinv, v->theta[k], T, M, M, M);
        matrix_multiply(T, Cbar, Th[k], M, M, M);
        free_matrix(T, 1, M, 1, M);
    }
    {
        real **T = matrix(1, M, 1, M);
        matrix_multiply(Cinv, v->qq, T, M, M, M);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                real ss = 0.0;
                for (k = 1; k <= M; k++) ss += T[i][k] * Cinv[j][k];
                Sg[i][j] = ss;
            }
        free_matrix(T, 1, M, 1, M);
    }

    /*  The affine system in Lambda: R(Lambda) = R0 + J vec(Lambda).         */
    for (i = 1; i <= M; i++) for (j = 1; j <= r; j++) Lam[i][j] = 0.0;
    wi_forward(PhB, Cbar, E, Lam, M, r, p, F, R0);
    c = 0;
    for (a = 1; a <= M; a++)
        for (b = 1; b <= r; b++) {
            c++;
            Lam[a][b] = 1.0;
            wi_forward(PhB, Cbar, E, Lam, M, r, p, F, R1);
            Lam[a][b] = 0.0;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    J[(i-1)*M + j][c] = R1[i][j] - R0[i][j];
        }
    for (a = 1; a <= nl; a++) {
        for (b = 1; b <= nl; b++) {
            real ss = 0.0;
            for (i = 1; i <= M * M; i++) ss += J[i][a] * J[i][b];
            N[a][b] = ss;
        }
        {
            real ss = 0.0;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) ss += J[(i-1)*M + j][a] * R0[i][j];
            rhs[a] = -ss;
        }
    }
    if (nl > 0) {
        ludcp(N, nl, ind);
        lusol(N, rhs, nl, ind);
        c = 0;
        for (a = 1; a <= M; a++)
            for (b = 1; b <= r; b++) Lam[a][b] = rhs[++c];
    }
    wi_forward(PhB, Cbar, E, Lam, M, r, p, F, R1);
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) res += R1[i][j] * R1[i][j];
    res = sqrt(res);

    free_ivector(ind, 1, (nl > 0 ? nl : 1));
    free_vector(rhs, 1, (nl > 0 ? nl : 1));
    free_matrix(N, 1, (nl > 0 ? nl : 1), 1, (nl > 0 ? nl : 1));
    free_matrix(J, 1, M * M, 1, (nl > 0 ? nl : 1));
    free_matrix(R1, 1, M, 1, M);
    free_matrix(R0, 1, M, 1, M);
    free_tensor(PhB, 1, p, 1, M, 1, M);
    free_matrix(E, 1, M, 1, M);
    free_matrix(Hbar, 1, M, 1, M);
    free_matrix(Cinv, 1, M, 1, M);
    free_matrix(Cbar, 1, M, 1, M);
    (void) nf;
    return res;
}

/*****************************************************************************/
/*  canonical_b2 — B2 BY THE CANONICAL REDUCED-RANK SOLUTION (Johansen).  */
/*                                                                           */
/*  WHAT IT IS.  Johansen's estimator solves the cointegrating vector in  */
/*  CLOSED FORM, by an eigenvalue problem, optimising nothing:            */
/*                                                                           */
/*    R0  residuals of regressing nabla Y_t on the lagged nabla Y             */
/*    R1  residuals of regressing Y_{t-1}   on the same                       */
/*    S_ij = R_i' R_j / T                                                     */
/*    |lambda S11 - S10 S00^-1 S01| = 0,  beta = the r largest eigenvectors   */
/*                                                                           */
/*  WHY AS A SEED.  Because how close it falls to this program's optimum  */
/*  is already measured, and was measured for something else:             */
/*  HOMOLOGATION.md 2.1b compares the two routes on the SAME              */
/*  specification (q = 0) over the eight pairs and finds them between     */
/*  0.0003 and 0.052 apart, in 24 comparisons.  No other seed this        */
/*  program has tried is at that distance: (C)'s starts 11 to 17 units of */
/*  logL below the optimum and (B)'s can get B2's sign wrong (4b, 4c).    */
/*                                                                           */
/*  CONVENTIONS, which is where this breaks if it breaks.  drvec's        */
/*  normalisation is B = [I_r ; B2] over Y = [Y1 ; Y2], i.e.              */
/*  W = Y1 + B2'Y2, so the canonical beta -- which comes out normalised   */
/*  however the eigenvector likes -- has to be RENORMALISED by dividing   */
/*  by its upper r x r block.  And alpha is not computed here: the        */
/*  conditional regression init_guess already does, with the canonical W, */
/*  IS Johansen's formula for alpha, alpha = S01 beta (beta' S11 beta)^-1,*/
/*  so asking for it twice would be writing two implementations of the    */
/*  regression: drvec carries -Lambda(W - E[W]), so Lambda = -alpha.          */
/*                                                                           */
/*  Returns 1 if it left a new B2, 0 if it could not (and then the static */
/*  OLS one stands, which is the route of always).                        */
/*****************************************************************************/
static int canonical_b2(real **B2)
{
    int M = nser, r = global_r, s = M - r, p = global_p;
    int nf = (p > 1) ? p - 1 : 0;
    int T  = nobs - p;
    int nd = nf * M;                 /* the lagged nabla Y, with no constant */
    int i, j, k, t, ok = 0;
    real **Y2lev = Y2_levels;
    real **R0, **R1, **S00, **S01, **S11, **A, **bet;
    gsl_matrix *Ag, *Bg, *evec;
    gsl_vector *eval;
    gsl_eigen_gensymmv_workspace *ws;

    if (r <= 0 || T <= nd + M + 1) return 0;

    R0 = matrix(1, T, 1, M);
    R1 = matrix(1, T, 1, M);
    for (t = p + 1; t <= nobs; t++) {
        int row = t - p;
        for (j = 1; j <= r; j++) {
            R0[row][j]     = datamat[t][s+j] - datamat[t-1][s+j];   /* nabla Y1 */
            R1[row][j]     = datamat[t-1][s+j];                     /* Y1_{t-1} */
        }
        for (i = 1; i <= s; i++) {
            R0[row][r+i]   = datamat[t][i];         /* nabla Y2, already differenced */
            R1[row][r+i]   = Y2lev[t-1][i];         /* Y2_{t-1} in levels        */
        }
    }

    /*  The two auxiliary regressions, with a constant: a constant restricted
     *  to the relation is drvec's case 2 and the det_order = 0 the external
     *  comparison was made with, so centring is what corresponds.            */
    {
        int nc = nd + 1;                        /* +1 for the constant          */
        real **D = matrix(1, T, 1, nc);
        real **XtX = matrix(1, nc, 1, nc);
        real  *Xty = vector(1, nc);
        int   *ind = ivector(1, nc);
        int    c, c2, e;

        for (t = p + 1; t <= nobs; t++) {
            int row = t - p; c = 1;
            D[row][c++] = 1.0;
            for (k = 1; k <= nf; k++) {
                for (j = 1; j <= r; j++)
                    D[row][c++] = datamat[t-k][s+j] - datamat[t-k-1][s+j];
                for (i = 1; i <= s; i++)
                    D[row][c++] = datamat[t-k][i];
            }
        }
        for (c = 1; c <= nc; c++)
            for (c2 = 1; c2 <= nc; c2++) {
                real ss = 0.0;
                for (t = 1; t <= T; t++) ss += D[t][c] * D[t][c2];
                XtX[c][c2] = ss;
            }
        ludcp(XtX, nc, ind);
        for (e = 1; e <= M; e++) {
            real **RR;
            int w;
            for (w = 0; w < 2; w++) {
                RR = w ? R1 : R0;
                for (c = 1; c <= nc; c++) {
                    Xty[c] = 0.0;
                    for (t = 1; t <= T; t++) Xty[c] += D[t][c] * RR[t][e];
                }
                lusol(XtX, Xty, nc, ind);
                for (t = 1; t <= T; t++) {
                    real fit = 0.0;
                    for (c = 1; c <= nc; c++) fit += Xty[c] * D[t][c];
                    RR[t][e] -= fit;
                }
            }
        }
        free_ivector(ind, 1, nc);
        free_vector(Xty, 1, nc);
        free_matrix(XtX, 1, nc, 1, nc);
        free_matrix(D, 1, T, 1, nc);
    }

    S00 = matrix(1, M, 1, M);
    S01 = matrix(1, M, 1, M);
    S11 = matrix(1, M, 1, M);
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) {
            real a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (t = 1; t <= T; t++) {
                a0 += R0[t][i] * R0[t][j];
                a1 += R0[t][i] * R1[t][j];
                a2 += R1[t][i] * R1[t][j];
            }
            S00[i][j] = a0 / T; S01[i][j] = a1 / T; S11[i][j] = a2 / T;
        }

    /*  A = S10 S00^-1 S01, simetrica semidefinida positiva.                  */
    A = matrix(1, M, 1, M);
    {
        real **S00i = matrix(1, M, 1, M);
        real  *col  = vector(1, M);
        int   *ind  = ivector(1, M);
        real **cp   = matrix(1, M, 1, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) cp[i][j] = S00[i][j];
        ludcp(cp, M, ind);
        for (j = 1; j <= M; j++) {
            for (i = 1; i <= M; i++) col[i] = (i == j) ? 1.0 : 0.0;
            lusol(cp, col, M, ind);
            for (i = 1; i <= M; i++) S00i[i][j] = col[i];
        }
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                real ss = 0.0;
                for (k = 1; k <= M; k++)
                    for (t = 1; t <= M; t++)
                        ss += S01[k][i] * S00i[k][t] * S01[t][j];
                A[i][j] = ss;
            }
        free_matrix(cp, 1, M, 1, M);
        free_ivector(ind, 1, M);
        free_vector(col, 1, M);
        free_matrix(S00i, 1, M, 1, M);
    }

    Ag = gsl_matrix_alloc(M, M); Bg = gsl_matrix_alloc(M, M);
    evec = gsl_matrix_alloc(M, M); eval = gsl_vector_alloc(M);
    ws = gsl_eigen_gensymmv_alloc(M);
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) {
            gsl_matrix_set(Ag, i-1, j-1, 0.5 * (A[i][j] + A[j][i]));
            gsl_matrix_set(Bg, i-1, j-1, 0.5 * (S11[i][j] + S11[j][i]));
        }
    if (gsl_eigen_gensymmv(Ag, Bg, eval, evec, ws) == 0) {
        gsl_eigen_gensymmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);
        bet = matrix(1, M, 1, r);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++) bet[i][j] = gsl_matrix_get(evec, i-1, j-1);

        /*  Renormalise on the upper r x r block: beta -> beta inv(Btop), which
         *  is what makes the first r rows I_r and the lower s ones B2.  If
         *  that block is singular, drvec's normalisation does not exist for
         *  these data, and then nothing is seeded: better than seeding a huge
         *  number.                                                            */
        {
            real **Bt = matrix(1, r, 1, r);
            real  *z  = vector(1, r);
            int   *ind = ivector(1, r);
            int    sing = 0;
            for (i = 1; i <= r; i++)
                for (j = 1; j <= r; j++) Bt[i][j] = bet[j][i];   /* Btop' */
            ludcp(Bt, r, ind);
            for (i = 1; i <= r; i++) if (fabs(Bt[i][i]) < 1.0e-12) sing = 1;
            if (!sing) {
                real **bn = matrix(1, M, 1, r);
                for (i = 1; i <= M; i++) {
                    for (j = 1; j <= r; j++) z[j] = bet[i][j];
                    lusol(Bt, z, r, ind);
                    for (j = 1; j <= r; j++) bn[i][j] = z[j];
                }
                for (i = 1; i <= s; i++)
                    for (j = 1; j <= r; j++) B2[i][j] = bn[r+i][j];
                ok = 1;
                free_matrix(bn, 1, M, 1, r);
            }
            free_ivector(ind, 1, r);
            free_vector(z, 1, r);
            free_matrix(Bt, 1, r, 1, r);
        }
        free_matrix(bet, 1, M, 1, r);
    }
    gsl_eigen_gensymmv_free(ws);
    gsl_vector_free(eval); gsl_matrix_free(evec);
    gsl_matrix_free(Bg); gsl_matrix_free(Ag);
    free_matrix(A, 1, M, 1, M);
    free_matrix(S11, 1, M, 1, M);
    free_matrix(S01, 1, M, 1, M);
    free_matrix(S00, 1, M, 1, M);
    free_matrix(R1, 1, T, 1, M);
    free_matrix(R0, 1, T, 1, M);
    return ok;
}

/*****************************************************************************/
/*  init_guess — conditional (Johansen-style) initial parameter values       */
/*                                                                           */
/*  Runs the concentrated regression  ∇Y_t = Σ F_i ∇Y_{t-i} − Λ W_{t-1} + e, */
/*  with W_t = Y_{1t} + B₂'Y_{2t} built from a static OLS B₂, to initialise   */
/*  Λ and F_i directly (this is the conditional estimator the paper mentions  */
/*  as the natural starting point, Remark 1.1).                              */

/*****************************************************************************/
/*  prelim_b2 — initial B₂ by static OLS with a constant.
 *
 *  Lifted out of init_guess unchanged, because TWO places need it: the seeding
 *  and F2's .inp writer, which has to build the same Ȳ that will be estimated.
 *  B2 is expected dimensioned (1..s, 1..max(r,1)).                           */
static void prelim_b2(real **B2)
{
    int M = nser, r = global_r, s = M - r;
    int i, j, t;
    real **Y2lev = Y2_levels;

    for (j = 1; j <= r; j++) {
        int nc = s + 1;
        real *yy = vector(1, nobs);
        for (t = 1; t <= nobs; t++) yy[t] = datamat[t][s + j];
        real **XX = matrix(1, nc, 1, nc);
        real  *Xy = vector(1, nc);
        for (i = 1; i <= nc; i++) {
            for (int ii = 1; ii <= nc; ii++) {
                real ss = 0.0;
                for (t = 1; t <= nobs; t++) {
                    real xi  = (i  == 1) ? 1.0 : Y2lev[t][i-1];
                    real xii = (ii == 1) ? 1.0 : Y2lev[t][ii-1];
                    ss += xi * xii;
                }
                XX[i][ii] = ss;
            }
            Xy[i] = 0.0;
            for (t = 1; t <= nobs; t++) {
                real xi = (i == 1) ? 1.0 : Y2lev[t][i-1];
                Xy[i] += xi * yy[t];
            }
        }
        int *ind = ivector(1, nc);
        ludcp(XX, nc, ind);
        lusol(XX, Xy, nc, ind);
        for (i = 1; i <= s; i++) B2[i][j] = -Xy[i+1];
        free_ivector(ind, 1, nc);
        free_vector(yy, 1, nobs);
        free_matrix(XX, 1, nc, 1, nc);
        free_vector(Xy, 1, nc);
    }
    (void) M;
}

/*  build_ybar — Ȳ_t = (∇Y_{2t}', W_t')' for a given B₂.
 *
 *  SAME construction as block [5] of vec_shootx; if the two stop agreeing,
 *  what drvec writes into the .inp files is not what it estimates.  Ybar is
 *  expected dimensioned (1..nobs, 1..M).                                     */
static void build_ybar(real **B2, real **Ybar)
{
    int M = nser, r = global_r, s = M - r;
    int i, j, t;
    for (t = 1; t <= nobs; t++) {
        for (i = 1; i <= s; i++) Ybar[t][i] = datamat[t][i];
        for (j = 1; j <= r; j++) {
            real w = datamat[t][s + j];
            for (i = 1; i <= s; i++) w += B2[i][j] * Y2_levels[t][i];
            Ybar[t][s + j] = w;
        }
    }
}

/*  free_case_data — what the case allocates and that survives every exit.
 *
 *  P3, 2026-08-21.  These three -- datamat, Y2_levels and the A of the
 *  alpha = A*psi restriction -- were end-of-program leaks that were INVISIBLE
 *  while matrix() returned the pointer at the base of its block.  Adopting
 *  drvarma's shared version brought all five out at once (with rawmat and
 *  cond_resid, in cleanup_names), and -lrtest and -rungs, which call
 *  build_y2_levels once per rank, had them twice over.
 *
 *  None of them does harm in a batch program.  What they do is hide the ones
 *  that would: a valgrind output with five known leaks is one in which the
 *  sixth cannot be seen.                                                     */
static void free_case_data(void)
{
    free_ladder_cache();      /* P12: the r = 0 ladder's cached optimum */
    if (datamat)   { free_matrix(datamat,   1, alloc_nobs, 1, nser); datamat = NULL; }
    if (Y2_levels) { free_matrix(Y2_levels, 1, alloc_nobs, 1, alloc_s); Y2_levels = NULL; }
    if (alpha_A)   { free_matrix(alpha_A, 1, nser, 1, (alpha_sa > 0 ? alpha_sa : 1));
                     alpha_A = NULL; }
    alloc_nobs = 0; alloc_s = 0;
}

/*  The residuals of the conditional regression, which init_guess publishes
 *  for -writeres.  They are declared HERE, and not next to their use, because
 *  cleanup_names frees them and sits above: that is the price of having a
 *  single cleanup.                                                           */
static real **cond_resid   = NULL;
static int    cond_resid_T = 0, cond_resid_M = 0;
/*  Forward declarations: -f H is declared with the other options below, and
 *  the back-transform (BUG-36) is released by the cleanup here.            */
static int  global_fcast;
static void bt_free(void);


/*  cleanup_names — what has to be released at the end, in ONE place.
 *
 *  main has four exits -- the normal one, -lrtest, -writeinp/-writeres and
 *  -eval -- and only the normal one freed anything.  valgrind catches it the
 *  moment it is asked: 350 bytes per -lrtest run.  It is an end-of-program leak
 *  and harms nobody, but having four exits and a single cleanup is how these
 *  things turn into something worse.                                         */
static void cleanup_names(char *outf, char *inf, char *basef)
{
    if (series_names) {
        for (int j = 1; j <= nser; j++) free(series_names[j]);
        free(series_names);
        series_names = NULL;
    }
    /*  P3 — THE TWO THAT ALIGNING nlatools UNCOVERED, on 2026-08-21.
     *
     *  rawmat (the raw .inp data) and cond_resid (the residuals of the
     *  conditional regression, which init_guess publishes for -writeres) are
     *  process-lifetime and nobody freed them.  They were INVISIBLE to valgrind
     *  while matrix() returned the pointer at the BASE of its block: a pointer
     *  to the base looks reachable.  Adopting drvarma's shared version -- which
     *  returns the offset pointer -- brought both out, 1416 and 1488 bytes.
     *  Same mechanism SUITE_INTEGRATION.md 5 already described for vector(),
     *  found a second time and by the same route.
     *
     *  They go here because here is where all seven exits of main pass.      */
    if (rawmat) {
        free_matrix(rawmat, 1, nobs_raw, 1, nser);
        rawmat = NULL;
    }
    if (cond_resid) {
        free_matrix(cond_resid, 1, cond_resid_T, 1, cond_resid_M);
        cond_resid = NULL;
    }
    free_case_data();
    bt_free();
    if (outf)  FREE_STR(outf);
    if (inf)   FREE_STR(inf);
    if (basef) FREE_STR(basef);
}



/*  THE BACK-TRANSFORM (BUG-36).  The system is estimated on
 *
 *      w_t = refactor * BoxCox(z_t) - det_t
 *
 *  -- the .pre route builds rawmat that way, and -interv subtracts det on the
 *  .inp route --, and the forecast and the rolling evaluation used to stay in
 *  w while the .forecast file said "the units of the .inp": mink/muskrat with
 *  refactor 100 forecast 1379.5 for a log level of ~13.8.  Levels are handed
 *  back in z, the convention drtran uses (drtran.c, the level forecast):
 *
 *      z = BoxCox^-1( (w + det) / refactor )     -- the MEDIAN under a log,
 *
 *  the band transformed at its two ends (so it is asymmetric when lambda != 1)
 *  and the s.e. by the delta method, dz/dw at the point.  det is kept for
 *  the sample AND the forecast horizon, because a step or a trend goes on.
 *  Indexed by the RAW observation, t = 1..bt_T.                              */
static int    bt_on = 0;
static int    bt_T = 0, bt_M = 0;
static real  *bt_lam = NULL, *bt_refac = NULL;
static real **bt_det = NULL;

static void bt_alloc(int M, int T)
{
    int i, t;
    bt_M = M; bt_T = T; bt_on = 1;
    bt_lam   = vector(1, M);
    bt_refac = vector(1, M);
    bt_det   = matrix(1, M, 1, T);
    for (i = 1; i <= M; i++) {
        bt_lam[i] = 1.0; bt_refac[i] = 1.0;
        for (t = 1; t <= T; t++) bt_det[i][t] = 0.0;
    }
}

static void bt_free(void)
{
    if (!bt_on) return;
    free_matrix(bt_det, 1, bt_M, 1, bt_T);
    free_vector(bt_refac, 1, bt_M);
    free_vector(bt_lam, 1, bt_M);
    bt_det = NULL; bt_lam = bt_refac = NULL; bt_on = 0; bt_T = bt_M = 0;
}

/*  w (the system's units) -> z (the series' own units) at raw observation t. */
static real bt_level(int i, int t, real w)
{
    real c, lam;
    if (!bt_on) return w;
    c   = (w + ((t >= 1 && t <= bt_T) ? bt_det[i][t] : 0.0)) / bt_refac[i];
    lam = bt_lam[i];
    if (fabs(lam) < 1.0e-8)         return exp(c);
    if (fabs(lam - 1.0) < 1.0e-12)  return c;
    if (lam * c + 1.0 <= 0.0)       return NAN;   /* outside the transform's range */
    return pow(lam * c + 1.0, 1.0 / lam);
}

/*  dz/dw at that point: the delta-method factor for the s.e.               */
static real bt_jac(int i, int t, real w)
{
    real c, lam;
    if (!bt_on) return 1.0;
    c   = (w + ((t >= 1 && t <= bt_T) ? bt_det[i][t] : 0.0)) / bt_refac[i];
    lam = bt_lam[i];
    if (fabs(lam) < 1.0e-8)         return exp(c) / bt_refac[i];
    if (fabs(lam - 1.0) < 1.0e-12)  return 1.0 / bt_refac[i];
    if (lam * c + 1.0 <= 0.0)       return NAN;
    return pow(lam * c + 1.0, 1.0 / lam - 1.0) / bt_refac[i];
}

/*  subtract_interventions — removes from the data the deterministic component
 *  each series declares in its .pre.
 *
 *  WHY IT IS NEEDED.  fue's cast admits interventions -- omega(B)/delta(B) on
 *  an impulse, step, ramp... -- and the univariate models of the ladder use
 *  them.  Estimating a VEC afterwards that ignores them is estimating a
 *  different model: in the exercise that motivated this, a series with a
 *  five-year response impulse over a sample of 76 gave a cointegrating vector
 *  with no economic sense (positive), and that omission was the cause.
 *
 *  HOW.  build_det_component comes with the vendored reader and computes
 *  nu(B) = omega(B)/delta(B) applied to the regressor, with the Box-Jenkins
 *  sign convention (omega_0 adds, the rest subtract) and the rational case
 *  included.  It is handed the .pre's model but THE DATES OF THIS SAMPLE,
 *  because a deterministic term is a function of time: align them by index
 *  instead of by date and the intervention lands in the wrong year.
 *
 *  LIMIT, declared: the omegas stay FIXED at whatever fue estimated
 *  separately; they are not re-estimated jointly.  drtran does carry them in
 *  its parameter vector (BRIDGE_DESIGN.md), and that is the natural next step.
 *  Meanwhile this is "the univariate model's interventions, applied", which is
 *  a good deal better than "no interventions" and worse than estimating them. */
static void subtract_interventions(const char *prefix)
{
    int M = nser, i, t, nsub = 0;

    for (i = 1; i <= M; i++) {
        char path[1024];
        struct Tusmodel Tm;
        struct Tseries  Ts, Tsx;
        real **DataMat = NULL;
        real  *det;

        snprintf(path, sizeof path, "%s.%d.pre", prefix, i);
        if (read_fue_pre(path, &Tm, &Ts, &DataMat) != 0) {
            char alt[1024];
            snprintf(alt, sizeof alt, "%s.%d.inp", prefix, i);
            if (read_fue_pre(alt, &Tm, &Ts, &DataMat) != 0) {
                fprintf(stderr, "WARNING: could not read %s or %s; series %d goes\n"
                                "         without deterministic terms\n", path, alt, i);
                continue;
            }
        }
        if (Tm.NdetVar > 0) {
            /* dates of THIS sample, model from the .pre */
            Tsx = Ts;
            Tsx.freq    = data_freq;
            Tsx.begyear = data_start_year;
            Tsx.begtime = data_start_sub;
            Tsx.nobs    = nobs_raw;
            {   /*  over the sample AND the horizon: the forecast needs det's
                 *  future path back (BUG-36).                               */
                int TT = nobs_raw + (global_fcast > 0 ? global_fcast : 0);
                det = vector(1, TT);
                build_det_component(&Tm, &Tsx, TT, det);
                if (!bt_on) bt_alloc(M, TT);
            }
            /*  THE REFACTOR, which until 2026-08-20 was not applied, and that
             *  is a defect.  The .pre's model is defined on
             *  w = refactor * BoxCox(z) (the format's FILE_CONTRACT, and
             *  fue_pre_reader.c reads it into Ts->refactor), so its omegas are
             *  in THE UNITS OF w and not in those of the datum drvec has in
             *  front of it.  Subtracting them as they come is right only when
             *  refactor = 1, which is what held in every case tried until a
             *  real application -- three monthly CPIs with refactor = 100 --
             *  showed otherwise: the "adjusted" series came out with an
             *  innovation variance a hundred times its own and the model was
             *  useless without anything warning about it.
             *
             *  Same family as the sibling program's scale defects: a
             *  coefficient taken from one source and applied in the units of
             *  another.  Divide, which is what takes det into the units of the
             *  datum.                                                        */
            {
                real rf = (Ts.refactor != 0.0) ? Ts.refactor : 1.0;
                if (rf != 1.0) {
                    for (t = 1; t <= bt_T; t++) det[t] /= rf;
                    if (!quiet_mode)
                        printf("  series %d: the .pre's refactor %.6g applied to the\n"
                               "             deterministic terms\n", i, rf);
                    fprintf(outputv, "Series %d: the .pre's refactor %.6g was "
                                     "applied to its deterministic terms.\n", i, rf);
                }
                /*  And the Box-Cox transformation CANNOT be checked from here --
                 *  the .pre says which lambda it was estimated with, but not
                 *  whether drvec's .inp carries the series already transformed
                 *  -- so it is said and left to the user, which is the only
                 *  honest thing to do.                                       */
                if (!quiet_mode && Tm.boxlam != 1.0)
                    printf("  series %d: the .pre was estimated with lambda = %.4g; "
                           "drvec's .inp must bring the series ALREADY "
                           "transformada\n", i, Tm.boxlam);
            }
            for (t = 1; t <= nobs_raw; t++) rawmat[t][i] -= det[t];
            for (t = 1; t <= bt_T; t++) bt_det[i][t] = det[t];
            if (!quiet_mode) {
                printf("  series %d: %d deterministic term(s) from the .pre subtracted (", i,
                       Tm.NdetVar);
                for (int k = 1; k <= Tm.NdetVar; k++)
                    printf("%s%s", (k > 1 ? "; " : ""),
                           Tm.detspec[k] ? Tm.detspec[k] : "?");
                printf(")\n");
            }
            fprintf(outputv, "Series %d: %d deterministic term(s) from %s "
                             "subtracted before estimation.\n",
                    i, Tm.NdetVar, path);
            free_vector(det, 1, bt_T);
            nsub++;
        }
        free_fue_pre(&Tm, &Ts, DataMat);
    }
    if (nsub == 0 && !quiet_mode)
        printf("  (ningun .pre declaraba deterministas)\n");
}

/*  residual_diagnostics — the multivariate diagnosis of the residuals.
 *
 *  WHY IT IS HERE.  drvec did NO diagnosis at all: main.h declares hosking_test
 *  and multivariate_diagnostics inherited from drvarma's header, but those
 *  routines do not exist in this project and the .out said nothing about the
 *  residuals.  An estimator that does not show its residuals cannot be used to
 *  identify, and in the application that motivated this the question is
 *  precisely one of identification: with the univariate ARMA inherited from
 *  clean ACF/PACF work, all that is left to decide is whether there are CROSS
 *  EFFECTS and of what order.  That is not answered by looking at the
 *  likelihood; it is answered by looking at the CROSS correlations of the
 *  residuals, lag by lag.
 *
 *  WHAT IT PRINTS, IN TWO PARTS
 *
 *   1. THE SUITE'S DIAGNOSIS, as it comes: multivariate_diagnostics from
 *      drtran -- Hosking's portmanteau and multivariate Jarque-Bera -- now
 *      ONE file, lib/diagmv, for the three engines (BUG-42, items 3-4); this
 *      program passes it its count of coefficients (diag_k).  That is
 *      deliberate: the same residual has to read the same in drvarma, in
 *      drtran and here.  The first version
 *      of this was a portmanteau written by hand in this file, and it was badly
 *      framed even though it was correct: it forced a comparison of apples with
 *      oranges.
 *
 *   2. WHAT DRVEC ADDS, fitted to its own reality: the cross-correlation matrix
 *      R(k) for k = 0..K, with whatever crosses the +-2/sqrt(n) band marked.
 *      The DIAGONAL of R(k) is each equation's own ACF (own dynamics badly
 *      captured); the OFF-DIAGONAL ones are the cross effect the model has not
 *      captured, and their k is THEIR ORDER.  That is what an aggregate
 *      portmanteau cannot say, and it is exactly the question that is left when
 *      the univariate ARMA arrives already identified from clean ACF/PACF work:
 *      whether there are cross effects and of what order.
 *
 *  R(k)[i][j] correlates a_i(t) with a_j(t-k), so a significant (i,j) element
 *  with k >= 1 says that equation i responds to the PAST innovation of j: it is
 *  a lagged cross effect of order k.  The upper and lower triangles are NOT the
 *  same thing, and that is where the direction is.                           */
/*  diag_k, diag_npar_ser -- WHAT THE RESIDUALS COST, for the degrees of
 *  freedom of their tests (BUG-42, item 3).
 *
 *  Hosking's Q is chi2(m^2 s - k) with k the DYNAMIC coefficients estimated:
 *  Lambda, F and Theta as the restrictions leave them, and the transfer
 *  weights of -xlink.  Not the mean, not Sigma, not B2: Bruggemann,
 *  Lutkepohl and Saikkonen (2006) give m^2 (s - p + 1) - m r for a VECM,
 *  which subtracts Gamma and alpha and leaves beta, superconsistent, out.
 *  The count is par_blocks', so it cannot disagree with the vector.
 *
 *  The Ljung-Box of each residual subtracts the coefficients of its equation
 *  on its own lags: the diagonal of each F_k and Theta_j that is free.  The
 *  residuals are Ybar's, [nabla Y2 ; W], and the coefficients are in the
 *  internal order [Y1 ; Y2]: nabla Y2_i is row r + i, W_j is row j.  It is
 *  the univariate reading, and an approximation in a system: the .out of
 *  art and fue subtracts p + q the same way.                              */
static int diag_k(void)
{
    int nmean, nlam, nmid, ntail, M = nser;
    par_blocks(&nmean, &nlam, &nmid, &ntail);
    return nlam + nmid - ((global_diag_cov ? M : M * (M + 1) / 2) - 1)
         + xl_npar();
}

static int *diag_npar_ser(void)
{
    int M = nser, r = global_r, s = M - r, i, k;
    int nf = (global_p > 1) ? global_p - 1 : 0, q = global_q;
    int *n = (int *) calloc((size_t) M + 1, sizeof(int));
    for (i = 1; i <= M; i++) {
        int row = (i <= s) ? r + i : i - s;
        int w_row = row <= r;               /* an equation of the Y1 block */
        if (global_warma) {
            n[i] = w_row ? nf + q : 0;
            continue;
        }
        for (k = 1; k <= nf; k++) n[i] += f_free(row, row);
        if ((mawarma_on() || marow_on()) && !w_row) continue;
        n[i] += q;
    }
    return n;
}

static void residual_diagnostics(struct Tvarma *v)
{
    int M = v->m, n = v->n, K, i, j, k, t;
    real band, ***C;
    real *mean = vector(1, M);

    if (n < 20 || M < 1) return;
    K = (n / 4 < 12) ? n / 4 : 12;
    if (K < 1) return;
    band = 2.0 / sqrt((real) n);

    /* means (should be ~0) and autocovariance matrices C(k) */
    for (i = 1; i <= M; i++) {
        real sm = 0.0;
        for (t = 1; t <= n; t++) sm += v->a[t][i];
        mean[i] = sm / n;
    }
    C = tensor(0, K, 1, M, 1, M);
    for (k = 0; k <= K; k++)
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                real sm = 0.0;
                for (t = k + 1; t <= n; t++)
                    sm += (v->a[t][i] - mean[i]) * (v->a[t-k][j] - mean[j]);
                C[k][i][j] = sm / n;
            }

    /*  The suite's STANDARD diagnosis, untouched (src/diagnose.c, vendored
     *  from drvarma), and it brings its own banner: this section is titled by
     *  the shared code, not by a second title of ours.                       */
    multivariate_diagnostics(v->a, n, M, data_freq, diag_k(), outputv);
    fprintf(outputv, "\n");
    /*  BUG-62.  With r >= 1 the residuals are those of (nabla Y2, W): the
     *  last r columns are the innovations of the equilibrium errors
     *  W = Y1 + B2'Y2, not of the Y1 series.  The block printed them with no
     *  names, and they were read as the Y1 series' (a residual sd of 4.4 on a
     *  series whose innovation sd is 21.8).  The columns are named here, as
     *  diagnose_ybar names them for the per-series diagnosis (BUG-19).        */
    if (global_r > 0 && series_names && M == nser) {
        int s = nser - global_r;
        fprintf(outputv, "  the columns:");
        for (i = 1; i <= M; i++) {
            if (i <= s) fprintf(outputv, " D.%s", series_names[i]);
            else        fprintf(outputv, " ec%d", i - s);
        }
        fprintf(outputv, "  (ec = the innovation of beta'Y, not of the Y1 series)\n");
    }
    fprintf(outputv, "  residual sd:");
    for (i = 1; i <= M; i++) fprintf(outputv, " %10.6f", sqrt(C[0][i][i]));
    fprintf(outputv, "\n  band = 2/sqrt(n) = %.4f;  * marks |r| > band\n", band);
    fprintf(outputv, "\nResidual cross-correlation matrices, "
                     "R(k)[i][j] = corr(a_i(t), a_j(t-k)):\n");
    for (k = 0; k <= K; k++) {
        fprintf(outputv, "  k=%-2d ", k);
        for (i = 1; i <= M; i++) {
            if (i > 1) fprintf(outputv, "\n       ");
            for (j = 1; j <= M; j++) {
                real r = C[k][i][j] / sqrt(C[0][i][i] * C[0][j][j]);
                fprintf(outputv, "%8.3f%s", r, (fabs(r) > band) ? "*" : " ");
            }
        }
        fprintf(outputv, "\n");
    }


    /* the verdict on cross effects, which is the question that matters */
    {
        int worst_k = -1, wi = 0, wj = 0, any = 0;
        real worst = 0.0;
        /* ONLY k >= 1.  The CONTEMPORANEOUS cross correlation (k = 0) is not a
           failure of the model: it is the off-diagonal of Sigma, which the model
           ESTIMATES -- except under -diagcov, where it would indeed be a badly
           placed restriction.  Counting it here would fire the alarm on any
           model with correlated innovations, which is the normal situation.  */
        for (k = 1; k <= K; k++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) {
                    real r;
                    if (i == j) continue;                    /* cross ones only */
                    r = C[k][i][j] / sqrt(C[0][i][i] * C[0][j][j]);
                    if (fabs(r) > band) any = 1;
                    if (fabs(r) > fabs(worst)) { worst = r; worst_k = k; wi = i; wj = j; }
                }
        {   /* the contemporaneous one is reported apart, not as a failure */
            real r0 = 0.0;
            for (i = 1; i <= M; i++)
                for (j = 1; j < i; j++) {
                    real r = C[0][i][j] / sqrt(C[0][i][i] * C[0][j][j]);
                    if (fabs(r) > fabs(r0)) r0 = r;
                }
            fprintf(outputv, "\n  Contemporaneous correlation, largest |r| "
                             "at k=0: %+.3f%s\n", r0,
                     (global_diag_cov && fabs(r0) > band)
                       ? "   ! -diagcov forces Sigma diagonal: an imposed "
                         "restriction the data do not support"
                       : "   (Sigma carries it)");
        }
        fprintf(outputv, "  Cross dynamics left at k >= 1: ");
        if (!any)
            fprintf(outputv, "none beyond the band\n");
        else
            fprintf(outputv, "YES, largest is equation %d against the "
                    "innovation of %d at lag %d, r = %+.3f\n",
                    wi, wj, worst_k, worst);
    }

    free_tensor(C, 0, K, 1, M, 1, M);
    free_vector(mean, 1, M);
}

/*  exact_hessian_se — standard errors from the Hessian AT THE OPTIMUM.
 *
 *  THE PROBLEM.  est() computes the covariance by inverting the Hessian that
 *  BFGS ACCUMULATES along the trajectory (raxopt leaves it in mtmp).  That is
 *  good for steering the search but is NOT the curvature at the optimum: it
 *  depends on the path taken and degrades precisely in the flattest directions,
 *  which are the ones with the largest standard errors.  This is not a
 *  suspicion -- drtran diagnosed and fixed it (BRIDGE_DESIGN.md 8c), and here
 *  the extreme symptom showed up: with -multistart, since the final est does
 *  not iterate, ALL the standard errors came out identical.
 *
 *  THE ALTERNATIVE WAS NOTED IN THE ENGINE ITSELF, commented out at
 *  drvmlest.c:104-107:  fdhess(objcfunc, ...) + choldcp.  That is what is used.
 *
 *  AND IT IS DONE WITHOUT TOUCHING THE ENGINE.  fdhess (qnewtopt.c) and
 *  objcfunc (drvmlest.c) are public symbols; they are called from here after
 *  est(), while their globals -- castx and varmax -- still point at this fit.
 *  The covariance formula is the SAME one est uses (drvmlest.c:111-119),
 *      cov = 2 * f * H^-1 / n,
 *  with H the Hessian of the concentrated objective; all that changes is where
 *  H comes from.
 *
 *  Returns 0 if it worked; leaves dev and cov overwritten.                   */
extern void fdhess(real (*func)(real *), int n, real *x, real f, real eta,
                   real **H);

/*  The objective, replicated here with ITS OWN structure.
 *
 *  The engine's objcfunc cannot be reused: est() ends by calling the cast with
 *  lastx = 1, which DEALLOCATES the structure, so calling it afterwards writes
 *  into freed memory -- checked, segfault.  And there is no entry point that
 *  would make it reallocate.
 *
 *  The formula is drvmlest.c:159-190's, and THE NORMALISING CONSTANT DOES NOT
 *  MATTER: if g = c*f then H_g = c*H_f and 2*g*H_g^-1 = 2*f*H_f^-1, i.e. the
 *  covariance does not depend on c.  It is normalised by the value at the
 *  optimum, which leaves the objective at 1 and is the most convenient
 *  numerically.                                                              */
static struct Tvarma  fdh_varma;
static int            hess_nneg = 0;
static long           fdh_rej = 0;
static int            se_from_fdhess = 0;   /* 1 if the reported s.e. are -fdhess' */
static real           hess_ratio = 0.0;
static real           fdh_norm1 = 1.0, fdh_norm2 = 1.0;

static real fdh_obj(real *x)
{
    real pi1, pi2, pi3;
    int ifault = 0;

    vec_shootx(x, &fdh_varma, &ifault, 0, 0);
    if (ifault > 0) { fdh_rej++; return 1.0e10; }   /* Sigma not positive definite */
    elf(fdh_varma.m, fdh_varma.n, fdh_varma.p, fdh_varma.q, fdh_varma.mu,
        fdh_varma.phi, fdh_varma.theta, fdh_varma.qq, fdh_varma.w, 1.0,
        fdh_varma.xitol, FALSE, fdh_varma.a, &pi1, &pi2, &pi3, &ifault);
    if (ifault > 0) { fdh_rej++; return 1.0e10; }   /* non-stationary / non-invertible */
    return pow(pi1 / fdh_norm1, (real) fdh_varma.m) * (pi2 / fdh_norm2);
}

/*  exact_hessian_se — standard errors from the Hessian AT THE OPTIMUM.
 *
 *  THE PROBLEM.  est() computes the covariance by inverting the Hessian that
 *  BFGS ACCUMULATES along the trajectory (raxopt leaves it in mtmp).  That is
 *  good for steering the search but is NOT the curvature at the optimum: it
 *  depends on the path taken and degrades precisely in the flattest directions,
 *  which are the ones with the largest standard errors.  This is not a
 *  suspicion -- drtran diagnosed and fixed it (BRIDGE_DESIGN.md 8c) -- and here
 *  the extreme symptom showed up: with -multistart, since the final est does
 *  not iterate, ALL the standard errors came out identical.
 *
 *  THE ALTERNATIVE WAS NOTED IN THE ENGINE ITSELF, commented out at
 *  drvmlest.c:104-107:  fdhess + choldcp.  That is what is used, without
 *  touching the engine: fdhess is a public symbol of qnewtopt.c and the
 *  objective is our own.
 *
 *  cov = 2 * f * H^-1 / n, the same formula est uses (drvmlest.c:111-119); all
 *  that changes is where H comes from.  Returns 0 if it worked.              */
static int exact_hessian_se(int npar, real *x, real *dev, real **cov, int neff)
{
    real **H = matrix(1, npar, 1, npar);
    real  *e = vector(1, npar);
    real d1, d2, pi1, pi2, pi3, f;
    int i, j, ifc = 0, ifault = 0;

    /* Allocate our own structure and fix the normalisation at the optimum. */
    fdh_varma.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    fdh_norm1 = fdh_norm2 = 1.0;
    vec_shootx(x, &fdh_varma, &ifault, 1, 0);
    if (ifault > 0) goto fail;
    elf(fdh_varma.m, fdh_varma.n, fdh_varma.p, fdh_varma.q, fdh_varma.mu,
        fdh_varma.phi, fdh_varma.theta, fdh_varma.qq, fdh_varma.w, 1.0,
        fdh_varma.xitol, FALSE, fdh_varma.a, &pi1, &pi2, &pi3, &ifault);
    if (ifault > 0) goto fail;
    fdh_norm1 = pi1; fdh_norm2 = pi2;

    f = fdh_obj(x);                      /* = 1 by construction */
    fdh_rej = 0;
    fdhess(fdh_obj, npar, x, f, macheps, H);
    {   /* Spectrum BEFORE the Cholesky, which destroys the matrix.  If it fails
           one has to be able to say BY HOW MUCH: one tiny negative eigenvalue is
           numerical noise in a flat direction, and several large ones are a
           saddle point -- i.e. the optimiser did not stop at a maximum.      */
        real **Hc = matrix(1, npar, 1, npar);
        real *wr = vector(1, npar), *wi = vector(1, npar);
        real mx = 0.0, mn = 0.0;
        int nneg = 0, k;
        for (i = 1; i <= npar; i++) for (j = 1; j <= npar; j++) Hc[i][j] = H[i][j];
        eigenqr(Hc, npar, wr, wi);
        for (k = 1; k <= npar; k++) {
            if (wr[k] > mx) mx = wr[k];
            if (wr[k] < mn) mn = wr[k];
            if (wr[k] <= 0.0) nneg++;
        }
        hess_nneg = nneg; hess_ratio = (mx > 0.0) ? -mn / mx : 0.0;
        free_vector(wi, 1, npar); free_vector(wr, 1, npar);
        free_matrix(Hc, 1, npar, 1, npar);
    }
    /*  BUG-34.  A perturbation that left the admissible region was answered
     *  with the 1e10 penalty, and that enters the second differences as an
     *  enormous "curvature" -- whether or not the Cholesky then succeeds.
     *  Checked only on its failure, it let VILL publish an MA coefficient of
     *  1.000043 with s.e. 0.000000 and t = 1.2e11 under the heading of the
     *  finite-difference Hessian.  So it is checked FIRST: any rejected
     *  evaluation means the optimum is on the boundary, and the BFGS s.e.
     *  are kept with the warning below.                                    */
    if (fdh_rej > 0) ifc = 1;
    else choldcp(H, npar, &d1, &d2, &ifc);
    if (ifc > 0) {
        /*  TWO distinct causes, and confusing them leads to saying something
         *  false.
         *
         *  (a) fdh_rej > 0: some finite-difference perturbation left the
         *      admissible region and was answered with the penalty.  Those rows
         *      and columns of the Hessian are NOT curvature -- they are the jump
         *      to the penalty -- so their spectrum means nothing and is not
         *      reported.  What it does say is that the optimum is ON the
         *      boundary: a constrained optimum, where the free curvature is not
         *      defined.  The roots reported above point at which one.
         *
         *  (b) fdh_rej == 0: the Hessian was formed entirely from valid
         *      evaluations and is indefinite even so.  There it IS informative,
         *      and the spectrum says by how much.                            */
        if (fdh_rej > 0)
            fprintf(stderr,
                "WARNING: -fdhess: the optimum lies ON the boundary of the admissible\n"
                "         region -- %ld of the finite-difference evaluations fell\n"
                "         outside it -- so the unconstrained Hessian is not defined\n"
                "         there and the BFGS standard errors are kept.  See the roots\n"
                "         of the AR and MA operators reported above: a modulus at one\n"
                "         identifies the binding direction.\n", fdh_rej);
        else
            fprintf(stderr,
                "WARNING: -fdhess: the Hessian at the reported optimum is not positive\n"
                "         definite: %d of %d eigenvalues are non-positive, the most\n"
                "         negative being %.3g times the largest positive one.  Every\n"
                "         evaluation was admissible, so the optimiser did not stop at\n"
                "         a maximum.  The BFGS standard errors are kept.\n",
                hess_nneg, npar, hess_ratio);
        goto fail;
    }
    for (i = 1; i <= npar; i++) {
        for (j = 1; j <= npar; j++) e[j] = 0.0;
        e[i] = 1.0;
        cholsol(H, npar, e);             /* e <- H^-1 e_i */
        for (j = 1; j <= npar; j++) cov[j][i] = (2.0 * f * e[j]) / neff;
        dev[i] = sqrt(cov[i][i] > 0.0 ? cov[i][i] : 0.0);
    }
    vec_shootx(x, &fdh_varma, &ifault, 0, 1);
    free_vector(e, 1, npar); free_matrix(H, 1, npar, 1, npar);
    return 0;
fail:
    vec_shootx(x, &fdh_varma, &ifault, 0, 1);
    free_vector(e, 1, npar); free_matrix(H, 1, npar, 1, npar);
    return 1;
}

/*****************************************************************************/
/*  report_operator_roots — moduli of the AR and MA roots at the optimum.     */
/*                                                                           */
/*  WHY THIS IS REPORTED.  drvec places nabla Y_2 in Ybar, so the second      */
/*  block is differenced by construction.  When the data do not need that     */
/*  differencing -- when the declared rank is too low -- the MA operator      */
/*  absorbs it with a root on the unit circle, which is the classical         */
/*  signature of overdifferencing.  The likelihood cannot go there: the       */
/*  engine's invertibility check (chekma, elfvarma.c) rejects any point whose */
/*  companion eigenvalue reaches 1.00005, so the optimiser stops ON the       */
/*  boundary.  The fit that results is a CONSTRAINED optimum, and standard    */
/*  errors from an unconstrained Hessian are not defined along that           */
/*  direction.  Reporting the roots is what lets the user see it.            */
/*                                                                           */
/*  The companion matrix is built exactly as chekma builds it, so the two     */
/*  agree by construction: for A(B) = I - A_1 B - ... - A_k B^k its           */
/*  eigenvalues are lambda = 1/z with z the roots of det A(z) = 0, whence the */
/*  modulus reported below is 1/|lambda|.                                     */
/*****************************************************************************/
/*  quiet = 1: computes only the smallest modulus and prints nothing.  The
 *  specification ladder needs it, wanting each rung's number without each
 *  rung's table of roots.                                                    */
static void report_operator_roots(const char *label, real ***A, int m, int k,
                                  real *minmod, int quiet)
{
    int mk = m * k, i, j, l;
    real **C, *wr, *wi;

    if (k <= 0) return;
    C  = matrix(1, mk, 1, mk);
    wr = vector(1, mk);
    wi = vector(1, mk);
    for (i = 1; i <= mk; i++)
        for (j = 1; j <= mk; j++) C[i][j] = 0.0;
    for (l = 1; l <= k; l++)
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++) C[i][j + (l - 1) * m] = A[l][i][j];
    for (l = 1; l <= k - 1; l++)
        for (j = 1; j <= m; j++) C[j + l * m][j + (l - 1) * m] = 1.0;

    eigenqr(C, mk, wr, wi);

    if (!quiet) fprintf(outputv, "  %-11s", label);
    for (i = 1; i <= mk; i++) {
        real lam = sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
        /* A null companion eigenvalue is an infinite root: it happens whenever
           the last coefficient matrix is singular, and it is no defect.      */
        if (lam <= 1.0e-12) {
            if (!quiet) fprintf(outputv, "  %8s ", "inf");
            continue;
        }
        if (1.0 / lam < *minmod) *minmod = 1.0 / lam;
        if (!quiet)
            fprintf(outputv, "  %8.5f%s", 1.0 / lam,
                    (1.0 / lam < 1.0001) ? "*" : " ");
    }
    if (!quiet) fprintf(outputv, "\n");

    free_vector(wi, 1, mk);
    free_vector(wr, 1, mk);
    free_matrix(C, 1, mk, 1, mk);
}

static void operator_roots(struct Tvarma *v)
{
    real minmod = 1.0e12, min_ar = 1.0e12, min_ma = 1.0e12;
    if (v->p <= 0 && v->q <= 0) return;
    /*  BUG-45: this said "Inverse roots" and prints the moduli of the ROOTS
     *  (1/|companion eigenvalue|), which is what "> 1" refers to.          */
    fprintf(outputv, "\nRoots of |phi(B)|=0 and |theta(B)|=0 "
                     "(moduli; > 1 is stationary/invertible):\n\n");
    report_operator_roots("AR (Phi)",   v->phi,   v->m, v->p, &min_ar, 0);
    report_operator_roots("MA (Theta)", v->theta, v->m, v->q, &min_ma, 0);
    minmod = (min_ar < min_ma) ? min_ar : min_ma;
    (void) minmod;
    if (min_ar < 1.0001)
        fprintf(outputv,
            "\n  * A root sits on the unit circle -- an AR root: the system is at\n"
            "    the edge of stationarity, which is where Lambda -> 0 puts it (the\n"
            "    rank-r model degenerating into the rank-(r-1) one).\n");
    if (min_ma < 1.0001) {
        /*  BUG-49.  WHICH direction carries the MA unit root.  Theta*(1) =
         *  I - sum Theta*_k is singular there; its left null vector u says which
         *  combination u'Ybar has the (1 - B) in its moving average, i.e. which
         *  combination of Ybar = [nabla Y2 ; W] the model is differencing once
         *  too often.  Its weights are printed by block, with names, and the
         *  reading depends on where they fall.                              */
        int M = v->m, r = global_r, s = M - r, i, k;
        gsl_matrix *T = gsl_matrix_alloc(M, M), *V = gsl_matrix_alloc(M, M);
        gsl_vector *sv = gsl_vector_alloc(M), *wk = gsl_vector_alloc(M);
        real w2 = 0.0, ww = 0.0;
        /*  u solves u'T = 0, i.e. T'u = 0: the right singular vector of T'
         *  for its smallest singular value.  gsl's SVD of A = T' gives
         *  A = U S V', and the last column of V is the null direction of A. */
        for (i = 0; i < M; i++)
            for (k = 0; k < M; k++) {
                real acc = (i == k) ? 1.0 : 0.0;
                for (int l = 1; l <= v->q; l++) acc -= v->theta[l][k + 1][i + 1];
                gsl_matrix_set(T, i, k, acc);            /* T' */
            }
        fprintf(outputv,
            "\n  * A root sits on the unit circle -- an MA root (modulus %.5f).  The\n"
            "    likelihood rose towards a non-invertible MA and the optimiser stopped\n"
            "    at the engine's invertibility gate: this is a CONSTRAINED point, not an\n"
            "    interior maximum.  The standard errors of the MA are not defined along\n"
            "    that direction, and no LR that uses this fit has its usual\n"
            "    distribution (BUG-49).\n", min_ma);
        if (gsl_linalg_SV_decomp(T, V, sv, wk) == 0) {
            fprintf(outputv,
                "    The combination the model differences once too often -- the left\n"
                "    null vector u of Theta*(1), smallest singular value %.2e -- has\n"
                "    weights (|u|, on Ybar = [nabla Y2 ; W]):\n",
                gsl_vector_get(sv, M - 1));
            for (i = 0; i < M; i++) {
                real ui = gsl_matrix_get(V, i, M - 1);
                if (i < s) w2 += ui * ui; else ww += ui * ui;
                if (i < s)
                    fprintf(outputv, "      nabla %-12s %8.4f\n",
                            series_names ? series_names[i + 1] : "Y2", fabs(ui));
                else
                    fprintf(outputv, "      W%-17d %8.4f\n", i - s + 1, fabs(ui));
            }
            if (r == 0 || w2 >= 0.8)
                fprintf(outputv,
                    "    It lies in the nabla Y2 block (share %.2f): that combination of the\n"
                    "    common-trend series looks OVER-DIFFERENCED -- stationary in levels.\n"
                    "    Read it as evidence that the rank is higher than %d, or that the\n"
                    "    series in that block are not I(1).\n", w2, r);
            else if (ww >= 0.8)
                fprintf(outputv,
                    "    It lies in the W block (share %.2f).  %s\n", ww,
                    (min_ar < 1.02)
                    ? "An AR root is near one as well: an\n"
                      "    AR/MA near-cancellation, i.e. a near common factor.  Read it as\n"
                      "    evidence that the equilibrium error is barely mean-reverting --\n"
                      "    that the rank may be LOWER than the one fitted."
                    : "The equilibrium error's own moving\n"
                      "    average is at the boundary: re-examine the MA order of W and the\n"
                      "    deterministic case before reading the estimates.");
            else
                fprintf(outputv,
                    "    It mixes both blocks (nabla Y2 share %.2f, W share %.2f): no single\n"
                    "    reading; re-examine the rank and the orders before reading the\n"
                    "    estimates.\n", w2, ww);
        }
        gsl_vector_free(wk); gsl_vector_free(sv);
        gsl_matrix_free(V); gsl_matrix_free(T);
    }
    /*  P4.4 — what those roots ARE in the structured class.  With the lower s
     *  rows of Theta zero, det Theta(x) = det(I_r - sum T11_k x^k): there are
     *  r*q finite roots and s*q at infinity, and the finite ones are the r x r
     *  block's.  Saying so changes what the reader has to check -- a scalar
     *  with M = 2, r = 1 -- and why that is enough.                          */
    if (v->q > 0 && ma_struct_on())
        fprintf(outputv,
            "  the %d finite MA root%s %s the %d x %d block's; the %d at "
            "infinity %s the zeroed rows\n",
            global_r * v->q, (global_r * v->q == 1) ? "" : "s",
            (global_r * v->q == 1) ? "is" : "are", global_r, global_r,
            (nser - global_r) * v->q,
            ((nser - global_r) * v->q == 1) ? "is" : "are");
}

/*****************************************************************************/
/*  F4 — parametric bootstrap for the rank test                              */
/*****************************************************************************/
/*  WHY.  Under H0 the rank statistic does NOT follow a chi2, and the
 *  asymptotic critical values -lrtest prints are measured to be insufficient at
 *  these sizes: over 20 replications of a process with a true r = 1 and
 *  n = 120, the test over-rejects by about THREE TIMES its nominal level
 *  (HOMOLOGATION.md 2.3).  Melard, Roy and Saidi say it for this very class of
 *  models: the MA terms do not alter the asymptotic distribution of the LR, but
 *  "finite sample performance of the test is affected by the MA terms".
 *
 *  HOW.  Parametric bootstrap: N samples are simulated UNDER H0 with the
 *  parameters estimated at rank r, the LR(r -> r+1) statistic is recomputed on
 *  each, and the empirical percentiles are the critical values.  It is what
 *  BVECM 6.3-6.5 prescribes.
 *
 *  THE SIMULATION EXPLOITS THE TRANSFORMATION, instead of reimplementing a VEC:
 *  the fitted model IS a stationary VARMA on Ybar, so it is simulated there --
 *  with elf's convention, (w-mu) = SUM phi (w-mu) + a - SUM theta a -- and the
 *  transformation is INVERTED to get back to levels:
 *
 *      nabla Y2 = Ybar[1..s]         -> Y2 by cumulating from the real level
 *      Y1       = W - B2' Y2          with W = Ybar[s+1..M]
 *
 *  That leaves a sample in the same format as the .inp, so the re-estimations
 *  are EXACTLY those of the normal route, with no parallel code that could
 *  drift from the one being calibrated.
 *
 *  The generator is deterministic with a fixed seed: a critical value that
 *  cannot be reproduced is no use for deciding anything.                     */

/*  uint64_t, not unsigned long: the generator is 64-bit by construction, and
 *  `unsigned long` is 32 bits on Windows (LLP64).  There the state was
 *  truncated, `>> 33` was wider than the type, and boot_normal's rejection
 *  loop never ended (BUG-61).  On LP64 the two spellings are the same type, so
 *  the sequence is the one every .boot and .irfboot was measured with.      */
static uint64_t boot_rng = 987654321ULL;

static real boot_normal(void)
{
    /* Box-Muller over an LCG of our own; deterministic and independent of libc. */
    static int have = 0;
    static real spare = 0.0;
    real u1, u2, r, th;
    if (have) { have = 0; return spare; }
    do {
        boot_rng = boot_rng * 6364136223846793005ULL + 1442695040888963407ULL;
        u1 = ((real)((boot_rng >> 33) & 0x7FFFFFFF)) / 2147483648.0;
    } while (u1 <= 1.0e-12);
    boot_rng = boot_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    u2 = ((real)((boot_rng >> 33) & 0x7FFFFFFF)) / 2147483648.0;
    r  = sqrt(-2.0 * log(u1));
    th = 2.0 * M_PI * u2;
    spare = r * sin(th); have = 1;
    return r * cos(th);
}

/*  simulate_h0 — one sample of LEVELS under v's model, with B2 given.
 *  out is expected dimensioned (1..nobs_raw, 1..M).  Returns 0 if it worked.  */
static int simulate_h0(struct Tvarma *v, real **B2, int r, real **out)
{
    int M = nser, s = M - r, n = v->n, p = v->p, q = v->q;
    int burn = 50 + 10 * p, T = n + burn;
    real **wb = matrix(1, T, 1, M);
    real **ab = matrix(1, T, 1, M);
    real **L  = matrix(1, M, 1, M);
    real d1, d2;
    int t, i, j, k, ifc = 0;

    /* Cholesky of Sigma* = sigma2 * qq to give the shocks their covariance. */
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) L[i][j] = v->sigma2 * v->qq[i][j];
    choldcp(L, M, &d1, &d2, &ifc);
    if (ifc > 0) { free_matrix(L,1,M,1,M); free_matrix(ab,1,T,1,M);
                   free_matrix(wb,1,T,1,M); return 1; }
    for (i = 1; i <= M; i++) for (j = i+1; j <= M; j++) L[i][j] = 0.0;

    for (t = 1; t <= T; t++) {
        real *z = vector(1, M);
        for (i = 1; i <= M; i++) z[i] = boot_normal();
        for (i = 1; i <= M; i++) {
            real acc = 0.0;
            for (k = 1; k <= i; k++) acc += L[i][k] * z[k];
            ab[t][i] = acc;
        }
        free_vector(z, 1, M);
        /* (w - mu) = SUM phi_j (w - mu)_{t-j} + a_t - SUM theta_j a_{t-j} */
        for (i = 1; i <= M; i++) {
            real acc = ab[t][i];
            for (j = 1; j <= p; j++) if (t-j >= 1)
                for (k = 1; k <= M; k++) acc += v->phi[j][i][k] * (wb[t-j][k] - v->mu[k]);
            for (j = 1; j <= q; j++) if (t-j >= 1)
                for (k = 1; k <= M; k++) acc -= v->theta[j][i][k] * ab[t-j][k];
            wb[t][i] = v->mu[i] + acc;
        }
    }

    /* Invert the transformation.  Y2's origin is the real one: in case 1 the
       constant is not free, so an arbitrary origin would contaminate W.      */
    for (i = 1; i <= M; i++) out[1][i] = rawmat[1][i];
    for (t = 1; t <= n; t++) {
        int tb = t + burn;
        for (i = 1; i <= s; i++) out[t+1][i] = out[t][i] + wb[tb][i];
        for (j = 1; j <= r; j++) {
            real w = wb[tb][s+j];
            for (i = 1; i <= s; i++) w -= B2[i][j] * out[t+1][i];
            out[t+1][s+j] = w;
        }
    }
    free_matrix(L, 1, M, 1, M);
    free_matrix(ab, 1, T, 1, M);
    free_matrix(wb, 1, T, 1, M);
    return 0;
}

/*  fit_ll — re-estimates at rank rr and returns the logL, or 0 with ok = 0. */
/*  Defined further down, with the ladder (P12).                             */
static int  ladder_wanted(void);
static int  gate_profile_seed(real *x, int np);
static void ladder_seed_r0(real *x, int np);

static real fit_ll(int rr, int *ok)
{
    int np, ifr = 0;
    real *xr, *devr, **covr, ll = 0.0;
    struct Tvarma vr;
    int save_r = global_r;

    global_r = rr;
    build_y2_levels();
    np = calc_nparametrs();
    xr = vector(1, np); devr = vector(1, np); covr = matrix(1, np, 1, np);
    /*  P12: a replication is fitted as the observed data are -- the best of
     *  several starts -- or the simulated distribution would be that of a
     *  worse estimator than the one it is compared with.                  */
    (void) vr;
    init_guess(xr, np);
    /*  ... and started as the observed data are: the ladder at every rank
     *  (BUG-50).  A replication fitted from a worse start than the data
     *  would put its L(0) in a local optimum, inflate LR* and push the
     *  bootstrap's critical values up.                                     */
    if (rr > 0 && ladder_wanted())       gate_profile_seed(xr, np);
    else if (rr == 0 && ladder_wanted()) ladder_seed_r0(xr, np);
    {
        real s2r;
        ifr = fit_search(xr, np, devr, covr, &ll, &s2r, 1, 0, NULL);
    }
    *ok = (ifr == 0);
    if (!*ok) ll = 0.0;
    free_matrix(covr, 1, np, 1, np); free_vector(devr, 1, np); free_vector(xr, 1, np);
    global_r = save_r;
    return ll;
}

/*  refresh_b2_fixed -- B2_fixed as init_guess builds it for rank rr on the data
 *  now in rawmat (the -fixb2 value, or the static OLS one).  Leaves global_r
 *  at rr.                                                                    */
static void refresh_b2_fixed(int rr)
{
    int np;
    real *xt;
    global_r = rr;
    build_y2_levels();
    np = calc_nparametrs();
    xt = vector(1, np);
    init_guess(xt, np);
    free_vector(xt, 1, np);
}

static int cmp_real(const void *a, const void *b)
{
    real x = *(const real *)a, y = *(const real *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/*  boot_dump -- <base>.boot, EVERY REPLICATION OF THE RANK BOOTSTRAP.
 *
 *  The .out gives three percentiles, a p-value and a count.  Homologating a
 *  port against that only says "the same within Monte Carlo error", and when
 *  one replication converges on one side and not on the other nothing points
 *  at which.  So each replication is written as it ends, in the order the
 *  generator produced it: the H0 rank, the replication number, what became of
 *  it (ok; simfail, Sigma* without a Cholesky factor, before any draw is
 *  consumed; fitfail, one of the two refits failed) and its LR* to 17
 *  digits.  It is what a replication-by-replication comparison needs.      */
static FILE *boot_dump = NULL;

/*  bootstrap_rank — critical values of LR(rr -> rr+1) under H0: rank = rr.
 *  x is the fit at rank rr.  Returns the number of usable replications.      */
static int bootstrap_rank(int rr, real *x, int npar, int N, real *cv, real *pval,
                          real lr_obs)
{
    int M = nser, s = M - rr, i, j, b, nok = 0, ifr = 0;
    real **B2 = matrix(1, (s > 0 ? s : 1), 1, (rr > 0 ? rr : 1));
    real **sim = matrix(1, nobs_raw, 1, M);
    real **saved = matrix(1, nobs_raw, 1, M);
    real *stat = vector(1, N);
    struct Tvarma vh;
    FILE *save_out = outputv;
    int save_quiet = quiet_mode, save_r = global_r, ge = 0;

    /*  With -fixb2, B2 is not in x[] but in B2_fixed, which init_guess
     *  reallocates for whatever rank ran last -- M-1 after -lrtest, and each
     *  replication's own inside the loop below.  So it is rebuilt for rank rr
     *  on the observed data before it is read, and again after them (BUG-31,
     *  case 2: the H0 model was the previous replication's, then a crash).  */
    if (global_fixb2) refresh_b2_fixed(rr);
    /* B2 of the fit: last s*rr entries of x[], column-major (or held). */
    if (rr > 0) {
        int idx = npar - s * rr + 1;
        for (j = 1; j <= rr; j++) for (i = 1; i <= s; i++)
            B2[i][j] = global_fixb2 ? B2_fixed[i][j] : x[idx++];
    }
    /* The model under H0, recovered from the fit. */
    global_r = rr; build_y2_levels();
    vh.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x, &vh, &ifr, 1, 0);
    {   /* fill sigma2/logelf by evaluating: vec_shootx does not set them */
        real pi1, pi2, pi3; int ife = 0;
        const real LOG2PI = 1.837877066;
        elf(vh.m, vh.n, vh.p, vh.q, vh.mu, vh.phi, vh.theta, vh.qq, vh.w, 1.0,
            vh.xitol, TRUE, vh.a, &pi1, &pi2, &pi3, &ife);
        vh.sigma2 = pi1 / (vh.n * vh.m);
        vh.logelf = -0.5*vh.m*vh.n*(LOG2PI - log((real)vh.m) - log((real)vh.n) + 1.0)
                    - 0.5*vh.n*(vh.m*log(pi1) + log(pi2));
    }

    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) saved[i][j] = rawmat[i][j];

    outputv = fopen("/dev/null", "w"); quiet_mode = 1;
    for (b = 1; b <= N; b++) {
        int ok0 = 0, ok1 = 0;
        real l0, l1;
        if (simulate_h0(&vh, B2, rr, sim) != 0) {
            if (boot_dump) {
                fprintf(boot_dump, "%d,%d,simfail,\n", rr, b);
                fflush(boot_dump);
            }
            continue;
        }
        for (i = 1; i <= nobs_raw; i++)
            for (j = 1; j <= M; j++) rawmat[i][j] = sim[i][j];
        l0 = fit_ll(rr,   &ok0);
        l1 = fit_ll(rr+1, &ok1);
        if (boot_dump) {
            if (ok0 && ok1)
                fprintf(boot_dump, "%d,%d,ok,%.17g\n", rr, b, 2.0 * (l1 - l0));
            else
                fprintf(boot_dump, "%d,%d,fitfail,\n", rr, b);
            /*  BUG-63: each line is flushed as it is written.  The .boot is
             *  also the PROGRESS of the bootstrap -- the Python port and sima
             *  read it while the C runs -- and the library's 4 KB buffer held
             *  some 160 replications, minutes of a trivariate test, before
             *  anything reached the file.                                    */
            fflush(boot_dump);
        }
        /* Every converged draw counts, negative ones included: the exact LR
           is not nested at Lambda = 0 and goes below zero under H0 in about
           40 % of samples.  Dropping them censored the null from below and
           pushed the quantiles up (BUG-26).                                 */
        if (ok0 && ok1) stat[++nok] = 2.0 * (l1 - l0);
    }
    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) rawmat[i][j] = saved[i][j];
    if (outputv) fclose(outputv);
    outputv = save_out; quiet_mode = save_quiet;
    if (global_fixb2) refresh_b2_fixed(rr);
    global_r = rr; build_y2_levels();
    vec_shootx(x, &vh, &ifr, 0, 1);

    if (nok >= 10) {
        /* Quantile (1-alpha): index ceil((1-alpha)*nok), bounded.  With the
           round-to-nearest that was there before, the 99th percentile of 97
           values landed in slot 96 and left TWO above it, i.e. a 2% where 1%
           was being asked for.                                              */
        int i90 = (int) ceil(0.90 * nok), i95 = (int) ceil(0.95 * nok),
            i99 = (int) ceil(0.99 * nok);
        qsort(&stat[1], (size_t) nok, sizeof(real), cmp_real);
        if (i90 < 1) i90 = 1;
        if (i90 > nok) i90 = nok;
        if (i95 < 1) i95 = 1;
        if (i95 > nok) i95 = nok;
        if (i99 < 1) i99 = 1;
        if (i99 > nok) i99 = nok;
        cv[0] = stat[i90]; cv[1] = stat[i95]; cv[2] = stat[i99];
        for (i = 1; i <= nok; i++) if (stat[i] >= lr_obs) ge++;
        *pval = (real) (ge + 1) / (real) (nok + 1);   /* p-valor bootstrap */
    }
    free_vector(stat, 1, N);
    free_matrix(saved, 1, nobs_raw, 1, M);
    free_matrix(sim, 1, nobs_raw, 1, M);
    free_matrix(B2, 1, (s > 0 ? s : 1), 1, (rr > 0 ? rr : 1));
    global_r = save_r;
    return nok;
}

/*****************************************************************************/
/*  bootstrap_ma — the distribution of the LR between the INHERITED MA and   */
/*  the free one, simulated under the restricted model.                      */
/*                                                                           */
/*  WHY THE chi2 IS NOT ENOUGH.  The statistic compares q*r*r moving-       */
/*  average parameters against q*M*M, i.e. df = q(M^2 - r^2), and read that  */
/*  way the restricted model is rejected in all eight pairs.  But that      */
/*  reading does not hold: the UNRESTRICTED optimum stops in the            */
/*  neighbourhood where the rank condition degenerates -- G between 0.016   */
/*  and 0.133 against ~1 for a well specified model, with Theta(1) singular */
/*  and an LR whose unrestricted estimator sits on the edge of the          */
/*  admissible region does not have its asymptotic distribution.  Same      */
/*  warning 3b carries for the standard errors, applied to the test.       */
/*                                                                           */
/*  What can be done without an asymptotic distribution is to simulate the  */
/*  one there is: generate under the RESTRICTED model -- which is H0 -- and */
/*  see where the observed statistic falls in that distribution.  Each      */
/*  replication costs TWO fits, restricted and free, just like the observed.*/
/*****************************************************************************/
/*  set_spec — THE LADDER, IN ONE PLACE ONLY.  0 warma, 1 mawarma, 2 marow,
 *  3 matri, 4 free.  It exists because the bootstrap has to be able to set and
 *  clear a whole specification without forgetting a flag, and because four
 *  flags set by hand in five places is how a combination nobody wanted gets
 *  in.                                                                       */
static void set_spec(int k)
{
    global_warma = (k == 0); global_mawarma = (k == 1);
    global_marow = (k == 2); global_matri   = (k == 3);
}

static int bootstrap_ma(real *x0, int npar0, int N, real *cv, real *pval,
                        real lr_obs, int k0, int k1)
{
    int M = nser, r = global_r, s = M - r, i, j, b, nok = 0, ifr = 0, ge = 0;
    real **B2  = matrix(1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
    real **sim = matrix(1, nobs_raw, 1, M);
    real **saved = matrix(1, nobs_raw, 1, M);
    real *stat = vector(1, N);
    struct Tvarma vh;
    FILE *save_out = outputv;
    int save_quiet = quiet_mode, save_wa = global_mawarma;

    /*  The model under H0 is the RESTRICTED one, so it is recovered with ITS
     *  specification in place: with another, vec_shootx would read another
     *  vector.                                                               */
    set_spec(k0);
    {
        int idx = npar0 - s * r + 1;
        for (j = 1; j <= r; j++) for (i = 1; i <= s; i++)
            B2[i][j] = global_fixb2 ? B2_fixed[i][j] : x0[idx++];
    }
    build_y2_levels();
    vh.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x0, &vh, &ifr, 1, 0);
    {
        real pi1, pi2, pi3; int ife = 0;
        const real LOG2PI = 1.837877066;
        elf(vh.m, vh.n, vh.p, vh.q, vh.mu, vh.phi, vh.theta, vh.qq, vh.w, 1.0,
            vh.xitol, TRUE, vh.a, &pi1, &pi2, &pi3, &ife);
        vh.sigma2 = pi1 / (vh.n * vh.m);
        vh.logelf = -0.5*vh.m*vh.n*(LOG2PI - log((real)vh.m) - log((real)vh.n) + 1.0)
                    - 0.5*vh.n*(vh.m*log(pi1) + log(pi2));
    }

    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) saved[i][j] = rawmat[i][j];

    outputv = fopen("/dev/null", "w"); quiet_mode = 1;
    for (b = 1; b <= N; b++) {
        int ok0 = 0, ok1 = 0;
        real l0, l1;
        set_spec(k0);
        if (simulate_h0(&vh, B2, r, sim) != 0) continue;
        for (i = 1; i <= nobs_raw; i++)
            for (j = 1; j <= M; j++) rawmat[i][j] = sim[i][j];
        set_spec(k0); l0 = fit_ll(r, &ok0);
        set_spec(k1); l1 = fit_ll(r, &ok1);
        /*  Replications where the free fit ends up BELOW the restricted one are
         *  discarded: the restricted model is nested, so a negative LR is a fit
         *  that did not converge, not a realisation of the statistic.        */
        /* Every converged draw counts, negative ones included: the exact LR
           is not nested at Lambda = 0 and goes below zero under H0 in about
           40 % of samples.  Dropping them censored the null from below and
           pushed the quantiles up (BUG-26).                                 */
        if (ok0 && ok1) stat[++nok] = 2.0 * (l1 - l0);
    }
    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) rawmat[i][j] = saved[i][j];
    if (outputv) fclose(outputv);
    outputv = save_out; quiet_mode = save_quiet;
    set_spec(k0);
    build_y2_levels();
    vec_shootx(x0, &vh, &ifr, 0, 1);
    set_spec(-1); global_mawarma = save_wa;

    if (nok >= 10) {
        int i90 = (int) ceil(0.90 * nok), i95 = (int) ceil(0.95 * nok),
            i99 = (int) ceil(0.99 * nok);
        qsort(&stat[1], (size_t) nok, sizeof(real), cmp_real);
        if (i90 < 1) i90 = 1;
        if (i90 > nok) i90 = nok;
        if (i95 < 1) i95 = 1;
        if (i95 > nok) i95 = nok;
        if (i99 < 1) i99 = 1;
        if (i99 > nok) i99 = nok;
        cv[0] = stat[i90]; cv[1] = stat[i95]; cv[2] = stat[i99];
        for (i = 1; i <= nok; i++) if (stat[i] >= lr_obs) ge++;
        *pval = (real) (ge + 1) / (real) (nok + 1);
    }
    free_vector(stat, 1, N);
    free_matrix(saved, 1, nobs_raw, 1, M);
    free_matrix(sim, 1, nobs_raw, 1, M);
    free_matrix(B2, 1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
    return nok;
}

/*  Convergence note — WHY it stopped, not just WHETHER it stopped.
 *
 *  In multivariate VARMA the reason for stopping is a first-order diagnostic:
 *  ill-conditioned likelihoods, near non-identification and common factors all
 *  show up as termination on steptol rather than on the gradient.  The suite
 *  had already established this -- drvarma fixed it in its
 *  report._convergence_block and drtran exposes it as Fit.convergence_note --
 *  and drvec printed the criterion without saying what it means.
 *
 *  TWO THINGS THIS NOTE FIXES, both inherited:
 *
 *   - termcode 2 (steptol) was announced as a flat "OPTIMIZER CONVERGED".  It
 *     is, in the program's sense, but it is the typical symptom of an
 *     ill-conditioned likelihood and the standard errors are not to be trusted.
 *   - "ESTIMATION SUCCESSFUL (ifault=0)" reads as convergence and IS NOT:
 *     ifault is MODEL adequacy, not the optimiser's (drvarma documents this
 *     explicitly).  A fit that stopped far from an optimum can perfectly well
 *     report ifault = 0.
 *
 *  HOW THE TERMCODE IS OBTAINED.  est() does not return it, and report() lives
 *  in qnewtopt.c, which is ENGINE and is not touched -- not by one line, not
 *  even to expose an observable.  So it is read back from the text report()
 *  already wrote into the .out itself.  That is fragile with respect to that
 *  string and nothing else, and the alternative was editing published, refereed
 *  code.
 *
 *  Returns the termcode 1..5, or 0 if it could not be determined.            */
static int termcode_from_out(const char *path)
{
    FILE *f;
    char line[512];
    int code = 0;

    fflush(outputv);
    f = fopen(path, "r");
    if (!f) return 0;
    while (fgets(line, sizeof line, f)) {
        if (!strstr(line, "Convergence criterion:")) continue;
        if      (strstr(line, "gradtol"))        code = 1;
        else if (strstr(line, "steptol"))        code = 2;
        else if (strstr(line, "lower point"))    code = 3;
        else if (strstr(line, "iteration limit"))code = 4;
        else if (strstr(line, "maximum length")) code = 5;
    }
    fclose(f);
    return code;
}

/*  convergence_note — the interpretation, in the output and on the console. */
static void convergence_note(int code)
{
    const char *note = NULL, *head = NULL;

    /*  One line each, in the shape drvarma's convergence block uses: a verdict
     *  and, where the verdict changes how the fit must be read, a marked
     *  warning.  What each termination code MEANS is argued in
     *  docs/CONVERGENCE.md, which is where an argument belongs.              */
    switch (code) {
    case 1:
        head = "clean convergence, on the scaled gradient";
        note = NULL;
        break;
    case 2:
        head = "stopped on steptol, NOT on the gradient";
        note = "the step collapsed while the gradient may still be appreciable; "
               "treat the standard errors with caution";
        break;
    case 3:
        head = "NOT a convergence: the line search failed to improve";
        note = "the estimates are a stationary-ish point, not a demonstrated "
               "maximum.  See docs/CONVERGENCE.md";
        break;
    case 4: case 5:
        head = "NOT a convergence: the optimiser gave up";
        note = "every criterion derived from this fit -- standard errors, "
               "AIC/BIC, any LR -- is unreliable";
        break;
    default:
        head = "the termination criterion could not be read";
        note = NULL;
        break;
    }

    fprintf(outputv, "Convergence      : %s\n", head);
    if (note) fprintf(outputv, "  ! %s\n", note);
    if (!quiet_mode && code != 1)
        printf("  Convergence note: %s.  See the .out and docs/CONVERGENCE.md\n",
               head);
}

/*  load_alpha_A — reads the matrix A of the alpha = A*psi restriction.
 *
 *  Format, deliberately simple and ASCII: a first line with  M sa  and then M
 *  rows of sa numbers.  Lines beginning with '*' or '#' are comments.  fue's
 *  positional format is not copied here because this is not a file of the
 *  ladder: it is a hypothesis of the user's.
 *
 *  A's rank is checked through its Gram matrix: if A'A is singular the
 *  restriction does not identify psi, and that has to be said before
 *  estimating, not after.                                                    */
static int load_alpha_A(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[1024];
    int mm = 0, sa = 0, i, j, got = 0;

    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return 1; }
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '*' || line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%d %d", &mm, &sa) == 2) { got = 1; break; }
    }
    if (!got || mm != nser || sa < 1 || sa > nser) {
        fprintf(stderr, "ERROR: %s must start with 'M sa', with M = %d and "
                        "1 <= sa <= %d (leido %d %d)\n", path, nser, nser, mm, sa);
        fclose(f); return 1;
    }
    alpha_A  = matrix(1, nser, 1, sa);
    alpha_sa = sa;
    for (i = 1; i <= nser; i++) {
        char *tok;
        do { if (!fgets(line, sizeof line, f)) {
                 fprintf(stderr, "ERROR: %s ends at row %d\n", path, i);
                 fclose(f); return 1; }
        } while (line[0] == '*' || line[0] == '#' || line[0] == '\n');
        tok = strtok(line, " \t\n");
        for (j = 1; j <= sa; j++) {
            if (!tok) { fprintf(stderr, "ERROR: %s, fila %d: faltan columnas\n",
                                path, i); fclose(f); return 1; }
            alpha_A[i][j] = atof(tok);
            tok = strtok(NULL, " \t\n");
        }
    }
    fclose(f);

    {   /* rango de A via A'A */
        real **G = matrix(1, sa, 1, sa);
        real d1, d2; int ifc = 0;
        for (i = 1; i <= sa; i++) for (j = 1; j <= sa; j++) {
            real acc = 0.0; int k;
            for (k = 1; k <= nser; k++) acc += alpha_A[k][i] * alpha_A[k][j];
            G[i][j] = acc;
        }
        choldcp(G, sa, &d1, &d2, &ifc);
        free_matrix(G, 1, sa, 1, sa);
        if (ifc > 0) {
            fprintf(stderr, "ERROR: A does not have rank %d (A'A is singular), so psi\n"
                            "       is not identified\n", sa);
            return 1;
        }
    }
    return 0;
}

/*  build_weakex_A — the A that declares equation `eq` weakly exogenous.
 *  It is the M x M identity without its column eq: alpha_eq = 0 for every j.
 *  Weak exogeneity needs no test of its own, it is H1(r) with this A.        */
static int build_weakex_A(int eq)
{
    int i, j, c;
    if (eq < 1 || eq > nser) {
        fprintf(stderr, "ERROR: -weakex %d is outside 1..%d\n", eq, nser);
        return 1;
    }
    /*  -weakex i NAMES THE SERIES BY ITS POSITION IN THE .inp, which is what a
     *  user has in front of them, and the restriction has to be put on the row
     *  of Lambda that belongs to it.  Those two are not the same index when
     *  r < M: Lambda is written in the internal order [Y1 ; Y2] (see inp2lam).
     *  Until 2026-08-24 this zeroed the internal row directly, so on any fit
     *  with r < M it declared the wrong series weakly exogenous.  Part of
     *  BUG-17.                                                              */
    /*  BUG-28/29: A is kept in the .inp's order -- the order the user writes
     *  it in -- and permuted to the internal one where it meets Lambda, with
     *  the r of the fit in hand (lam2inp).  Permuting it HERE, once, fixed it
     *  to the command-line r: -lrtest then restricted a different series at
     *  every other rank.                                                    */
    alpha_sa = nser - 1;
    alpha_A  = matrix(1, nser, 1, (alpha_sa > 0 ? alpha_sa : 1));
    for (i = 1; i <= nser; i++)
        for (j = 1; j <= alpha_sa; j++) alpha_A[i][j] = 0.0;
    for (i = 1, c = 0; i <= nser; i++) {
        if (i == eq) continue;
        alpha_A[i][++c] = 1.0;
    }
    return 0;
}

/*****************************************************************************/
/*  F2 — the bridge to the suite: drvec writes .inp and reads .pre          */
/*                                                                           */
/*  See docs/PLAN_BETA.md F2 for the full study.  What governs this block,  */
/*  in three sentences:                                                    */
/*                                                                           */
/*   - drvec WRITES .inp (a specification) and never .pre (a claim of       */
/*     optimality, which only whoever estimated can make).                  */
/*   - fue's parser is POSITIONAL and validates nothing, so the sections    */
/*     all go, and in order, including the annual-difference factor one,    */
/*     which with annual data carries a literal " 0" but MUST be there.     */
/*   - Everything written is pure ASCII: fue's Python parser does not read  */
/*     Latin-1 (BUG-0010, open) and this engine's sources are.              */
/*****************************************************************************/

static int   global_writeinp = 0;    /* -writeinp <prefix>  (components of Ȳ) */
static int   global_writeres = 0;    /* -writeres <prefijo>  (residuos)         */
static char *inp_prefix      = NULL;   /* -writeinp's */
static char *res_prefix      = NULL;   /* -writeres's: its own (BUG-45) */
static int   global_eval     = 0;    /* -eval: evaluate and exit, without optimising */
static int   global_fdhess   = 0;    /* -fdhess: standard errors from a
                                        finite-difference Hessian at the optimum */
static int   global_boot     = 0;    /* -bootstrap N: critical values by a
                                        parametric bootstrap under H0         */
static int   global_interv   = 0;    /* -interv <prefix>: deterministics from the .pre */
static char *interv_prefix   = NULL;
static int   global_multistart = 0;  /* -multistart n: n starts, keep the best */
static int   global_fcast = 0;       /* -f H: forecast horizon (P5)                   */
static const char *fc_csv = NULL;    /* -C file: per-origin errors, for the DM test    */
/*  P6.7 — THE BASE NAME, GLOBAL.  The suite's file system names every
 *  product after the same prefix -- <base>.out, <base>.forecast,
 *  <base>.recursive -- and until now that prefix lived only inside main(), so
 *  the forecast had no way to write its own file and ended up inside the
 *  ESTIMATION report.  See docs/PLAN_PRODUCCION.md 7.1.                      */
static char out_base[512] = "";

/*  P1 — the exit code reflects a fit that did not converge.  It is at file
 *  scope because report_fit() sets it and main() reads it: the two used to be
 *  one function, and this is the only piece of state that crossed the cut.   */
static int estimation_failed = 0;

/*  P10 — THE SEPARATORS OF THE SUITE'S .out.  61 '=' and 68 '-', which is what
 *  drvarma's report uses (report.py, EQ and DASH) and what the C engine's own
 *  report() writes.  A .out that separates its sections some other way is a
 *  .out a reader of the suite has to learn again.                            */
#define EQBAR   "============================================================="
#define DASHBAR "------------------------------------------------------------------------"

/*  banner — a titled section, in the suite's shape.                          */
static void banner(const char *title)
{
    fprintf(outputv, "\n\n%s\n  %s\n%s\n", EQBAR, title, EQBAR);
}

/*  sig_code / par_row — one row of the parameter table: name, estimate,
 *  standard error, t and its two-sided p, with the significance codes
 *  drvarma prints (report.py:_parameters_block).  drvec used to print
 *  `0.608562 (sd 0.263137)` glued to the matrix -- no t, no p, and impossible
 *  to read down a column.                                                    */
static const char *sig_code(real p)
{
    return (p < 0.001) ? "***" : (p < 0.01) ? "**"
         : (p < 0.05)  ? "*"   : (p < 0.10) ? "." : " ";
}

/*  inp2lam — the row of Lambda (and of psi, and of alpha) that belongs to the
 *  series sitting at position i of the .inp.
 *
 *  THE TWO ORDERS, and this is where they meet.  The data, the series names,
 *  Theta, Gamma, the responses and the forecast are all in the .inp's order,
 *  [Y2 block ; Y1 block].  Lambda and B are NOT: the cast writes the system in
 *  the INTERNAL order [Y1 ; Y2] -- Cbar maps [Y1 ; Y2] to Ybar = [nabla Y2 ; W]
 *  and PhBar[1] = Cinv*Hbar - LamBar, whose rows are Cinv's, which are
 *  internal.  So Lambda's row 1..r is the Y1 block and r+1..M the Y2 block.
 *
 *  Until 2026-08-24 the report labelled Lambda's rows with series_names[i]
 *  straight, so on any fit with r < M it named the WRONG SERIES -- the weak
 *  exogeneity tests included, since P6.8.  Found while building the
 *  beta' gain = 0 certificate, which is the first thing in the program that
 *  had to multiply beta by something in the .inp's order and therefore the
 *  first that could not paper over the mismatch.
 *
 *  Everything the report shows is put in the .inp's order through here, so the
 *  reader sees one order and only one.                                       */
static int inp2lam(int i)
{
    int s = nser - global_r;
    return (i <= s) ? global_r + i : i - s;
}

/*  The inverse: the .inp column of internal row a.  Used to NAME a row of
 *  something that is walked in the internal order (the parameter vector's Q
 *  block, whose [1][1] is Y1's variance and not the first column's).       */
static int lam2inp(int a)
{
    int s = nser - global_r;
    return (a <= global_r) ? s + a : a - global_r;
}

/*  sname / cname — the names the rows are labelled with.  A parameter that
 *  says `Lambda[2,1]` needs a legend; one that says `Lambda[mink <- ec1]` does
 *  not, and that is the whole difference between a table a VECM reader can use
 *  and one they have to decode.  cname is for the components of
 *  Ybar = (nabla Y2', W')': the differenced series, and the equilibrium
 *  errors, which is what the W block is.                                     */
static const char *sname(int i)
{
    static char buf[4][24];
    static int  turn = 0;
    char *b = buf[turn = (turn + 1) & 3];
    snprintf(b, 24, "%.20s", (series_names && i >= 1 && i <= nser)
                             ? series_names[i] : "y");
    return b;
}

static int ma_on_boundary = 0;   /* the fitted MA has a unit root (BUG-49) */

static void par_row(const char *label, real est, real se)
{
    if (se > 0.0) {
        real t  = est / se;
        real pv = 2.0 * gsl_cdf_ugaussian_Q(fabs(t));
        fprintf(outputv, "%-28s %13.6f %12.6f %8.3f %7.4f %s\n",
                label, est, se, t, pv, sig_code(pv));
    } else {
        fprintf(outputv, "%-28s %13.6f %12s %8s %7s\n",
                label, est, "-", "-", "-");
    }
}

/*  P8 — the three file names, at file scope.  They were locals of main(), and
 *  every mode that was extracted out of main() needed all three: the .out to
 *  write to, the input to name in it, and the base to free at the end.  Three
 *  parameters repeated on six signatures say less than one declaration here,
 *  and they are exactly as global as out_base already was.                   */
static STRING inputf = NULL, outputf = NULL, base_name = NULL;

/*  P8 — the optimiser's stopping rules, at file scope.  They were locals of
 *  main() and every mode that estimates needs them; they are constants, they
 *  are the same for every fit the program makes, and passing four of them down
 *  six signatures said nothing that this declaration does not.               */
static const int  maxits = 500, nrits = 200;
static const real gradtol = 1e-5, sptol = 1e-7;

/*  P9 — THE .pre ROUTE'S STATE.  pre_route says which of the two interfaces was
 *  used; i_pqr and first_opt say where p q r and the options begin, which on
 *  the .inp route are 2 and 5 and on the .pre one depend on how many files were
 *  given.  model_name is what the products are named after: <name>.out,
 *  <name>.forecast, <name>.recursive.  drtran calls that option -m; here -m is
 *  already the estimation method, inherited from drvarma, so the option is
 *  -name and the collision is declared instead of resolved by moving a letter
 *  that appears in the register.                                             */
static int    pre_route  = 0;
static char **pre_files  = NULL;
static int    n_pre      = 0;
static int    i_pqr      = 2;
static int    first_opt  = 5;
static const char *model_name = NULL;

/*  ends_with — the suffix test the route detection is made of.               */
static int ends_with(const char *s, const char *suf)
{
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

/*  path_stem — "dir/ES_CPI.pre" -> "ES_CPI".  Used to name the products after
 *  the files they came from, which is what drtran does with <output>_<input>. */
static void path_stem(const char *path, char *out, size_t n)
{
    const char *b = strrchr(path, '/');
    const char *dot;
    size_t k, lim;
    b = b ? b + 1 : path;
    /*  The LAST dot, not the first: `mmpre.muskrat.pre` is one series called
     *  mmpre.muskrat, and cutting at the first dot named every fixture of a set
     *  the same thing -- so two runs wrote over each other's .out.           */
    dot = strrchr(b, '.');
    lim = dot ? (size_t) (dot - b) : strlen(b);
    for (k = 0; k + 1 < n && k < lim; k++) out[k] = b[k];
    out[k] = '\0';
}

static int   global_seed     = 0;    /* -seed <prefijo> */
static char *pre_prefix      = NULL;

/*  Residuals of the conditional regression, published by init_guess so that
 *  -writeres can write them out.  e_t = Θ(L)A_t.                             */

/*  The seed read from the .pre files: the DIAGONAL of Θ̄_k, k=1..q.
 *
 *  ONLY Θ, and not out of convenience: the .pre **does not carry σ²** —
 *  checked against the grammar (FILE_CONTRACT.md: the innovation variance
 *  appears only in fuf files) and against a real .pre written by fue.  So Σ's
 *  ratios still come from the residuals of the conditional regression, where F1
 *  put them.  And the AR is not seeded either: the univariate models give Φ*_k
 *  for k=1..p, but the model only has F_1..F_{p-1} and Φ̄_p = −F_{p-1}C̄⁻¹H̄ is
 *  determined, so with r ≥ 1 the system is overdetermined and there is no
 *  consistent way to split it.  Θ is exactly what starts at zero today.      */
static real **seed_tbar  = NULL;     /* [1..q][1..M]   diagonal of ThetaBar */
static real **seed_phi   = NULL;     /* [1..p-1][1..M] diagonal of Phi*      */
static real  *seed_var   = NULL;     /* [1..M]  sigma^2 of each univariate    */
static real  *seed_logl  = NULL;     /* [1..M]  logL of each univariate       */
static int    seed_have_uv = 0;      /* 1 if sigma2/logL could be evaluated */
static real   seed_logl_sum = 0.0;   /* sum of the univariate logLs          */
static real   gate_ll_start  = 0.0;  /* logL AT the values brought in (pre-fit)  */
static int    gate_have_start = 0;   /* 1 if it could be evaluated before optimising */
static real   gate_move      = 0.0;  /* largest displacement of a coefficient    */
static int    gate_have_move = 0;

/*  free_seed_pre — releases the four seeding buffers.
 *
 *  It did not exist: they were allocated in load_seed_pre and never freed.  The
 *  leak was invisible while vector() returned the base of the block, because
 *  valgrind saw a pointer to the start and called it "still reachable"; on
 *  aligning vector() with the suite the stored pointer points inside the block
 *  and the leak shows up as definitely lost.  So the offset allocator DETECTS
 *  better, and this is the first thing it brought to light.
 *
 *  The bounds have to be the allocation's, not the logical ones: with q = 0 or
 *  p = 1 a row was reserved all the same.                                    */
static void free_seed_pre(void)
{
    int M = nser, q = global_q;
    int nf = (global_p > 1) ? global_p - 1 : 0;

    if (seed_tbar) { free_matrix(seed_tbar, 1, (q  > 0 ? q  : 1), 1, M);
                     seed_tbar = NULL; }
    if (seed_phi)  { free_matrix(seed_phi,  1, (nf > 0 ? nf : 1), 1, M);
                     seed_phi  = NULL; }
    if (seed_var)  { free_vector(seed_var,  1, M); seed_var  = NULL; }
    if (seed_logl) { free_vector(seed_logl, 1, M); seed_logl = NULL; }
}

/*****************************************************************************/
/*  gate_contract — the entry gate verifies ITSELF, by the ladder's contract. */
/*                                                                           */
/*  WHY.  At the diagonal rung -- r = 0 with diagonal Phi, Theta and Sigma -- */
/*  the exact likelihood FACTORISES: the joint model is M independent         */
/*  univariate models, so                                                    */
/*                                                                           */
/*      logL(joint diagonal fit)  =  SUM_i logL(univariate i).               */
/*                                                                           */
/*  That is an identity, not an approximation, and it is the sharpest check   */
/*  the program has on everything upstream of the likelihood: the            */
/*  transformation, the differencing implied by the rank, the layout of the   */
/*  parameter vector, the deterministic terms subtracted, and the scaling.    */
/*  If any of them is wrong the two sides part company; if all are right they */
/*  agree to rounding.  It is the same contract the suite states for its own  */
/*  entry gate, and it is checked here rather than only in the test suite,    */
/*  so that any run at the diagonal rung certifies itself.                    */
/*                                                                           */
/*  HOW.  The univariate side is computed from the FITTED diagonal blocks --  */
/*  no external file is involved -- by evaluating the same elf() with m = 1   */
/*  on each component in turn.  The scale is concentrated in both cases, so   */
/*  the two sides are on the same footing.                                    */
/*****************************************************************************/
static void gate_contract(struct Tvarma *v)
{
    const real LOG2PI = 1.837877066;
    int M = v->m, n = v->n, p = v->p, q = v->q, i, k, t;
    real sum = 0.0;
    int failed = 0;

    if (global_r != 0 || !global_diag_ar || !global_diag_ma || !global_diag_cov)
        return;                       /* the factorisation only holds here */

    /*  A readable signal next to the gap: how far the coefficient that moved
     *  most actually moved.  The seed is compared with the fit in the SAME
     *  convention, the engine's, so that a sign change is not read as
     *  movement.                                                             */
    if (seed_have_uv && seed_tbar) {
        gate_move = 0.0;
        for (k = 1; k <= q; k++)
            for (i = 1; i <= M; i++) {
                real d = fabs(seed_tbar[k][i] - v->theta[k][i][i]);
                if (d > gate_move) gate_move = d;
            }
        gate_have_move = 1;
    }

    fprintf(outputv,
        "\n=== Entry gate: the factorisation contract ===\n\n"
        "At r = 0 with diagonal Phi, Theta and Sigma the likelihood factorises,\n"
        "so the joint fit must equal the sum of the univariate fits exactly.\n\n");

    for (i = 1; i <= M; i++) {
        struct Tvarma u;
        real pi1, pi2, pi3, ll = 0.0;
        int ifault = 0;

        u.m = 1; u.n = n; u.p = p; u.q = q; u.xitol = -fabs(v->xitol);
        u.mu    = vector(1, 1);
        u.phi   = tensor(0, (p > 0 ? p : 1), 1, 1, 1, 1);
        u.theta = tensor(0, (q > 0 ? q : 1), 1, 1, 1, 1);
        u.qq    = matrix(1, 1, 1, 1);
        u.w     = matrix(1, n, 1, 1);
        u.a     = matrix(1, n, 1, 1);

        u.mu[1] = v->mu[i];
        u.qq[1][1] = 1.0;                 /* the scale is concentrated, as above */
        u.phi[0][1][1] = 1.0; u.theta[0][1][1] = 1.0;
        for (k = 1; k <= p; k++) u.phi[k][1][1]   = v->phi[k][i][i];
        for (k = 1; k <= q; k++) u.theta[k][1][1] = v->theta[k][i][i];
        for (t = 1; t <= n; t++) u.w[t][1] = v->w[t][i];

        elf(u.m, u.n, u.p, u.q, u.mu, u.phi, u.theta, u.qq, u.w, 1.0,
            u.xitol, FALSE, u.a, &pi1, &pi2, &pi3, &ifault);

        if (ifault == 0) {
            ll = -0.5 * u.n * (LOG2PI - log((real) u.n) + 1.0)
                 - 0.5 * u.n * (log(pi1) + log(pi2));
            sum += ll;
            fprintf(outputv, "  univariate %d (p=%d, q=%d)   logL = %18.10f\n",
                    i, p, q, ll);
        } else {
            failed = 1;
            fprintf(outputv, "  univariate %d              elf ifault = %d\n",
                    i, ifault);
        }

        free_matrix(u.a, 1, n, 1, 1);
        free_matrix(u.w, 1, n, 1, 1);
        free_matrix(u.qq, 1, 1, 1, 1);
        free_tensor(u.theta, 0, (q > 0 ? q : 1), 1, 1, 1, 1);
        free_tensor(u.phi,   0, (p > 0 ? p : 1), 1, 1, 1, 1);
        free_vector(u.mu, 1, 1);
    }

    if (failed) {
        fprintf(outputv, "\n  Contract NOT verified: a univariate evaluation "
                         "failed.\n");
        return;
    }

    /*  BOTH SIDES UNTRUNCATED, and the tolerance is rounding (BUG-47).
     *
     *  The identity is exact in algebra.  What used to separate the two sides
     *  is that elf truncates the xi sequence once its term falls below xitol
     *  (elfvarma.c, cxi), and the joint system and the univariate ones do not
     *  truncate at the same term.  The tolerance was then xitol itself, which
     *  was measured on Milan (gap ~0.15 xitol) but is not a bound: the
     *  truncation error grows like xitol / (1 - |theta|), and an independent
     *  pair of ARIMA(0,1,1) with theta = 0.9 and 0.95 (tests/repro/fixtures/
     *  gate_hi.inp) failed a correct model by 1.8e-3.
     *
     *  A negative xitol switches the truncation off (that is all -m 2 does).
     *  So the certificate evaluates both sides at the fitted point with the
     *  whole sequence -- one more likelihood evaluation -- and the contract is
     *  then exact to rounding whatever -m the fit used.  The failures this
     *  gate has to catch -- the sign of Lambda, B2 read transposed -- open
     *  gaps of units.                                                        */
    {
        real pi1, pi2, pi3, joint = v->logelf;
        int  ife = 0;
        real **abuf = matrix(1, n, 1, M);
        elf(M, n, p, q, v->mu, v->phi, v->theta, v->qq, v->w, 1.0,
            -fabs(v->xitol), FALSE, abuf, &pi1, &pi2, &pi3, &ife);
        free_matrix(abuf, 1, n, 1, M);
        if (ife == 0)
            joint = -0.5 * M * n * (LOG2PI - log((real) M) - log((real) n) + 1.0)
                    - 0.5 * n * (M * log(pi1) + log(pi2));
        {
        real gap = joint - sum;
        real tol = 1.0e-6;

        fprintf(outputv, "  %-28s   sum  = %18.10f\n", "", sum);
        fprintf(outputv, "  %-28s   joint= %18.10f\n", "", joint);
        if (v->xitol > 0.0)
            fprintf(outputv, "  (both sides untruncated; the fit's own logL, "
                             "xi truncated at %.0e, is %.10f)\n",
                    v->xitol, v->logelf);
        fprintf(outputv, "\n  crossing identity (joint - sum) = %.3e   "
                         "(tolerance %.1e)   %s\n", gap, tol,
                (ife == 0 && fabs(gap) < tol) ? "VERIFIED" : "*** NOT VERIFIED ***");
        if (ife != 0 || fabs(gap) >= tol)
            fprintf(outputv,
                "\n  The two sides must agree at this rung to rounding: both are\n"
                "  evaluated with the whole xi sequence.  A gap is then not\n"
                "  truncation, and the fault is upstream of the likelihood --\n"
                "  the transformation, the rank's differencing, the parameter\n"
                "  walk, the deterministic terms or the scaling -- and never in\n"
                "  elf() itself.\n");
        }
    }

    /*  THE OPTIMALITY CERTIFICATE.  Second of the two contracts, and it costs
     *  one likelihood evaluation that has already been made.  The gap between
     *  the fit and the values that were brought in is non-negative by
     *  construction and is zero if and only if those values were the
     *  univariate optima.  So it answers a question the files themselves
     *  cannot: was this a `.pre` -- an optimum in re-runnable form -- or a
     *  specification that still needed estimating?  Neither is a defect; both
     *  are legitimate inputs.  What was missing was being told which.
     *
     *  The threshold is the suite's, and it is MEASURED rather than chosen: a
     *  genuine `.pre` does not return exactly to its own values, because the
     *  format stores six decimals and the optimiser stops inside its own
     *  tolerance, which leaves a residue of order 1e-5; a specification moves
     *  by orders of magnitude more.  1e-3 sits in between with room to spare.  */
    if (gate_have_start) {
        real gap = v->logelf - gate_ll_start;

        fprintf(outputv, "\n  --- optimality certificate ---\n");
        fprintf(outputv, "  logL AT the values brought in   = %18.10f\n",
                gate_ll_start);
        fprintf(outputv, "  logL of the diagonal fit        = %18.10f\n",
                v->logelf);
        fprintf(outputv, "  optimality gap (fit - brought)  = %+.3e\n", gap);
        if (gate_have_move)
            fprintf(outputv, "  largest coefficient movement    = %.3e\n",
                    gate_move);
        fprintf(outputv,
            "\n  The gap is >= 0 always, and zero exactly when what came in were\n"
            "  the univariate optima.  Here they %s: this input is %s.\n",
            (gap <= 1.0e-3) ? "were" : "were not",
            (gap <= 1.0e-3) ? "AN OPTIMUM" : "A SPECIFICATION");
        if (gap < -1.0e-6)
            fprintf(outputv,
                "\n  A NEGATIVE gap cannot happen at a converged fit: the fitted\n"
                "  point is worse than the one it started from, so the optimiser\n"
                "  did not converge here and the fit should not be read.\n");
    }
}

static int    seed_loaded = 0;

/*  Which route the seed comes from, which decides what coordinates it is in:
 *
 *    SEED_RESID (-seed)      the .pre files are of the RESIDUALS of the
 *                            conditional regression.  e_t = Θ(L)A_t, so the
 *                            univariate θ estimates Θ DIRECTLY, in ∇Y
 *                            coordinates.  It is not transformed.
 *    SEED_YBAR  (-seedybar)  the .pre files are of the COMPONENTS OF Ȳ.  What a
 *                            univariate model of Ȳ sees is Θ̄ = C̄ΘC̄⁻¹, so that
 *                            has to be undone: Θ_k = C̄⁻¹Θ̄_kC̄.
 *
 *  Confusing the two is a silent error: the same θ put into the wrong
 *  coordinate gives a different model with nothing complaining.  The Ȳ route is
 *  measured and it is WORSE (see PLAN_BETA.md F2.7); it is kept so that the
 *  measurement can be reproduced.                                            */
#define SEED_NONE  0
#define SEED_RESID 1
#define SEED_YBAR  2
static int seed_route = SEED_NONE;

/*  ybar_start_date — date of the first observation of Ȳ.
 *  In the levels layout datamat[t] corresponds to rawmat[t+1], because the
 *  first observation is consumed by differencing; with -differenced there is no
 *  offset.  ObsToDate comes from the bridge (src/fue_bridge.c).              */
static void ybar_start_date(int *year, int *sub)
{
    int first = global_levels ? 2 : 1;
    ObsToDate(data_start_year, data_start_sub, first, data_freq, year, sub);
}

/*  ar_ols_seed — seed for a component's AR: OLS on its own lags.  It is not
 *  an identification (that is ART); it is a starting point better than a
 *  constant, and fue will re-estimate it anyway.                             */
static void ar_ols_seed(real *y, int n, int k, real *phi)
{
    int i, j, t;
    real mean = 0.0;
    for (t = 1; t <= n; t++) mean += y[t];
    mean /= n;
    for (i = 1; i <= k; i++) phi[i] = 0.0;
    if (k < 1 || n <= k + 1) return;

    real **XX = matrix(1, k, 1, k);
    real  *Xy = vector(1, k);
    int   *ind = ivector(1, k);
    for (i = 1; i <= k; i++) {
        for (j = 1; j <= k; j++) {
            real ss = 0.0;
            for (t = k + 1; t <= n; t++) ss += (y[t-i] - mean) * (y[t-j] - mean);
            XX[i][j] = ss;
        }
        Xy[i] = 0.0;
        for (t = k + 1; t <= n; t++) Xy[i] += (y[t-i] - mean) * (y[t] - mean);
    }
    /* A degenerate component (zero variance) would make XX singular; in that
       case the AR is left at zero, which is a legitimate seed.               */
    for (i = 1; i <= k; i++) if (XX[i][i] <= 1.0e-30) goto done;
    ludcp(XX, k, ind);
    lusol(XX, Xy, k, ind);
    for (i = 1; i <= k; i++) phi[i] = Xy[i];
done:
    free_ivector(ind, 1, k);
    free_vector(Xy, 1, k);
    free_matrix(XX, 1, k, 1, k);
}

/*  ascii_name — a series name fit for an .inp: ASCII, no blanks.           */
static void ascii_name(const char *src, const char *prefix, char *dst, int cap)
{
    int n = 0;
    while (*prefix && n < cap - 1) dst[n++] = *prefix++;
    while (src && *src && n < cap - 1) {
        unsigned char c = (unsigned char) *src++;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '_') dst[n++] = (char) c;
    }
    if (n == 0) dst[n++] = 'y';
    dst[n] = '\0';
}

/*  write_inp_series — writes ONE .inp for an already stationary series.
 *
 *  par = order of the AR factor (0 = no AR), qma = order of the MA factor.
 *  They are asked for as ONE factor of order k ("1 k") and not as k first-order
 *  factors ("k 1 1 ..."), so that the .pre's coefficients map directly onto
 *  phi_1..phi_k / theta_1..theta_k when expanded (FILE_CONTRACT.md 2.3).     */
static int write_inp_series(const char *path, const char *name,
                            real *col, int n, int year, int sub,
                            int par, int qma, int mu_free, const char *what)
{
    int t, k;
    real mean = 0.0, refactor;
    real *phi = vector(1, (par > 0 ? par : 1));
    FILE *f;

    for (t = 1; t <= n; t++) mean += col[t];
    mean /= n;

    /* refactor: the rule measured in the suite (drtran-python pre.py:check_scale
       and BRIDGE_DESIGN.md).  cdgrad uses an ABSOLUTE finite-difference step of
       ~6e-6, so a tiny series gives a gradient that is noise; the comfortable
       band is a typical |w| between 0.01 and 100 with the objective around 1.
       It is measured on the series itself and rounded to a power of ten.  That
       each component carries its own is harmless because the only thing seeded
       back is Theta, which is scale-invariant.                                */
    {
        real med = 0.0;
        int cnt = 0;
        for (t = 1; t <= n; t++) { real a = fabs(col[t]);
                                  if (a > 0.0) { med += a; cnt++; } }
        med = (cnt > 0) ? med / cnt : 1.0;
        refactor = (med > 0.0) ? pow(10.0, floor(log10(1.0 / med) + 0.5)) : 1.0;
        if (refactor < 1.0e-6) refactor = 1.0e-6;
        if (refactor > 1.0e+6) refactor = 1.0e+6;
    }

    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path);
              free_vector(phi, 1, (par > 0 ? par : 1)); return 1; }

    ar_ols_seed(col, n, par, phi);

    /* Free header: the parser skips lines up to the separator that says
       "frequency".  Free, but ASCII (fue's BUG-0010).                        */
    fprintf(f, "************************************************\n");
    fprintf(f, "* Input file for program FUE                   *\n");
    fprintf(f, "* written by drvec: %-26s *\n", what);
    fprintf(f, "************************************************\n");
    fprintf(f, "** Frequency of time series: either 1(A), 4(Q) or 12(M):\n");
    fprintf(f, " %d\n", data_freq);
    fprintf(f, "** Number of observations and starting date of time series:\n");
    fprintf(f, " %d %d %d %s\n", n, sub, year, name);
    fprintf(f, "** Number of deterministic variables"
               " (including seasonal components):\n0\n");
    fprintf(f, "** Number and orders of regular AR operators:\n");
    if (par > 0) {
        fprintf(f, "1 %d\n**\n", par);
        for (k = 1; k <= par; k++) fprintf(f, "%.10g  1\n", phi[k] * 1.0);
    } else fprintf(f, "0\n");
    fprintf(f, "** Number and orders of annual AR operators:\n0\n");
    fprintf(f, "** Number and orders of regular MA operators:\n");
    if (qma > 0) {
        fprintf(f, "1 %d\n**\n", qma);
        for (k = 1; k <= qma; k++) fprintf(f, "%.6f  1\n", 0.1);
    } else fprintf(f, "0\n");
    fprintf(f, "** Number and orders of anual MA operators:\n0\n");
    fprintf(f, "** Number and frequencies of regular AR(2) operators"
               " with fixed frequency:\n0\n");
    fprintf(f, "** Number and frequencies of regular MA(2) operators"
               " with fixed frequency:\n0\n");
    /* mu: the mean of the ALREADY differenced variable, which is what these
       series are.  Seeding it wrongly is expensive -- fue's BUG-0012: a mu_0 of
       2.5 against a series of mean 17.06 leaves fue 6.86 of logL below the
       optimum.  It goes scaled by refactor, like the rest of the series.
       And mu_free MUST follow the deterministic case of the joint model: if a
       mean the joint model cannot represent is estimated here, the theta fue
       returns is conditioned on something that does not exist.  The format
       distinguishes the two by the FLAG, not by the value: "value 1" means
       estimate, and a single "0" means the mean is not part of the model.    */
    fprintf(f, "** Mean parameter (mu):\n");
    if (mu_free) fprintf(f, "%.17g  1\n", mean * refactor);
    else         fprintf(f, "0\n");
    /* Series already stationary, and already in logs if the original .inp was:
       identity and zero differences.                                        */
    fprintf(f, "** Box-Cox lambda, regular differences and complete"
               " annual differences:\n1.00 0 0\n");
    /* This section is ALWAYS written by both of fue's writers (fue.c:3485 and
       report.py:1203) and the Python parser ALWAYS reads it: with annual data,
       a literal " 0".  Omitting it shifts everything that comes after with no
       error -- the same bug the C reader had (see F2.1).                     */
    fprintf(f, "** Individual factors of the annual difference"
               " (from freq 0.0): \n");
    if (data_freq > 1) {
        for (k = 0; k <= data_freq / 2; k++) fprintf(f, " 0");
        fprintf(f, "\n");
    } else fprintf(f, " 0\n");
    fprintf(f, "** ACF/PACF bands (0 Automatic) and reescaling factor: \n");
    /*  BUG-44: %.2f wrote any refactor <= 0.005 as 0.00, which fue reads as
     *  1 -- while mu above had been scaled by the true factor.  And %.10f
     *  kept 3-4 significant digits of a series around 1e-7.  The writer-side
     *  twin of fue BUG-0021 / art BUG-0188.                                */
    fprintf(f, " 0.00 %.10g\n", refactor);
    fprintf(f, "** Time series (stochastic and non-standard deterministic"
               " variables): \n");
    /* The data go RAW: refactor is a directive and fue applies it itself
       (w = refactor * BoxCox(z), BRIDGE_DESIGN.md).  Pre-multiplying here would
       apply it twice.  mu does go scaled, because it is the mean of w.       */
    for (t = 1; t <= n; t++) fprintf(f, "%.17g\n", col[t]);
    fclose(f);

    if (!quiet_mode)
        printf("  escrito %s  (%s, n=%d, ARMA(%d,%d), refactor=%.2f)\n",
               path, name, n, par, qma, refactor);
    free_vector(phi, 1, (par > 0 ? par : 1));
    return 0;
}

/*  write_component_inps — one .inp per COMPONENT OF Ybar.
 *
 *  The model asked for is ARMA(p-1, q) with a mean: p is the AR order on Ybar,
 *  so on nabla Y the effective order is p-1 (section 2 of the plan).
 *
 *  MEASURED WARNING: seeding Theta from these files MAKES THE FIT WORSE (see
 *  PLAN_BETA.md F2.7).  The univariate marginal of a component of a VARMA is
 *  not Theta_ii: marginalising mixes AR and MA and inflates the orders.  This
 *  mode is kept because it is the natural unit for ART to identify the model
 *  -- which is information about p and q -- not as a source of seeds.  For that
 *  there is -writeres.                                                       */
static int write_component_inps(const char *prefix)
{
    int M = nser, r = global_r, s = M - r;
    int p = global_p, q = global_q;
    int par = (p > 1) ? p - 1 : 0;
    int i, t, year, sub, nbad = 0;

    real **B2   = matrix(1, s, 1, (r > 0 ? r : 1));
    real **Ybar = matrix(1, nobs, 1, M);
    real  *col  = vector(1, nobs);
    prelim_b2(B2);
    build_ybar(B2, Ybar);
    ybar_start_date(&year, &sub);

    for (i = 1; i <= M; i++) {
        char name[64], path[1024], what[64];
        for (t = 1; t <= nobs; t++) col[t] = Ybar[t][i];
        if (i <= s) ascii_name(series_names[i], "d", name, sizeof name);
        else        ascii_name(series_names[i], "W", name, sizeof name);
        snprintf(path, sizeof path, "%s.%d.inp", prefix, i);
        snprintf(what, sizeof what, "Ybar component %d", i);
        /* Case 1: E[nabla Y2] = 0 and E[W] = 0, so no mean at all.
           Case 2: E[W] != 0 but E[nabla Y2] = 0.
           Case 3: both free.  (Mauricio 2006, Remark 6.)                     */
        {
            int mu_free = (i <= s) ? y2mu_free(i) : (global_case >= 2);
            nbad += write_inp_series(path, name, col, nobs, year, sub,
                                     par, q, mu_free, what);
        }
    }

    free_vector(col, 1, nobs);
    free_matrix(Ybar, 1, nobs, 1, M);
    free_matrix(B2, 1, s, 1, (r > 0 ? r : 1));
    return nbad ? 1 : 0;
}

/*  write_resid_inps — one .inp per component of the RESIDUALS of the
 *  conditional regression.  This is the good route for seeding Theta, and the
 *  reason is fundamental: the residuals are e_t = Theta(L)A_t, whose marginal
 *  per component is EXACTLY MA(q), without the order inflation of Ybar's
 *  marginal.  So ARMA(0, q) is asked for -- pure MA, with no AR, which has
 *  already been netted out.
 *
 *  Requires init_guess to have run (it is what publishes cond_resid).        */
static int write_resid_inps(const char *prefix)
{
    int M = nser, q = global_q, p = global_p;
    int i, t, year, sub, nbad = 0;
    int T = cond_resid_T;
    real *col;

    if (!cond_resid || T < 4) {
        fprintf(stderr, "ERROR: there are no residuals to write\n");
        return 1;
    }
    /* The residuals start at t = p+1 on Ybar's index. */
    {
        int first = (global_levels ? 2 : 1) + p;
        ObsToDate(data_start_year, data_start_sub, first, data_freq, &year, &sub);
    }
    col = vector(1, T);
    for (i = 1; i <= M; i++) {
        char name[64], path[1024], what[64];
        for (t = 1; t <= T; t++) col[t] = cond_resid[t][i];
        /*  BUG-19: the columns of cond_resid are internal ([nabla Y1 ;
         *  nabla Y2]); the file keeps its number -- the -seed route reads it
         *  back by that number, in that order -- and takes the NAME of the
         *  series it belongs to.                                            */
        ascii_name(series_names[lam2inp(i)], "e", name, sizeof name);
        snprintf(path, sizeof path, "%s.%d.inp", prefix, i);
        snprintf(what, sizeof what, "residual %d, MA(%d)", i, q);
        /* The residual's mean is a nuisance of the preliminary fit --- the
           conditional regression carries no constant --- and not a parameter of
           the joint model, so in general it goes free: all that is wanted from
           here is the MA structure.
           EXCEPT in case 1, where the joint model ADMITS NO MEAN at all
           (E[nabla Y2] = 0 and E[W] = 0).  Leaving it free there returns a theta
           conditioned on something the model cannot represent, and it is
           measured: with mu free the seed starts 3.40 below the cold start, and
           fixing it at 0 recovers 0.53 of those 3.40.  See F2.7.             */
        nbad += write_inp_series(path, name, col, T, year, sub, 0, q,
                                 (global_case != 1), what);
    }
    free_vector(col, 1, T);
    return nbad ? 1 : 0;
}

/*  load_seed_pre — reads one .pre per component and leaves the diagonal of
 *  Θ̄_k in seed_tbar.  Returns 0 if it could read all M files.
 *
 *  The .pre's factors are expanded with expand_ma_factors, which comes from
 *  drtran and is the C mirror of fue's cast _unscramble: the .pre stores
 *  FACTORISED operators, and "2 1 1" (two first-order factors) is not "1 2"
 *  (one of second order).  Expanding with the suite's routine, instead of
 *  reimplementing the convolution, is what guarantees that drvec reads the same
 *  model fue estimated.                                                      */
/*  pre_univariate — evaluates ONE `.pre`'s model on its own series and returns
 *  its exact log-likelihood and its σ².
 *
 *  The `.pre` does not carry σ² — that is true and checked against the format —
 *  but **it does carry the model and the data**, so σ² is *derivable*: it is
 *  enough to evaluate the univariate likelihood with the same `elf` the whole
 *  suite uses.  Saying "Σ cannot be seeded from the .pre" was stopping at the
 *  first half of the argument.
 *
 *  And it gives, in passing, what the ladder's two contracts need
 *  (`drtran-python/docs/LADDER_AS_OPTIMISATION.md` §2.1 and §3): the sum of the
 *  univariate logLs, which with r = 0 and diagonal structure must coincide with
 *  the joint one, and the optimality certificate, which is the gap between
 *  evaluating and fitting.
 *
 *  σ² comes out in the RESCALED units (w = refactor·z), so it is returned
 *  divided by refactor² for the ratios between components with different
 *  refactors to be comparable.  What we do not know how to handle is refused:
 *  a Box-Cox other than the identity, differencing, or deterministic terms.  */
/*  pre_ar_order / pre_ma_order -- the expanded order of a .pre's AR / MA:
 *  regular factors, ANNUAL ones (order p acts at lags sper .. p*sper) and the
 *  fixed-frequency ones (order 2 each).  Ported from drtran (total_ar_order);
 *  counting only the regular factors, as drvec did, made the annual and
 *  fixed-frequency factors vanish when there was no regular one and
 *  truncated them when there was (BUG-38).                                   */
static int pre_ar_order(const struct Tusmodel *Tm)
{
    int ord = 0, i;
    for (i = 1; i <= Tm->NumAr1;  i++) ord += Tm->p1[i];
    for (i = 1; i <= Tm->NumAr2;  i++) ord += Tm->p2[i] * Tm->sper;
    for (i = 1; i <= Tm->NumAr1f; i++) ord += 2;
    return ord;
}
static int pre_ma_order(const struct Tusmodel *Tm)
{
    int ord = 0, i;
    for (i = 1; i <= Tm->NumMa1;  i++) ord += Tm->q1[i];
    for (i = 1; i <= Tm->NumMa2;  i++) ord += Tm->q2[i] * Tm->sper;
    for (i = 1; i <= Tm->NumMa1f; i++) ord += 2;
    return ord;
}

static int pre_univariate(struct Tusmodel *Tm, struct Tseries *Ts,
                          real *logl_out, real *sigma2_out)
{
    const real LOG2PI = 1.837877066;
    int n = Ts->nobs, p = 0, q = 0, k, t, ifault = 0;
    real pi1, pi2, pi3, refac = (Ts->refactor != 0.0) ? Ts->refactor : 1.0;
    struct Tvarma uv;

    /*  ornsop, not only nrdiff/nadiff: the ifadf factors are differences too,
     *  and were ignored (BUG-38).                                            */
    if (Tm->NdetVar != 0 || Tm->nrdiff != 0 || Tm->nadiff != 0
        || Tm->ornsop != 0 || Tm->boxlam != 1.0) {
        fprintf(stderr, "WARNING: the .pre carries deterministic terms, differencing or a\n"
                        "         Box-Cox; its sigma2 is not evaluated\n");
        return 1;
    }
    p = pre_ar_order(Tm);
    q = pre_ma_order(Tm);

    uv.m = 1; uv.n = n; uv.p = p; uv.q = q;
    uv.xitol = 1.0e-3;
    uv.mu    = vector(1, 1);
    uv.phi   = tensor(0, (p > 0 ? p : 1), 1, 1, 1, 1);
    uv.theta = tensor(0, (q > 0 ? q : 1), 1, 1, 1, 1);
    uv.qq    = matrix(1, 1, 1, 1);
    uv.w     = matrix(1, n, 1, 1);
    uv.a     = matrix(1, n, 1, 1);

    /*  The value whatever the flag, as fue does (fue.c:2749): the flag says
     *  whether mu is FREE, not whether it is in the model.  A fixed mean was
     *  evaluated as 0 (BUG-38); fue's writer emits it as `<mu> 0' since fue
     *  BUG-0021.                                                             */
    uv.mu[1]      = Tm->mu;
    uv.qq[1][1]   = 1.0;
    uv.phi[0][1][1]   = 1.0;
    uv.theta[0][1][1] = 1.0;
    if (p > 0) { real *ph = vector(1, p);
                 for (k = 1; k <= p; k++) ph[k] = 0.0;
                 expand_ar_factors(Tm, ph, p);
                 for (k = 1; k <= p; k++) uv.phi[k][1][1] = ph[k];
                 free_vector(ph, 1, p); }
    if (q > 0) { real *th = vector(1, q);
                 for (k = 1; k <= q; k++) th[k] = 0.0;
                 expand_ma_factors(Tm, th, q);
                 for (k = 1; k <= q; k++) uv.theta[k][1][1] = th[k];
                 free_vector(th, 1, q); }
    for (t = 1; t <= n; t++) uv.w[t][1] = Ts->data[t] * refac;

    elf(uv.m, uv.n, uv.p, uv.q, uv.mu, uv.phi, uv.theta, uv.qq, uv.w,
        1.0, uv.xitol, TRUE, uv.a, &pi1, &pi2, &pi3, &ifault);

    if (ifault == 0) {
        *logl_out = -0.5 * uv.m * uv.n * (LOG2PI - log((real) uv.m)
                    - log((real) uv.n) + 1.0)
                    - 0.5 * uv.n * (uv.m * log(pi1) + log(pi2));
        /* Back to the ORIGINAL units.  fue estimates on w = refactor*z, and a
           logL is not scale-invariant: the jacobian n*log(refactor) has to come
           off.  Without this the sum of univariate logLs is not comparable with
           the joint one and the crossing identity looks as though it fails by
           hundreds of units (on mink-muskrat, by 2*61*log(10) = 280.92).
           The sign: if w = c*z then p_z(z) = c^n * p_w(w), so
           logL_z = logL_w + n*log(c).                                        */
        *logl_out += uv.n * log(refac);
        *sigma2_out = (pi1 / (uv.n * uv.m)) / (refac * refac);
    }

    free_matrix(uv.a, 1, n, 1, 1);
    free_matrix(uv.w, 1, n, 1, 1);
    free_matrix(uv.qq, 1, 1, 1, 1);
    free_tensor(uv.theta, 0, (q > 0 ? q : 1), 1, 1, 1, 1);
    free_tensor(uv.phi,   0, (p > 0 ? p : 1), 1, 1, 1, 1);
    free_vector(uv.mu, 1, 1);
    return (ifault == 0) ? 0 : 1;
}

/*  Each .pre read is released with free_fue_pre (src/fue_bridge.c).  The
 *  reader brings no deallocator -- neither here nor in drtran, which is its
 *  BUG-12 -- so one was written; not freeing is a fault even if the program
 *  ends straight away, and the same reader is used from processes that do not
 *  end.                                                                      */
static int load_seed_pre(const char *prefix)
{
    int M = nser, q = global_q;
    int i, k;

    int nf = (global_p > 1) ? global_p - 1 : 0;

    seed_tbar = matrix(1, (q  > 0 ? q  : 1), 1, M);
    seed_phi  = matrix(1, (nf > 0 ? nf : 1), 1, M);
    seed_var  = vector(1, M);
    seed_logl = vector(1, M);
    for (k = 1; k <= q;  k++) for (i = 1; i <= M; i++) seed_tbar[k][i] = 0.0;
    for (k = 1; k <= nf; k++) for (i = 1; i <= M; i++) seed_phi[k][i]  = 0.0;
    for (i = 1; i <= M; i++) { seed_var[i] = 1.0; seed_logl[i] = 0.0; }
    seed_have_uv = 1;

    for (i = 1; i <= M; i++) {
        char path[1024];
        struct Tusmodel Tm;
        struct Tseries  Ts;
        real **DataMat = NULL;
        int qpre;

        /* .pre is tried and, failing that, .inp.  The two are the SAME format
           and a different claim: the .pre says "this is an optimum" and the .inp
           "this is a specification" (FILE_CONTRACT.md §4).  Accepting both is
           what fue's reader and drtran's load_pre do, and it is what allows the
           seeding to be tested with a hand-edited file without having to
           manufacture a .pre, which would be claiming an optimum that does not
           exist.                                                             */
        snprintf(path, sizeof path, "%s.%d.pre", prefix, i);
        if (read_fue_pre(path, &Tm, &Ts, &DataMat) != 0) {
            char alt[1024];
            snprintf(alt, sizeof alt, "%s.%d.inp", prefix, i);
            if (read_fue_pre(alt, &Tm, &Ts, &DataMat) != 0) {
                fprintf(stderr, "ERROR: could not read %s or %s\n", path, alt);
                free_matrix(seed_tbar, 1, q, 1, M); seed_tbar = NULL;
                return 1;
            }
            if (!quiet_mode)
                printf("  (%s is not there; %s is used, which is a specification"
                       " and not an optimum)\n", path, alt);
        }
        if (Ts.nobs != nobs)
            fprintf(stderr, "WARNING: %s brings %d observations and Ȳ has %d\n",
                    path, Ts.nobs, nobs);

        /* MA order the file brings, every factor expanded (BUG-38). */
        qpre = pre_ma_order(&Tm);
        if (qpre > 0) {
            real *th = vector(1, qpre);
            for (k = 1; k <= qpre; k++) th[k] = 0.0;
            expand_ma_factors(&Tm, th, qpre);
            for (k = 1; k <= q && k <= qpre; k++) seed_tbar[k][i] = th[k];
            free_vector(th, 1, qpre);
        }
        if (qpre != q)
            fprintf(stderr, "WARNING: %s brings an MA of order %d and the model asks for %d;\n"
                            "         the missing lags are seeded at 0\n",
                    path, qpre, q);

        /* AR: the diagonal of Phi*.  With r = 0 one has Phi*_k = F_k, so this
           seeds F directly; with r >= 1 Cbar has to be undone, which is what
           init_guess does.                                                   */
        {
            int ppre = pre_ar_order(&Tm);
            if (ppre > 0) {
                real *ph = vector(1, ppre);
                for (k = 1; k <= ppre; k++) ph[k] = 0.0;
                expand_ar_factors(&Tm, ph, ppre);
                for (k = 1; k <= nf && k <= ppre; k++) seed_phi[k][i] = ph[k];
                free_vector(ph, 1, ppre);
            }
            if (ppre != nf)
                fprintf(stderr, "WARNING: %s brings an AR of order %d and the model "
                                "asks for %d\n", path, ppre, nf);
        }

        /* sigma^2 and logL: NOT in the file, but the file brings the model AND
           the data, so they are derived by evaluating with elf.              */
        {
            real ll = 0.0, s2 = 1.0;
            if (pre_univariate(&Tm, &Ts, &ll, &s2) == 0) {
                seed_logl[i] = ll; seed_var[i] = s2;
            } else {
                seed_have_uv = 0;
            }
        }
        free_fue_pre(&Tm, &Ts, DataMat);
    }

    if (seed_have_uv) {
        real tot = 0.0;
        for (i = 1; i <= M; i++) tot += seed_logl[i];
        seed_logl_sum = tot;
        if (!quiet_mode) {
            printf("  univariate logL from the .pre files:");
            for (i = 1; i <= M; i++) printf(" %.6f", seed_logl[i]);
            printf("   sum = %.6f\n", tot);
        }
    }
    seed_loaded = 1;
    return 0;
}

static void init_guess(real *x, int npar)
{
    int M = nser, r = global_r, s = M - r;
    int p = global_p, q = global_q;
    int i, j, k, t, idx = 1;
    int nf = (p > 1) ? p - 1 : 0;

    /* --- 0. Levels of Y_{2t}: built once by build_y2_levels() ---------- */
    real **Y2lev = Y2_levels;

    /*  -warma: the seed is its own regression, because the system being
     *  parameterised is ANOTHER one -- the equations are Ybar = [nabla Y2 ; W]'s
     *  and not nabla Y's.  Each component of Ybar_t is regressed on
     *  W_{t-1}..W_{t-p} and a constant, which is exactly the form of
     *  definition 3, and Theta starts at zero.                               */
    if (global_warma) {
        real **B2w = matrix(1, s, 1, (r > 0 ? r : 1));
        real **Yb, **X, **Yd, **XtX;
        real *Xty, *EWv;
        int *ind, nreg2, T2, t2, e2, c2, c3, nf2 = (p > 1) ? p - 1 : 0;

        prelim_b2(B2w);
        if (global_seedjoh && r > 0) canonical_b2(B2w);
        if (global_fixb2 && global_fixb2_given && r > 0 && seed_fixb2_consistent)
            for (i = 1; i <= s; i++)
                for (j = 1; j <= r; j++) B2w[i][j] = global_fixb2_value;

        Yb = matrix(1, nobs, 1, M);
        for (t2 = 1; t2 <= nobs; t2++) {
            for (i = 1; i <= s; i++) Yb[t2][i] = datamat[t2][i];
            for (j = 1; j <= r; j++) {
                real wv = datamat[t2][s + j];
                for (i = 1; i <= s; i++) wv += B2w[i][j] * Y2_levels[t2][i];
                Yb[t2][s + j] = wv;
            }
        }
        EWv = vector(1, (r > 0 ? r : 1));
        for (j = 1; j <= r; j++) {
            real sm = 0.0;
            for (t2 = 1; t2 <= nobs; t2++) sm += Yb[t2][s + j];
            EWv[j] = sm / nobs;
        }
        nreg2 = p * r;
        T2 = nobs - p; if (T2 < 1) T2 = 1;
        X   = matrix(1, T2, 1, (nreg2 > 0 ? nreg2 : 1));
        Yd  = matrix(1, T2, 1, M);
        for (t2 = p + 1; t2 <= nobs; t2++) {
            int row = t2 - p; c2 = 1;
            for (k = 1; k <= p; k++)
                for (j = 1; j <= r; j++) X[row][c2++] = Yb[t2-k][s+j] - EWv[j];
            for (i = 1; i <= M; i++) Yd[row][i] = Yb[t2][i];
        }
        XtX = matrix(1, (nreg2 > 0 ? nreg2 : 1), 1, (nreg2 > 0 ? nreg2 : 1));
        Xty = vector(1, (nreg2 > 0 ? nreg2 : 1));
        ind = ivector(1, (nreg2 > 0 ? nreg2 : 1));
        {
            real ***Cw = tensor(1, p, 1, M, 1, (r > 0 ? r : 1));
            real **E2  = matrix(1, T2, 1, M);
            real **Sg2 = matrix(1, M, 1, M);
            for (c2 = 1; c2 <= nreg2; c2++)
                for (c3 = 1; c3 <= nreg2; c3++) {
                    real ss = 0.0;
                    for (t2 = 1; t2 <= T2; t2++) ss += X[t2][c2] * X[t2][c3];
                    XtX[c2][c3] = ss;
                }
            if (nreg2 > 0) ludcp(XtX, nreg2, ind);
            for (e2 = 1; e2 <= M; e2++) {
                real **XX = matrix(1, nreg2, 1, nreg2);
                for (c2 = 1; c2 <= nreg2; c2++)
                    for (c3 = 1; c3 <= nreg2; c3++) XX[c2][c3] = XtX[c2][c3];
                for (c2 = 1; c2 <= nreg2; c2++) {
                    Xty[c2] = 0.0;
                    for (t2 = 1; t2 <= T2; t2++) Xty[c2] += X[t2][c2] * Yd[t2][e2];
                }
                lusol(XX, Xty, nreg2, ind);
                for (k = 1; k <= p; k++)
                    for (j = 1; j <= r; j++) Cw[k][e2][j] = Xty[(k-1)*r + j];
                for (t2 = 1; t2 <= T2; t2++) {
                    real ei = Yd[t2][e2];
                    for (c2 = 1; c2 <= nreg2; c2++) ei -= Xty[c2] * X[t2][c2];
                    E2[t2][e2] = ei;
                }
                free_matrix(XX, 1, nreg2, 1, nreg2);
            }
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) {
                    real ss = 0.0;
                    for (t2 = 1; t2 <= T2; t2++) ss += E2[t2][i] * E2[t2][j];
                    Sg2[i][j] = ss / T2;
                }
            /* --- writing, in the order the cast expects --- */
            if (global_case == 2) { for (j = 1; j <= r; j++) x[idx++] = EWv[j]; }
            else if (global_case == 3) {
                for (i = 1; i <= s; i++) {
                    real sm = 0.0;
                    for (t2 = 1; t2 <= nobs; t2++) sm += Yb[t2][i];
                    x[idx++] = sm / nobs;
                }
                for (j = 1; j <= r; j++) x[idx++] = EWv[j];
            }
            for (i = 1; i <= M; i++)
                for (j = 1; j <= r; j++) x[idx++] = Cw[1][i][j];
            for (k = 1; k <= nf2; k++)
                for (i = 1; i <= M; i++)
                    for (j = 1; j <= r; j++) x[idx++] = Cw[k+1][i][j];
            for (k = 1; k <= q; k++)
                for (i = 1; i <= r; i++)
                    for (j = 1; j <= r; j++) x[idx++] = 0.0;
            {
                real s11 = (Sg2[1][1] > 1.0e-12) ? Sg2[1][1] : 1.0;
                if (global_diag_cov) {
                    for (i = 2; i <= M; i++) x[idx++] = Sg2[i][i] / s11;
                } else {
                    for (i = 2; i <= M; i++) x[idx++] = Sg2[i][i] / s11;
                    for (i = 2; i <= M; i++)
                        for (j = 1; j < i; j++) x[idx++] = Sg2[i][j] / s11;
                }
            }
            if (!global_fixb2)
                for (j = 1; j <= r; j++)
                    for (i = 1; i <= s; i++) x[idx++] = B2w[i][j];
            else {
                if (!B2_fixed || b2f_s != s || b2f_r != r) {
                    if (B2_fixed) free_matrix(B2_fixed, 1, b2f_s, 1, b2f_r);
                    B2_fixed = matrix(1, s, 1, (r > 0 ? r : 1));
                    b2f_s = s; b2f_r = (r > 0 ? r : 1);
                }
                for (j = 1; j <= r; j++)
                    for (i = 1; i <= s; i++)
                        B2_fixed[i][j] = global_fixb2_given ? global_fixb2_value
                                                            : B2w[i][j];
            }
            free_matrix(Sg2, 1, M, 1, M);
            free_matrix(E2, 1, T2, 1, M);
            free_tensor(Cw, 1, p, 1, M, 1, (r > 0 ? r : 1));
        }
        if (idx - 1 != npar)
            fprintf(stderr, "ERROR init_guess (-warma): idx=%d, npar=%d\n",
                    idx-1, npar);
        free_ivector(ind, 1, (nreg2 > 0 ? nreg2 : 1));
        free_vector(Xty, 1, (nreg2 > 0 ? nreg2 : 1));
        free_matrix(XtX, 1, (nreg2 > 0 ? nreg2 : 1), 1, (nreg2 > 0 ? nreg2 : 1));
        free_matrix(Yd, 1, T2, 1, M);
        free_matrix(X, 1, T2, 1, (nreg2 > 0 ? nreg2 : 1));
        free_vector(EWv, 1, (r > 0 ? r : 1));
        free_matrix(Yb, 1, nobs, 1, M);
        free_matrix(B2w, 1, s, 1, (r > 0 ? r : 1));
        return;
    }

    /* --- 1. Initial B₂ via static OLS with intercept ------------------- */
    real **B2 = matrix(1, s, 1, (r > 0 ? r : 1));
    prelim_b2(B2);

    /*  -seedjoh: the canonical solution in its place.  It is asked for AFTER
     *  the static OLS and not instead of it, so that if the eigenvalue problem
     *  cannot be solved what is left is exactly the route of always and not a
     *  third thing halfway between.  Everything that follows -- Lambda, F,
     *  Sigma and E[W] -- then comes out of the conditional regression WITH THIS
     *  W, which is Johansen's own formula for alpha.                         */
    canon_used = 0;
    if (global_seedjoh && r > 0) {
        canon_used = canonical_b2(B2);
        if (canon_used) {
            fprintf(outputv, "\n-seedjoh: B2 seeded from the canonical "
                             "reduced-rank solution:\n");
            for (i = 1; i <= s; i++) {
                fprintf(outputv, "   ");
                for (j = 1; j <= r; j++) fprintf(outputv, " %12.6f", B2[i][j]);
                fprintf(outputv, "\n");
            }
            if (!quiet_mode) {
                printf("  -seedjoh: B2 canonico =");
                for (i = 1; i <= s; i++)
                    for (j = 1; j <= r; j++) printf(" %.6f", B2[i][j]);
                printf("\n");
            }
        } else {
            fprintf(outputv, "\n-seedjoh: the canonical solution could not be "
                             "formed; the static OLS seed stands.\n");
            if (!quiet_mode)
                printf("  -seedjoh: the canonical solution could not be formed\n");
        }
    }

    /*  -fixb2 v: the rest of the seed -- Lambda, F, Sigma, E[W] -- comes from
     *  the W that is going to be FITTED, W = Y1 + v'Y2, and not from the static
     *  OLS one.  Seeding them from the OLS W and then swapping B2 for v handed
     *  the optimiser an inconsistent start: on UK consumption-income the fit
     *  ended 766 log-units below the one reached from the same model with one
     *  column demeaned (BUG-33).                                            */
    if (global_fixb2 && global_fixb2_given && r > 0 && seed_fixb2_consistent)
        for (i = 1; i <= s; i++)
            for (j = 1; j <= r; j++) B2[i][j] = global_fixb2_value;
    /*  -seedb2 v: the same, for the same reason -- a B2 started at v with the
     *  rest of the seed built for another B2 is the inconsistency of BUG-33,
     *  only with B2 free.  main() still writes v into the B2 slots after this,
     *  which now changes nothing and keeps the slot-order check meaningful. */
    else if (global_seedb2 && r > 0 && seed_fixb2_consistent)
        for (i = 1; i <= s; i++)
            for (j = 1; j <= r; j++) B2[i][j] = global_seedb2_value;
    /*  -fixb2row: the held rows at their value BEFORE W is built, so that the
     *  rest of the seed is consistent with the model that will be fitted
     *  (the lesson of BUG-33).                                              */
    if (b2row_on && !global_fixb2 && r > 0)
        for (i = 1; i <= s; i++)
            if (b2_held(i))
                for (j = 1; j <= r; j++) B2[i][j] = b2row_val[i];

    /* --- 2. Build W_t = Y_{1t} + B₂'Y_{2t} and ∇Y_t = [∇Y_{1t};∇Y_{2t}] - */
    real **W  = matrix(1, nobs, 1, (r > 0 ? r : 1));
    real **dY = matrix(1, nobs, 1, M);   /* rows: [∇Y₁ (r); ∇Y₂ (s)] */
    for (t = 1; t <= nobs; t++) {
        for (j = 1; j <= r; j++) {
            real w = datamat[t][s + j];
            for (i = 1; i <= s; i++) w += B2[i][j] * Y2lev[t][i];
            W[t][j] = w;
        }
        for (j = 1; j <= r; j++)
            dY[t][j] = (t > 1) ? datamat[t][s+j] - datamat[t-1][s+j] : 0.0;
        for (i = 1; i <= s; i++)
            dY[t][r + i] = datamat[t][i];
    }

    /*  -trend: the slope of each W by OLS on the centred time, and W
     *  detrended with it, so that the regression below sees what the cast
     *  will build.                                                       */
    real *ET = vector(1, (r > 0 ? r : 1));
    for (j = 1; j <= r; j++) ET[j] = 0.0;
    if (global_trend) {
        real tc = 0.5 * (nobs + 1), stt = 0.0, swt = 0.0;
        for (t = 1; t <= nobs; t++) stt += (t - tc) * (t - tc);
        for (j = 1; j <= r; j++) {
            swt = 0.0;
            for (t = 1; t <= nobs; t++) swt += W[t][j] * (t - tc);
            ET[j] = (stt > 0.0) ? swt / stt : 0.0;
            for (t = 1; t <= nobs; t++) W[t][j] -= ET[j] * (t - tc);
        }
    }

    /* --- 3. Mean E[Ȳ_t] (Remark 6) --------------------------------------- */
    real *EW   = vector(1, (r > 0 ? r : 1));   /* sample mean of W */
    real *EdY2 = vector(1, s);   /* sample mean of ∇Y₂ */
    for (j = 1; j <= r; j++) { real sm=0.0; for (t=1;t<=nobs;t++) sm+=W[t][j]; EW[j]=sm/nobs; }
    for (i = 1; i <= s; i++) { real sm=0.0; for (t=1;t<=nobs;t++) sm+=dY[t][r+i]; EdY2[i]=sm/nobs; }

    /* --- 4. Conditional regression: ∇Y_t = Σ F_i ∇Y_{t-i} − Λ (W_{t−1}−E[W]) */
    int nreg = r + nf * M;
    int T = nobs - p;              /* rows t = p+1 .. nobs */
    if (T < 1) T = 1;
    real **X = matrix(1, T, 1, (nreg > 0 ? nreg : 1));
    real **Ydep = matrix(1, T, 1, M);
    for (t = p + 1; t <= nobs; t++) {
        int row = t - p, col = 1;
        for (j = 1; j <= r; j++) X[row][col++] = W[t-1][j] - EW[j];
        for (k = 1; k <= nf; k++)
            for (i = 1; i <= M; i++)
                X[row][col++] = dY[t-k][i];
        for (i = 1; i <= M; i++) Ydep[row][i] = dY[t][i];
    }
    /* With r = 0 and p <= 1 there is nothing to regress on (no error-correction
       term, no F_i), so the whole conditional regression is skipped and the
       residuals are the differences themselves.                              */
    int nalloc = (nreg > 0) ? nreg : 1;
    real **XtX = matrix(1, nalloc, 1, nalloc);
    real  *Xty = vector(1, nalloc);
    int *indx = ivector(1, nalloc);
    if (nreg > 0) {
        for (i = 1; i <= nreg; i++)
            for (j = 1; j <= nreg; j++) {
                real ss = 0.0;
                for (t = 1; t <= T; t++) ss += X[t][i]*X[t][j];
                XtX[i][j] = ss;
            }
        ludcp(XtX, nreg, indx);
    }

    real **Lambda = matrix(1, M, 1, (r > 0 ? r : 1));
    real ***F = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
    for (int eq = 1; eq <= M; eq++) {
        if (nreg == 0) break;
        for (i = 1; i <= nreg; i++) {
            Xty[i] = 0.0;
            for (t = 1; t <= T; t++) Xty[i] += X[t][i]*Ydep[t][eq];
        }
        lusol(XtX, Xty, nreg, indx);
        for (j = 1; j <= r; j++) Lambda[eq][j] = -Xty[j];          /* −Λ */
        for (k = 1; k <= nf; k++)
            for (i = 1; i <= M; i++)
                F[k][eq][i] = Xty[r + (k-1)*M + i];                /* F_k */
    }
    /* Residuals of the conditional regression, and from them Σ.
       e_t used to be recomputed inside the double (i,j) loop, which repeated the
       same calculation M² times; now it is computed ONCE, in the same order of
       subtractions, so the result is identical bit for bit.  They are also kept
       in cond_resid because they are what -writeres needs: e_t = Θ(L)A_t, so
       each component's marginal is EXACTLY MA(q) — without the order inflation
       a component of Ȳ's marginal suffers.                                    */
    real **E = matrix(1, T, 1, M);
    for (t = 1; t <= T; t++)
        for (i = 1; i <= M; i++) {
            real ei = Ydep[t][i];
            for (int c = 1; c <= nreg; c++) {
                real bi = (c <= r) ? -Lambda[i][c] : F[(c-r-1)/M+1][i][(c-r-1)%M+1];
                ei -= bi * X[t][c];
            }
            E[t][i] = ei;
        }
    real **Sig = matrix(1, M, 1, M);
    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) {
        real ss = 0.0;
        for (t = 1; t <= T; t++) ss += E[t][i] * E[t][j];
        Sig[i][j] = (T > 0) ? ss / T : 1.0;
    }
    /* Publish the residuals for -writeres.  This module owns them. */
    if (cond_resid) free_matrix(cond_resid, 1, cond_resid_T, 1, cond_resid_M);
    cond_resid = matrix(1, T, 1, M);
    cond_resid_T = T; cond_resid_M = M;
    for (t = 1; t <= T; t++) for (i = 1; i <= M; i++) cond_resid[t][i] = E[t][i];
    free_matrix(E, 1, T, 1, M);

    /* --- 4b. The .pre's seed, carried into drvec's coordinates -------------
       Only for the Ybar route: the .pre files describe the COMPONENTS OF Ybar,
       and the model is parameterised in Lambda / F / Theta / Sigma on nabla Y.
       The three returns, with Phi*_k = Cbar*PhiBar_k and
       Theta*_k = Cbar*Theta_k*Cinv:

         Theta_k = Cinv * diag(theta_k) * Cbar
         F_1     = (Cinv*diag(phi_1) - Cinv*Hbar + LamBar) * Cbar
         F_i     = (Cinv*diag(phi_i) + F_{i-1}*Cinv*Hbar) * Cbar     i = 2..p-1
         Sigma   = Cinv * diag(sigma^2) * Cinv'

       With r = 0 all of this collapses to the identity (Cbar = I, Hbar = 0),
       which is the diagonal rung where the ladder's contracts live.
       Phi*_p is determined by F_{p-1} and cannot be imposed, so with r >= 1 the
       AR is overdetermined and this is a projection, not an exact return.
       See docs/PLAN_BETA.md F2.8.                                            */
    real ***Fseed = NULL, ***Tseed = NULL, **Sigseed = NULL;
    if (seed_loaded && seed_route == SEED_YBAR) {
        real **Cb = matrix(1, M, 1, M), **Ci = matrix(1, M, 1, M);
        real **Hb = matrix(1, M, 1, M), **Lb = matrix(1, M, 1, M);
        real **T1 = matrix(1, M, 1, M), **T2 = matrix(1, M, 1, M);
        int a, b;
        for (a = 1; a <= M; a++) for (b = 1; b <= M; b++) {
            Cb[a][b] = 0.0; Ci[a][b] = 0.0; Hb[a][b] = 0.0; Lb[a][b] = 0.0; }
        for (a = 1; a <= s; a++) Cb[a][r + a] = 1.0;
        for (b = 1; b <= r; b++) Cb[s + b][b] = 1.0;
        for (b = 1; b <= r; b++) for (a = 1; a <= s; a++) Cb[s + b][r + a] = B2[a][b];
        for (a = 1; a <= r; a++) for (b = 1; b <= s; b++) Ci[a][b] = -B2[b][a];
        for (a = 1; a <= r; a++) Ci[a][s + a] = 1.0;
        for (a = 1; a <= s; a++) Ci[r + a][a] = 1.0;
        for (a = 1; a <= r; a++) Hb[s + a][s + a] = 1.0;
        for (a = 1; a <= M; a++) for (b = 1; b <= r; b++) Lb[a][s + b] = Lambda[a][b];

        if (q > 0) {
            Tseed = tensor(1, q, 1, M, 1, M);
            for (k = 1; k <= q; k++) {
                for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                    T1[a][b] = (a == b) ? seed_tbar[k][a] : 0.0;
                matrix_multiply(Ci, T1, T2, M, M, M);
                matrix_multiply(T2, Cb, Tseed[k], M, M, M);
            }
        }
        if (nf > 0) {
            real **CiH = matrix(1, M, 1, M), **W1 = matrix(1, M, 1, M);
            matrix_multiply(Ci, Hb, CiH, M, M, M);
            Fseed = tensor(1, nf, 1, M, 1, M);
            for (k = 1; k <= nf; k++) {
                for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                    T1[a][b] = (a == b) ? seed_phi[k][a] : 0.0;
                matrix_multiply(Ci, T1, W1, M, M, M);        /* Cinv*diag(phi) */
                if (k == 1) {
                    for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                        W1[a][b] += -CiH[a][b] + Lb[a][b];
                } else {
                    matrix_multiply(Fseed[k-1], CiH, T2, M, M, M);
                    for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                        W1[a][b] += T2[a][b];
                }
                matrix_multiply(W1, Cb, Fseed[k], M, M, M);
            }
            free_matrix(W1, 1, M, 1, M);
            free_matrix(CiH, 1, M, 1, M);
        }
        if (seed_have_uv) {
            Sigseed = matrix(1, M, 1, M);
            for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                T1[a][b] = (a == b) ? seed_var[a] : 0.0;
            matrix_multiply(Ci, T1, T2, M, M, M);
            for (a = 1; a <= M; a++) for (b = 1; b <= M; b++) {
                real acc = 0.0;
                for (k = 1; k <= M; k++) acc += T2[a][k] * Ci[b][k];  /* *Cinv' */
                Sigseed[a][b] = acc;
            }
        }
        free_matrix(T2, 1, M, 1, M); free_matrix(T1, 1, M, 1, M);
        free_matrix(Lb, 1, M, 1, M); free_matrix(Hb, 1, M, 1, M);
        free_matrix(Ci, 1, M, 1, M); free_matrix(Cb, 1, M, 1, M);
    }

    /* --- 5. Write x[] in the canonical VEC order ------------------------- */
    if (global_case == 2) { for (j = 1; j <= r; j++) x[idx++] = EW[j]; }
    else if (global_case == 3) {
        for (i = 1; i <= s; i++) if (y2mu_free(i)) x[idx++] = EdY2[i];
        for (j = 1; j <= r; j++) x[idx++] = EW[j];
    }
    if (global_trend) for (j = 1; j <= r; j++) x[idx++] = ET[j];
    if (global_alpha) {
        /* psi = (A'A)^-1 A' Lambda_ols: the projection of the free seed onto the
           subspace the restriction allows.  It is the best seed available and
           costs nothing.                                                     */
        real **AtA = matrix(1, alpha_sa, 1, alpha_sa);
        real **psi_seed = matrix(1, alpha_sa, 1, (r > 0 ? r : 1));
        real  *AtL = vector(1, alpha_sa);
        int   *ind = ivector(1, alpha_sa);
        int    a, b;
        for (a = 1; a <= alpha_sa; a++)
            for (b = 1; b <= alpha_sa; b++) {
                real acc = 0.0;
                for (i = 1; i <= M; i++) acc += alpha_A[i][a] * alpha_A[i][b];
                AtA[a][b] = acc;
            }
        ludcp(AtA, alpha_sa, ind);
        for (j = 1; j <= r; j++) {
            for (a = 1; a <= alpha_sa; a++) {
                real acc = 0.0;
                for (i = 1; i <= M; i++) acc += alpha_A[lam2inp(i)][a] * Lambda[i][j];
                AtL[a] = acc;
            }
            lusol(AtA, AtL, alpha_sa, ind);
            for (a = 1; a <= alpha_sa; a++) psi_seed[a][j] = AtL[a];
        }
        for (a = 1; a <= alpha_sa; a++)
            for (j = 1; j <= r; j++) x[idx++] = psi_seed[a][j];
        free_ivector(ind, 1, alpha_sa);
        free_vector(AtL, 1, alpha_sa);
        free_matrix(psi_seed, 1, alpha_sa, 1, (r > 0 ? r : 1));
        free_matrix(AtA, 1, alpha_sa, 1, alpha_sa);
    } else {
        for (i = 1; i <= M; i++) for (j = 1; j <= r; j++) x[idx++] = Lambda[i][j];
    }
    for (k = 1; k <= nf; k++) {
        real **Fk = (Fseed ? Fseed[k] : F[k]);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) if (f_free(i, j)) x[idx++] = Fk[i][j];
    }
    /* MA block.  Without -seed it starts at EXACT ZERO, which is the only real
       gap of the cold start (everything else comes from data: B₂ by OLS, Λ and
       F by the conditional regression, Σ from its residuals).
       With -seed what fue estimated on each component of Ȳ is used, and it has
       to be brought back into drvec's coordinates: what a univariate model of Ȳ
       sees is Θ̄ = C̄ΘC̄⁻¹ —that is how vec_shootx builds armax->theta—, so
                              Θ_k = C̄⁻¹ Θ̄_k C̄
       with Θ̄_k diagonal.  Seeding the .pre's θ directly into Θ works only if
       C̄ = I (r = 0) and is a silent error as soon as r ≥ 1.                  */
    if (marow_on() && q > 0) {
        for (k = 1; k <= q; k++)
            for (i = 1; i <= r; i++)
                for (j = 1; j <= M; j++) x[idx++] = 0.0;
    } else if (global_matri && q > 0) {
        for (k = 1; k <= q; k++) {
            for (i = 1; i <= r; i++)
                for (j = 1; j <= M; j++) x[idx++] = 0.0;
            for (i = r + 1; i <= M; i++)
                for (j = r + 1; j <= M; j++) x[idx++] = 0.0;
        }
    } else if (mawarma_on() && q > 0) {
        /*  -mawarma: the free block is only Theta11 (r x r), and the cast builds
         *  the rest.  The seed is the diagonal of whatever there is: with the
         *  residual route, the univariate theta of the cointegrated block;
         *  otherwise zero, which is the start of always.                     */
        for (k = 1; k <= q; k++)
            for (i = 1; i <= r; i++)
                for (j = 1; j <= r; j++)
                    x[idx++] = (i == j && seed_loaded && seed_route == SEED_RESID
                                && seed_tbar) ? seed_tbar[k][i] : 0.0;
    } else if (seed_loaded && q > 0 && seed_route == SEED_RESID) {
        /* Residual route: the θ read IS the diagonal of Θ, untransformed. */
        for (k = 1; k <= q; k++) {
            if (global_diag_ma) {
                for (i = 1; i <= M; i++) x[idx++] = seed_tbar[k][i];
            } else {
                for (i = 1; i <= M; i++)
                    for (j = 1; j <= M; j++)
                        x[idx++] = (i == j) ? seed_tbar[k][i] : 0.0;
            }
        }
    } else if (seed_loaded && q > 0 && seed_route == SEED_YBAR && Tseed) {
        /* Ybar route: Theta_k = Cinv * diag(theta_k) * Cbar, already built above. */
        for (k = 1; k <= q; k++) {
            if (global_diag_ma) {
                /* With -diagma only the diagonal of Theta_k is a parameter, and
                   Cinv*diag(.)*Cbar is not diagonal in general: its diagonal is
                   taken, which is an approximation and not the exact return. */
                for (i = 1; i <= M; i++) x[idx++] = Tseed[k][i][i];
            } else {
                for (i = 1; i <= M; i++)
                    for (j = 1; j <= M; j++) x[idx++] = Tseed[k][i][j];
            }
        }
    } else {
        for (k = 1; k <= q; k++) {
            if (global_diag_ma) { for (i = 1; i <= M; i++) x[idx++] = 0.0; }
            else { for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) x[idx++] = 0.0; }
        }
    }
    /* Covariance block.  The concentrated objective is scale-invariant in qq
       (f1 -> f1/c, f2 -> c^m f2), so only the RATIOS are identified and the
       scale is reported through sigma2.  Divide the residual covariance by its
       (1,1) entry: that pins the scale at var_1 = 1 and, crucially, KEEPS the
       ratios var_i/var_1 that the data provides.
       Seeding them at 1 instead -- which is what normalising to the correlation
       matrix does -- throws that information away.  drtran measured the cost on
       its canonical case: logL -1371 instead of -767, with scales differing by
       1098x.  Here the spread is milder (1.06 on mink-muskrat, up to 15x between
       components on UK consumption) but the failure mode is the same.          */
    /* With the .pre's seed, Sigma comes from the univariate variances
       (derived by evaluating each .pre with elf) instead of from the residuals
       of the conditional regression.  It is what closes the whole univariate
       block: seeding Theta alone leaves a point that is nobody's optimum.    */
    if (Sigseed) {
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Sig[i][j] = Sigseed[i][j];
    }
    {
        real s11 = (Sig[1][1] > 1.0e-24) ? Sig[1][1] : 1.0e-24;
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++)
                Sig[i][j] /= s11;
        Sig[1][1] = 1.0;
    }
    /* Sig[1][1] = 1 is the normalisation, not a parameter. */
    if (global_diag_cov) {
        for (i = 2; i <= M; i++) x[idx++] = Sig[i][i];
    } else {
        for (i = 2; i <= M; i++) x[idx++] = Sig[i][i];
        for (i = 2; i <= M; i++)
            for (j = 1; j < i; j++) x[idx++] = Sig[i][j];
    }
    if (global_fixb2 || b2row_on) {
        /* Hand the held entries to vec_shootx through B2_fixed; they are not
           in x[].  Under -fixb2row the held rows already carry their value. */
        if (B2_fixed) free_matrix(B2_fixed, 1, b2f_s, 1, (b2f_r > 0 ? b2f_r : 1));
        B2_fixed = matrix(1, s, 1, (r > 0 ? r : 1));
        b2f_s = s; b2f_r = r;
        for (j = 1; j <= r; j++) for (i = 1; i <= s; i++)
            B2_fixed[i][j] = (global_fixb2 && global_fixb2_given)
                           ? global_fixb2_value : B2[i][j];
    }
    for (j = 1; j <= r; j++) for (i = 1; i <= s; i++)
        if (!b2_held(i)) x[idx++] = B2[i][j];
    /*  -xlink: the transfer starts at omega = delta = 0 (drtran's seeding):
     *  with omega = 0 delta is not identified, so it starts at 0 too.     */
    for (i = 1; i <= xl_npar(); i++) x[idx++] = 0.0;

    if (idx != npar + 1)
        fprintf(stderr, "ERROR init_guess: idx=%d, npar=%d\n", idx-1, npar);

    if (Sigseed) free_matrix(Sigseed, 1, M, 1, M);
    if (Tseed)   free_tensor(Tseed, 1, q, 1, M, 1, M);
    if (Fseed)   free_tensor(Fseed, 1, nf, 1, M, 1, M);
    free_matrix(Sig, 1, M, 1, M);
    free_tensor(F, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
    free_matrix(Lambda, 1, M, 1, (r > 0 ? r : 1));
    free_ivector(indx, 1, nalloc);
    free_matrix(XtX, 1, nalloc, 1, nalloc);
    free_vector(Xty, 1, nalloc);
    free_matrix(X, 1, T, 1, (nreg > 0 ? nreg : 1));
    free_matrix(Ydep, 1, T, 1, M);
    free_vector(EdY2, 1, s);
    free_vector(EW, 1, (r > 0 ? r : 1));
    free_vector(ET, 1, (r > 0 ? r : 1));
    free_matrix(dY, 1, nobs, 1, M);
    free_matrix(W, 1, nobs, 1, (r > 0 ? r : 1));
    free_matrix(B2, 1, s, 1, (r > 0 ? r : 1));
    /* Y2lev aliases the global Y2_levels; it is not owned here. */
}

/*****************************************************************************/
/*  vec_shootx — Mauricio (2006) transformation: x[] → Tvarma + Ȳ_t         */
/*                                                                             */
/*  This is the "user function" (Remark 4 of Mauricio 2006).  On every        */
/*  call it:                                                                   */
/*    1. Unpacks parameters from x[]                                           */
/*    2. Builds Ȳ_t = (∇Y_{2t}', W_t')' using current B₂                    */
/*    3. Places the VARMA parameters into the Tvarma structure                */
/*    4. Normalises (Mauricio's standard normalisation)                        */
/*  The estimator then computes the exact log-likelihood of Ȳ_t.              */
/*****************************************************************************/
static void vec_shootx(real *x, struct Tvarma *armax,
                       int *ifaultx, int firstx, int lastx)
{
    int M = nser, r = global_r, s = M - r;
    int p = global_p, q = global_q;
    int i, j, k, idx = 1;

    *ifaultx = 0;

    /* [1] Set dimensions --------------------------------------------------- */
    armax->m = M;
    armax->n = nobs;
    armax->p = p;
    armax->q = q;

    /*  P8 — AND THE TRUNCATION TOLERANCE, WHICH IS WHAT CLOSES BUG-14.  Every
     *  site that uses a Tvarma set this field itself -- a dozen of them, all
     *  with the same expression -- and rolling_eval did not, so elf ran the
     *  out-of-sample route with whatever was in that word of the stack
     *  (docs/BUGS.md BUG-14).  Patching that one site fixed the symptom and
     *  left the hole: a field no filling function fills is a field the next
     *  site will forget too.  Setting it here makes forgetting impossible.
     *
     *  It cannot move a number: every one of those sites sets it to exactly
     *  this expression, and they set it AFTER this call, so they write the same
     *  value over the same value.  The golden set (tools/golden.sh) is what
     *  says so rather than this comment.                                     */
    armax->xitol = (met == 2) ? -1.0e-3 : 1.0e-3;

    /* [2] Allocate memory on first call ------------------------------------ */
    if (firstx) {
        armax->mu    = vector(1, M);
        armax->phi   = tensor(0, p, 1, M, 1, M);
        armax->theta = tensor(0, q, 1, M, 1, M);
        armax->qq    = matrix(1, M, 1, M);
        armax->w     = matrix(1, nobs, 1, M);
        armax->a     = matrix(1, nobs, 1, M);

        for (i = 1; i <= M; i++) {
            armax->mu[i] = 0.0;
            for (j = 1; j <= M; j++) {
                for (k = 0; k <= p; k++) armax->phi[k][i][j] = 0.0;
                for (k = 0; k <= q; k++) armax->theta[k][i][j] = 0.0;
                armax->qq[i][j] = 0.0;
            }
            for (j = 1; j <= nobs; j++) {
                armax->w[j][i] = 0.0;
                armax->a[j][i] = 0.0;
            }
        }
        for (i = 1; i <= M; i++) {
            armax->phi[0][i][i]   = 1.0;
            armax->theta[0][i][i] = 1.0;
        }
    }

    /*  -warma: the parameterisation in Ybar coordinates.  Phi*, Theta* and
     *  Sigma* are written DIRECTLY and no C̄ is built: there is nothing to
     *  transform because the parameters already are the transformed system's.
     *  See -warma.  The vector w is assembled as always, at the end, with B2,
     *  which is where -- and only where -- the cointegrating vector enters.  */
    if (global_warma) {
        int nf_w = (p > 1) ? p - 1 : 0;
        int idw = 1, kk, ii, jj, tt2;
        real **B2w = matrix(1, s, 1, (r > 0 ? r : 1));

        for (i = 1; i <= M; i++) armax->mu[i] = 0.0;
        if (global_case == 2) {
            for (j = 1; j <= r; j++) armax->mu[s + j] = x[idw++];
        } else if (global_case == 3) {
            for (i = 1; i <= s; i++) armax->mu[i] = x[idw++];
            for (j = 1; j <= r; j++) armax->mu[s + j] = x[idw++];
        }
        for (kk = 1; kk <= p; kk++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) armax->phi[kk][i][j] = 0.0;
        for (kk = 1; kk <= q; kk++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) armax->theta[kk][i][j] = 0.0;
        /*  The coefficients of W_{t-k}: one M x r block per lag, k = 1..p-1 plus
         *  the first, which occupies Lambda's slot.                          */
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++) armax->phi[1][i][s + j] = x[idw++];
        for (kk = 1; kk <= nf_w; kk++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= r; j++) armax->phi[kk + 1][i][s + j] = x[idw++];
        for (kk = 1; kk <= q; kk++)
            for (i = 1; i <= r; i++)
                for (j = 1; j <= r; j++) armax->theta[kk][s + i][s + j] = x[idw++];
        {
            real **Sg = matrix(1, M, 1, M);
            real **Sc = matrix(1, M, 1, M);
            real d1, d2; int ifc = 0;
            for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Sg[i][j] = 0.0;
            Sg[1][1] = 1.0;
            if (global_diag_cov) {
                for (i = 2; i <= M; i++) Sg[i][i] = x[idw++];
            } else {
                for (i = 2; i <= M; i++) Sg[i][i] = x[idw++];
                for (i = 2; i <= M; i++)
                    for (j = 1; j < i; j++) { Sg[i][j] = x[idw++]; Sg[j][i] = Sg[i][j]; }
            }
            for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Sc[i][j] = Sg[i][j];
            choldcp(Sc, M, &d1, &d2, &ifc);
            if (ifc > 0) *ifaultx = 1;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) armax->qq[i][j] = Sg[i][j];
            free_matrix(Sc, 1, M, 1, M);
            free_matrix(Sg, 1, M, 1, M);
        }
        for (j = 1; j <= r; j++)
            for (i = 1; i <= s; i++)
                B2w[i][j] = global_fixb2 ? B2_fixed[i][j] : x[idw++];

        for (tt2 = 1; tt2 <= nobs; tt2++) {
            for (i = 1; i <= s; i++) armax->w[tt2][i] = datamat[tt2][i];
            for (j = 1; j <= r; j++) {
                real wv = datamat[tt2][s + j];
                for (i = 1; i <= s; i++) wv += B2w[i][j] * Y2_levels[tt2][i];
                armax->w[tt2][s + j] = wv;
            }
        }
        granger_sv = -1.0;
        free_matrix(B2w, 1, s, 1, (r > 0 ? r : 1));
        if (lastx == 1) {
            free_matrix(armax->a, 1, armax->n, 1, armax->m);
            free_matrix(armax->w, 1, armax->n, 1, armax->m);
            free_matrix(armax->qq, 1, armax->m, 1, armax->m);
            free_tensor(armax->theta, 0, armax->q, 1, armax->m, 1, armax->m);
            free_tensor(armax->phi, 0, armax->p, 1, armax->m, 1, armax->m);
            free_vector(armax->mu, 1, armax->m);
        }
        (void) ii; (void) jj;
        return;
    }

    /* [3] Unpack VEC parameters in the canonical order ---------------------- */
    /*   1. Mean E[Ȳ_t] (Remark 6): Ȳ_t = [∇Y₂; W], so μ = [E[∇Y₂]; E[W]]  */
    real *mu = vector(1, M);
    for (i = 1; i <= M; i++) mu[i] = 0.0;
    if (global_case == 2) {
        for (j = 1; j <= r; j++) mu[s + j] = x[idx++];
    } else if (global_case == 3) {
        for (i = 1; i <= s; i++) mu[i] = y2mu_free(i) ? x[idx++] : 0.0;
        for (j = 1; j <= r; j++) mu[s + j] = x[idx++];
    }
    real *dtr = vector(1, (r > 0 ? r : 1));       /* -trend: slopes of W */
    for (j = 1; j <= r; j++) dtr[j] = global_trend ? x[idx++] : 0.0;

    /*   2. Adjustment matrix Lambda (M x r).  With r = 0 there is no
           error-correction term at all: Lambda and B2 are empty, Cbar and
           Cinv collapse to the identity, Hbar to zero, and Ybar_t = nabla Y_t.
           That is the no-cointegration null of the rank test.               */
    real **Lambda = matrix(1, M, 1, (r > 0 ? r : 1));
    if (global_alpha) {
        /* Lambda = A * psi.  psi (sa x r) is what the optimiser sees; A is the
           user's datum.  Same pattern as -fixb2: the restriction lives in the
           cast, not in the optimiser.                                        */
        real **psi = matrix(1, alpha_sa, 1, (r > 0 ? r : 1));
        for (i = 1; i <= alpha_sa; i++)
            for (j = 1; j <= r; j++) psi[i][j] = x[idx++];
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++) {
                real acc = 0.0;
                for (int kk = 1; kk <= alpha_sa; kk++) acc += alpha_A[lam2inp(i)][kk] * psi[kk][j];
                Lambda[i][j] = acc;       /* internal row i: A's row lam2inp(i) */
            }
        free_matrix(psi, 1, alpha_sa, 1, (r > 0 ? r : 1));
    } else {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++)
                Lambda[i][j] = x[idx++];
    }

    /*   3. F_i (M x M, i=1..p-1) */
    int nf = (p > 1) ? p - 1 : 0;
    real ***F = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
    for (k = 1; k <= (nf > 0 ? nf : 1); k++)
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) F[k][i][j] = 0.0;
    for (k = 1; k <= nf; k++) {
        if (prof_hold) {          /* held at the r = 0 optimum; see -seedgate */
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) F[k][i][j] = hold_F[k][i][j];
        } else {
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    if (f_free(i, j)) F[k][i][j] = x[idx++];
        }
    }

    /*   4. Theta_i (M x M, i=1..q) */
    real ***Theta = tensor(1, (q > 0 ? q : 1), 1, M, 1, M);
    for (k = 1; k <= (q > 0 ? q : 1); k++)
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Theta[k][i][j] = 0.0;
    for (k = 1; k <= q; k++) {
        if (prof_hold) {
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = hold_Th[k][i][j];
        } else if (mawarma_on()) {
            /*  Theta = [Theta11  Theta11 B2' ; 0  0].  Mind the order: B2 is read
             *  further down, so only the free block is stored here and the rest
             *  is completed AFTER B2 is in hand.  See -mawarma.              */
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = 0.0;
            for (i = 1; i <= r; i++)
                for (j = 1; j <= r; j++) Theta[k][i][j] = x[idx++];
        } else if (marow_on()) {
            /*  Theta = [T11  T12 ; 0  0]: the lower s rows, zero.  See
             *  -marow.                                                       */
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = 0.0;
            for (i = 1; i <= r; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = x[idx++];
        } else if (global_matri) {
            /*  Theta = [T11  T12 ; 0  T22]: only the lower-left block is zeroed.
             *  See -matri.                                                   */
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = 0.0;
            for (i = 1; i <= r; i++)
                for (j = 1; j <= M; j++) Theta[k][i][j] = x[idx++];
            for (i = r + 1; i <= M; i++)
                for (j = r + 1; j <= M; j++) Theta[k][i][j] = x[idx++];
        } else if (global_diag_ma) {
            for (i = 1; i <= M; i++) Theta[k][i][i] = x[idx++];
        } else {
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    Theta[k][i][j] = x[idx++];
        }
    }

    /*   5. Sigma (M x M) — innovation covariance of A_t, up to the scale.
           Sigma[1][1] = 1 (see calc_nparametrs); the rest goes in RAW, in units
           where var_1 = 1: the ratios var_i/var_1 on the diagonal and the
           covariances off it.
           drtran carries the diagonal as log(var_i/var_1) and applies exp() here,
           which makes positivity structural.  That was tried and REVERTED: it is
           not free.  Measured, exp() drove the optimiser into regions where each
           likelihood evaluation is very slow — the legacy-layout case 3 stopped
           finishing at all, and with correlation seeding the M=5 case did — and it
           lost log-likelihood on three of four configurations against the raw
           diagonal.  Positivity is enforced by the check below instead, which
           costs one Cholesky and does not touch the geometry.                  */
    real **Sigma = matrix(1, M, 1, M);
    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Sigma[i][j] = 0.0;
    Sigma[1][1] = 1.0;
    if (prof_hold) {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) Sigma[i][j] = hold_S[i][j];
    } else if (global_diag_cov) {
        for (i = 2; i <= M; i++) Sigma[i][i] = x[idx++];
    } else {
        for (i = 2; i <= M; i++) Sigma[i][i] = x[idx++];
        for (i = 2; i <= M; i++)
            for (j = 1; j < i; j++) {
                Sigma[i][j] = x[idx++];
                Sigma[j][i] = Sigma[i][j];
            }
    }

    /* Reject a non-PD Sigma at translation time.  With the raw diagonal the
       optimiser CAN step a variance negative, so this is the guard that keeps
       the parameterisation honest — it is not a redundant extra.  elf() would
       also catch it on qq = Cbar*Sigma*Cbar' (Cbar is nonsingular, so the two
       are equivalent), but here we can say WHICH matrix is the problem.
       objcfunc answers 1.0 on ifault > 0 and the search moves away.           */
    {
        real **Schk = matrix(1, M, 1, M);
        real d1, d2;
        int ifchol = 0;
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Schk[i][j] = Sigma[i][j];
        choldcp(Schk, M, &d1, &d2, &ifchol);
        free_matrix(Schk, 1, M, 1, M);
        if (ifchol > 0) *ifaultx = 1;      /* Sigma not positive definite */
    }

    /*   6. Cointegration matrix B₂ (s x r) */
    real **B2 = matrix(1, s, 1, (r > 0 ? r : 1));
    for (j = 1; j <= r; j++)
        for (i = 1; i <= s; i++)
            B2[i][j] = b2_held(i) ? B2_fixed[i][j] : x[idx++];

    /*  -xlink: the transfer block, last in x[] */
    if (xl_on) {
        if (xl_compute(x + idx - 1) != 0) *ifaultx = 1;
        idx += xl_npar();
    }

    /*  -mawarma: the upper-right block of Theta, which is NOT free.  It is
     *  completed here and not above because it needs B2, which has just been
     *  read: Theta[k][i][r+jj] = sum_ii Theta11[k][i][ii] * B2'[ii][jj].     */
    if (mawarma_on() && !prof_hold) {
        for (k = 1; k <= q; k++)
            for (i = 1; i <= r; i++)
                for (int jj = 1; jj <= s; jj++) {
                    real acc = 0.0;
                    for (int ii = 1; ii <= r; ii++)
                        acc += Theta[k][i][ii] * B2[jj][ii];
                    Theta[k][i][r + jj] = acc;
                }
    }

    /*  THE RANK CONDITION, measured at every evaluation and optionally
     *  imposed.  It goes here because it is the first point at which Lambda,
     *  Theta and B2 all exist, and before anything is built with them.       */
    granger_sv = (r > 0 && q > 0) ? granger_smin(Lambda, B2, Theta, M, r, q) : -1.0;
    if (global_rankadm && granger_sv >= 0.0 && granger_sv < global_rankadm_tol)
        *ifaultx = 1;          /* the point denies the rank: out, like a non-PD Sigma */

    /* [4] Mauricio transformation matrices (eq. 10-14) ---------------------- */
    real **Cbar   = matrix(1, M, 1, M);
    real **Cinv   = matrix(1, M, 1, M);
    real **Hbar   = matrix(1, M, 1, M);
    real **LamBar = matrix(1, M, 1, M);
    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) {
        Cbar[i][j] = 0.0; Cinv[i][j] = 0.0; Hbar[i][j] = 0.0; LamBar[i][j] = 0.0;
    }
    /* Cbar = [0_{s x r}  I_s ; I_r  B2'] */
    for (i = 1; i <= s; i++) Cbar[i][r + i] = 1.0;
    for (j = 1; j <= r; j++) Cbar[s + j][j] = 1.0;
    for (j = 1; j <= r; j++) for (i = 1; i <= s; i++) Cbar[s + j][r + i] = B2[i][j];
    /* Cinv = [-B2'  I_r ; I_s  0] */
    for (i = 1; i <= r; i++) for (j = 1; j <= s; j++) Cinv[i][j] = -B2[j][i];
    for (i = 1; i <= r; i++) Cinv[i][s + i] = 1.0;
    for (i = 1; i <= s; i++) Cinv[r + i][i] = 1.0;
    /* Hbar = [0 0 ; 0 I_r] */
    for (i = 1; i <= r; i++) Hbar[s + i][s + i] = 1.0;
    /* LamBar = [0, Lambda] (Lambda in the last r columns) */
    for (i = 1; i <= M; i++) for (j = 1; j <= r; j++) LamBar[i][s + j] = Lambda[i][j];

    /* [5] PhiBar_i (eq. 16) ------------------------------------------------- */
    real ***PhBar = tensor(0, p, 1, M, 1, M);
    for (k = 0; k <= p; k++)
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) PhBar[k][i][j] = 0.0;
    /* PhBar[0] = Cinv */
    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) PhBar[0][i][j] = Cinv[i][j];
    /* PhBar[1] = Cinv*Hbar - LamBar + F1*Cinv (F1 term only if p>=2) */
    {
        real **T1 = matrix(1, M, 1, M);
        matrix_multiply(Cinv, Hbar, T1, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
            PhBar[1][i][j] = T1[i][j] - LamBar[i][j];
        if (p >= 2) {
            real **T2 = matrix(1, M, 1, M);
            matrix_multiply(F[1], Cinv, T2, M, M, M);
            for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
                PhBar[1][i][j] += T2[i][j];
            free_matrix(T2, 1, M, 1, M);
        }
        free_matrix(T1, 1, M, 1, M);
    }
    /* PhBar[i] = F_i*Cinv - F_{i-1}*Cinv*Hbar  (i=2..p-1) */
    for (k = 2; k <= p - 1; k++) {
        real **T1 = matrix(1, M, 1, M);
        real **T2 = matrix(1, M, 1, M);
        real **T3 = matrix(1, M, 1, M);
        matrix_multiply(F[k],   Cinv, T1, M, M, M);
        matrix_multiply(F[k-1], Cinv, T2, M, M, M);
        matrix_multiply(T2, Hbar, T3, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
            PhBar[k][i][j] = T1[i][j] - T3[i][j];
        free_matrix(T3, 1, M, 1, M);
        free_matrix(T2, 1, M, 1, M);
        free_matrix(T1, 1, M, 1, M);
    }
    /* PhBar[p] = -F_{p-1}*Cinv*Hbar */
    if (p >= 2) {
        real **T1 = matrix(1, M, 1, M);
        real **T2 = matrix(1, M, 1, M);
        matrix_multiply(F[p-1], Cinv, T1, M, M, M);
        matrix_multiply(T1, Hbar, T2, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
            PhBar[p][i][j] = -T2[i][j];
        free_matrix(T2, 1, M, 1, M);
        free_matrix(T1, 1, M, 1, M);
    }

    /* [6] Standard VARMA (eq. 18) ------------------------------------------- */
    /* Phi*_i = Cbar * PhBar_i */
    for (k = 1; k <= p; k++) {
        real **T = matrix(1, M, 1, M);
        matrix_multiply(Cbar, PhBar[k], T, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
            armax->phi[k][i][j] = T[i][j];
        free_matrix(T, 1, M, 1, M);
    }
    /* Theta*_i = Cbar * Theta_i * Cinv */
    for (k = 1; k <= q; k++) {
        real **T1 = matrix(1, M, 1, M);
        real **T2 = matrix(1, M, 1, M);
        matrix_multiply(Cbar, Theta[k], T1, M, M, M);
        matrix_multiply(T1, Cinv, T2, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++)
            armax->theta[k][i][j] = T2[i][j];
        free_matrix(T2, 1, M, 1, M);
        free_matrix(T1, 1, M, 1, M);
    }
    /* Sigma* = Cbar * Sigma * Cbar'  (covariance of A*_t = Cbar A_t) */
    {
        real **T1 = matrix(1, M, 1, M);
        matrix_multiply(Cbar, Sigma, T1, M, M, M);
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) {
            real ss = 0.0;
            for (int k1 = 1; k1 <= M; k1++) ss += T1[i][k1] * Cbar[j][k1];
            armax->qq[i][j] = ss;
        }
        free_matrix(T1, 1, M, 1, M);
    }
    /* Mean E[Ȳ_t] */
    for (i = 1; i <= M; i++) armax->mu[i] = mu[i];

    free_matrix(LamBar, 1, M, 1, M);
    free_matrix(Hbar, 1, M, 1, M);
    free_matrix(Cinv, 1, M, 1, M);
    free_matrix(Cbar, 1, M, 1, M);
    free_tensor(PhBar, 0, p, 1, M, 1, M);
    free_matrix(Sigma, 1, M, 1, M);
    free_tensor(Theta, 1, (q > 0 ? q : 1), 1, M, 1, M);
    free_tensor(F, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
    free_matrix(Lambda, 1, M, 1, (r > 0 ? r : 1));
    free_vector(mu, 1, M);

    /* [5] Build Ybar_t = (nabla Y_{2t}', W_t')'                            */
    /* datamat cols 1..s = nabla Y_{2t}, cols s+1..M = Y_{1t} in levels.    */
    /* W_t = Y_{1t} + B2' Y_{2t}; the Y_{2t} levels are parameter-free and  */
    /* were built once by build_y2_levels(), so nothing is recomputed here. */
    int tt;
    real **Y2_level = Y2_levels;

    for (tt = 1; tt <= nobs; tt++) {
        /* nabla Y_{2t} */
        for (i = 1; i <= s; i++)
            armax->w[tt][i] = datamat[tt][i];
        /* W_t = Y_{1t} + B2' Y_{2t} */
        for (j = 1; j <= r; j++) {
            real w = datamat[tt][s + j];   /* Y_{1t,j} in levels */
            for (i = 1; i <= s; i++)
                w += B2[i][j] * Y2_level[tt][i];
            armax->w[tt][s + j] = w - dtr[j] * (tt - 0.5 * (nobs + 1));
        }
        /*  -xlink: the same Ybar built from N = Y - nu(B)X.  The levels layout
         *  is guaranteed (-differenced is refused), so datamat row tt is
         *  rawmat row tt+1 and nabla N_2 = nabla Y_2 - (T_{tt+1} - T_tt).    */
        if (xl_on) {
            for (i = 1; i <= s; i++)
                armax->w[tt][i] -= xl_T[tt + 1][i] - xl_T[tt][i];
            for (j = 1; j <= r; j++) {
                real dw = xl_T[tt + 1][s + j];
                for (i = 1; i <= s; i++) dw += B2[i][j] * xl_T[tt + 1][i];
                armax->w[tt][s + j] -= dw;
            }
        }
    }

    /* Y2_level aliases the global Y2_levels; it is not owned here. */
    free_matrix(B2, 1, s, 1, (r > 0 ? r : 1));
    free_vector(dtr, 1, (r > 0 ? r : 1));

    /* [7] Deallocate on last call ------------------------------------------ */
    if (lastx == 1) {
        free_matrix(armax->a, 1, armax->n, 1, armax->m);
        free_matrix(armax->w, 1, armax->n, 1, armax->m);
        free_matrix(armax->qq, 1, armax->m, 1, armax->m);
        free_tensor(armax->theta, 0, armax->q, 1, armax->m, 1, armax->m);
        free_tensor(armax->phi, 0, armax->p, 1, armax->m, 1, armax->m);
        free_vector(armax->mu, 1, armax->m);
    }
}

/*****************************************************************************/
/*  P12 — THE SEARCH                                                          */
/*                                                                           */
/*  WHY.  Until 2026-09-23 every fit in the program was ONE call to est()    */
/*  from ONE starting point: the main fit, each rank of -lrtest, each        */
/*  replication of the bootstraps.  The review of that date measured what    */
/*  it cost (docs/BUGS.md BUG-25, 35, 43): on UKconsumption the r = 1 fit of  */
/*  -lrtest stopped 22 log-units below the optimum a Johansen seed reaches,   */
/*  and the two LRs of the table swapped rows without anybody noticing; on   */
/*  the Danish M = 5 system the rank flipped from 2 to 0; on the wheat pairs */
/*  the richer MA classes came out BELOW the poorer ones they contain, which */
/*  is impossible at the maxima.  And -multistart, the documented remedy,    */
/*  was ignored by -lrtest.                                                  */
/*                                                                           */
/*  WHAT IT DOES.  fit_search() keeps the best of several starts:            */
/*                                                                           */
/*    seed      the one the caller prepared (init_guess, -seedgate, ...)    */
/*    cold      init_guess, when the seed was something else                  */
/*    johansen  the canonical seed of -seedjoh, when r >= 1                  */
/*    nested    THE CHAIN: the same model with q = 0, then -marow, -matri    */
/*              and free, each started from the optimum of the one below     */
/*              EMBEDDED with zeros where the richer class has more.  est()  */
/*              only accepts points that lower the objective, so the richer  */
/*              fit can never end below the poorer one it contains: the     */
/*              nesting the theory promises becomes a property of the output.*/
/*    jitter    with -multistart n, n - 1 perturbations of the seed, exactly */
/*              as the old run_multistart made them (same generator).      */
/*                                                                           */
/*  Every start first goes through make_admissible(): if the engine rejects  */
/*  it (a non-stationary AR, a non-invertible MA), F and Theta are shrunk    */
/*  towards zero -- and Lambda halved if that is not enough -- until it      */
/*  accepts it.  That is the ladder -warma already had, now for everybody    */
/*  (BUG-35, BUG-43).  Lambda is never taken to zero: there the transformed  */
/*  system has an AR root of modulus one (VEC_EMBEDDING_PLAN.md 3).          */
/*                                                                           */
/*  The engine is NOT touched (P3.1).  How est() stopped is known only from  */
/*  the text report() writes to outputv, so est_run() points outputv at a    */
/*  temporary file for the duration of the call and reads the criterion      */
/*  there; the winner's text is then written to the real .out, which is what */
/*  makes termcode_from_out() read the winner's code and not the last start's*/
/*  (the -multistart defect of BUG-45).                                      */
/*****************************************************************************/

/*  The moving-average classes the chain walks, as zero patterns on Theta.   */
enum { MA_Q0 = 0, MA_DIAG, MA_MAROW, MA_MATRI, MA_FREE, MA_OTHER };

static int ma_class_now(void)
{
    if (global_q == 0) return MA_Q0;
    if (global_warma || mawarma_on() || prof_hold) return MA_OTHER;
    if (marow_on())       return MA_MAROW;
    if (global_matri)     return MA_MATRI;
    if (global_diag_ma)   return MA_DIAG;
    return MA_FREE;
}

static void ma_class_set(int c)
{
    global_marow   = (c == MA_MAROW);
    global_matri   = (c == MA_MATRI);
    global_diag_ma = (c == MA_DIAG);
}

/*  Lengths of the blocks around Theta, for the CURRENT settings.  Head =     */
/*  mean, Lambda (or psi) and F; tail = Sigma and B2.  Neither depends on the */
/*  MA class, which is what makes the embedding a copy plus a remap.         */
static void search_blocks(int *head, int *tail)
{
    int nmean, nlam, nmid, ntail, M = nser;
    int nf = (global_p > 1) ? global_p - 1 : 0;
    int nsig = (global_diag_cov ? M : M * (M + 1) / 2) - 1;
    par_blocks(&nmean, &nlam, &nmid, &ntail);
    *head = nmean + nlam + nf * f_nfree();
    *tail = nsig + ntail;
}

static int theta_len(int c, int q)
{
    int M = nser, r = global_r, s = M - r;
    switch (c) {
        case MA_Q0:    return 0;
        case MA_DIAG:  return q * M;
        case MA_MAROW: return q * r * M;
        case MA_MATRI: return q * (M * M - s * r);
        case MA_FREE:  return q * M * M;
    }
    return -1;
}

/*  Theta (q x M x M, zeros where the class has none) <-> its slots in x.    */
static void theta_unpack(const real *x, int off, int c, int q, real ***Th)
{
    int M = nser, r = global_r, i, j, k, idx = off + 1;
    for (k = 1; k <= q; k++) {
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Th[k][i][j] = 0.0;
        if (c == MA_DIAG) {
            for (i = 1; i <= M; i++) Th[k][i][i] = x[idx++];
        } else if (c == MA_MAROW) {
            for (i = 1; i <= r; i++) for (j = 1; j <= M; j++) Th[k][i][j] = x[idx++];
        } else if (c == MA_MATRI) {
            for (i = 1; i <= r; i++) for (j = 1; j <= M; j++) Th[k][i][j] = x[idx++];
            for (i = r + 1; i <= M; i++) for (j = r + 1; j <= M; j++) Th[k][i][j] = x[idx++];
        } else if (c == MA_FREE) {
            for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Th[k][i][j] = x[idx++];
        }
    }
}

static void theta_pack(real ***Th, int c, int q, real *x, int off)
{
    int M = nser, r = global_r, i, j, k, idx = off + 1;
    for (k = 1; k <= q; k++) {
        if (c == MA_DIAG) {
            for (i = 1; i <= M; i++) x[idx++] = Th[k][i][i];
        } else if (c == MA_MAROW) {
            for (i = 1; i <= r; i++) for (j = 1; j <= M; j++) x[idx++] = Th[k][i][j];
        } else if (c == MA_MATRI) {
            for (i = 1; i <= r; i++) for (j = 1; j <= M; j++) x[idx++] = Th[k][i][j];
            for (i = r + 1; i <= M; i++) for (j = r + 1; j <= M; j++) x[idx++] = Th[k][i][j];
        } else if (c == MA_FREE) {
            for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) x[idx++] = Th[k][i][j];
        }
    }
}

/*  The optimum of class cs (q = qs) as a point of class cd (q = qd), where  */
/*  cd contains cs: same head and tail, Theta completed with zeros.  It is   */
/*  the SAME model, so its likelihood is the same number.                    */
static void embed_ma(const real *xs, int cs, int qs, real *xd, int cd, int qd)
{
    int head, tail, i, M = nser;
    int ls = theta_len(cs, qs), ld = theta_len(cd, qd);
    int qm = (qd > 0 ? qd : 1);
    real ***Th = tensor(1, qm, 1, M, 1, M);
    search_blocks(&head, &tail);
    for (i = 1; i <= head; i++) xd[i] = xs[i];
    for (int k = 1; k <= qm; k++)
        for (int a = 1; a <= M; a++) for (int b = 1; b <= M; b++) Th[k][a][b] = 0.0;
    if (qs > 0) theta_unpack(xs, head, cs, (qs < qd ? qs : qd), Th);
    if (qd > 0) theta_pack(Th, cd, qd, xd, head);
    for (i = 1; i <= tail; i++) xd[head + ld + i] = xs[head + ls + i];
    free_tensor(Th, 1, qm, 1, M, 1, M);
}

/*  Does the engine accept x as a starting point?  If not, walk the ladder.  */
/*  Returns 0 when it accepts x as given, k > 0 when it needed step k of     */
/*  the ladder, and -1 when nothing made it admissible (x left as given).   */
static int make_admissible(real *x, int npar)
{
    static const real shr[6] = { 1.0, 0.8, 0.5, 0.3, 0.1, 0.0 };
    int nmean, nlam, nmid, ntail, M = nser;
    int nsig = (global_diag_cov ? M : M * (M + 1) / 2) - 1;
    int lo, hi, i, pass, mi, res = -1;
    real *x0 = vector(1, npar);
    struct Tvarma vt;
    int ift = 0;

    par_blocks(&nmean, &nlam, &nmid, &ntail);
    for (i = 1; i <= npar; i++) x0[i] = x[i];
    /*  F and Theta (or, under -warma, the W-lag and MA blocks): everything  */
    /*  between Lambda and Sigma.  Taken to zero they leave an admissible   */
    /*  system as long as Lambda itself is.                                  */
    /*  Under -warma the first block is not Lambda but the coefficients of
     *  W_{t-1}: taken to zero with the rest they leave Ybar = A*, trivially
     *  admissible -- the ladder main() always applied to -warma (BUG-35).  */
    lo = global_warma ? nmean + 1 : nmean + nlam + 1;
    hi = (prof_hold ? nmean + nlam : nmean + nlam + nmid - nsig);
    vt.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x, &vt, &ift, 1, 0);
    for (pass = 0; pass < 3 && res < 0; pass++) {
        real lamf = (pass == 0) ? 1.0 : (pass == 1 ? 0.5 : 0.25);
        for (mi = 0; mi < 6; mi++) {
            real pi1, pi2, pi3;
            int ifev = 0, ifc = 0;
            for (i = 1; i <= npar; i++) x[i] = x0[i];
            for (i = lo; i <= hi; i++) x[i] = shr[mi] * x0[i];
            if (!global_warma)
                for (i = nmean + 1; i <= nmean + nlam; i++) x[i] = lamf * x0[i];
            vec_shootx(x, &vt, &ifc, 0, 0);
            if (ifc != 0) continue;
            elf(vt.m, vt.n, vt.p, vt.q, vt.mu, vt.phi, vt.theta, vt.qq, vt.w,
                1.0, vt.xitol, TRUE, vt.a, &pi1, &pi2, &pi3, &ifev);
            if (ifev == 0) { res = pass * 6 + mi; break; }
        }
    }
    if (res < 0) for (i = 1; i <= npar; i++) x[i] = x0[i];
    vec_shootx(x, &vt, &ift, 0, 1);
    free_vector(x0, 1, npar);
    return res;
}

/*  ONE optimisation from x, with the stop reason read where the engine      */
/*  prints it.  Returns est()'s ifault; *tc is the termination code (1..5),  */
/*  or 0 when it could not be read (quiet runs print nothing).  *text gets   */
/*  the engine's report, for the caller to copy into the .out if it wants.   */
static int est_run(real *x, int npar, real *dev, real **cov,
                   real *ll, real *s2, int *tc, int *iters, char **text)
{
    struct Tvarma v;
    int ifr = 0;
    FILE *real_out = outputv, *tmp = NULL;
    int saved_stdout = -1;

    *tc = 0; *iters = -1; if (text) *text = NULL;
    v.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x, &v, &ifr, 1, 0);
    if (!quiet_mode && (tmp = tmpfile()) != NULL) {
        outputv = tmp;
        fflush(stdout);
        saved_stdout = dup(fileno(stdout));
        if (saved_stdout >= 0) {
            FILE *nul = fopen(
#ifdef _WIN32
                "NUL",
#else
                "/dev/null",
#endif
                "w");
            if (nul) { dup2(fileno(nul), fileno(stdout)); fclose(nul); }
        }
    }
    est(&vec_shootx, npar, x, dev, cov, maxits, nrits, gradtol, sptol,
        v.xitol, v.a, &v.sigma2, &v.logelf, &ifr);
    if (tmp) {
        long len;
        fflush(stdout);
        if (saved_stdout >= 0) { dup2(saved_stdout, fileno(stdout)); close(saved_stdout); }
        outputv = real_out;
        fflush(tmp);
        len = ftell(tmp);
        rewind(tmp);
        if (len > 0) {
            char *buf = (char *) malloc((size_t) len + 1);
            size_t got = fread(buf, 1, (size_t) len, tmp);
            char *p;
            buf[got] = '\0';
            if ((p = strstr(buf, "after ")) != NULL) sscanf(p, "after %d", iters);
            if ((p = strstr(buf, "Convergence criterion:")) != NULL) {
                if      (strstr(p, "gradtol"))         *tc = 1;
                else if (strstr(p, "steptol"))         *tc = 2;
                else if (strstr(p, "lower point"))     *tc = 3;
                else if (strstr(p, "iteration limit")) *tc = 4;
                else if (strstr(p, "maximum length"))  *tc = 5;
            }
            if (text) *text = buf; else free(buf);
        }
        fclose(tmp);
    }
    /*  A SINGULAR Sigma is not an optimum: the likelihood is unbounded there.
     *  With two exactly collinear series the ladder reaches such a point in a
     *  step (Q = [[1, .5], [.5, .25]], logL 830 after one iteration), where
     *  before every start was refused.  The engine accepts it because Q is
     *  still positive definite in floating point, so the check is made here,
     *  on the relative determinant det Q / prod Q_ii (1 for a diagonal Q, 0
     *  for a singular one).  Refused as ifault 1, "Q not positive definite",
     *  which is what it is.                                                 */
    if (ifr == 0 && v.m > 1) {
        int ifq = 0;
        vec_shootx(x, &v, &ifq, 0, 0);    /* Q at the FINAL point, not the start */
        if (ifq != 0) ifr = 1;
    }
    if (ifr == 0 && v.m > 1) {
        int m = v.m, a, b, c;
        real **L = matrix(1, m, 1, m), det = 1.0, pd = 1.0;
        for (a = 1; a <= m; a++) for (b = 1; b <= m; b++) L[a][b] = v.qq[a][b];
        for (a = 1; a <= m && ifr == 0; a++) {
            real d = L[a][a];
            for (c = 1; c < a; c++) d -= L[a][c] * L[a][c];
            if (!(d > 0.0)) { ifr = 1; break; }
            L[a][a] = sqrt(d);
            for (b = a + 1; b <= m; b++) {
                real e = L[b][a];
                for (c = 1; c < a; c++) e -= L[b][c] * L[a][c];
                L[b][a] = e / L[a][a];
            }
            det *= d; pd *= v.qq[a][a];
        }
        if (ifr == 0 && pd > 0.0 && det / pd < 1e-10) ifr = 1;
        free_matrix(L, 1, m, 1, m);
    }
    *ll = v.logelf; *s2 = v.sigma2;
    {   /*  the deallocating call resets its own ifault argument: est()'s
         *  verdict has to survive it, or a start the engine refused reads
         *  as converged with logL 0.                                        */
        int ifd = 0;
        vec_shootx(x, &v, &ifd, 0, 1);
    }
    return ifr;
}

/*  What fit_search found, for the caller that has to print or test it.      */
struct search_out {
    int  ntried, nok;
    int  tc, iters;            /* of the winning start                        */
    char label[24];            /* which start won                             */
    real worst;                /* lowest logL among the starts that converged */
};

#define SEARCH_MAXC 64

/*  THE SEARCH.  x comes in as the caller's seed and leaves as the best point */
/*  found; dev and cov are those of the winning run (they come from the      */
/*  BFGS factor of THAT run: est() called again at the optimum does not      */
/*  iterate and would leave them at their initialisation).  *ll and *s2 are the winner's.  Returns   */
/*  est()'s ifault for the winner, or the seed's when no start converged.   */
/*  With echo, a table of the starts and the winner's optimizer report are   */
/*  written to the .out.                                                    */
static int fit_search(real *x, int npar, real *dev, real **cov,
                      real *ll, real *s2, int njitter, int echo,
                      struct search_out *so)
{
    int cls = ma_class_now();
    int nc = 0, best = -1, k, i;
    int ifault_seed = 0;
    real *xc, *xbest, *devc, **covc;
    struct { char lab[24]; int ok, tc, it, adm; real ll; } C[SEARCH_MAXC];
    char *best_text = NULL;
    real best_s2 = 0.0;

    xc = vector(1, npar); xbest = vector(1, npar);
    devc = vector(1, npar); covc = matrix(1, npar, 1, npar);

    /*  One candidate: made admissible, optimised, compared with the best.  */
#define TRY(LABEL, XSTART) do {                                                \
        int _tc, _it, _if, _adm; real _ll, _s2; char *_txt = NULL;             \
        if (nc >= SEARCH_MAXC) break;                                          \
        for (i = 1; i <= npar; i++) xc[i] = (XSTART)[i];                       \
        _adm = make_admissible(xc, npar);                                      \
        /*  A start nothing made admissible is not handed to est(): it would  \
         *  refuse it anyway, and on that path it leaks (BUG-41).           */ \
        if (_adm < 0) { _if = 6; _ll = 0.0; _s2 = 0.0; _tc = 0; _it = -1; }    \
        else _if = est_run(xc, npar, devc, covc, &_ll, &_s2, &_tc, &_it, &_txt); \
        snprintf(C[nc].lab, sizeof C[nc].lab, "%s", (LABEL));                  \
        C[nc].ok = (_if == 0); C[nc].tc = _tc; C[nc].it = _it;                 \
        C[nc].adm = _adm; C[nc].ll = _ll;                                      \
        if (nc == 0) ifault_seed = _if;                                        \
        if (_if == 0 && (best < 0 || _ll > C[best].ll)) {                      \
            best = nc; best_s2 = _s2;                                          \
            for (i = 1; i <= npar; i++) {                                      \
                xbest[i] = xc[i]; dev[i] = devc[i];                            \
                for (int _j = 1; _j <= npar; _j++) cov[i][_j] = covc[i][_j];   \
            }                                                                  \
            free(best_text); best_text = _txt; _txt = NULL;                    \
        }                                                                      \
        free(_txt);                                                            \
        nc++;                                                                  \
    } while (0)

    /*  1. the caller's seed                                                */
    real *seed = vector(1, npar);
    for (i = 1; i <= npar; i++) seed[i] = x[i];
    TRY("seed", seed);

    /*  The other starts only make sense in the VEC parameterisation, and not */
    /*  while -seedgate holds F, Theta and Sigma at the rung below.          */
    int vec_layout = (!global_warma && !prof_hold);

    /*  2. the cold start, when the seed was something else                 */
    if (vec_layout) {
        int differs = 0;
        init_guess(xc, npar);
        for (i = 1; i <= npar; i++) if (fabs(xc[i] - seed[i]) > 1e-12) { differs = 1; break; }
        if (differs) { real *xg = vector(1, npar);
                       for (i = 1; i <= npar; i++) xg[i] = xc[i];
                       TRY("cold", xg); free_vector(xg, 1, npar); }
    }

    /*  2b. under -fixb2 v, the old seed too: Lambda, F and Sigma from the
     *  static-OLS W.  It is inconsistent with v (BUG-33) and usually worse,
     *  but not always -- and a search must never end below the single start
     *  it replaced.                                                         */
    if (vec_layout && global_fixb2 && global_fixb2_given && global_r > 0) {
        real *xo = vector(1, npar);
        seed_fixb2_consistent = 0; init_guess(xo, npar); seed_fixb2_consistent = 1;
        TRY("ols-b2", xo);
        free_vector(xo, 1, npar);
    }

    /*  3. Johansen's canonical seed                                        */
    if (vec_layout && global_r > 0 && !global_seedjoh && !global_fixb2) {
        real *xj = vector(1, npar);
        /*  init_guess announces -seedjoh in the .out and on the console; here
         *  it is one start among several, not the user's request.           */
        FILE *o_save = outputv; int q_quiet = quiet_mode;
        FILE *nul = tmpfile();
        if (nul) outputv = nul;
        quiet_mode = 1;
        global_seedjoh = 1; init_guess(xj, npar); global_seedjoh = 0;
        quiet_mode = q_quiet; outputv = o_save;
        if (nul) fclose(nul);
        TRY("johansen", xj);
        free_vector(xj, 1, npar);
    }

    /*  4. the nested chain.  The class immediately below the target is fitted
     *  by a SEARCH of its own -- which in turn starts from the class below it
     *  -- and its optimum, embedded, is the start here.  So the chain carries
     *  the best point found at every rung, not the first one: a single fit per
     *  rung left -mafree below -matri on PLL.  The recursion is a chain, not a
     *  tree: every level makes one recursive call, and it ends at q = 0.     */
    if (vec_layout && global_q > 0 && cls != MA_OTHER && cls != MA_Q0) {
        int q_save = global_q;
        int f_marow = global_marow, f_matri = global_matri, f_diag = global_diag_ma;
        int low;
        if      (cls == MA_FREE)  low = (global_r > 0) ? MA_MATRI : MA_Q0;
        else if (cls == MA_MATRI) low = (global_r > 0) ? MA_MAROW : MA_Q0;
        else                      low = MA_Q0;          /* MAROW, DIAG */
        global_q = (low == MA_Q0) ? 0 : q_save;
        ma_class_set(low == MA_Q0 ? MA_FREE : low);
        {
            int npl = calc_nparametrs();
            real *xl = vector(1, npl), *dl = vector(1, npl), **cl = matrix(1, npl, 1, npl);
            real lll, s2l;
            int ifl;
            init_guess(xl, npl);
            ifl = fit_search(xl, npl, dl, cl, &lll, &s2l, 1, 0, NULL);
            global_q = q_save;
            global_marow = f_marow; global_matri = f_matri; global_diag_ma = f_diag;
            if (ifl == 0) {
                real *xn = vector(1, npar);
                embed_ma(xl, low, (low == MA_Q0 ? 0 : q_save), xn, cls, q_save);
                TRY("nested", xn);
                free_vector(xn, 1, npar);
            }
            free_matrix(cl, 1, npl, 1, npl); free_vector(dl, 1, npl); free_vector(xl, 1, npl);
        }
        global_q = q_save;
        global_marow = f_marow; global_matri = f_matri; global_diag_ma = f_diag;
    }

    /*  5. -multistart: jitter around the caller's seed, as run_multistart did (P8) */
    if (njitter > 1) {
        uint64_t rng = 20260818ULL;          /* 64-bit on every ABI (BUG-61) */
        real *xt = vector(1, npar);
        for (k = 1; k < njitter; k++) {
            real amp = 0.05 * (real) (1 + (k - 1) % 20);
            for (i = 1; i <= npar; i++) {
                real u;
                rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                u = ((real) ((rng >> 33) & 0x7FFFFFFF)) / 2147483647.0;
                u = 2.0 * u - 1.0;
                xt[i] = seed[i] + amp * u * (fabs(seed[i]) > 1.0e-8 ? fabs(seed[i]) : 0.1);
            }
            char lab[24]; snprintf(lab, sizeof lab, "jitter %d", k);
            TRY(lab, xt);
        }
        free_vector(xt, 1, npar);
    }
#undef TRY

    /*  The winner                                                          */
    int nok = 0; real worst = 0.0;
    for (k = 0; k < nc; k++) if (C[k].ok) { if (nok == 0 || C[k].ll < worst) worst = C[k].ll; nok++; }
    if (best >= 0) {
        for (i = 1; i <= npar; i++) x[i] = xbest[i];
        *ll = C[best].ll; *s2 = best_s2;
    } else {
        *ll = 0.0; *s2 = 0.0;
    }
    if (so) {
        so->ntried = nc; so->nok = nok; so->worst = worst;
        so->tc = (best >= 0) ? C[best].tc : 0;
        so->iters = (best >= 0) ? C[best].it : -1;
        snprintf(so->label, sizeof so->label, "%s", best >= 0 ? C[best].lab : "none");
    }
    if (echo && nc > 1) {
        fprintf(outputv, "\nSearch: %d starting points, %d converged; logL from "
                         "%.6f to %.6f.  The best is `%s'.\n",
                nc, nok, worst, (best >= 0 ? C[best].ll : 0.0),
                (best >= 0 ? C[best].lab : "none"));
        fprintf(outputv, "  start        logL            stop  iters  admissible start\n");
        for (k = 0; k < nc; k++) {
            fprintf(outputv, "  %-10s ", C[k].lab);
            if (C[k].ok) fprintf(outputv, "%15.6f   ", C[k].ll);
            else         fprintf(outputv, "  (failed)        ");
            fprintf(outputv, "  %4s  ",
                    C[k].tc ? (C[k].tc == 1 ? "grad" : C[k].tc == 2 ? "step" :
                               C[k].tc == 3 ? "tc3" : C[k].tc == 4 ? "iter" : "maxl") : "  - ");
            if (C[k].it >= 0) fprintf(outputv, "%5d", C[k].it);
            else              fprintf(outputv, "%5s", "-");
            fprintf(outputv, "  %s\n",
                    C[k].adm == 0 ? "as given" : C[k].adm > 0 ? "shrunk" : "not found");
        }
        fprintf(outputv, "  The SPREAD is the diagnostic: on a well-behaved surface "
                         "every start lands in the same place.\n");
    }
    if (echo && best_text) fputs(best_text, outputv);
    free(best_text);
    free_vector(seed, 1, npar);
    free_matrix(covc, 1, npar, 1, npar);
    free_vector(devc, 1, npar); free_vector(xbest, 1, npar); free_vector(xc, 1, npar);
    return (best >= 0) ? 0 : ifault_seed;
}

/*****************************************************************************/
/*  fit_r0_ladder -- THE RUNGS BELOW THE RANK, EACH FROM THE ONE BELOW IT.    */
/*                                                                           */
/*  The suite's convention (drtran, the ladder of SUITE_INTEGRATION.md): the */
/*  univariate models are optima; the diagonal gate -- r = 0, diagonal F,    */
/*  Theta and Sigma -- is where the joint likelihood factorises into theirs  */
/*  (Theorem 9), so its optimum is theirs and its certificate says so; then  */
/*  structure is added one rung at a time, each estimated from the optimum   */
/*  of the rung below with the new entries at zero.  Below the rank those    */
/*  are ORDINARY nested steps: the rung below is an interior point of the    */
/*  one above.                                                               */
/*                                                                           */
/*    rung 0   F, Theta, Sigma diagonal     the gate, certified              */
/*    rung 1   Sigma free                                                    */
/*    rung 2   the structure asked for (F, Theta free unless -diag* flags)   */
/*                                                                           */
/*  Rung 2's optimum is what route (B) holds while it profiles Lambda and B2 */
/*  (gate_profile_seed): the VEC cannot be carried up by adding zeros --     */
/*  Lambda = 0 is on the boundary of the rung above (VEC_EMBEDDING_PLAN.md   */
/*  3) -- which is why the crossing is a profile and not an embedding.       */
/*                                                                           */
/*  Returns 1 and leaves the r = 0 optimum in F0, Th0, S0 (full M x M, in    */
/*  the .inp's order, which at r = 0 is also the engine's) and its logL in   */
/*  *ll0; 0 if a rung failed.  The result is cached for the rest of the run  */
/*  (-lrtest asks for it once per rank) and keyed on what determines it.    */
/*****************************************************************************/
/*  When the ladder is the start.  By default for r >= 1; not under -warma
 *  (another parameterisation) or -noladder; and not when the user gave a seed
 *  of their own (-seedb2, -seedjoh, -seed, -seedybar): an explicit seed is a
 *  request, and replacing it in silence would be answering another one.
 *  -seedgate asks for the ladder whatever else was given.                  */
static int ladder_wanted(void)
{
    if (global_warma) return 0;
    if (global_seedgate) return 1;
    if (global_noladder) return 0;
    if (global_seedb2 || global_seedjoh || global_seed) return 0;
    return 1;
}

static real ***lad_F = NULL, ***lad_Th = NULL, **lad_S = NULL, lad_ll = 0.0;
static int     lad_ok = 0, lad_nf = 0, lad_q = 0, lad_M = 0;
static double  lad_key = 0.0;

static double ladder_key(void)
{
    double k = global_p * 1e6 + global_q * 1e4 + global_case * 1e3
             + global_diag_ar * 100 + global_diag_ma * 10 + global_diag_cov
             + nobs_raw * 1e-3 + global_matri * 0.5;
    int t, i;
    for (i = 1; i <= nser && i <= ND_MAX; i++)       /* the drift mask */
        if (global_case == 3 && !y2mu_free(i)) k += 1e12 * (double) (1 << (i - 1));
    for (i = 1; i <= nser && i <= XS_MAX; i++)       /* the -xsys mask */
        if (xs_on && xs_inp[i]) k += 1e9 * (double) (1 << (i - 1));
    if (gz_on) {                                     /* the -gzero mask */
        int j;
        for (i = 1; i <= nser && i <= GZ_MAX; i++)
            for (j = 1; j <= nser && j <= GZ_MAX; j++)
                if (gz[i][j]) k += 3.7e8 * (double) ((i - 1) * GZ_MAX + j);
    }
    for (t = 1; t <= nobs_raw; t++)
        for (i = 1; i <= nser; i++) k += rawmat[t][i] * (1e-7 * (t % 97 + i));
    return k;
}

/*  Pack an r = 0 point from full matrices, in the walk of the CURRENT flags. */
static void pack_r0(real *x, real ***F, real ***Th, real **S, real *mu)
{
    int M = nser, nf = (global_p > 1) ? global_p - 1 : 0, q = global_q;
    int i, j, k, idx = 1, head, tail;
    real s11 = (S[1][1] > 1e-24) ? S[1][1] : 1.0;
    if (global_case == 3)
        for (i = 1; i <= M; i++) if (y2mu_free(i)) x[idx++] = mu[i];
    for (k = 1; k <= nf; k++) {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) if (f_free(i, j)) x[idx++] = F[k][i][j];
    }
    search_blocks(&head, &tail);
    if (q > 0) theta_pack(Th, ma_class_now(), q, x, idx - 1);
    idx += theta_len(ma_class_now(), q);
    for (i = 2; i <= M; i++) x[idx++] = S[i][i] / s11;
    if (!global_diag_cov)
        for (i = 2; i <= M; i++) for (j = 1; j < i; j++) x[idx++] = S[i][j] / s11;
}

static int fit_r0_ladder(real ***F0, real ***Th0, real **S0, real *ll0, int echo)
{
    int M = nser, nf = (global_p > 1) ? global_p - 1 : 0, q = global_q;
    int r_save = global_r;
    int t_dar = global_diag_ar, t_dma = global_diag_ma, t_dcov = global_diag_cov;
    int rungs[3][3], nr = 0, st, i, j, k, ok = 1;
    double key;
    real ***F = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
    real ***Th = tensor(1, (q > 0 ? q : 1), 1, M, 1, M);
    real **S = matrix(1, M, 1, M), *mu = vector(1, M), ll = 0.0;

    global_r = 0;
    build_y2_levels();
    key = ladder_key();
    if (lad_ok && key == lad_key && lad_M == M && lad_nf == nf && lad_q == q) {
        for (k = 1; k <= nf; k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) F0[k][i][j] = lad_F[k][i][j];
        for (k = 1; k <= q; k++)  for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Th0[k][i][j] = lad_Th[k][i][j];
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) S0[i][j] = lad_S[i][j];
        *ll0 = lad_ll;
        global_r = r_save; build_y2_levels();
        free_tensor(F, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
        free_tensor(Th, 1, (q > 0 ? q : 1), 1, M, 1, M);
        free_matrix(S, 1, M, 1, M); free_vector(mu, 1, M);
        return 1;
    }

    rungs[nr][0] = 1; rungs[nr][1] = 1; rungs[nr][2] = 1; nr++;
    if (!t_dcov) { rungs[nr][0] = 1; rungs[nr][1] = 1; rungs[nr][2] = 0; nr++; }
    if (!t_dar || !t_dma) { rungs[nr][0] = t_dar; rungs[nr][1] = t_dma; rungs[nr][2] = t_dcov; nr++; }

    if (echo)
        fprintf(outputv, "\n=== The ladder below the rank: r = 0, from the gate up ===\n");
    for (st = 0; st < nr && ok; st++) {
        int np, ifr, ifs = 0;
        real s2, *x, *dev, **cov;
        struct Tvarma v;
        global_diag_ar = rungs[st][0]; global_diag_ma = rungs[st][1];
        global_diag_cov = rungs[st][2];
        np = calc_nparametrs();
        x = vector(1, np); dev = vector(1, np); cov = matrix(1, np, 1, np);
        if (st == 0) init_guess(x, np);
        else         pack_r0(x, F, Th, S, mu);
        ifr = fit_search(x, np, dev, cov, &ll, &s2, 1, 0, NULL);
        if (ifr != 0) {
            ok = 0;
            if (echo) fprintf(outputv, "  rung %d did not converge (ifault = %d)\n", st, ifr);
        } else {
            v.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
            vec_shootx(x, &v, &ifs, 1, 0);
            /*  At r = 0 Cbar = I and Hbar = 0, so Phi*_k = F_k, Theta*_k =
             *  Theta_k and Sigma* = Sigma term by term (vec_shootx [4]).     */
            for (k = 1; k <= nf; k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) F[k][i][j] = v.phi[k][i][j];
            for (k = 1; k <= q; k++)  for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Th[k][i][j] = v.theta[k][i][j];
            for (i = 1; i <= M; i++) { mu[i] = v.mu[i]; for (j = 1; j <= M; j++) S[i][j] = v.qq[i][j]; }
            if (echo) {
                fprintf(outputv, "  rung %d  (F %s, Theta %s, Sigma %s)   logL = %.10f\n", st,
                        global_diag_ar ? "diag" : "free", global_diag_ma ? "diag" : "free",
                        global_diag_cov ? "diag" : "free", ll);
                if (st == 0) { v.logelf = ll; gate_contract(&v); }
            }
            vec_shootx(x, &v, &ifs, 0, 1);
        }
        free_matrix(cov, 1, np, 1, np); free_vector(dev, 1, np); free_vector(x, 1, np);
    }
    global_diag_ar = t_dar; global_diag_ma = t_dma; global_diag_cov = t_dcov;
    if (ok) {
        for (k = 1; k <= nf; k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) F0[k][i][j] = F[k][i][j];
        for (k = 1; k <= q; k++)  for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Th0[k][i][j] = Th[k][i][j];
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) S0[i][j] = S[i][j];
        *ll0 = ll;
        /* the cache */
        if (lad_ok) {
            free_tensor(lad_F, 1, (lad_nf > 0 ? lad_nf : 1), 1, lad_M, 1, lad_M);
            free_tensor(lad_Th, 1, (lad_q > 0 ? lad_q : 1), 1, lad_M, 1, lad_M);
            free_matrix(lad_S, 1, lad_M, 1, lad_M);
        }
        lad_F = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
        lad_Th = tensor(1, (q > 0 ? q : 1), 1, M, 1, M);
        lad_S = matrix(1, M, 1, M);
        for (k = 1; k <= nf; k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) lad_F[k][i][j] = F[k][i][j];
        for (k = 1; k <= q; k++)  for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) lad_Th[k][i][j] = Th[k][i][j];
        for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) lad_S[i][j] = S[i][j];
        lad_ll = ll; lad_key = key; lad_ok = 1; lad_M = M; lad_nf = nf; lad_q = q;
    }
    global_r = r_save; build_y2_levels();
    free_tensor(F, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
    free_tensor(Th, 1, (q > 0 ? q : 1), 1, M, 1, M);
    free_matrix(S, 1, M, 1, M); free_vector(mu, 1, M);
    return ok;
}

/*  ladder_seed_r0 -- at r = 0 the ladder's top rung IS the model, so its
 *  optimum is the start.  Found 2026-09-24 through -rungs: on mink-muskrat
 *  (2 1 0 -case 1) the plain fit stops at logL -6.3311 from every one of 21
 *  starts, and climbing from the certified gate reaches -1.0495 -- the same
 *  model, 5.28 higher.  -lrtest took its L(0) from the plain fit, which
 *  inflated LR(0 -> 1) by ~10.6.  The gate itself (all diagonal) is its own
 *  rung and needs nothing.                                                   */
static void ladder_seed_r0(real *x, int np)
{
    int M = nser, nf = (global_p > 1) ? global_p - 1 : 0, q = global_q, i;
    real ***F, ***Th, **S, *mu, ll = 0.0;
    (void) np;
    if (global_r != 0) return;
    if (global_diag_ar && global_diag_ma && global_diag_cov) return;
    F  = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
    Th = tensor(1, (q > 0 ? q : 1), 1, M, 1, M);
    S  = matrix(1, M, 1, M);
    mu = vector(1, M);
    {
        int im = 1;
        for (i = 1; i <= M; i++) mu[i] = y2mu_free(i) ? x[im++] : 0.0;
    }
    if (fit_r0_ladder(F, Th, S, &ll, 1)) pack_r0(x, F, Th, S, mu);
    free_vector(mu, 1, M);
    free_matrix(S, 1, M, 1, M);
    free_tensor(Th, 1, (q > 0 ? q : 1), 1, M, 1, M);
    free_tensor(F, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
}

static void free_ladder_cache(void)
{
    if (!lad_ok) return;
    free_tensor(lad_F, 1, (lad_nf > 0 ? lad_nf : 1), 1, lad_M, 1, lad_M);
    free_tensor(lad_Th, 1, (lad_q > 0 ? lad_q : 1), 1, lad_M, 1, lad_M);
    free_matrix(lad_S, 1, lad_M, 1, lad_M);
    lad_ok = 0;
}

/*****************************************************************************/
/*  gate_profile_seed — ROUTE (B) OF THE PLAN: the r = 0 optimum, and on top */
/*  of it, Lambda and B2 by likelihood.                                       */
/*                                                                           */
/*  WHY IT EXISTS.  The whole construction of the suite goes from OPTIMA    */
/*  TO OPTIMA: a rung is estimated, certified, and the one above starts     */
/*  there.  That bridge does NOT reach the r = 1 rung, and the reason is    */
/*  measured in docs/VEC_EMBEDDING_PLAN.md 3: with Lambda = 0 the           */
/*  transformed system has an AR root of modulus exactly 1.000000, i.e. the */
/*  base is ON the boundary of the space above and not in its interior, the */
/*  likelihood is not defined there, and B2 is not identified either,       */
/*  because Pi = Lambda B' = 0 whatever B2 is.  Seeding with the rung below */
/*  not merely inexact.                                                       */
/*                                                                           */
/*  HOW IT CROSSES.  By holding F, Theta and Sigma at the r = 0 optimum and */
/*  estimating ONLY Lambda and B2 (and the mean, which at r = 1 has an      */
/*  E[W] that does not exist below).  The conditional problem is small and  */
/*  far better conditioned than the joint one, and its solution is          */
/*  admissible BY CONSTRUCTION: the step off the boundary is chosen by the  */
/*  likelihood and not by whoever writes the program.  That is the reason   */
/*  for preferring (B) to (A) -- entering along the direction that fits,    */
/*  with a calibrated step --: (A) needs a constant, and a fixed constant   */
/*                                                                           */
/*  WHAT IT HOLDS, EXACTLY.  The rung below is estimated with the SAME      */
/*  structure that was asked for above: with no diagonal flags it is rung 2*/
/*  of the ladder (F, Theta and Sigma free), and with them it is the        */
/*  certified diagonal gate.  The plan says "the gate's values"; the r = 0  */
/*  optimum of the requested structure is taken because it is the rung      */
/*  immediately below and the one the ladder mandates, and with the flags     */
/*  set, the two coincide.                                                    */
/*                                                                           */
/*  Returns 1 if it left a new starting point in x, and 0 if it did not, in    */
/*  which case x is still init_guess's -- route (C) -- and that is said.       */
/*****************************************************************************/
static int gate_profile_seed(real *x, int npar)
{
    int M = nser, r0 = global_r, p = global_p, q = global_q;
    int nf = (p > 1) ? p - 1 : 0;
    int nmean, nlam, nhead, nmid, ntail, np2, i, j, k, idx, ifr = 0, ifc;
    real *x2, *dev2, **cov2;
    struct Tvarma v2;

    if (r0 <= 0) return 0;                  /* with no VEC matrix there is nothing to cross */
    par_blocks(&nmean, &nlam, &nmid, &ntail);
    nhead = nmean + nlam;
    if (nhead + nmid + ntail != npar) return 0;      /* the vector is not the one */
                                                     /* this walk expects  */

    /* ---- 1. the rung below: the ladder from the gate up (fit_r0_ladder) --- */
    {
        real ***F0 = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
        real ***T0 = tensor(1, (q  > 0 ? q  : 1), 1, M, 1, M);
        real **S0  = matrix(1, M, 1, M);
        int s0 = M - r0, a, b;
        real s11;
        for (k = 1; k <= (nf > 0 ? nf : 1); k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) F0[k][i][j] = 0.0;
        for (k = 1; k <= (q  > 0 ? q  : 1); k++) for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) T0[k][i][j] = 0.0;
        if (!fit_r0_ladder(F0, T0, S0, &gate_seed_ll0, 1)) {
            fprintf(outputv, "\n-seedgate: the ladder below the rank did not converge; "
                             "falling back to the cold start.\n");
            if (!quiet_mode)
                printf("  ladder: the r = 0 rungs did not converge; carrying on cold\n");
            free_tensor(F0, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
            free_tensor(T0, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
            free_matrix(S0, 1, M, 1, M);
            return 0;
        }
        /*  BUG-32.  The r = 0 optimum is in the .inp's order -- at r = 0 that
         *  is the engine's too -- but at r0 > 0 the cast reads F, Theta and
         *  Sigma in the INTERNAL order [Y1 ; Y2]: internal a is .inp s0 + a
         *  for a <= r0, and a - r0 after.  Copying them straight, as this did,
         *  held another model: on mink-muskrat the profiled start was -60.37
         *  where the permuted one is -1.37, above the r = 0 rung itself.
         *  Sigma is then renormalised so that its [1][1] -- now Y1's -- is 1,
         *  which is the convention of the parameter vector (the scale is
         *  concentrated, so the likelihood does not see it).                */
        #define PERM(z) ((z) <= r0 ? s0 + (z) : (z) - r0)
        hold_M = M; hold_nf = nf; hold_q = q;
        hold_F  = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
        hold_Th = tensor(1, (q  > 0 ? q  : 1), 1, M, 1, M);
        hold_S  = matrix(1, M, 1, M);
        for (k = 1; k <= (nf > 0 ? nf : 1); k++)
            for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                hold_F[k][a][b] = (k <= nf) ? F0[k][PERM(a)][PERM(b)] : 0.0;
        for (k = 1; k <= (q > 0 ? q : 1); k++)
            for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
                hold_Th[k][a][b] = (k <= q) ? T0[k][PERM(a)][PERM(b)] : 0.0;
        s11 = S0[PERM(1)][PERM(1)];
        if (s11 <= 1e-24) s11 = 1.0;
        for (a = 1; a <= M; a++) for (b = 1; b <= M; b++)
            hold_S[a][b] = S0[PERM(a)][PERM(b)] / s11;
        #undef PERM
        free_tensor(F0, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
        free_tensor(T0, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
        free_matrix(S0, 1, M, 1, M);
    }
    /*  BUG-32 (b).  Nothing above may leave the rank-r0 state changed: the
     *  ladder ran init_guess at r = 0, which reallocates B2_fixed (a fixed B2
     *  came back as 0) and republishes cond_resid.  One init_guess at r0 into
     *  a scratch vector restores both, from the same data and flags.        */
    global_r = r0;
    build_y2_levels();
    {
        real *xs = vector(1, npar);
        init_guess(xs, npar);
        free_vector(xs, 1, npar);
    }

    /* ---- 2. the conditional step: only the mean, Lambda and B2 ------------ */
    global_r = r0;
    build_y2_levels();
    prof_hold = 1;
    np2  = calc_nparametrs();               /* = nhead + ntail, from par_blocks */
    x2   = vector(1, np2);
    dev2 = vector(1, np2);
    cov2 = matrix(1, np2, 1, np2);
    for (i = 1; i <= nhead; i++) x2[i] = x[i];
    for (i = 1; i <= ntail; i++) x2[nhead + i] = x[nhead + nmid + i];
    v2.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x2, &v2, &ifr, 1, 0);

    /*  A STARTING POINT THAT IS ADMISSIBLE, AND THE LIKELIHOOD CHOOSES IT.
     *
     *  This was forced by the measurement and was not in the plan: with F,
     *  Theta and Sigma held at the r = 0 optimum and Lambda at the value of the
     *  conditional regression, the transformed system comes out NON-STATIONARY
     *  -- elf returns ifault = 3 -- and the optimiser does not even start,
     *  because est refuses if the initial point is not admissible
     *  (drvmlest.c, "bad initial estimates").  It is the other face of what the
     *  plan measured in 3: at Lambda = 0 the root is EXACTLY at 1, and of the
     *  two directions leaving that point only one is admissible.  The sign the
     *  conditional regression brings need not be that one.
     *
     *  So a ladder of multiples of the Lambda the regression brought is walked
     *  -- the DIRECTION is chosen by the data, not by the program -- and the
     *  start is the best ADMISSIBLE point among those evaluated.  Which is not
     *  a fixed constant, which is what the plan forbids: it is a multiple of
     *  something estimated, and the conditional step moves it afterwards
     *  anyway.  If none is admissible, nothing is crossed and that is said.  */
    {
        const real LOG2PI = 1.837877066;
        static const real mult[18] = { 1.0, -1.0, 0.5, -0.5, 0.25, -0.25,
                                       0.1, -0.1, 0.05, -0.05, 0.02, -0.02,
                                       0.01, -0.01, 2.0, -2.0, 4.0, -4.0 };
        real *lam0 = vector(1, (nlam > 0 ? nlam : 1));
        real best = 0.0, bestc = 0.0;
        int  have = 0, mi;

        for (i = 1; i <= nlam; i++) lam0[i] = x2[nmean + i];
        for (mi = 0; mi < 18; mi++) {
            real pi1, pi2, pi3, ll;
            int ifev = 0, ifc2 = 0;
            for (i = 1; i <= nlam; i++) x2[nmean + i] = mult[mi] * lam0[i];
            vec_shootx(x2, &v2, &ifc2, 0, 0);
            if (ifc2 != 0) continue;                  /* Sigma not positive def. */
            elf(v2.m, v2.n, v2.p, v2.q, v2.mu, v2.phi, v2.theta, v2.qq, v2.w,
                1.0, v2.xitol, TRUE, v2.a, &pi1, &pi2, &pi3, &ifev);
            if (ifev != 0) continue;                  /* not admissible: 1..5    */
            ll = -0.5 * v2.m * v2.n * (LOG2PI - log((real) v2.m)
                 - log((real) v2.n) + 1.0)
                 - 0.5 * v2.n * (v2.m * log(pi1) + log(pi2));
            if (!have || ll > best) { have = 1; best = ll; bestc = mult[mi]; }
        }
        for (i = 1; i <= nlam; i++) x2[nmean + i] = bestc * lam0[i];
        free_vector(lam0, 1, (nlam > 0 ? nlam : 1));
        gate_seed_lam = bestc;
        gate_seed_ll_start = have ? best : 0.0;
        if (!have) {
            fprintf(outputv, "\n-seedgate: no admissible starting point for the "
                             "conditional step -- every multiple of the "
                             "conditional-regression Lambda leaves the transformed "
                             "system non-stationary or non-invertible.  Falling "
                             "back to the cold start.\n");
            if (!quiet_mode)
                printf("  -seedgate: no admissible start; carrying on cold\n");
            vec_shootx(x2, &v2, &ifr, 0, 1);
            prof_hold = 0;
            free_matrix(cov2, 1, np2, 1, np2);
            free_vector(dev2, 1, np2);
            free_vector(x2, 1, np2);
            free_matrix(hold_S, 1, M, 1, M);
            free_tensor(hold_Th, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
            free_tensor(hold_F,  1, (nf > 0 ? nf : 1), 1, M, 1, M);
            hold_F = NULL; hold_Th = NULL; hold_S = NULL;
            return 0;
        }
    }

    est(&vec_shootx, np2, x2, dev2, cov2, 500, 200, 1e-5, 1e-7,
        v2.xitol, v2.a, &v2.sigma2, &v2.logelf, &ifr);
    ifc = ifr;                     /* est leaves its code here, and the release */
    gate_seed_ll1 = v2.logelf;     /* below would overwrite it                  */
    vec_shootx(x2, &v2, &ifr, 0, 1);
    prof_hold = 0;

    if (ifc != 0) {
        fprintf(outputv, "\n-seedgate: the conditional step for Lambda and B2 "
                         "did not converge (ifault = %d); falling back to the "
                         "cold start.\n", ifc);
        if (!quiet_mode)
            printf("  -seedgate: the conditional step did not converge; carrying on cold\n");
        free_matrix(cov2, 1, np2, 1, np2);
        free_vector(dev2, 1, np2);
        free_vector(x2, 1, np2);
        free_matrix(hold_S, 1, M, 1, M);
        free_tensor(hold_Th, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
        free_tensor(hold_F,  1, (nf > 0 ? nf : 1), 1, M, 1, M);
        hold_F = NULL; hold_Th = NULL; hold_S = NULL;
        return 0;
    }

    /* ---- 3. the full vector: head and tail from the conditional step, middle
              stretch from the r = 0 optimum.  The order of writing is
              vec_shootx's walk, and it has to be: this is the fifth site that
              walks this vector.                                              */
    for (i = 1; i <= nhead; i++) x[i] = x2[i];
    for (i = 1; i <= ntail; i++) x[nhead + nmid + i] = x2[nhead + i];
    idx = nhead + 1;
    for (k = 1; k <= nf; k++) {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) if (f_free(i, j)) x[idx++] = hold_F[k][i][j];
    }
    /*  The rung below is estimated at r = 0, where the moving average is FREE
     *  (the structured classes collapse there; see ma_struct_on()).  The rung
     *  above may be in a class with fewer parameters, so what is carried over
     *  is the PROJECTION of the retained Theta onto that class: the entries the
     *  class carries are kept and the ones it zeroes are dropped.  Writing M*M
     *  here, which is what used to be done, misaligned the vector as soon as
     *  the default stopped being the free class.  The walk has to be the SAME
     *  as the cast's, and that is why it goes in the same order.             */
    for (k = 1; k <= q; k++) {
        if (mawarma_on()) {
            for (i = 1; i <= r0; i++)
                for (j = 1; j <= r0; j++) x[idx++] = hold_Th[k][i][j];
        } else if (marow_on()) {
            for (i = 1; i <= r0; i++)
                for (j = 1; j <= M; j++) x[idx++] = hold_Th[k][i][j];
        } else if (global_matri) {
            for (i = 1; i <= r0; i++)
                for (j = 1; j <= M; j++) x[idx++] = hold_Th[k][i][j];
            for (i = r0 + 1; i <= M; i++)
                for (j = r0 + 1; j <= M; j++) x[idx++] = hold_Th[k][i][j];
        } else if (global_diag_ma) {
            for (i = 1; i <= M; i++) x[idx++] = hold_Th[k][i][i];
        } else {
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) x[idx++] = hold_Th[k][i][j];
        }
    }
    if (global_diag_cov) {
        for (i = 2; i <= M; i++) x[idx++] = hold_S[i][i];
    } else {
        for (i = 2; i <= M; i++) x[idx++] = hold_S[i][i];
        for (i = 2; i <= M; i++)
            for (j = 1; j < i; j++) x[idx++] = hold_S[i][j];
    }
    if (idx - 1 != nhead + nmid) {          /* the walk does not add up: not used */
        fprintf(outputv, "\n-seedgate: internal walk mismatch (%d vs %d); "
                         "falling back to the cold start.\n",
                idx - 1, nhead + nmid);
        free_matrix(cov2, 1, np2, 1, np2);
        free_vector(dev2, 1, np2);
        free_vector(x2, 1, np2);
        free_matrix(hold_S, 1, M, 1, M);
        free_tensor(hold_Th, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
        free_tensor(hold_F,  1, (nf > 0 ? nf : 1), 1, M, 1, M);
        hold_F = NULL; hold_Th = NULL; hold_S = NULL;
        return 0;
    }

    free_matrix(cov2, 1, np2, 1, np2);
    free_vector(dev2, 1, np2);
    free_vector(x2, 1, np2);
    free_matrix(hold_S, 1, M, 1, M);
    free_tensor(hold_Th, 1, (q  > 0 ? q  : 1), 1, M, 1, M);
    free_tensor(hold_F,  1, (nf > 0 ? nf : 1), 1, M, 1, M);
    hold_F = NULL; hold_Th = NULL; hold_S = NULL;

    gate_seed_ok = 1;
    fprintf(outputv,
        "\n=== -seedgate: the VEC block profiled on the rung below ===\n\n"
        "  r = 0 optimum (F, Theta, Sigma)        logL = %18.10f\n"
        "  admissible entry, Lambda x %-6.2f        logL = %18.10f\n"
        "  + Lambda and B2 profiled on it         logL = %18.10f\n\n"
        "  The starting point for the free fit below is that second value, not\n"
        "  a conditional regression.  Holding the rung below and estimating only\n"
        "  Lambda and B2 lands in the INTERIOR by construction: at Lambda = 0 the\n"
        "  transformed system has an AR root of modulus exactly one, so the rung\n"
        "  below cannot be carried up unchanged, and the step off that boundary\n"
        "  is chosen here by the likelihood rather than by a constant.\n",
        gate_seed_ll0, gate_seed_lam, gate_seed_ll_start, gate_seed_ll1);
    /*  THE PRE-ESTIMATES, WRITTEN OUT.  They are the product of the conditional
     *  step and one has to be able to see them: if (B) ends up worse than (C),
     *  the next question is always whether the starting point is reasonable or
     *  absurd, and that one is not answered by the logL.  The order is
     *  vec_shootx's walk: Lambda by rows (i outer, j inner) and B2 by columns.*/
    {
        int nl = (global_alpha ? alpha_sa : M), s0 = M - r0, c;
        fprintf(outputv, "\n  the pre-estimates the conditional step produced\n");
        fprintf(outputv, "    %s (%d x %d):\n",
                global_alpha ? "psi, with Lambda = A*psi" : "Lambda", nl, r0);
        c = nhead - nlam;                       /* where Lambda starts in x */
        for (i = 1; i <= nl; i++) {
            fprintf(outputv, "     ");
            for (j = 1; j <= r0; j++)
                fprintf(outputv, " %12.6f", x[c + (i - 1) * r0 + j]);
            fprintf(outputv, "\n");
        }
        if (global_fixb2)
            fprintf(outputv, "    B2 (%d x %d): held fixed by -fixb2\n", s0, r0);
        else {
            int nfr = s0 - b2_nheld(s0), cc;
            fprintf(outputv, "    B2 (%d x %d)%s:\n", s0, r0,
                    b2row_on ? ", rows held by -fixb2row marked *" : "");
            c = nhead + nmid;                   /* where B2 starts in x     */
            for (i = 1; i <= s0; i++) {
                fprintf(outputv, "     ");
                for (j = 1; j <= r0; j++) {
                    if (b2_held(i)) {
                        fprintf(outputv, " %11.6f*", B2_fixed[i][j]);
                        continue;
                    }
                    /* position of (i, j) among the free entries */
                    cc = 0;
                    for (int ii = 1; ii < i; ii++) cc += !b2_held(ii);
                    fprintf(outputv, " %12.6f", x[c + (j - 1) * nfr + cc + 1]);
                }
                fprintf(outputv, "\n");
            }
        }
        if (nmean > 0) {
            fprintf(outputv, "    mean block (%d):  ", nmean);
            for (i = 1; i <= nmean; i++) fprintf(outputv, " %12.6f", x[i]);
            fprintf(outputv, "\n");
        }
    }

    if (!quiet_mode)
        printf("  -seedgate: r=0 logL = %.6f  ->  perfilando Lambda y B2: %.6f\n",
               gate_seed_ll0, gate_seed_ll1);
    return 1;
}

/*****************************************************************************/
/*  P5 — THE FORECAST, IN LEVELS, WITH ITS BANDS                              */
/*****************************************************************************/
/*  WHY IT EXISTS.  Until 2026-08-20 drvec could not forecast: none of its
 *  options did it, there was no out-of-sample evaluation, and the register had
 *  seventeen hundred lines about estimation and no measurement of the one thing
 *  that decides whether a multivariate model is worth its parameters -- whether
 *  it improves on a univariate forecast.  See docs/PLAN_PRODUCCION.md P5.
 *
 *  HOW.  The fitted model IS a stationary VARMA on Ybar = (nabla Y2 ; W), so
 *  the forecast is made there with elf's recursion and the transformation is
 *  INVERTED, exactly as simulate_h0 does for the bootstrap.  There is no
 *  parallel code: it is the same inversion.
 *
 *      nabla Y2_{n+h} = Ybar_{n+h}[1..s]  ->  Y2 by cumulating from Y2_n
 *      W_{n+h}        = Ybar_{n+h}[s+1..M]
 *      Y1_{n+h}       = W_{n+h} - B2' Y2_{n+h}
 *
 *  THE VARIANCE, AND THE LESSON OF BUG-10.  In the sibling program the level
 *  was integrated correctly in the MEAN and incorrectly in the VARIANCE,
 *  because each reached the level by its own route: the mean asked its
 *  authoritative source for the operator and the variance rebuilt one of its
 *  own from (d, D, s).  The defect is invisible in the point forecast and comes
 *  out whole in the bands -- a factor of 19 in variance in the measured case --
 *  and the bias always runs the dangerous way: omitting non-stationary factors
 *  can only NARROW the band.
 *
 *  Here the integration is a single one (nabla on the Y2 block) and the W block
 *  is not integrated, but the trap is the same: Y1 = W - B2'Y2 INHERITS the
 *  cumulated error of Y2, and computing its band from W's alone leaves it
 *  systematically narrow.  So the map from innovations to level error is
 *  written ONCE, in level_error_map(), and the band comes from there.  With
 *  Psi_m the MA(inf) weights of the transformed system and C_m = sum_{k<=m}
 *  Psi_k:
 *
 *      G_m = [           rows 1..s of C_m             ]
 *            [ rows s+1..M of Psi_m - B2' (rows 1..s of C_m) ]
 *
 *      Var(h) = sum_{m=0}^{h-1} G_m Sigma* G_m',    Sigma* = sigma2 * qq
 *
 *  At h = 1, C_0 = Psi_0 = I and G_0 = [I_s 0 ; -B2' I_r], so the one-step band
 *  is the innovation covariance read in levels.  The suite checks that, and
 *  also checks the identity that really ties the calculation down: the one-step
 *  forecast from origin n-1 reproduces the datum minus the stored residual.   */

/*  compute_psi_weights — THE MA(inf) WEIGHTS, AND THEY ARE NOT OURS.
 *
 *  PROVENANCE, read before copying.  It is drvarma's function, which lives in
 *  drtran/src/forecast.c and whose header says "part of drvarma": the same
 *  lineage as elfvarma.c.  The first version of this section rewrote it, which
 *  is a third copy of a shared function and exactly the drift
 *  docs/PLAN_PRODUCCION.md P3 exists to prevent.  It was replaced by the
 *  suite's, character for character, on 2026-08-20.  If it is fixed there, it
 *  has to be fixed here.
 *
 *  Psi_0 = I;  Psi_l = sum_{i<=min(l,p)} Phi_i Psi_{l-i} - Theta_l (l <= q).
 *  Here it is applied to the TRANSFORMED VARMA on Ybar, which is where the
 *  model is stationary; the step to levels is another matter and is not
 *  borrowed -- see level_error_map().                                         */
void compute_psi_weights(int m, int p, int q, real ***phi, real ***theta,
                                int L, real ***psi)
{
    int l, i, j, k, i1, j1;
    for (l = 0; l <= L; l++)
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++)
                psi[l][i][j] = 0.0;
    for (i = 1; i <= m; i++)
        psi[0][i][i] = 1.0;

    for (l = 1; l <= L; l++) {
        for (i = 1; i <= l; i++) {
            if (i <= p) {
                for (i1 = 1; i1 <= m; i1++) {
                    for (j1 = 1; j1 <= m; j1++) {
                        real s = 0.0;
                        for (k = 1; k <= m; k++)
                            s += phi[i][i1][k] * psi[l-i][k][j1];
                        psi[l][i1][j1] += s;
                    }
                }
            }
        }
        if (l <= q) {
            for (i1 = 1; i1 <= m; i1++)
                for (j1 = 1; j1 <= m; j1++)
                    psi[l][i1][j1] -= theta[l][i1][j1];
        }
    }
}

/*  level_error_map — G_m, the ONLY source of truth for the step to levels.
 *  Csum is C_m = sum_{k<=m} Psi_k, already accumulated by the caller.         */
static void level_error_map(real **Csum, real **Psi_m, real **B2,
                            int M, int r, real **G)
{
    int s = M - r, i, j, k;
    for (i = 1; i <= s; i++)
        for (j = 1; j <= M; j++) G[i][j] = Csum[i][j];
    for (i = 1; i <= r; i++)
        for (j = 1; j <= M; j++) {
            real acc = Psi_m[s + i][j];
            for (k = 1; k <= s; k++) acc -= B2[k][i] * Csum[k][j];
            G[s + i][j] = acc;
        }
}

/*  forecast_core — the recursion and the step to levels, from ANY origin.
 *
 *  It exists as a separate function because two things use it: -f, which
 *  forecasts from the end of the sample, and -estwin, which does it from every
 *  origin of a rolling window.  Writing it twice would repeat the mistake this
 *  file already made with compute_psi_weights.
 *
 *  Yb receives the H forecasts of Ybar and lev those of the LEVEL, in the
 *  .inp's column order: Y2 (1..s) and then Y1 (s+1..M).  With r = 0 there is no
 *  W block, s = M, and the step to levels is the pure cumulation of the M
 *  differences: the Y1 loop is empty and no separate case is needed.          */
static void forecast_core(struct Tvarma *v, real **B2, int o, int H,
                          real **Yb, real **lev)
{
    int M = v->m, r = global_r, s = M - r, p = v->p, q = v->q;
    int h, i, j, k, l;

    for (h = 1; h <= H; h++) {
        for (i = 1; i <= M; i++) {
            real acc = 0.0;
            for (k = 1; k <= p; k++) {
                int t = h - k;
                for (l = 1; l <= M; l++)
                    acc += v->phi[k][i][l]
                         * ((t >= 1 ? Yb[t][l] : v->w[o + t][l]) - v->mu[l]);
            }
            for (k = 1; k <= q; k++) {
                int t = h - k;
                if (t >= 1) continue;               /* a futuro = 0           */
                for (l = 1; l <= M; l++) acc -= v->theta[k][i][l] * v->a[o + t][l];
            }
            Yb[h][i] = v->mu[i] + acc;
        }
    }
    for (h = 1; h <= H; h++) {
        for (i = 1; i <= s; i++)
            lev[h][i] = (h == 1 ? Y2_levels[o][i] : lev[h-1][i]) + Yb[h][i];
        for (j = 1; j <= r; j++) {
            real acc = Yb[h][s + j];
            for (i = 1; i <= s; i++) acc -= B2[i][j] * lev[h][i];
            lev[h][s + j] = acc;
        }
    }
}

/*  rolling_eval — P5.2: ROLLING-ORIGIN EVALUATION, WHICH IS THE ONLY
 *  MEASUREMENT THAT SAYS WHETHER THE MODEL IS ANY USE.
 *
 *  The protocol is the suite's, not a new one: estimate ONCE on 1..E, hold the
 *  parameters FIXED, and advance the origin one datum at a time over E..n-H,
 *  comparing each forecast with what actually happened.  The sibling program's
 *  port describes it as "the only way to decide EMPIRICALLY whether one model
 *  forecasts better than another", and it is right: the likelihood, the AIC and
 *  the theoretical bands do not say so.
 *
 *  WHY THE PARAMETERS ARE HELD.  Re-estimating at every origin would still be
 *  an honest measurement but would cost n-E-H optimisations; and above all it
 *  would mix two things -- what the model predicts and what the re-estimation
 *  learns -- that are better kept apart.  What matters is that the parameters
 *  have NOT seen the datum they are scored against, and the window gives that.
 *
 *  THE RESIDUALS.  The recursion needs the shocks up to the origin, and at
 *  origins later than E they do not exist yet.  They are obtained in ONE pass:
 *  Ybar is rebuilt over the whole sample with the parameters from 1..E and elf
 *  is called with atf = TRUE.  No information is brought forward: each origin
 *  uses only what there is up to it, and the parameters come from 1..E.       */
static int rolling_eval(real *x, int E, int H)
{
    int M = nser, r = global_r, s = M - r, i, h, o, nor = 0;
    int save_nobs = nobs, ifr = 0;
    struct Tvarma vf;
    FILE *csv = NULL;
    real p1, p2, p3;
    real **Yb, **lev, **sae, **sse, **spe, **B2r;
    int  **cnt;

    if (E + H > nobs_full) return 1;

    nobs = nobs_full;                      /* filter over the WHOLE sample    */
    vec_shootx(x, &vf, &ifr, 1, 0);
    /*  THE TRUNCATION TOLERANCE, WHICH WAS MISSING.  vec_shootx fills the
     *  structure except for this field -- every site that uses it sets it, and
     *  there are a dozen -- and here it was not being set: vf is on the stack,
     *  so elf received as xitol whatever was in that word, and valgrind caught
     *  it in cxi (elfvarma.c:773) with 60 jumps on an uninitialised value.
     *  This is the out-of-sample evaluation, i.e. the measurement that decides
     *  this program's version number (HOMOLOGATION.md 4t), running with an
     *  undefined truncation.  Found on 2026-08-22 by running valgrind over P6's
     *  new paths.  BUG-14.                                                   */
    vf.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    if (ifr == 0)
        elf(vf.m, vf.n, vf.p, vf.q, vf.mu, vf.phi, vf.theta, vf.qq, vf.w,
            1.0, vf.xitol, TRUE, vf.a, &p1, &p2, &p3, &ifr);
    if (ifr > 0) {
        vec_shootx(x, &vf, &ifr, 0, 1);
        nobs = save_nobs;
        return 1;
    }

    /*  B2 of the fit: the last block of the vector, unless -fixb2 holds it. */
    B2r = matrix(1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
    {
        int nmean, nlam, nmid, ntail, idx;
        par_blocks(&nmean, &nlam, &nmid, &ntail);
        idx = nmean + nlam + nmid + 1;
        for (int j = 1; j <= r; j++)
            for (i = 1; i <= s; i++)
                B2r[i][j] = b2_held(i) ? B2_fixed[i][j] : x[idx++];
    }

    Yb  = matrix(1, H, 1, M);   lev = matrix(1, H, 1, M);
    sae = matrix(1, H, 1, M);   sse = matrix(1, H, 1, M);
    spe = matrix(1, H, 1, M);   cnt = imatrix(1, H, 1, M);
    for (h = 1; h <= H; h++)
        for (i = 1; i <= M; i++) {
            sae[h][i] = sse[h][i] = spe[h][i] = 0.0; cnt[h][i] = 0;
        }

    /*  -C: the ORIGIN-BY-ORIGIN errors, which is what a test of equal
     *  predictive ability needs.  An aggregate RMSE does not allow a
     *  Diebold-Mariano: the series of losses is required.  The letter is the
     *  sibling program's, which carries the same option for the same reason. */
    /*  P6.7 — and it is written ALWAYS, in <base>.recursive, without having to
     *  be named.  It used to exist only if the user remembered -C, and the
     *  measurement that sustains this program's version number (§4t of the
     *  register) was made that way, against a temporary file.  A measurement
     *  that decides the version cannot depend on somebody remembering an
     *  option.  -C still works, now as a REDIRECTION.                        */
    {
        static char rec_path[600];
        if (!fc_csv) {
            snprintf(rec_path, sizeof rec_path, "%s.recursive", out_base);
            fc_csv = rec_path;
        }
        csv = fopen(fc_csv, "w");
        if (csv) {
            fprintf(csv, "# DRVEC %s  rolling-origin errors, one row per "
                         "(origin, horizon, series)\n", DRVEC_VERSION);
            fprintf(csv, "# estwin=%d horizon=%d p=%d q=%d r=%d case=%d\n",
                    E, H, global_p, global_q, r, global_case);
            fprintf(csv, "origin,h,series,actual,forecast,error\n");
        } else fprintf(stderr, "WARNING: cannot write %s\n", fc_csv);
    }

    for (o = E; o + H <= nobs_full; o++) {
        forecast_core(&vf, B2r, o, H, Yb, lev);
        nor++;
        for (h = 1; h <= H; h++)
            for (i = 1; i <= M; i++) {
                /*  The realised level: the Y2 block is in Y2_levels and the Y1 block
                 *  in datamat, which build_y2_levels left in LEVELS.         */
                real act = (i <= s) ? Y2_levels[o + h][i] : datamat[o + h][i];
                real e;
                /*  Scored in the series' own units, like the forecast (BUG-36):
                 *  the MAPE of a log level times refactor is no MAPE at all.   */
                {
                    int tr = o + h + (global_levels ? 1 : 0);
                    act       = bt_level(i, tr, act);
                    lev[h][i] = bt_level(i, tr, lev[h][i]);
                }
                e = act - lev[h][i];
                sae[h][i] += fabs(e);
                sse[h][i] += e * e;
                if (fabs(act) > 1.0e-12) spe[h][i] += fabs(e / act);
                cnt[h][i]++;
                if (csv) fprintf(csv, "%d,%d,%s,%.10f,%.10f,%.10f\n", o, h,
                                 series_names ? series_names[i] : "y",
                                 act, lev[h][i], e);
            }
    }

    fprintf(outputv,
        "\n=== Rolling-origin evaluation (out of sample) ===\n"
        "  Estimated ONCE on observations 1..%d; parameters held FIXED.\n"
        "  %d origins, %d..%d, each compared with what actually happened.\n"
        "  The parameters have not seen the data they are scored against.\n\n",
        E, nor, E, E + nor - 1);
    fprintf(outputv, "   h  series             MAE           RMSE          MAPE%%\n");
    for (h = 1; h <= H; h++)
        for (i = 1; i <= M; i++)
            fprintf(outputv, "%4d  %-12s %13.6f  %13.6f  %13.4f\n", h,
                    series_names ? series_names[i] : "y",
                    sae[h][i] / cnt[h][i], sqrt(sse[h][i] / cnt[h][i]),
                    100.0 * spe[h][i] / cnt[h][i]);
    fprintf(outputv,
        "\n  These are the numbers that decide whether the model earns its\n"
        "  parameters.  A likelihood, an AIC and a theoretical band do not.\n");
    if (!quiet_mode)
        printf("Rolling origin: %d origins from %d, H = %d, written to the .out\n",
               nor, E, H);

    if (csv) { fclose(csv); if (!quiet_mode) printf("Per-origin errors: %s\n", fc_csv); }
    free_imatrix(cnt, 1, H, 1, M);
    free_matrix(spe, 1, H, 1, M);  free_matrix(sse, 1, H, 1, M);
    free_matrix(sae, 1, H, 1, M);
    free_matrix(lev, 1, H, 1, M);  free_matrix(Yb, 1, H, 1, M);
    free_matrix(B2r, 1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
    vec_shootx(x, &vf, &ifr, 0, 1);
    nobs = save_nobs;
    return 0;
}


/*****************************************************************************/
/*  P6.8 — THE HYPOTHESES A VEC ANSWERS, AND THAT COST NO ESTIMATION       */
/*                                                                           */
/*  Until now this program printed by default the diagnosis (Hosking,      */
/*  Jarque-Bera, the R(k)) and the rank condition -- all about the         */
/*  RESIDUALS -- and no hypothesis about the RELATIONS, which is what the  */
/*  model is about.  The ones it had were all behind an option and         */
/*  todas exigian reestimar: -lrtest, -weakex, -matest, -artest, -fixb2.      */
/*                                                                           */
/*  The cheap rung was missing, and not for want of material: est()        */
/*  returns the covariance of the parameters in cov, and its diagonal is   */
/*  where the sd values the .out already prints beside Lambda and B2 come  */
/*  from.  With that matrix a Wald on a subvector is arithmetic.  The      */
/*  sibling program does it always (drvarma report.py:_wald_blocks); here  */
/*  there are also Lambda and B2, the part where a VEC is informative and  */
/*                                                                           */
/*  See docs/PLAN_PRODUCCION.md 7.2.                                         */
/*****************************************************************************/

/*  wald_sub — chi2 = th' S^+ th on the subvector idx[1..k], with S^+ the
 *  SVD PSEUDO-INVERSE and df = rank(S).  It is the port of the sibling's
 *  wald_test, and the pseudo-inverse is not an ornament: with -diagcov,
 *  -marow or -fixb2 there are directions the likelihood cannot see, and an
 *  ordinary inverse would turn them into a huge chi2 instead of discounting
 *  them from the df.                                                        */
static int wald_sub(real *x, real **cov, int *idx, int k, real *chi2, int *df)
{
    real **S, **V, *w, *th, *y;
    real tol, smax = 0.0, acc = 0.0;
    int i, j, rk = 0;

    *chi2 = 0.0; *df = 0;
    if (k < 1) return 1;

    S  = matrix(1, k, 1, k);
    V  = matrix(1, k, 1, k);
    w  = vector(1, k);
    th = vector(1, k);
    y  = vector(1, k);

    for (i = 1; i <= k; i++) {
        th[i] = x[idx[i]];
        for (j = 1; j <= k; j++) S[i][j] = cov[idx[i]][idx[j]];
    }
    svdcp(S, k, k, w, V);                    /* S <- U */
    for (i = 1; i <= k; i++) if (w[i] > smax) smax = w[i];
    tol = smax * 1.0e-8;

    for (i = 1; i <= k; i++) {               /* y = diag(1/w) U' th          */
        real u = 0.0;
        for (j = 1; j <= k; j++) u += S[j][i] * th[j];
        if (w[i] > tol) { y[i] = u / w[i]; rk++; } else y[i] = 0.0;
    }
    for (i = 1; i <= k; i++) {               /* chi2 = th' (V y)             */
        real v = 0.0;
        for (j = 1; j <= k; j++) v += V[i][j] * y[j];
        acc += th[i] * v;
    }

    free_vector(y, 1, k); free_vector(th, 1, k); free_vector(w, 1, k);
    free_matrix(V, 1, k, 1, k); free_matrix(S, 1, k, 1, k);

    if (!(acc >= 0.0)) return 1;             /* nan or negative: not emitted  */
    *chi2 = acc;
    *df   = rk;
    return (rk > 0) ? 0 : 1;
}

/*  emit_wald — one hypothesis, with its reading.  Returns 0 if emitted.    */
static int emit_wald(real *x, real **cov, int *idx, int k,
                     const char *title, const char *h0,
                     const char *reject, const char *accept)
{
    real chi2; int df;
    if (k < 1) return 1;
    if (wald_sub(x, cov, idx, k, &chi2, &df) != 0) {
        fprintf(outputv, "\n%s\n  not computable (the covariance of this block "
                         "is not usable)\n", title);
        return 1;
    }
    {
        real pv = gsl_cdf_chisq_Q(chi2, df);
        fprintf(outputv, "\n%s\n", title);
        if (h0) fprintf(outputv, "  %s\n", h0);
        fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pv);
        fprintf(outputv, "  %s\n", (pv < 0.05) ? reject : accept);
    }
    return 0;
}

/*  hypothesis_block — the whole block.  ix_* are the indices into x[] that
 *  the printer's walk noted down; 0 means that entry is not free (the
 *  structure, -fixb2 or -alpha fixes it) and therefore cannot be tested: it is
 *  imposed, not estimated.                                                   */
static void hypothesis_block(real *x, real **cov, int **ix_lam, int **ix_B2,
                             int **ix_F, int **ix_Th)
{
    int M = nser, r = global_r, s = M - r, p = global_p, q = global_q;
    int nf = (p > 1) ? p - 1 : 0;
    int *idx = ivector(1, (M * M * (nf + q) + M * r + s * r) + 1);
    int k, i, j, kk;
    char title[256], h0[256], acc_s[256], rej_s[256];
    const char *nm;

    banner("Joint Hypothesis Tests (Wald)");

    if (r > 0) {
        /* ---- 1. Lambda = 0, and why it is NOT a test --------------------- */
        k = 0;
        for (i = 1; i <= M; i++)
            for (j = 1; j <= r; j++)
                if (ix_lam[i][j] > 0) idx[++k] = ix_lam[i][j];
        if (k > 0) {
            real chi2; int df;
            if (wald_sub(x, cov, idx, k, &chi2, &df) == 0)
                fprintf(outputv,
                    "\nError-correction term as a whole\n"
                    "  H0: alpha = 0\n"
                    "  Wald chi2(%d) = %.4f, p-value = %.4f\n"
                    "  ! NOT A TEST: under H0, beta is unidentified "
                    "(Davies).  Use -lrtest -bootstrap\n",
                    df, chi2, gsl_cdf_chisq_Q(chi2, df));
        } else {
            fprintf(outputv,
                "\nError-correction term as a whole\n"
                "  alpha is not free (-alpha imposes alpha = A*psi): "
                "nothing to test\n");
        }

        /* ---- 2. Exogeneidad debil, fila a fila --------------------------- */
        fprintf(outputv,
            "\nWeak exogeneity, one variable at a time"
            "   [H0: row i of alpha = 0, chi2(%d)]\n", r);
        /*  i walks the .inp's order and inp2lam takes it to the row of
         *  Lambda that belongs to that series.  Walking Lambda directly, which
         *  is what this loop did until 2026-08-24, tested the right row and
         *  named the WRONG SERIES on every fit with r < M.                   */
        for (i = 1; i <= M; i++) {
            int li = inp2lam(i);
            k = 0;
            for (j = 1; j <= r; j++)
                if (ix_lam[li][j] > 0) idx[++k] = ix_lam[li][j];
            if (k < 1) continue;
            nm = series_names ? series_names[i] : "y";
            snprintf(title, sizeof title, "%s (%s block):", nm,
                     (i <= s) ? "nabla Y2" : "Y1");
            snprintf(h0, sizeof h0, "H0: this variable does not adjust");
            snprintf(rej_s, sizeof rej_s,
                     "REJECT H0 -> %s adjusts to the disequilibrium.", nm);
            snprintf(acc_s, sizeof acc_s,
                     "Cannot reject H0 -> %s is weakly exogenous for B.", nm);
            emit_wald(x, cov, idx, k, title, h0, rej_s, acc_s);
        }

        /* ---- 3. Exclusion from the long-run relation --------------------- */
        k = 0;
        for (i = 1; i <= s; i++)
            for (j = 1; j <= r; j++)
                if (ix_B2[i][j] > 0) idx[++k] = ix_B2[i][j];
        if (k > 0) {
            fprintf(outputv,
                "\nExclusion from the cointegrating relations"
                "   [H0: row i of beta_2 = 0, chi2(%d)]\n"
                "  only the %d nabla Y2 variable%s can be excluded: "
                "beta = [I_r ; beta_2] normalises on Y1\n",
                r, s, (s == 1) ? "" : "s");
            for (i = 1; i <= s; i++) {
                k = 0;
                for (j = 1; j <= r; j++)
                    if (ix_B2[i][j] > 0) idx[++k] = ix_B2[i][j];
                if (k < 1) continue;
                nm = series_names ? series_names[i] : "y";
                snprintf(title, sizeof title, "%s:", nm);
                snprintf(h0, sizeof h0, "H0: row %d of beta_2 = 0", i);
                snprintf(rej_s, sizeof rej_s,
                         "REJECT H0 -> %s belongs in the long-run relation.", nm);
                snprintf(acc_s, sizeof acc_s,
                         "Cannot reject H0 -> %s can be dropped from it.", nm);
                emit_wald(x, cov, idx, k, title, h0, rej_s, acc_s);
            }
            if (s > 1) {
                k = 0;
                for (i = 1; i <= s; i++)
                    for (j = 1; j <= r; j++)
                        if (ix_B2[i][j] > 0) idx[++k] = ix_B2[i][j];
                emit_wald(x, cov, idx, k, "All of B2 jointly:",
                          "H0: B2 = 0 (the relation involves only the Y1 block)",
                          "REJECT H0 -> the nabla Y2 block belongs in it.",
                          "Cannot reject H0 -> the relation is inside Y1 alone.");
            }
        } else if (global_fixb2) {
            fprintf(outputv,
                "\nExclusion from the cointegrating relations\n"
                "  beta_2 is held fixed (-fixb2): imposed, not estimated, so there "
                "is nothing to test\n");
        }
    } else {
        fprintf(outputv,
            "\n  r = 0: no error-correction term, no Lambda and no B; the "
            "short-run block only\n");
    }

    /* ---- 4. The short-run dynamics -------------------------------------- */
    if (nf > 0 || q > 0) {
        /*  WHAT F AND Theta ACT ON, said correctly.  This block used to warn
         *  that they act on Ybar = (nabla Y2', W')'; they do NOT.  The
         *  parameter vector printed here is the VEC form's, where
         *  F(L) nabla Y_t = -Lambda(...) + Theta(L) A_t: F multiplies nabla Y
         *  and Theta multiplies the innovations.  It is the -warma
         *  parameterisation that lives in Ybar coordinates, and that route
         *  prints its own block and never reaches here.  Corrected 2026-08-24;
         *  the old line told the reader to distrust a Granger reading that is
         *  in fact exactly what these tests are.                             */
        fprintf(outputv,
            "\nShort-run dynamics   [Gamma on nabla Y, Theta on the "
            "innovations]\n"
            "  ! statements about nabla Y, not about the levels: a variable "
            "can drive\n    another's DIFFERENCES and still be tied to it only "
            "through ec\n");

        if (nf > 0) {                              /* last lag of F     */
            k = 0;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    if (ix_F[(nf - 1) * M + i][j] > 0)
                        idx[++k] = ix_F[(nf - 1) * M + i][j];
            snprintf(title, sizeof title,
                     "Joint significance of the last short-run lag, "
                     "Gamma(%d):", nf);
            emit_wald(x, cov, idx, k, title, "H0: Gamma(last) = 0",
                      "REJECT H0 -> the last AR lag is significant.",
                      "Cannot reject H0 -> the last AR lag is not significant.");
        }
        if (q > 0) {                               /* last lag of Theta */
            k = 0;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    if (ix_Th[(q - 1) * M + i][j] > 0)
                        idx[++k] = ix_Th[(q - 1) * M + i][j];
            snprintf(title, sizeof title,
                     "Joint significance of the last MA lag, Theta(%d):", q);
            emit_wald(x, cov, idx, k, title, "H0: Theta(last) = 0",
                      "REJECT H0 -> the last MA lag is significant.",
                      "Cannot reject H0 -> the last MA lag is not significant.");
        }

        k = 0;                                     /* all the cross ones        */
        for (kk = 1; kk <= nf; kk++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    if (i != j && ix_F[(kk - 1) * M + i][j] > 0)
                        idx[++k] = ix_F[(kk - 1) * M + i][j];
        for (kk = 1; kk <= q; kk++)
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++)
                    if (i != j && ix_Th[(kk - 1) * M + i][j] > 0)
                        idx[++k] = ix_Th[(kk - 1) * M + i][j];
        if (k > 0)
            emit_wald(x, cov, idx, k, "All cross effects jointly:",
                      "H0: every off-diagonal coefficient of Gamma and "
                      "Theta = 0",
                      "REJECT H0 -> the cross structure earns its parameters.",
                      "Cannot reject H0 -> a diagonal short run would do "
                      "(-diagar / -diagma).");
        else
            fprintf(outputv, "\nNo free cross effects to test (the short run "
                             "is already diagonal or structured).\n");

        for (i = 1; i <= M; i++) {                 /* the two directions  */
            /*  BUG-23: i names a series (the .inp's order); F and Theta are
             *  indexed in the internal order, so the row and the column that
             *  belong to that series are li = inp2lam(i).  Indexing them by i,
             *  as this did, tested one series and named another: on a DGP
             *  where x is driven by y it concluded that x drives y.         */
            int li = inp2lam(i);
            nm = series_names ? series_names[i] : "y";
            k = 0;
            for (kk = 1; kk <= nf; kk++)
                for (j = 1; j <= M; j++)
                    if (j != li && ix_F[(kk - 1) * M + li][j] > 0)
                        idx[++k] = ix_F[(kk - 1) * M + li][j];
            for (kk = 1; kk <= q; kk++)
                for (j = 1; j <= M; j++)
                    if (j != li && ix_Th[(kk - 1) * M + li][j] > 0)
                        idx[++k] = ix_Th[(kk - 1) * M + li][j];
            if (k > 0) {
                snprintf(title, sizeof title,
                         "D.%s: what the others do to it:", nm);
                snprintf(h0, sizeof h0,
                         "H0: no other variable enters the equation of D.%s", nm);
                snprintf(rej_s, sizeof rej_s,
                         "REJECT H0 -> D.%s is driven by the others.", nm);
                snprintf(acc_s, sizeof acc_s,
                         "Cannot reject H0 -> D.%s is not driven by the others.", nm);
                emit_wald(x, cov, idx, k, title, h0, rej_s, acc_s);
            }
            k = 0;
            for (kk = 1; kk <= nf; kk++)
                for (j = 1; j <= M; j++)
                    if (j != li && ix_F[(kk - 1) * M + j][li] > 0)
                        idx[++k] = ix_F[(kk - 1) * M + j][li];
            for (kk = 1; kk <= q; kk++)
                for (j = 1; j <= M; j++)
                    if (j != li && ix_Th[(kk - 1) * M + j][li] > 0)
                        idx[++k] = ix_Th[(kk - 1) * M + j][li];
            if (k > 0) {
                snprintf(title, sizeof title,
                         "D.%s: what it does to the others:", nm);
                snprintf(h0, sizeof h0,
                         "H0: D.%s enters no other equation", nm);
                snprintf(rej_s, sizeof rej_s,
                         "REJECT H0 -> D.%s drives the others.", nm);
                snprintf(acc_s, sizeof acc_s,
                         "Cannot reject H0 -> D.%s does not drive the others.", nm);
                emit_wald(x, cov, idx, k, title, h0, rej_s, acc_s);
            }
        }
    }

    if (!global_fdhess)
        fprintf(outputv, "\n  ! p-values from the BFGS-accumulated covariance; "
                         "use -fdhess before quoting them\n");

    free_ivector(idx, 1, (M * M * (nf + q) + M * r + s * r) + 1);
}

/*  forecast_vec — H steps from the end of the sample, in levels.
 *  Returns 0 if it worked.  The columns are the .inp's: Y2 (1..s), Y1
 *  (s+1..M).                                                                 */
static int forecast_vec(struct Tvarma *v, real **B2, int H, real conf)
{
    int M = v->m, r = global_r, s = M - r, n = v->n, p = v->p, q = v->q;
    int h, i, j, k, l;
    real ***Psi, **Csum, **G, **Sig, **Var, **Yb, **lev, **SE;
    real z;
    /*  The RAW index of the origin: with the series in levels, row t of
     *  datamat is row t+1 of the .inp (one observation is consumed by
     *  differencing), and with -differenced it is the same row.          */
    int raw_origin = global_levels ? n + 1 : n;

    if (H < 1 || r < 0 || s < 1) return 1;
    if (!Y2_levels) return 1;

    Psi  = tensor(0, H, 1, M, 1, M);
    Csum = matrix(1, M, 1, M);
    G    = matrix(1, M, 1, M);
    Sig  = matrix(1, M, 1, M);
    Var  = matrix(1, M, 1, M);
    Yb   = matrix(1, H, 1, M);          /* Ybar previsto                     */
    lev  = matrix(1, H, 1, M);          /* niveles: [Y2 ; Y1]                */
    SE   = matrix(1, H, 1, M);          /* the standard error of each level   */

    compute_psi_weights(M, p, q, v->phi, v->theta, H, Psi);
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) Sig[i][j] = v->sigma2 * v->qq[i][j];

    /*  [1] The mean and [2] the levels: computed by forecast_core(), which is
     *      the same function the rolling-origin evaluation uses.  One single
     *      copy of the recursion, which is what this file already learned the
     *      hard way with compute_psi_weights.                                */
    forecast_core(v, B2, n, H, Yb, lev);

    banner("Forecast");

    /*  [1b] THE CERTIFICATE.  The recursion of [1] is applied backwards,
     *  inside the sample: the one-step prediction of Ybar_t with the
     *  information up to t-1 has to be Ybar_t - a_t, with a_t the residual elf
     *  returned.  It is not circular -- the residuals are computed by the
     *  engine through AS 311, not by this function -- and it ties down at once
     *  the recursion, the mean convention and the indexing.  It is measured
     *  over the second half of the sample, where the exact start no longer
     *  weighs.                                                                */
    {
        real worst = 0.0;
        int t0 = n / 2 + 1, t;
        if (t0 < p + q + 1) t0 = p + q + 1;
        for (t = t0; t <= n; t++)
            for (i = 1; i <= M; i++) {
                real pred = v->mu[i];
                for (k = 1; k <= p; k++)
                    for (l = 1; l <= M; l++)
                        pred += v->phi[k][i][l] * (v->w[t-k][l] - v->mu[l]);
                for (k = 1; k <= q; k++)
                    for (l = 1; l <= M; l++)
                        pred -= v->theta[k][i][l] * v->a[t-k][l];
                {
                    real d = fabs(v->w[t][i] - v->a[t][i] - pred);
                    if (d > worst) worst = d;
                }
            }
        fprintf(outputv,
            "\n  one-step self-check: max |Ybar_t - a_t - pred(t|t-1)| = %.3e "
            "over %d obs\n"
            "  (the residue is the xi truncation, of order xitol = %.0e)\n",
            worst, n - t0 + 1, fabs(v->xitol));
    }

    /*  [3] The bands, through the map of [4] above and by no other route.    */
    z = (conf >= 0.99) ? 2.575829 : (conf >= 0.95) ? 1.959964 : 1.644854;
    fprintf(outputv, "\n%d step%s ahead, in levels.  Columns are the .inp's: "
                     "Y2 block (1..%d), then Y1.\n"
                     "s.e. are THEORETICAL; a %.0f%% band is +/- %.4f s.e.\n",
        H, (H == 1) ? "" : "s", s, 100.0 * conf, z);
    if (bt_on)
        fprintf(outputv, "Levels are in each series' OWN units: deterministic path added\n"
                         "back, refactor divided out, Box-Cox inverted (the median under\n"
                         "a log); s.e. by the delta method (BUG-36).\n");
    fprintf(outputv, "\n");
    fprintf(outputv, "   h");
    for (i = 1; i <= M; i++)
        fprintf(outputv, "  %14s %10s", series_names ? series_names[i] : "y", "s.e.");
    fprintf(outputv, "\n");

    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Var[i][j] = 0.0;
    for (i = 1; i <= M; i++) for (j = 1; j <= M; j++) Csum[i][j] = 0.0;

    for (h = 1; h <= H; h++) {
        int m = h - 1;
        /*  C_m = C_{m-1} + Psi_m, and Var(h) = Var(h-1) + G_m Sigma* G_m'.
         *  Both are accumulations of one term per horizon: the band at h
         *  contains every shock from n+1..n+h, each with its due weight.     */
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) Csum[i][j] += Psi[m][i][j];
        level_error_map(Csum, Psi[m], B2, M, r, G);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                real acc = 0.0;
                for (k = 1; k <= M; k++)
                    for (l = 1; l <= M; l++) acc += G[i][k] * Sig[k][l] * G[j][l];
                Var[i][j] += acc;
            }
        fprintf(outputv, "%4d", h);
        for (i = 1; i <= M; i++) {
            SE[h][i] = (Var[i][i] > 0.0) ? sqrt(Var[i][i]) : 0.0;
            fprintf(outputv, "  %14.6f %10.6f",
                    bt_level(i, raw_origin + h, lev[h][i]),
                    bt_jac(i, raw_origin + h, lev[h][i]) * SE[h][i]);
        }
        fprintf(outputv, "\n");
    }

    /*  P6.7 — <base>.forecast, THE SUITE'S FILE.  What is above is the
     *  estimation report; this is the product, and it carries what the .out
     *  never did: the DATE of each row and the band already built.  With no
     *  date, whoever reads the forecast has to rebuild the calendar from the
     *  .inp's header, and that is an error waiting to happen.  The format is
     *  the sibling's (drvarma v.04.1, drvarma.c:627).                        */
    {
        char fname[600];
        FILE *ff;
        snprintf(fname, sizeof fname, "%s.forecast", out_base);
        ff = fopen(fname, "w");
        if (!ff) fprintf(stderr, "WARNING: cannot write %s\n", fname);
        else {
            int per, sub;
            fprintf(ff, "DRVEC %s -- forecasts from a VEC(%d) model\n",
                    DRVEC_VERSION, r);
            fprintf(ff, "input=%s.inp p=%d q=%d r=%d case=%d freq=%d "
                        "horizon=%d bands=%.0f%%\n",
                    out_base, p, q, r, global_case, data_freq, H, 100.0 * conf);
            fprintf(ff, "Levels in each series' own units%s.  Columns 1..%d are "
                        "the Y2 block, %d..%d the Y1 block.\n",
                    bt_on ? " (deterministic path added back, refactor divided "
                            "out, Box-Cox inverted: the median under a log)" : "",
                    s, s + 1, M);
            fprintf(ff, "Low/High are +/- %.4f standard errors%s and are "
                        "THEORETICAL: they assume the specification is right.\n\n",
                    z, bt_on ? " in the model's units, transformed at both ends; "
                               "s.e. by the delta method" : "");
            for (i = 1; i <= M; i++) {
                fprintf(ff, "Series %d (%s):\n", i,
                        series_names ? series_names[i] : "y");
                fprintf(ff, "  %-9s %14s %14s %14s %12s\n",
                        "date", "Level", "Low", "High", "s.e.");
                for (h = 1; h <= H; h++) {
                    ObsToDate(data_start_year, data_start_sub, raw_origin + h,
                              data_freq, &per, &sub);
                    if (data_freq > 1)
                        fprintf(ff, "  %3d/%-5d", sub, per);
                    else
                        fprintf(ff, "  %-9d", per);
                    {
                        int tr = raw_origin + h;
                        fprintf(ff, " %14.6f %14.6f %14.6f %12.6f\n",
                                bt_level(i, tr, lev[h][i]),
                                bt_level(i, tr, lev[h][i] - z * SE[h][i]),
                                bt_level(i, tr, lev[h][i] + z * SE[h][i]),
                                bt_jac(i, tr, lev[h][i]) * SE[h][i]);
                    }
                }
                fprintf(ff, "\n");
            }
            fclose(ff);
            if (!quiet_mode) printf("Forecasts written to %s\n", fname);
        }
    }
    fprintf(outputv, "\n  ! the Y1 s.e. inherits the CUMULATED error of the "
                     "nabla Y2 block,\n    because the two are tied by the "
                     "cointegrating relation\n");

    if (!quiet_mode)
        printf("Forecast: %d steps written to the .out\n", H);

    free_matrix(SE, 1, H, 1, M);
    free_matrix(lev, 1, H, 1, M);
    free_matrix(Yb, 1, H, 1, M);
    free_matrix(Var, 1, M, 1, M);
    free_matrix(Sig, 1, M, 1, M);
    free_matrix(G, 1, M, 1, M);
    free_matrix(Csum, 1, M, 1, M);
    free_tensor(Psi, 0, H, 1, M, 1, M);
    return 0;
}

/*****************************************************************************/
/*  P8 — read_inp_input: the .inp route's reader, out of main().             */
/*                                                                           */
/*  It was 53 lines inline in a function of 2320, next to the .pre route's    */
/*  one-line call to read_pre_inputs.  The two do the same job -- fill        */
/*  rawmat, nser, nobs_raw, series_names and the calendar -- and one of them  */
/*  was a function and the other was not, so the symmetry that makes the two  */
/*  routes interchangeable was invisible in the source.  Now both are         */
/*  functions and main() picks one.                                          */
/*                                                                           */
/*  Reader of the same shape as drvarma v.04.1's.  Exits on a malformed file: */
/*  there is nothing to recover to.                                          */
/*****************************************************************************/
static void read_inp_input(const char *path)
{
    FILE *inputv;

    if (NULL == (inputv = fopen(path, "r"))) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    {
        /* InpReader: same as drvarma v.04.1 */
        /* Minimal reader: read nser, nobs_raw, start, names, lambda, d, D, data */
        char line[512];
        /* skip comments */
        do { if (!fgets(line, sizeof line, inputv)) { fprintf(stderr,"ERROR: unexpected EOF\n"); exit(1); } }
        while (line[0] == '*');
        data_freq = atoi(line);
        /* next non-comment line: nser nobs start_sub start_year */
        do { if (!fgets(line, sizeof line, inputv)) { fprintf(stderr,"ERROR: unexpected EOF\n"); exit(1); } }
        while (line[0] == '*');
        sscanf(line, "%d %d %d %d", &nser, &nobs_raw, &data_start_sub, &data_start_year);
        nobs = nobs_raw;

        if (global_r >= nser) {
            fprintf(stderr, "ERROR: r=%d must be < M=%d\n", global_r, nser);
            exit(1);
        }

        series_names = (char **) malloc((nser + 1) * sizeof(char *));
        do { if (!fgets(line, sizeof line, inputv)) { fprintf(stderr,"ERROR: missing series names\n"); exit(1); } }
        while (line[0] == '*');
        {
            char *tok = strtok(line, " \t\n");
            for (int j = 1; j <= nser; j++) {
                if (!tok) { fprintf(stderr,"ERROR: need %d series names\n", nser); exit(1); }
                series_names[j] = strdup(tok);
                tok = strtok(NULL, " \t\n");
            }
        }
        /* Box-Cox lambda, d, D */
        do { if (!fgets(line, sizeof line, inputv)) { fprintf(stderr,"ERROR: missing transform line\n"); exit(1); } }
        while (line[0] == '*');
        sscanf(line, "%lf %d %d", &trans_lambda, &trans_d, &trans_D);

        /* Read data: nobs_raw rows, nser cols, kept untouched in rawmat */
        rawmat = matrix(1, nobs_raw, 1, nser);
        for (int t = 1; t <= nobs_raw; t++) {
            do { if (!fgets(line, sizeof line, inputv)) { fprintf(stderr,"ERROR: data too short at obs %d\n", t); exit(1); } }
            while (line[0] == '*');
            char *tok = strtok(line, " \t\n");
            for (int j = 1; j <= nser; j++) {
                if (!tok) { fprintf(stderr,"ERROR: missing value obs %d col %d\n", t, j); exit(1); }
                rawmat[t][j] = atof(tok);
                tok = strtok(NULL, " \t\n");
            }
        }
        fclose(inputv);
    }
}

/*****************************************************************************/
/*  P9 — THE .pre INPUT ROUTE: one univariate model per series, as in drtran  */
/*                                                                           */
/*  WHY drvec AND NOT drvarma.  drvarma reads a single .inp with every series */
/*  in it, and that is right for drvarma: it does not share fue's ladder --   */
/*  it is the engine the ladder is built on.  drtran does share it, and its   */
/*  interface says so: `drtran output.pre input1.pre ...`, one already        */
/*  identified univariate model per series.  drvec is on drtran's side of     */
/*  that line: its whole design (docs/PLAN_BETA.md F2) is that the univariate */
/*  work is done in fue and arrives here already done.  Until now it arrived  */
/*  by hand -- export the series to an .inp, then -interv to subtract the     */
/*  deterministic terms, then -seed for the moving average -- which is three  */
/*  steps a user has to remember and one file format to fill in by hand.      */
/*                                                                           */
/*  WHAT IS TAKEN FROM EACH .pre                                              */
/*                                                                           */
/*    the series          Ts.data, the raw z                                  */
/*    the transformation  w = refactor * BoxCox(z), which is the format's own */
/*                        contract (BRIDGE_DESIGN.md) and exactly what        */
/*                        drtran.c:885 does.  drvec works in w for the same   */
/*                        reason drtran does: the deterministic coefficients  */
/*                        the file carries are in the units of w              */
/*    the deterministics  subtracted, through build_det_component, with the   */
/*                        DATES of this sample -- a deterministic term is a   */
/*                        function of time, so aligning by index instead of   */
/*                        by date puts the intervention in the wrong year     */
/*    the calendar        frequency and start; the sample used is the         */
/*                        intersection of the files' calendars                */
/*                                                                           */
/*  WHAT IS NOT.  The moving average is NOT seeded from the models, even      */
/*  though they are right there and it would be free: it is measured to make  */
/*  the fit WORSE with r >= 1 (docs/PLAN_BETA.md F2.7 and F2.8), and a route  */
/*  that silently does a thing measured to be harmful is worse than one that  */
/*  makes you ask.  -seed still asks.                                         */
/*                                                                           */
/*  THE COLUMN ORDER IS THE .inp's, and it has to be: the files on the        */
/*  command line are the columns, so the first M-r are the nabla Y2 block and */
/*  the last r are the Y1 block, the one B = [I_r ; B2] normalises on.  A     */
/*  second order would be a silent trap of exactly the kind this program      */
/*  already warns about.  Which file went into which block is printed.        */
/*****************************************************************************/

/*  boxcox_w — the format's transformation: w = refactor * BoxCox(z).  Same
 *  branch and the same 1e-8 threshold as drtran.c:885, deliberately.        */
static real boxcox_w(real z, real lam, real refac)
{
    if (fabs(lam) < 1.0e-8) return log(z) * refac;
    if (fabs(lam - 1.0) < 1.0e-12) return z * refac;
    return ((pow(z, lam) - 1.0) / lam) * refac;
}

/*  abs_period — a single index on a common calendar, so that files that start
 *  on different dates can be lined up by DATE and not by position.  With
 *  annual data (freq = 1) the subperiod is meaningless and only the year
 *  counts, which is the same convention drtran's obs_to_date uses.          */
static long abs_period(int year, int sub, int freq)
{
    if (freq <= 1) return (long) year;
    return (long) year * freq + (sub - 1);
}

/*  read_pre_inputs — fill rawmat, series_names and the calendar from the M
 *  .pre files.  Returns 0 on success.  Everything it sets is what the .inp
 *  reader sets, so the rest of the program cannot tell the two apart.       */
static int read_pre_inputs(char **files, int nfiles)
{
    struct Tusmodel *Tm;
    struct Tseries  *Ts;
    real ***DM;
    real **det;
    long *start;
    long first = 0, last = 0;
    int i, t, ok = 1, freq0 = 0;
    int syear = 0, ssub = 1;

    if (nfiles < 2) {
        fprintf(stderr, "ERROR: a VEC model needs at least two series, got %d\n",
                nfiles);
        return 1;
    }

    Tm  = (struct Tusmodel *) calloc((size_t) nfiles + 1, sizeof *Tm);
    Ts  = (struct Tseries  *) calloc((size_t) nfiles + 1, sizeof *Ts);
    DM  = (real ***) calloc((size_t) nfiles + 1, sizeof *DM);
    det = (real **)  calloc((size_t) nfiles + 1, sizeof *det);
    start = (long *) calloc((size_t) nfiles + 1, sizeof *start);
    if (!Tm || !Ts || !DM || !det || !start) {
        fprintf(stderr, "ERROR: out of memory reading the .pre files\n");
        return 1;
    }

    for (i = 1; i <= nfiles; i++) {
        if (read_fue_pre(files[i], &Tm[i], &Ts[i], &DM[i]) != 0) {
            fprintf(stderr, "ERROR: cannot read %s\n", files[i]);
            ok = 0;
            break;
        }
        if (Ts[i].nobs < 4) {
            fprintf(stderr, "ERROR: %s carries %d observations\n",
                    files[i], Ts[i].nobs);
            ok = 0; break;
        }
        if (freq0 == 0) freq0 = Ts[i].freq;
        else if (Ts[i].freq != freq0) {
            fprintf(stderr,
                "ERROR: %s has frequency %d and %s has %d.  A VEC model is one\n"
                "       system on one calendar; the series cannot be mixed.\n",
                files[i], Ts[i].freq, files[1], freq0);
            ok = 0; break;
        }
        start[i] = abs_period(Ts[i].begyear, Ts[i].begtime, Ts[i].freq);
        if (i == 1 || start[i] > first) first = start[i];
        if (i == 1 || start[i] + Ts[i].nobs - 1 < last)
            last = start[i] + Ts[i].nobs - 1;
    }

    if (ok && last - first + 1 < 4) {
        fprintf(stderr,
            "ERROR: the .pre files overlap in %ld observation(s); at least 4 are\n"
            "       needed.  They are lined up by DATE, not by position.\n",
            last - first + 1);
        ok = 0;
    }

    if (ok) {
        data_freq       = freq0;
        nser            = nfiles;
        nobs_raw        = (int) (last - first + 1);
        nobs            = nobs_raw;
        if (freq0 <= 1) { syear = (int) first;            ssub = 1; }
        else            { syear = (int) (first / freq0);  ssub = (int) (first % freq0) + 1; }
        data_start_year = syear;
        data_start_sub  = ssub;

        series_names = (char **) malloc(((size_t) nser + 1) * sizeof *series_names);
        rawmat = matrix(1, nobs_raw, 1, nser);
        bt_alloc(nser, nobs_raw + (global_fcast > 0 ? global_fcast : 0));

        printf("\nSeries, in the .inp's column order (the first %d are the "
               "nabla Y2 block):\n", nser - global_r);
        printf("  %-3s %-22s %-12s %8s %6s %3s %3s %s\n",
               "#", "file", "name", "refactor", "lambda", "d", "D", "obs");
        for (i = 1; i <= nser; i++) {
            int off = (int) (first - start[i]);       /* rows to skip at the head */
            real refac = (Ts[i].refactor != 0.0) ? Ts[i].refactor : 1.0;
            real lam   = Tm[i].boxlam;

            series_names[i] = strdup(Ts[i].name ? Ts[i].name : files[i]);

            /*  The deterministic component over THIS file's own sample, which
             *  is where its dates are; the head is skipped afterwards.  It
             *  comes back in the units of w, which is the units this route
             *  works in, so unlike -interv there is nothing to divide by --
             *  the mismatch BUG-15 was about cannot arise here.             */
            det[i] = vector(1, Ts[i].nobs);
            for (t = 1; t <= Ts[i].nobs; t++) det[i][t] = 0.0;
            if (Tm[i].NdetVar > 0)
                build_det_component(&Tm[i], &Ts[i], Ts[i].nobs, det[i]);

            /*  What the forecast needs to come back to z (BUG-36): this
             *  series' lambda and refactor, and det on the COMMON sample and
             *  over the horizon -- a step or a trend goes on after the end.  */
            bt_lam[i] = lam; bt_refac[i] = refac;
            if (Tm[i].NdetVar > 0) {
                int Lx = off + bt_T;
                real *dx = vector(1, Lx);
                for (t = 1; t <= Lx; t++) dx[t] = 0.0;
                build_det_component(&Tm[i], &Ts[i], Lx, dx);
                for (t = 1; t <= bt_T; t++) bt_det[i][t] = dx[off + t];
                free_vector(dx, 1, Lx);
            }

            for (t = 1; t <= nobs_raw; t++) {
                real z = Ts[i].data[off + t];
                if (fabs(lam - 1.0) > 1.0e-12 && z <= 0.0) {
                    fprintf(stderr,
                        "ERROR: %s asks for a Box-Cox with lambda = %g and "
                        "observation %d is %g\n", files[i], lam, off + t, z);
                    ok = 0; break;
                }
                rawmat[t][i] = boxcox_w(z, lam, refac) - det[i][off + t];
            }
            if (!ok) break;

            printf("  %-3d %-22s %-12s %8.4g %6.3g %3d %3d %d%s\n",
                   i, files[i], series_names[i], refac, lam,
                   Tm[i].nrdiff, Tm[i].nadiff, Ts[i].nobs,
                   (i <= nser - global_r) ? "" : "   <- Y1");

            /*  A univariate model that does NOT difference is a model that says
             *  its series is stationary; putting it in a VEC says the opposite.
             *  It is a warning and not an error because the rank test exists
             *  precisely to settle the question.                            */
            /*  -case pre: the drift is free iff the univariate model frees
             *  its mean.  A mean HELD at a non-zero value cannot be carried
             *  (the drift block holds at 0 or frees), so it goes free and
             *  the reader is told.                                         */
            if (case_pre && i <= ND_MAX) {
                nodrift[i] = (Tm[i].Imu == 0);
                if (nodrift[i]) nodrift_on = 1;
                if (Tm[i].Imu == 0 && fabs(Tm[i].mu) > 1.0e-12) {
                    nodrift[i] = 0;
                    fprintf(stderr,
                        "WARNING: %s holds its mean at %g (not 0); -case pre\n"
                        "         can hold a drift only at 0, so it goes free.\n",
                        files[i], Tm[i].mu);
                }
                printf("      drift (-case pre): %s\n",
                       nodrift[i] ? "none, held at 0 (the .pre has no mean)"
                                  : "free (the .pre estimates its mean)");
            }

            if (Tm[i].nrdiff == 0 && Tm[i].nadiff == 0)
                fprintf(stderr,
                    "WARNING: %s carries no differencing, so its own univariate\n"
                    "         model says the series is stationary.  A VEC model\n"
                    "         assumes I(1) series.\n", files[i]);
        }
    }

    if (ok) {
        /*  The refactors, and what they do to B2.  W = Y1 + B2'Y2 is formed in
         *  w units, so a coefficient b_i is the z-unit one times
         *  refactor_i / refactor_{Y1}.  When every file carries the same
         *  factor -- which is the suite's norm -- they cancel and B2 reads
         *  directly.  When they do not, saying so is the difference between a
         *  number and a number in unknown units.                            */
        int same = 1;
        real r1 = (Ts[1].refactor != 0.0) ? Ts[1].refactor : 1.0;
        for (i = 2; i <= nser; i++) {
            real ri = (Ts[i].refactor != 0.0) ? Ts[i].refactor : 1.0;
            if (fabs(ri - r1) > 1.0e-9 * r1) same = 0;
        }
        if (!same)
            fprintf(stderr,
                "WARNING: the .pre files do not share one refactor.  The system is\n"
                "         built in w = refactor*BoxCox(z) units, as in drtran, so\n"
                "         each entry of B2 is its z-unit value times\n"
                "         refactor_i / refactor_(Y1 block).  Compare fits on Pi,\n"
                "         which the normalisation does not touch.\n");
        printf("Common sample: %d observations, %d/%d onwards, frequency %d\n",
               nobs_raw, ssub, syear, data_freq);
    }

    for (i = 1; i <= nfiles; i++) {
        if (det[i]) free_vector(det[i], 1, Ts[i].nobs);
        if (Ts[i].data) free_fue_pre(&Tm[i], &Ts[i], DM[i]);
    }
    free(start); free(det); free(DM); free(Ts); free(Tm);
    return ok ? 0 : 1;
}


/*  xl_backcast -- drtran's backcast_arma (drtran.c:447), ported: the ARMA in
 *  elf's sign convention is symmetric under time reversal, so backcasting is
 *  forecasting the reversed series with the same phi and theta (TASTE's
 *  BackForeCast).  out[k] is the value k periods before w[1].              */
static void xl_backcast(const real *w, int n, const real *ph, int p,
                        const real *th, int q, real mu, int L, real *out)
{
    real *u, *e;
    int t, i, j, k;
    if (L <= 0 || n <= 0) return;
    u = vector(1, n);
    e = vector(1, n);
    for (t = 1; t <= n; t++) u[t] = w[n - t + 1] - mu;
    for (t = 1; t <= n; t++) {
        real acc = u[t];
        for (i = 1; i <= p && i < t; i++) acc -= ph[i] * u[t - i];
        for (j = 1; j <= q && j < t; j++) acc += th[j] * e[t - j];
        e[t] = acc;
    }
    for (k = 1; k <= L; k++) {
        real acc = 0.0;
        for (i = 1; i <= p; i++) {
            int ix = k - i;
            acc += ph[i] * (ix >= 1 ? out[ix] - mu : u[n + ix]);
        }
        for (j = 1; j <= q; j++) {
            int ix = n + k - j;
            if (ix <= n) acc -= th[j] * e[ix];
        }
        out[k] = acc + mu;
    }
    free_vector(u, 1, n);
    free_vector(e, 1, n);
}

/*  xl_load -- the exogenous inputs and the network of links.  Called after
 *  read_pre_inputs, so the system's calendar and names exist.  Returns 0.  */
static int xl_load(void)
{
    struct Tusmodel Tm; struct Tseries Ts; real **DM;
    real *wv[XL_MAXIN + 1], *phv[XL_MAXIN + 1], *thv[XL_MAXIN + 1];
    int   nv[XL_MAXIN + 1], pv[XL_MAXIN + 1], qv[XL_MAXIN + 1];
    long  stv[XL_MAXIN + 1];
    real  muv[XL_MAXIN + 1];
    long  first = abs_period(data_start_year, data_start_sub, data_freq);
    int i, t, k, maxbs = 0, anyr = 0;
    FILE *f;
    char line[512];

    for (i = 1; i <= xl_nin; i++) {
        real refac, lam, *det;
        if (read_fue_pre(xl_infile[i], &Tm, &Ts, &DM) != 0) {
            fprintf(stderr, "ERROR: cannot read the input %s\n", xl_infile[i]);
            return 1;
        }
        if (Ts.freq != data_freq) {
            fprintf(stderr, "ERROR: the input %s has frequency %d and the system %d\n",
                    xl_infile[i], Ts.freq, data_freq);
            return 1;
        }
        if (Tm.nadiff != 0 || Tm.nrdiff > 1 || Tm.ornsop != Tm.nrdiff) {
            fprintf(stderr,
                "ERROR: the input %s is not at most I(1) at frequency zero with no\n"
                "       seasonal unit roots (d=%d, D=%d).  Its level would not be\n"
                "       a transfer input W could stay stationary with.\n",
                xl_infile[i], Tm.nrdiff, Tm.nadiff);
            return 1;
        }
        xl_inname[i] = strdup(Ts.name ? Ts.name : xl_infile[i]);
        xl_d[i] = Tm.nrdiff;
        refac = (Ts.refactor != 0.0) ? Ts.refactor : 1.0;
        lam = Tm.boxlam;
        nv[i] = Ts.nobs;
        stv[i] = abs_period(Ts.begyear, Ts.begtime, Ts.freq);
        det = vector(1, Ts.nobs);
        for (t = 1; t <= Ts.nobs; t++) det[t] = 0.0;
        if (Tm.NdetVar > 0) build_det_component(&Tm, &Ts, Ts.nobs, det);
        wv[i] = vector(1, Ts.nobs);
        for (t = 1; t <= Ts.nobs; t++) {
            real z = Ts.data[t];
            if (fabs(lam - 1.0) > 1.0e-12 && z <= 0.0) {
                fprintf(stderr, "ERROR: %s: Box-Cox with lambda %g at a value %g\n",
                        xl_infile[i], lam, z);
                return 1;
            }
            wv[i][t] = boxcox_w(z, lam, refac) - det[t];
        }
        free_vector(det, 1, Ts.nobs);
        pv[i] = pre_ar_order(&Tm);
        qv[i] = pre_ma_order(&Tm);
        phv[i] = vector(1, pv[i] > 0 ? pv[i] : 1);
        thv[i] = vector(1, qv[i] > 0 ? qv[i] : 1);
        for (k = 1; k <= pv[i]; k++) phv[i][k] = 0.0;
        for (k = 1; k <= qv[i]; k++) thv[i][k] = 0.0;
        if (pv[i] > 0) expand_ar_factors(&Tm, phv[i], pv[i]);
        if (qv[i] > 0) expand_ma_factors(&Tm, thv[i], qv[i]);
        muv[i] = Tm.mu;           /* the value, whatever the flag (BUG-38) */
        if (stv[i] + nv[i] - 1 < first + nobs_raw - 1) {
            fprintf(stderr,
                "ERROR: the input %s ends before the system's sample does.  An\n"
                "       exogenous input must be observed over the whole sample.\n",
                xl_infile[i]);
            return 1;
        }
        free_fue_pre(&Tm, &Ts, DM);
    }

    /*  The network: lines `OUTPUT <- INPUT b r s', drtran's syntax; `#'
     *  starts a comment.  OUTPUT is a series of the system, INPUT an -xpre. */
    f = fopen(xl_netfile, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", xl_netfile); return 1; }
    while (fgets(line, sizeof line, f)) {
        char o[128], arrow[8], in[128];
        int b, rr, ss, n2, a;
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        n2 = sscanf(line, "%127s %7s %127s %d %d %d", o, arrow, in, &b, &rr, &ss);
        if (n2 <= 0) continue;
        if (n2 != 6 || strcmp(arrow, "<-") != 0) {
            fprintf(stderr, "ERROR: %s: a link is `OUTPUT <- INPUT b r s', got: %s",
                    xl_netfile, line);
            fclose(f); return 1;
        }
        if (xl_nlnk >= XL_MAXLNK) {
            fprintf(stderr, "ERROR: more than %d links\n", XL_MAXLNK);
            fclose(f); return 1;
        }
        if (b < 0 || rr < 0 || ss < 0 || rr > 2 || ss > XL_MAXORD) {
            fprintf(stderr, "ERROR: %s: b, r, s must be >= 0, r <= 2 and s <= %d\n",
                    xl_netfile, XL_MAXORD);
            fclose(f); return 1;
        }
        xl_nlnk++;
        xl_lnk[xl_nlnk].out = xl_lnk[xl_nlnk].in = 0;
        for (a = 1; a <= nser; a++)
            if (strcmp(series_names[a], o) == 0) xl_lnk[xl_nlnk].out = a;
        for (a = 1; a <= xl_nin; a++)
            if (strcmp(xl_inname[a], in) == 0) xl_lnk[xl_nlnk].in = a;
        if (!xl_lnk[xl_nlnk].out || !xl_lnk[xl_nlnk].in) {
            fprintf(stderr,
                "ERROR: %s: `%s <- %s': the output must be a series of the system\n"
                "       and the input the name of an -xpre file.\n",
                xl_netfile, o, in);
            fclose(f); return 1;
        }
        xl_lnk[xl_nlnk].b = b; xl_lnk[xl_nlnk].r = rr; xl_lnk[xl_nlnk].s = ss;
        if (b + ss > maxbs) maxbs = b + ss;
        if (rr > 0) anyr = 1;
    }
    fclose(f);
    if (xl_nlnk == 0) {
        fprintf(stderr, "ERROR: %s declares no link\n", xl_netfile);
        return 1;
    }

    /*  As many presample values as nu can reach: b + s with r = 0, the whole
     *  sample with a rational (infinite-tail) filter, which is never
     *  truncated (drtran's rule, not fue's 40 lags).                       */
    xl_npre = anyr ? nobs_raw + maxbs : maxbs;
    xl_len  = xl_npre + nobs_raw;
    xl_X = matrix(1, xl_nin, 1, xl_len);
    xl_T = matrix(1, nobs_raw, 1, nser);
    for (i = 1; i <= xl_nin; i++) {
        int k0 = 0, L;
        for (k = 1; k <= xl_len; k++) {
            long per = first + (k - xl_npre - 1);
            long ix = per - stv[i] + 1;
            if (ix >= 1 && ix <= nv[i]) {
                xl_X[i][k] = wv[i][ix];
                if (!k0) k0 = k;
            } else xl_X[i][k] = 0.0;
        }
        L = k0 - 1;                               /* still missing        */
        xl_nreal[i] = xl_npre - L;
        xl_nbc[i] = L;
        if (L > 0) {
            real *out = vector(1, L);
            if (xl_d[i] == 0) {
                xl_backcast(wv[i], nv[i], phv[i], pv[i], thv[i], qv[i],
                            muv[i], L, out);
                for (k = 1; k <= L; k++) xl_X[i][k0 - k] = out[k];
            } else {
                real *u = vector(1, nv[i] - 1);
                for (t = 1; t < nv[i]; t++) u[t] = wv[i][t + 1] - wv[i][t];
                xl_backcast(u, nv[i] - 1, phv[i], pv[i], thv[i], qv[i],
                            muv[i], L, out);
                for (k = 1; k <= L; k++)
                    xl_X[i][k0 - k] = xl_X[i][k0 - k + 1] - out[k];
                free_vector(u, 1, nv[i] - 1);
            }
            free_vector(out, 1, L);
        }
        free_vector(wv[i], 1, nv[i]);
        free_vector(phv[i], 1, pv[i] > 0 ? pv[i] : 1);
        free_vector(thv[i], 1, qv[i] > 0 ? qv[i] : 1);
    }

    printf("Exogenous transfer inputs (-xlink %s):\n", xl_netfile);
    for (i = 1; i <= xl_nin; i++)
        printf("  %-12s %s, d = %d; presample %d: %d observed, %d backcast\n",
               xl_inname[i], xl_infile[i], xl_d[i], xl_npre, xl_nreal[i],
               xl_nbc[i]);
    for (k = 1; k <= xl_nlnk; k++)
        printf("  link %d: %s <- %s  (b, r, s) = (%d, %d, %d)\n", k,
               series_names[xl_lnk[k].out], xl_inname[xl_lnk[k].in],
               xl_lnk[k].b, xl_lnk[k].r, xl_lnk[k].s);
    return 0;
}

/*  xl_eval_ll -- the exact logL at x, evaluated (not optimised): the bridge
 *  check, logL(VEC optimum, omega = 0) == logL(VEC), which is the ladder's
 *  crossing identity for this rung.                                        */
static real xl_eval_ll(real *x, real xitol, int *ok)
{
    const real LOG2PI = 1.837877066;
    struct Tvarma v;
    int ifs = 0;
    real pi1, pi2, pi3, ll = 0.0;
    *ok = 0;
    v.xitol = xitol;
    vec_shootx(x, &v, &ifs, 1, 0);
    if (ifs == 0) {
        elf(v.m, v.n, v.p, v.q, v.mu, v.phi, v.theta, v.qq, v.w, 1.0, v.xitol,
            FALSE, v.a, &pi1, &pi2, &pi3, &ifs);
        if (ifs == 0) {
            ll = -0.5 * v.m * v.n * (LOG2PI - log((real) v.m) - log((real) v.n) + 1.0)
                 - 0.5 * v.n * (v.m * log(pi1) + log(pi2));
            *ok = 1;
        }
    }
    vec_shootx(x, &v, &ifs, 0, 1);
    return ll;
}

/*****************************************************************************/
/*  P11 — LDL' OF THE INNOVATION COVARIANCE, IN ONE PLACE.                    */
/*                                                                           */
/*  Sigma = P D P' with P unit lower triangular is used twice: the report     */
/*  prints it, and the impulse responses need it to orthogonalise the shocks. */
/*  Two copies of a factorisation is how two answers to one question appear,  */
/*  so it is written once.  Returns 1 if Sigma is positive definite.          */
/*                                                                           */
/*  The ordering is the .inp's column order, which is the user's choice and   */
/*  not a property of the fit; both callers say so.                           */
/*****************************************************************************/
static int ldl_sigma(struct Tvarma *v, real **P, real *D)
{
    int M = v->m, a, b, k;
    real **Sg = matrix(1, M, 1, M);
    int ok = 1;

    for (a = 1; a <= M; a++)
        for (b = 1; b <= M; b++) {
            Sg[a][b] = v->sigma2 * v->qq[a][b];
            P[a][b]  = (a == b) ? 1.0 : 0.0;
        }
    for (b = 1; b <= M; b++) {
        real acc = Sg[b][b];
        for (k = 1; k < b; k++) acc -= P[b][k] * P[b][k] * D[k];
        D[b] = acc;
        if (D[b] <= 0.0) { ok = 0; break; }
        for (a = b + 1; a <= M; a++) {
            real s2 = Sg[a][b];
            for (k = 1; k < b; k++) s2 -= P[a][k] * P[b][k] * D[k];
            P[a][b] = s2 / D[b];
        }
    }
    free_matrix(Sg, 1, M, 1, M);
    return ok;
}

/*****************************************************************************/
/*  P11 — IMPULSE RESPONSES AND VARIANCE DECOMPOSITION, IN LEVELS            */
/*                                                                           */
/*  WHY NOT THE SUITE'S.  drvarma has impulse_response() and                  */
/*  variance_decomposition() and they are right there in the vendored         */
/*  diagnose.c.  Calling them would give the responses of Ybar =              */
/*  (nabla Y2', W')' -- the differenced block and the equilibrium errors --    */
/*  which is not what anyone asks a cointegrated model.  What is asked is     */
/*  what a shock does to the LEVELS, because that is where the answer splits  */
/*  into a permanent part and a transitory one.                              */
/*                                                                           */
/*  AND drvec ALREADY HAS THE MAP.  level_error_map() carries an innovation   */
/*  into the level error, and it is the single source of truth the forecast   */
/*  bands are built on (docs/FORECAST.md 4).  The response of Y_{t+k} to an   */
/*  innovation at t IS G_k: the level error at horizon h is                    */
/*  sum_{m<h} G_m A_{t+h-m}, so the shock at t enters the level at t+k with   */
/*  weight G_k.  Nothing new is derived here; the same map is read forwards.  */
/*                                                                           */
/*  ORTHOGONALISED, with Sigma = P D P': shock j is one standard deviation of */
/*  A*_j, so the response is G_k P e_j sqrt(D_j).  The ordering is the .inp's */
/*  columns and a different order gives different shocks -- the same silent   */
/*  decision the P matrix carries, said in the same place.                    */
/*                                                                           */
/*  THE DECOMPOSITION FALLS OUT OF THE SAME NUMBERS, and that is the check    */
/*  worth having: with R_k = G_k P D^(1/2), Var(h) = sum_{m<h} R_m R_m', so   */
/*  the forecast standard error at h is the square root of the sum of squared */
/*  responses.  The suite verifies exactly that identity against the forecast */
/*  table, which is what says the two blocks describe one model.             */
/*****************************************************************************/
/*  level_irf_R -- the orthogonalised level responses R_k = G_k P D^(1/2),
 *  k = 0..K, into R (0..K x M x M), and P, D if asked.  The one computation
 *  both the report and the -irfboot replications use.  Returns 0 if Sigma is
 *  not positive definite.                                                   */
static int level_irf_R(struct Tvarma *v, real **B2, int K, real ***R,
                       real **P, real *D)
{
    int M = v->m, r = global_r, i, j, k, l;
    real ***Psi, **Csum, **G;

    if (!ldl_sigma(v, P, D)) return 0;
    Psi  = tensor(0, K, 1, M, 1, M);
    Csum = matrix(1, M, 1, M);
    G    = matrix(1, M, 1, M);
    compute_psi_weights(M, v->p, v->q, v->phi, v->theta, K, Psi);
    for (i = 1; i <= M; i++)
        for (j = 1; j <= M; j++) Csum[i][j] = 0.0;

    for (k = 0; k <= K; k++) {
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) Csum[i][j] += Psi[k][i][j];
        level_error_map(Csum, Psi[k], B2, M, r, G);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                real acc = 0.0;
                for (l = j; l <= M; l++) acc += G[i][l] * P[l][j];
                R[k][i][j] = acc * sqrt(D[j]);
            }
    }
    free_matrix(G, 1, M, 1, M);
    free_matrix(Csum, 1, M, 1, M);
    free_tensor(Psi, 0, K, 1, M, 1, M);
    return 1;
}

static void level_irf_fevd(struct Tvarma *v, real **B2, int K)
{
    int M = v->m, r = global_r, s = M - r, i, j, k, l;
    real ***R, **P, *D;

    if (K < 1 || s < 1) return;

    P = matrix(1, M, 1, M);
    D = vector(1, M);
    R = tensor(0, K, 1, M, 1, M);
    if (!level_irf_R(v, B2, K, R, P, D)) {
        free_tensor(R, 0, K, 1, M, 1, M);
        free_vector(D, 1, M); free_matrix(P, 1, M, 1, M);
        fprintf(outputv, "\n(impulse responses not computed: Sigma is not "
                         "positive definite)\n");
        return;
    }

    banner("Impulse Response of the Levels");
    fprintf(outputv,
        "\n  Response of Y_{t+k} to a one-s.d. orthogonalized shock at t.\n"
        "  Shocks are orthogonalized by Sigma = P D P', so the ORDERING is the\n"
        "  .inp's column order: a different order gives different shocks.\n"
        "  ! in a cointegrated system the response does not die out: what it\n"
        "    converges to is the permanent effect, of rank M - r = %d.\n", s);
    for (j = 1; j <= M; j++) {
        fprintf(outputv, "\nShock to %s:\n    k", series_names ? series_names[j] : "y");
        for (i = 1; i <= M; i++)
            fprintf(outputv, " %14s", series_names ? series_names[i] : "y");
        fprintf(outputv, "\n");
        for (k = 0; k <= K; k++) {
            fprintf(outputv, "%5d", k);
            for (i = 1; i <= M; i++) fprintf(outputv, " %14.6f", R[k][i][j]);
            fprintf(outputv, "\n");
        }
    }

    /*  ---- LONG-RUN GAIN AND MEAN LAG ---------------------------------- *
     *
     *  drvarma accumulates the response of the STATIONARY system and calls the
     *  total the long-run gain (diagnose.c:1626).  Here the response is already
     *  in levels, so accumulating it again would diverge: G_k does not go to
     *  zero, it converges to the PERMANENT effect.  The two programs compute
     *  the same object all the same, and it is worth seeing why: drvec's level
     *  response is the accumulation of drvarma's, so drvec's INCREMENTS
     *  delta_k = G_k - G_{k-1} are what drvarma calls the response.  The gain
     *  and the mean lag are then drvarma's formulas applied to delta_k, and
     *
     *      gain     = sum_k delta_k = lim_k G_k        the permanent effect
     *      mean lag = sum_k k delta_k / sum_k delta_k  how long it takes
     *
     *  THE CERTIFICATE, and it is sharp: beta'Y_t is stationary, so a permanent
     *  shock cannot move it.  Therefore beta' C(1) = 0 EXACTLY, and the worst
     *  element of beta' gain is reported.  It exercises the whole chain at once
     *  -- the Psi weights, the level map, the orthogonalisation and the
     *  accumulation -- and it cannot be satisfied by accident.
     *
     *  The horizon for the limit is its own, and much longer than the printed
     *  one: a gain read off a table that stops at 20 is not a limit.          */
    {
        int KL = 400, conv = 0;
        real ***PsiL = tensor(0, KL, 1, M, 1, M);
        real **CsL = matrix(1, M, 1, M), **GL = matrix(1, M, 1, M);
        real **Gp = matrix(1, M, 1, M);
        real **gain = matrix(1, M, 1, M), **wsum = matrix(1, M, 1, M);
        real worst = 0.0, tail = 0.0;

        compute_psi_weights(M, v->p, v->q, v->phi, v->theta, KL, PsiL);
        for (i = 1; i <= M; i++)
            for (j = 1; j <= M; j++) {
                CsL[i][j] = 0.0; Gp[i][j] = 0.0;
                gain[i][j] = 0.0; wsum[i][j] = 0.0;
            }
        for (k = 0; k <= KL; k++) {
            real mx = 0.0;
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) CsL[i][j] += PsiL[k][i][j];
            level_error_map(CsL, PsiL[k], B2, M, r, GL);
            /*  In the structural coordinates, so that the numbers are the ones
             *  the response table shows.                                     */
            for (i = 1; i <= M; i++)
                for (j = 1; j <= M; j++) {
                    real acc = 0.0, d;
                    for (l = j; l <= M; l++) acc += GL[i][l] * P[l][j];
                    acc *= sqrt(D[j]);
                    d = acc - Gp[i][j];
                    gain[i][j] += d;
                    wsum[i][j] += k * d;
                    Gp[i][j] = acc;
                    if (fabs(d) > mx) mx = fabs(d);
                }
            if (k > 20 && mx < 1.0e-10) { conv = 1; tail = mx; break; }
            tail = mx;
        }

        banner("Long-Run Gain and Mean Lag, in Levels");
        fprintf(outputv,
            "\n  gain     = lim_k G_k, the PERMANENT effect of a one-s.d. shock\n"
            "  mean lag = sum k delta_k / sum delta_k, with delta_k the "
            "increment of\n             the response: how long the level takes "
            "to get there\n");
        fprintf(outputv, "  %s at k = %d (largest increment %.2e)\n",
                conv ? "converged" : "NOT converged", k, tail);
        fprintf(outputv, "\nStructural shock standard deviations "
                         "(one-s.d. shock size):\n");
        for (j = 1; j <= M; j++)
            fprintf(outputv, "  %-12s : %.6f\n",
                    series_names ? series_names[j] : "y", sqrt(D[j]));
        for (j = 1; j <= M; j++) {
            fprintf(outputv, "\nShock to %s:\n",
                    series_names ? series_names[j] : "y");
            for (i = 1; i <= M; i++) {
                real gross = fabs(gain[i][j]);
                if (gross > 1.0e-12 && fabs(gain[i][j]) > 1.0e-8)
                    fprintf(outputv, "  %-12s : gain = %10.6f   mean lag = "
                                     "%7.2f periods\n",
                            series_names ? series_names[i] : "y",
                            gain[i][j], wsum[i][j] / gain[i][j]);
                else
                    fprintf(outputv, "  %-12s : gain = %10.6f   mean lag = "
                                     "undefined (the permanent effect is zero)\n",
                            series_names ? series_names[i] : "y", gain[i][j]);
            }
        }
        /*  beta' gain = 0: the equilibrium error cannot be moved permanently. */
        for (j = 1; j <= M; j++)
            for (k = 1; k <= r; k++) {
                real acc = 0.0;
                /*  beta IN THE .inp's ROW ORDER, which is [Y2 block ; Y1
                 *  block] -- not the internal [Y1 ; Y2] that B = [I_r ; B2] is
                 *  written in.  gain, alpha and the responses are all in the
                 *  .inp's order, so beta has to be put in it too; pairing them
                 *  the other way is what made this check fail the first time
                 *  it was run, and it turned out the report had the same
                 *  mismatch in its beta matrix.                              */
                for (i = 1; i <= M; i++) {
                    real bik = (i <= s) ? B2[i][k]
                                        : ((i - s == k) ? 1.0 : 0.0);
                    acc += bik * gain[i][j];
                }
                if (fabs(acc) > worst) worst = fabs(acc);
            }
        fprintf(outputv,
            "\n  self-check  max |beta' gain| = %.3e\n"
            "  beta'Y_t is stationary, so no shock can move it permanently: "
            "this is\n  zero by construction, and it exercises the weights, the "
            "level map and\n  the orthogonalisation at once.  The gain matrix "
            "has rank M - r = %d.\n", worst, s);

        free_matrix(wsum, 1, M, 1, M); free_matrix(gain, 1, M, 1, M);
        free_matrix(Gp, 1, M, 1, M);   free_matrix(GL, 1, M, 1, M);
        free_matrix(CsL, 1, M, 1, M);
        free_tensor(PsiL, 0, KL, 1, M, 1, M);
    }

    banner("Forecast Error Variance Decomposition, in Levels");
    fprintf(outputv,
        "\n  Percentage of the h-step level forecast error variance due to "
        "each shock.\n  Same orthogonalization, so the same caveat on the "
        "ordering.\n");
    for (i = 1; i <= M; i++) {
        real *acc = vector(1, M);
        for (j = 1; j <= M; j++) acc[j] = 0.0;
        fprintf(outputv, "\n%s:\n    h", series_names ? series_names[i] : "y");
        for (j = 1; j <= M; j++)
            fprintf(outputv, " %11s", series_names ? series_names[j] : "y");
        fprintf(outputv, "\n");
        for (k = 0; k <= K; k++) {
            real tot = 0.0;
            for (j = 1; j <= M; j++) { acc[j] += R[k][i][j] * R[k][i][j];
                                       tot += acc[j]; }
            fprintf(outputv, "%5d", k + 1);
            for (j = 1; j <= M; j++)
                fprintf(outputv, " %10.1f%%",
                        (tot > 0.0) ? 100.0 * acc[j] / tot : 0.0);
            fprintf(outputv, "\n");
        }
        free_vector(acc, 1, M);
    }

    free_tensor(R, 0, K, 1, M, 1, M);
    free_vector(D, 1, M);
    free_matrix(P, 1, M, 1, M);
}

/*****************************************************************************/
/*  -irfboot N -- BANDS FOR THE LEVEL RESPONSES AND THE DECOMPOSITION, BY     */
/*  PARAMETRIC BOOTSTRAP.                                                    */
/*                                                                           */
/*  WHY NOT THE DELTA METHOD.  The responses are a long nonlinear chain of    */
/*  the parameters -- Psi weights, the level map through B2, the LDL' of      */
/*  Sigma -- and in a cointegrated system they converge to the permanent      */
/*  effect, whose distribution near the boundary is nothing like a normal.    */
/*  With n = 67 a linearisation is exactly what should not be trusted.        */
/*                                                                           */
/*  WHAT IS DONE.  The fitted model is simulated N times with the generator   */
/*  of the rank bootstrap (simulate_h0: the stationary VARMA on Ybar, Gaussian*/
/*  innovations with the fitted Sigma*, the transformation inverted to        */
/*  levels from the real origin).  Each replication is REFITTED exactly as    */
/*  the data were -- same rank, same restrictions (-weakex/-alpha, -fixb2row, */
/*  -xsys, the drift mask), the same ladder start -- and its responses and    */
/*  decomposition are computed with the report's own level_irf_R.  The bands */
/*  are the percentile intervals of those replications (Efron), 90 % and     */
/*  95 %, printed for every response and written whole to <base>.irfboot.    */
/*                                                                           */
/*  The rank is held: the bands are conditional on r, as the responses are.  */
/*  The generator is deterministic (boot_rng), so the bands are reproducible. */
/*****************************************************************************/
static int global_irfboot = 0;            /* -irfboot N: replications        */

/*  B2 as the walk of x[] has it: first in the tail, column-major over the
 *  rows -fixb2/-fixb2row do not hold.                                        */
static void x_b2(const real *x, real **B2)
{
    int nmean, nlam, nmid, ntail, s = nser - global_r, r = global_r, i, j;
    int idx;
    par_blocks(&nmean, &nlam, &nmid, &ntail);
    idx = nmean + nlam + nmid + 1;
    for (j = 1; j <= r; j++)
        for (i = 1; i <= s; i++)
            B2[i][j] = b2_held(i) ? B2_fixed[i][j] : x[idx++];
}

/*  The model at x, with sigma2 filled (vec_shootx does not set it).  The
 *  caller deallocates with vec_shootx(x, v, &ifs, 0, 1).                     */
static int model_at(real *x, struct Tvarma *v)
{
    int ifs = 0, ife = 0;
    real pi1, pi2, pi3;
    v->xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
    vec_shootx(x, v, &ifs, 1, 0);
    if (ifs != 0) return 0;
    elf(v->m, v->n, v->p, v->q, v->mu, v->phi, v->theta, v->qq, v->w, 1.0,
        v->xitol, TRUE, v->a, &pi1, &pi2, &pi3, &ife);
    if (ife != 0) return 0;
    v->sigma2 = pi1 / (v->n * v->m);
    return 1;
}

static void irf_bootstrap(real *x, int npar, int K, int N)
{
    int M = nser, r = global_r, s = M - r, b, i, j, k, nok = 0, nfail = 0;
    long cells = (long) (K + 1) * M * M;
    real **B2 = matrix(1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
    real **sim = matrix(1, nobs_raw, 1, M), **saved = matrix(1, nobs_raw, 1, M);
    real ***R0 = tensor(0, K, 1, M, 1, M), **P = matrix(1, M, 1, M), *D = vector(1, M);
    real *irf = (real *) malloc(sizeof(real) * (size_t) cells * (size_t) N);
    real *fev = (real *) malloc(sizeof(real) * (size_t) cells * (size_t) N);
    real *col = vector(1, N);
    struct Tvarma vh;
    FILE *save_out = outputv;
    int save_quiet = quiet_mode;
    (void) npar;

    if (!irf || !fev) { fprintf(stderr, "ERROR: -irfboot: out of memory\n"); return; }
    x_b2(x, B2);
    if (!model_at(x, &vh) || !level_irf_R(&vh, B2, K, R0, P, D)) {
        fprintf(stderr, "ERROR: -irfboot: the fitted model cannot be simulated\n");
        free(irf); free(fev); return;
    }
    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) saved[i][j] = rawmat[i][j];

    printf("IRF bootstrap (-irfboot %d): ", N); fflush(stdout);
    outputv = fopen("/dev/null", "w"); quiet_mode = 1;
    for (b = 1; b <= N; b++) {
        int np, ifr;
        real *xr, *dev, **cov, ll = 0.0, s2 = 0.0;
        real **B2r = matrix(1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
        real ***Rb = tensor(0, K, 1, M, 1, M), **Pb = matrix(1, M, 1, M), *Db = vector(1, M);
        struct Tvarma vr;
        int good = 0;

        if (simulate_h0(&vh, B2, r, sim) == 0) {
            for (i = 1; i <= nobs_raw; i++)
                for (j = 1; j <= M; j++) rawmat[i][j] = sim[i][j];
            build_y2_levels();
            np = calc_nparametrs();
            xr = vector(1, np); dev = vector(1, np); cov = matrix(1, np, 1, np);
            init_guess(xr, np);
            if (r > 0 && ladder_wanted())       gate_profile_seed(xr, np);
            else if (r == 0 && ladder_wanted()) ladder_seed_r0(xr, np);
            ifr = fit_search(xr, np, dev, cov, &ll, &s2, 1, 0, NULL);
            if (ifr == 0) {
                x_b2(xr, B2r);
                if (model_at(xr, &vr)) {
                    if (level_irf_R(&vr, B2r, K, Rb, Pb, Db)) good = 1;
                    { int ifs = 0; vec_shootx(xr, &vr, &ifs, 0, 1); }
                }
            }
            free_matrix(cov, 1, np, 1, np); free_vector(dev, 1, np); free_vector(xr, 1, np);
        }
        if (good) {
            long base = (long) nok * cells;
            for (i = 1; i <= M; i++) {
                real *acc = vector(1, M);
                for (j = 1; j <= M; j++) acc[j] = 0.0;
                for (k = 0; k <= K; k++) {
                    real tot = 0.0;
                    for (j = 1; j <= M; j++) {
                        acc[j] += Rb[k][i][j] * Rb[k][i][j];
                        tot += acc[j];
                    }
                    for (j = 1; j <= M; j++) {
                        long c = base + ((long) k * M + (i - 1)) * M + (j - 1);
                        irf[c] = Rb[k][i][j];
                        fev[c] = (tot > 0.0) ? 100.0 * acc[j] / tot : 0.0;
                    }
                }
                free_vector(acc, 1, M);
            }
            nok++;
        } else nfail++;
        free_vector(Db, 1, M); free_matrix(Pb, 1, M, 1, M);
        free_tensor(Rb, 0, K, 1, M, 1, M);
        free_matrix(B2r, 1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
        if (b % 50 == 0) { printf("%d ", b); fflush(stdout); }
    }
    if (outputv) fclose(outputv);
    outputv = save_out; quiet_mode = save_quiet;
    for (i = 1; i <= nobs_raw; i++)
        for (j = 1; j <= M; j++) rawmat[i][j] = saved[i][j];
    build_y2_levels();
    printf("\n  %d replications refitted, %d failed\n", nok, nfail);

    banner("Bands for the Level Responses, Parametric Bootstrap");
    fprintf(outputv,
        "\n  %d replications simulated from the fitted model and REFITTED as the\n"
        "  data were (same rank, restrictions and ladder start); %d failed and\n"
        "  are left out.  Percentile intervals (Efron), conditional on r = %d.\n"
        "  Same orthogonalisation as the responses above.  The whole table, every\n"
        "  k, is in %s.irfboot.\n", nok, nfail, r, out_base);
    if (nok < 20) {
        fprintf(outputv, "\n  Too few replications for bands (%d).\n", nok);
    } else {
        char fname[600];
        FILE *fb;
        int pick[] = { 0, 1, 2, 4, 8, 12, 16, 20, 30, 40 };
        snprintf(fname, sizeof fname, "%s.irfboot", out_base);
        fb = fopen(fname, "w");
        if (fb) fprintf(fb, "# drvec -irfboot %d: %d replications, percentile bands\n"
                            "what,shock,response,k,point,p2.5,p5,p50,p95,p97.5\n", N, nok);
        for (j = 1; j <= M; j++)
            for (i = 1; i <= M; i++) {
                fprintf(outputv, "\nResponse of %s to a shock to %s:\n"
                        "    k        point       5%%      95%%     2.5%%    97.5%%\n",
                        series_names ? series_names[i] : "y",
                        series_names ? series_names[j] : "y");
                for (k = 0; k <= K; k++) {
                    real q[5];
                    int t, w, pr = 0;
                    for (w = 0; w < 2; w++) {
                        real *src = w ? fev : irf;
                        for (t = 0; t < nok; t++)
                            col[t + 1] = src[(long) t * cells
                                             + ((long) k * M + (i - 1)) * M + (j - 1)];
                        qsort(&col[1], (size_t) nok, sizeof(real), cmp_real);
                        {
                            const real pp[5] = { 0.025, 0.05, 0.50, 0.95, 0.975 };
                            int m;
                            for (m = 0; m < 5; m++) {
                                int ix = (int) ceil(pp[m] * nok);
                                if (ix < 1) ix = 1;
                                if (ix > nok) ix = nok;
                                q[m] = col[ix];
                            }
                        }
                        if (fb) {
                            real pt;
                            if (!w) pt = R0[k][i][j];
                            else {
                                real a = 0.0, tot = 0.0; int kk, jj;
                                for (kk = 0; kk <= k; kk++)
                                    for (jj = 1; jj <= M; jj++) {
                                        tot += R0[kk][i][jj] * R0[kk][i][jj];
                                        if (jj == j) a += R0[kk][i][jj] * R0[kk][i][jj];
                                    }
                                pt = (tot > 0.0) ? 100.0 * a / tot : 0.0;
                            }
                            fprintf(fb, "%s,%s,%s,%d,%.8g,%.8g,%.8g,%.8g,%.8g,%.8g\n",
                                    w ? "fevd" : "irf",
                                    series_names ? series_names[j] : "y",
                                    series_names ? series_names[i] : "y",
                                    w ? k + 1 : k, pt, q[0], q[1], q[2], q[3], q[4]);
                        }
                        if (!w) {
                            int m;
                            for (m = 0; m < (int) (sizeof pick / sizeof pick[0]); m++)
                                if (pick[m] == k) pr = 1;
                            if (k == K) pr = 1;
                            if (pr)
                                fprintf(outputv, "%5d %12.6f %8.4f %8.4f %8.4f %8.4f\n",
                                        k, R0[k][i][j], q[1], q[3], q[0], q[4]);
                        }
                    }
                }
            }
        /*  The decomposition's bands, for each series at a few horizons.   */
        for (i = 1; i <= M; i++) {
            fprintf(outputv, "\nFEVD of %s, %% due to each shock: point [5%%, 95%%]\n    h",
                    series_names ? series_names[i] : "y");
            for (j = 1; j <= M; j++)
                fprintf(outputv, " %26s", series_names ? series_names[j] : "y");
            fprintf(outputv, "\n");
            for (k = 0; k <= K; k++) {
                int hh = k + 1, m, show = 0;
                for (m = 0; m < (int) (sizeof pick / sizeof pick[0]); m++)
                    if (pick[m] == hh) show = 1;
                if (hh == K + 1) show = 1;
                if (!show) continue;
                fprintf(outputv, "%5d", hh);
                for (j = 1; j <= M; j++) {
                    real a = 0.0, tot = 0.0, lo, hi; int kk, jj, t;
                    for (kk = 0; kk <= k; kk++)
                        for (jj = 1; jj <= M; jj++) {
                            tot += R0[kk][i][jj] * R0[kk][i][jj];
                            if (jj == j) a += R0[kk][i][jj] * R0[kk][i][jj];
                        }
                    for (t = 0; t < nok; t++)
                        col[t + 1] = fev[(long) t * cells + ((long) k * M + (i - 1)) * M + (j - 1)];
                    qsort(&col[1], (size_t) nok, sizeof(real), cmp_real);
                    { int i5 = (int) ceil(0.05 * nok), i95 = (int) ceil(0.95 * nok);
                      if (i5 < 1) i5 = 1;
                      if (i95 > nok) i95 = nok;
                      lo = col[i5]; hi = col[i95]; }
                    fprintf(outputv, "   %6.1f%% [%5.1f%%, %5.1f%%]",
                            (tot > 0.0) ? 100.0 * a / tot : 0.0, lo, hi);
                }
                fprintf(outputv, "\n");
            }
        }
        if (fb) fclose(fb);
        printf("  bands written to the .out and to %s.irfboot\n", out_base);
    }
    { int ifs = 0; vec_shootx(x, &vh, &ifs, 0, 1); }
    free_vector(col, 1, N);
    free(irf); free(fev);
    free_vector(D, 1, M); free_matrix(P, 1, M, 1, M);
    free_tensor(R0, 0, K, 1, M, 1, M);
    free_matrix(saved, 1, nobs_raw, 1, M); free_matrix(sim, 1, nobs_raw, 1, M);
    free_matrix(B2, 1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
}

/*****************************************************************************/
/*  P8 — report_fit: everything the .out says about a finished fit.          */
/*                                                                           */
/*  It was 793 lines inline in main(), which is why finding anything in this  */
/*  program meant reading main() from the top: the parameter printer, the     */
/*  roots, the diagnostics, Pi, the hypothesis block and the forecast all     */
/*  lived at the same indentation as the argument parsing.  Nothing here is   */
/*  new and nothing here moved -- the golden set (tools/golden.sh) is what    */
/*  says so, byte for byte over twenty-four configurations.                   */
/*                                                                           */
/*  ifault comes in by value: main() does not read it back, it reads          */
/*  estimation_failed, which this sets.                                      */
/*  BUG-19.  The per-series diagnosis is vendored whole from drvarma and labels
 *  residual i with series_names[i].  The components of Ybar are nabla Y2 --
 *  those ARE series, and the name is right -- and then W = Y1 + B2'Y2, whose
 *  innovation is not the Y1 series' and was read as one (a residual sd of 7.1
 *  on a series whose differences have sd 2.7).  diagnose.c is not touched
 *  (P3.1: byte-identical to drvarma); the W names are swapped for ec1..ecr --
 *  what the Lambda table already calls them -- for the duration of the call. */
static void diagnose_ybar(struct Tvarma *vp)
{
    int s = nser - global_r, j;
    char **save = NULL;
    if (series_names && global_r > 0) {
        save = (char **) malloc((size_t) (nser + 1) * sizeof(char *));
        for (j = 1; j <= nser; j++) save[j] = series_names[j];
        for (j = 1; j <= global_r; j++) {
            char lab[32];
            snprintf(lab, sizeof lab, "ec%d", j);
            series_names[s + j] = strdup(lab);
        }
    }
    /*  BUG-37.  diagnose dates the residuals from trans_d + trans_D*freq, the
     *  .inp's declared differencing -- which this route does NOT apply --,
     *  while the sample drvec estimates starts nobs_raw - nobs observations
     *  in (one, in the levels layout, for nabla Y2).  Every date of the
     *  diagnosis was one period early.  The engine is not touched: the
     *  offset it reads is set to the true one for the call.               */
    {
        int sd = trans_d, sD = trans_D;
        trans_d = nobs_raw - nobs; trans_D = 0;
        int *np = diag_npar_ser();
        diagnose(vp, np);
        free(np);
        trans_d = sd; trans_D = sD;
    }
    if (save) {
        for (j = s + 1; j <= nser; j++) { free(series_names[j]); series_names[j] = save[j]; }
        free(save);
    }
}

/*****************************************************************************/
static void report_fit(real *x, real *dev, real **cov, int npar,
                       struct Tvarma *vp, int ifault, const char *outname,
                       real lr_free, int lr_free_ok)
{
    if (ifault == 0) {
        /* Standard errors from the Hessian at the optimum, if asked for.  It goes
           before recovering the final structure because objcfunc fills varmax
           with whatever point it is handed.                                  */
        se_from_fdhess = 0;
        if (global_fdhess) {
            if (exact_hessian_se(npar, x, dev, cov, nobs) == 0) {
                se_from_fdhess = 1;
                fprintf(outputv, "Standard errors  : finite-difference Hessian "
                                 "at the optimum (-fdhess)\n");
            } else
                /*  BUG-34: said in the .out too, not only on the terminal.  */
                fprintf(outputv, "Standard errors  : BFGS-accumulated factor -- "
                                 "-fdhess was asked for and NOT used:\n"
                                 "                   %s\n",
                        fdh_rej > 0
                          ? "the optimum is on the boundary (finite-difference "
                            "steps left the admissible region)"
                          : "the Hessian at the optimum is not positive definite");
        }
        vec_shootx(x, vp, &ifault, 0, 0);  /* retrieve final */

        /* Fill in the RESIDUALS of the final fit.  elf writes them with
           atf = TRUE, and until now they arrived by rebound because est made
           that call on finishing.  With -multistart there is no final est --
           the best point is already chosen -- and the residuals were left
           UNCOMPUTED: the diagnosis came out with Q = nan and "the residuals
           appear to be white noise", which is the worst possible way to be
           wrong.  They are computed here, which is where the final fit is
           known.                                                             */
        {
            real pi1, pi2, pi3;
            int ifr = 0;
            elf(vp->m, vp->n, vp->p, vp->q, vp->mu, vp->phi,
                vp->theta, vp->qq, vp->w, 1.0, vp->xitol,
                TRUE, vp->a, &pi1, &pi2, &pi3, &ifr);
        }

        /*  The convergence note goes with the optimizer's own banner, which the
         *  engine already wrote into this file: WHY it stopped belongs next to
         *  WHETHER it stopped.  The fit statistics follow it, and they are
         *  printed for EVERY parameterisation -- -warma included, which is
         *  what the branch below skips.                                      */
        convergence_note(termcode_from_out(outname));
        fprintf(outputv, "sigma2           : %15.10f\n", vp->sigma2);
        fprintf(outputv, "logelf           : %15.10f\n", vp->logelf);
        fprintf(outputv, "npar             : %d\n", npar);
        fprintf(outputv, "AIC              : %15.10f   (-2logL + 2k, /n)\n",
                (-2.0 * vp->logelf + 2.0 * npar) / nobs);
        fprintf(outputv, "BIC              : %15.10f   (-2logL + k log n, /n)\n",
                (-2.0 * vp->logelf + npar * log((real) nobs)) / nobs);


        /* The LR of H1(r) against H(r).  Johansen and Swensen (2024): the degrees
           of freedom are (M - sa)*r, which is how many free entries of alpha the
           restriction removes.                                               */
        if (xl_on && xl_base_ok) {
            int df = xl_npar();
            real lr = 2.0 * (vp->logelf - xl_base_ll);
            real pv = (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0;
            fprintf(outputv, "\n--- the transfer inputs (-xlink), against the VEC "
                             "without them ---\n");
            fprintf(outputv, "logL VEC without inputs      : %15.10f\n", xl_base_ll);
            if (xl_bridge_ok)
                fprintf(outputv, "logL at the bridge (omega=0) : %15.10f   "
                        "(difference %.2e: the crossing identity of this rung)\n",
                        xl_bridge_ll, xl_bridge_ll - xl_base_ll);
            fprintf(outputv, "logL with the inputs         : %15.10f\n", vp->logelf);
            fprintf(outputv, "LR = 2(inputs - VEC)         : %15.10f\n", lr);
            fprintf(outputv, "degrees of freedom           : %d   sum of (s+1+r)\n", df);
            fprintf(outputv, "p-value (chi2)               : %15.10f\n", pv);
            fprintf(outputv, "  The chi2 needs the rank to be the same under both "
                             "models and X fixed;\n  the rank test itself is NOT "
                             "recalibrated for exogenous inputs (TODO.md MEJORA-2).\n");
            printf("  LR (inputs vs VEC) = %.6f, %d df, p = %.6f\n", lr, df, pv);
        }
        if (b2row_on && lr_free_ok) {
            int nh = b2_nheld(nser - global_r), df, i_;
            int dfa = (global_alpha && !xs_on) ? (nser - alpha_sa) * global_r : 0;
            real lr = 2.0 * (lr_free - vp->logelf), pv;
            df = nh * global_r + dfa;
            pv = (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0;
            fprintf(outputv, "\n--- beta restricted by -fixb2row%s, against "
                             "H(r) ---\n", xs_on ? " (inputs embedded by -xsys in both)"
                             : (global_alpha ? " and alpha = A*psi" : ""));
            for (i_ = 1; i_ <= nser - global_r; i_++)
                if (b2_held(i_))
                    fprintf(outputv, "row %d of B2 (%s) held at %g in every "
                            "relation\n", i_, sname(i_), b2row_val[i_]);
            fprintf(outputv, "logL H(r)  free        : %15.10f\n", lr_free);
            fprintf(outputv, "logL restricted        : %15.10f\n", vp->logelf);
            fprintf(outputv, "LR = 2(free - restr.)  : %15.10f\n", lr);
            fprintf(outputv, "degrees of freedom     : %d   (rows held)*r%s\n", df,
                    (global_alpha && !xs_on) ? " + (M - sa)*r" : "");
            fprintf(outputv, "p-value (chi2)         : %15.10f\n", pv);
            printf("  restricted         : logL = %15.10f\n", vp->logelf);
            printf("  LR = %.6f, %d df, p = %.6f%s\n", lr, df, pv,
                   (lr < -1.0e-6) ? "   <- NEGATIVE: the restricted beats the free one,"
                                    " so one of the two did not converge" : "");
        }
        else if (xs_on && lr_free_ok) {
            int df = xs_df, i_;
            real lr = 2.0 * (lr_free - vp->logelf);
            real pv = (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0;
            fprintf(outputv, "\n--- the inputs embedded by -xsys: strong exogeneity, "
                             "against the free VEC ---\n");
            for (i_ = 1; i_ <= nser; i_++)
                if (xs_inp[i_])
                    fprintf(outputv, "%s: alpha = 0 and no other series in its "
                            "short run (Gamma(k) row)\n", sname(i_));
            fprintf(outputv, "logL VEC free          : %15.10f\n", lr_free);
            fprintf(outputv, "logL inputs embedded   : %15.10f\n", vp->logelf);
            fprintf(outputv, "LR = 2(free - restr.)  : %15.10f\n", lr);
            fprintf(outputv, "degrees of freedom     : %d   r*(inputs) + "
                             "(p-1)*(Gamma entries held)\n", df);
            fprintf(outputv, "p-value (chi2)         : %15.10f\n", pv);
            fprintf(outputv, "  Not rejecting is what licenses reading the rows of "
                             "the outputs as a\n  transfer from the inputs: their "
                             "entry in beta (the gain), their lags\n  in Gamma and "
                             "the covariance Q (omega_0, the SVAR identity).\n");
            printf("  inputs embedded    : logL = %15.10f\n", vp->logelf);
            printf("  LR (strong exogeneity) = %.6f, %d df, p = %.6f%s\n", lr, df, pv,
                   (lr < -1.0e-6) ? "   <- NEGATIVE: the restricted beats the free one,"
                                    " so one of the two did not converge" : "");
        }
        else if (gz_on && lr_free_ok) {
            int df = gz_df, i_, j_;
            int dfa = global_alpha ? (nser - alpha_sa) * global_r : 0;
            real lr = 2.0 * (lr_free - vp->logelf);
            real pv = (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0;
            fprintf(outputv, "\n--- Gamma(k) held by -gzero%s, against the free VEC ---\n",
                    global_alpha ? " and alpha = A*psi" : "");
            for (i_ = 1; i_ <= nser; i_++)
                for (j_ = 1; j_ <= nser; j_++)
                    if (gz[i_][j_])
                        fprintf(outputv, "D.%s <- D.%s held at 0 for k = 1..%d\n",
                                sname(i_), sname(j_), global_p - 1);
            fprintf(outputv, "logL VEC free          : %15.10f\n", lr_free);
            fprintf(outputv, "logL restricted        : %15.10f\n", vp->logelf);
            fprintf(outputv, "LR = 2(free - restr.)  : %15.10f\n", lr);
            fprintf(outputv, "degrees of freedom     : %d   (p-1)*(entries held) = %d\n",
                    df, df - dfa);
            if (dfa)
                fprintf(outputv, "                             + (M - sa)*r = %d\n", dfa);
            fprintf(outputv, "p-value (chi2)         : %15.10f\n", pv);
            printf("  restricted         : logL = %15.10f\n", vp->logelf);
            printf("  LR (Gamma%s) = %.6f, %d df, p = %.6f%s\n",
                   global_alpha ? " and alpha" : "", lr, df, pv,
                   (lr < -1.0e-6) ? "   <- NEGATIVE: the restricted beats the free one,"
                                    " so one of the two did not converge" : "");
        }
        else if (global_alpha && lr_free_ok) {
            int df = (nser - alpha_sa) * global_r;
            real lr = 2.0 * (lr_free - vp->logelf);
            real pv = (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0;
            fprintf(outputv, "\n--- H1(r): alpha = A*psi, against H(r) ---\n");
            fprintf(outputv, "logL H(r)  free        : %15.10f\n", lr_free);
            fprintf(outputv, "logL H1(r) restricted  : %15.10f\n", vp->logelf);
            fprintf(outputv, "LR = 2(free - restr.)  : %15.10f\n", lr);
            fprintf(outputv, "degrees of freedom     : %d   (M - sa)*r\n", df);
            fprintf(outputv, "p-value (chi2)         : %15.10f\n", pv);
            printf("  H1(r) restricted   : logL = %15.10f\n", vp->logelf);
            printf("  LR = %.6f, %d df, p = %.6f%s\n", lr, df, pv,
                   (lr < -1.0e-6) ? "   <- NEGATIVE: the restricted beats the free one,"
                                    " so one of the two did not converge" : "");
        }

        /* --- Structured VEC output --------------------------------------- */
        int s = nser - global_r, r = global_r;
        int ii = 1, warma_done = 0;
        real **Lam_m = matrix(1, nser, 1, (r > 0 ? r : 1));

        /*  -warma: the parameters are NOT the VEC's, so they are not printed
         *  as if they were.  What has been estimated is published, in the
         *  coordinates it was estimated in, and they are named.              */
        if (global_warma) {
            int nf_w = (global_p > 1) ? global_p - 1 : 0, kk;
            fprintf(outputv,
              "\nTriangular (WARMA) parameterisation, on Ybar_t = [nabla Y2 ; W]:\n"
              "  Ybar_t = sum_k Phi*_k Ybar_{t-k} + (I - sum_k Theta*_k L^k) A*_t\n"
              "  with Phi*_k = [0  Psi_k ; 0  Phi_k] and Theta*_k in the W block\n"
              "  only.  B2 enters ONLY through W = Y1 + B2'Y2, by subtraction.\n\n");
            if (global_case == 2) {
                fprintf(outputv, "E[W] =\n");
                for (int j = 1; j <= r; j++)
                    fprintf(outputv, "  %12.6f  (sd = %10.6f)\n", x[ii], dev[ii]), ii++;
            } else if (global_case == 3) {
                fprintf(outputv, "E[nabla Y2] and E[W] =\n");
                for (int i = 1; i <= nser; i++)
                    fprintf(outputv, "  %12.6f  (sd = %10.6f)\n", x[ii], dev[ii]), ii++;
            }
            for (kk = 1; kk <= nf_w + 1; kk++) {
                fprintf(outputv, "coefficients of W_{t-%d}  (M x r) =\n", kk);
                for (int i = 1; i <= nser; i++) {
                    fprintf(outputv, "  ");
                    for (int j = 1; j <= r; j++)
                        fprintf(outputv, "%12.6f (sd %9.6f)", x[ii], dev[ii]), ii++;
                    fprintf(outputv, "\n");
                }
            }
            for (kk = 1; kk <= global_q; kk++) {
                fprintf(outputv, "Theta*[%d] in the W block (r x r) =\n", kk);
                for (int i = 1; i <= r; i++) {
                    fprintf(outputv, "  ");
                    for (int j = 1; j <= r; j++)
                        fprintf(outputv, "%12.6f (sd %9.6f)", x[ii], dev[ii]), ii++;
                    fprintf(outputv, "\n");
                }
            }
            fprintf(outputv, "Q (M x M, lower triangle; Q[1][1] = 1) =\n  1.000000\n");
            for (int i = 2; i <= nser; i++) fprintf(outputv, "  %12.6f\n", x[ii++]);
            if (!global_diag_cov)
                for (int i = 2; i <= nser; i++)
                    for (int j = 1; j < i; j++)
                        fprintf(outputv, "  off(%d,%d) %12.6f\n", i, j, x[ii++]);
            fprintf(outputv, "B2 (s x r) =\n");
            for (int i = 1; i <= s; i++) {
                fprintf(outputv, "  ");
                for (int j = 1; j <= r; j++)
                    fprintf(outputv, "%12.6f%s", global_fixb2 ? B2_fixed[i][j] : x[ii],
                            global_fixb2 ? "" : "");
                if (!global_fixb2) ii += r;
                fprintf(outputv, "\n");
            }
            if (ii != npar + 1)
                fprintf(stderr, "ERROR output (-warma): consumed %d of %d\n",
                        ii - 1, npar);

            /*  AND BACK TO VEC COORDINATES, once only, at the end.  It is what
             *  makes this route usable: it is estimated where the class is a
             *  zero pattern and REPORTED where the user reads.               */
            if (r > 0) {
                real **B2w = matrix(1, s, 1, (r > 0 ? r : 1));
                real **Lw  = matrix(1, nser, 1, r);
                real ***Fw = tensor(1, (global_p > 1 ? global_p - 1 : 1),
                                    1, nser, 1, nser);
                real ***Tw = tensor(1, (global_q > 0 ? global_q : 1),
                                    1, nser, 1, nser);
                real **Sw  = matrix(1, nser, 1, nser);
                real resid, gg;
                int nfw = (global_p > 1) ? global_p - 1 : 0;

                for (int j2 = 1; j2 <= r; j2++)
                    for (int i2 = 1; i2 <= s; i2++)
                        B2w[i2][j2] = global_fixb2 ? B2_fixed[i2][j2]
                                    : x[npar - s * r + (j2 - 1) * s + i2];
                for (int k2 = 1; k2 <= (global_q > 0 ? global_q : 1); k2++)
                    for (int i2 = 1; i2 <= nser; i2++)
                        for (int j2 = 1; j2 <= nser; j2++) Tw[k2][i2][j2] = 0.0;
                resid = warma_inverse(vp, B2w, Lw, Fw, Tw, Sw);

                /*  BUG-45: this block printed Lambda (Mauricio's sign) and Pi
                 *  in the INTERNAL order, unlabelled, while the main report
                 *  uses Johansen's alpha = -Lambda in the .inp's order.  Same
                 *  conventions here: rows and columns in the .inp's order,
                 *  named; alpha = -Lambda; Pi = alpha beta'.                 */
                fprintf(outputv,
                  "\n--- the same fit in VEC coordinates ---\n"
                  "  nabla Y_t = alpha (beta'Y_{t-1} - E[W]) + Gamma_1 nabla Y_{t-1}"
                  " + ... + Theta(L) A_t\n"
                  "  recovered by inverting the transformation once, not "
                  "estimated again.\n"
                  "  Rows and columns in the .inp's order; alpha = -Lambda "
                  "(negative = error-correcting).\n");
#define WPRINT_HDR() do { fprintf(outputv, "  %-12s", ""); \
        for (int j3 = 1; j3 <= nser; j3++) fprintf(outputv, "%12.12s", \
            series_names ? series_names[j3] : "y"); fprintf(outputv, "\n"); } while (0)
                fprintf(outputv, "\nalpha (M x r) =\n");
                for (int i2 = 1; i2 <= nser; i2++) {
                    fprintf(outputv, "  %-12.12s", series_names ? series_names[i2] : "y");
                    for (int j2 = 1; j2 <= r; j2++)
                        fprintf(outputv, "%12.6f", -Lw[inp2lam(i2)][j2]);
                    fprintf(outputv, "\n");
                }
                for (int k2 = 1; k2 <= nfw; k2++) {
                    fprintf(outputv, "Gamma[%d] (M x M) =\n", k2);
                    WPRINT_HDR();
                    for (int i2 = 1; i2 <= nser; i2++) {
                        fprintf(outputv, "  %-12.12s", series_names ? series_names[i2] : "y");
                        for (int j2 = 1; j2 <= nser; j2++)
                            fprintf(outputv, "%12.6f", Fw[k2][inp2lam(i2)][inp2lam(j2)]);
                        fprintf(outputv, "\n");
                    }
                }
                for (int k2 = 1; k2 <= global_q; k2++) {
                    fprintf(outputv, "Theta[%d] (M x M) =\n", k2);
                    WPRINT_HDR();
                    for (int i2 = 1; i2 <= nser; i2++) {
                        fprintf(outputv, "  %-12.12s", series_names ? series_names[i2] : "y");
                        for (int j2 = 1; j2 <= nser; j2++)
                            fprintf(outputv, "%12.6f", Tw[k2][inp2lam(i2)][inp2lam(j2)]);
                        fprintf(outputv, "\n");
                    }
                }
                fprintf(outputv, "Pi = alpha beta' (M x M) =\n");
                WPRINT_HDR();
                for (int i2 = 1; i2 <= nser; i2++) {
                    int a2 = inp2lam(i2);
                    fprintf(outputv, "  %-12.12s", series_names ? series_names[i2] : "y");
                    for (int j2 = 1; j2 <= nser; j2++) {
                        int b2 = inp2lam(j2);
                        real acc = 0.0;
                        for (int k2 = 1; k2 <= r; k2++)
                            acc -= Lw[a2][k2] * ((b2 <= r) ? (b2 == k2 ? 1.0 : 0.0)
                                                           : B2w[b2 - r][k2]);
                        fprintf(outputv, "%12.6f", acc);
                    }
                    fprintf(outputv, "\n");
                }
#undef WPRINT_HDR
                gg = granger_smin(Lw, B2w, Tw, nser, r, global_q);
                if (gg >= 0.0)
                    fprintf(outputv,
                      "\nRank condition (Granger): sigma_s(Lambda_perp' "
                      "Theta(1)) = %.3e\n%s", gg,
                      (gg < GRANGER_ZERO)
                        ? "  *** ZERO to working precision: this fit denies the "
                          "rank it was estimated at.\n"
                      : (gg < global_rankadm_tol)
                        ? "  *** below the floor: NEAR the set where the fit would "
                          "deny its rank.\n"
                        : "  Away from zero: the fit is a model of the rank it "
                          "was estimated at.\n");
                /*  The residual of the map: if the point were not in the image of
                 *  the transformation, this would say so instead of letting an
                 *  invented Lambda be published.                             */
                fprintf(outputv, "  inversion residual = %.3e%s\n", resid,
                        (resid > 1.0e-6)
                          ? "   *** the fitted point is NOT in the image of the "
                            "transformation; the VEC parameters above are a "
                            "least-squares projection, not the fit"
                          : "   (exact: the two coordinate systems describe the "
                            "same fit)");
                free_matrix(Sw, 1, nser, 1, nser);
                free_tensor(Tw, 1, (global_q > 0 ? global_q : 1), 1, nser, 1, nser);
                free_tensor(Fw, 1, (global_p > 1 ? global_p - 1 : 1), 1, nser, 1, nser);
                free_matrix(Lw, 1, nser, 1, r);
                free_matrix(B2w, 1, s, 1, (r > 0 ? r : 1));
            }
            /*  It exits through the SAME cleanup as the rest, and not by a
             *  return of its own: a new exit path is a new set of leaks, and
             *  valgrind found them as soon as it was written (2112 bytes in 9
             *  blocks).  The rest of the VEC printing is skipped with the flag.*/
            warma_done = 1;
        }
        if (!warma_done) {

        /*  P6.8 — THE INDICES INTO x[], NOTED IN THE WALK THAT ALREADY EXISTS.
         *  The hypothesis block needs to know where in the vector each
         *  parameter lives.  This file already warns, further up, that it has
         *  FOUR walks of the same vector and that adding a fifth loose counting
         *  rule is exactly how the §4.1 bug was opened.  So there is no fifth
         *  walk: it is noted here, where the walking is already going on and
         *  where the walk itself is checked against npar at the end.  0 = that
         *  entry is NOT free -- the structure, -fixb2 or -alpha fixes it -- and
         *  is therefore imposed, not testable.                               */
        int nf_ix = (global_p > 1) ? global_p - 1 : 1;
        int nq_ix = (global_q > 0) ? global_q : 1;
        int **ix_lam = imatrix(1, nser, 1, (r > 0 ? r : 1));
        int **ix_B2  = imatrix(1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
        int **ix_F   = imatrix(1, nf_ix * nser, 1, nser);
        int **ix_Th  = imatrix(1, nq_ix * nser, 1, nser);
        for (int a = 1; a <= nser; a++)
            for (int b = 1; b <= (r > 0 ? r : 1); b++) ix_lam[a][b] = 0;
        for (int a = 1; a <= (s > 0 ? s : 1); a++)
            for (int b = 1; b <= (r > 0 ? r : 1); b++) ix_B2[a][b] = 0;
        for (int a = 1; a <= nf_ix * nser; a++)
            for (int b = 1; b <= nser; b++) ix_F[a][b] = 0;
        for (int a = 1; a <= nq_ix * nser; a++)
            for (int b = 1; b <= nser; b++) ix_Th[a][b] = 0;

        /*  P10 — THE WALK ONLY READS.  It used to read and print at the same
         *  time, which is why the matrices came out interleaved with their
         *  standard errors and there was no table anywhere.  Now it fills these
         *  arrays and records the indices, and the two printing passes below --
         *  the parameter table and the model in VEC form -- work off them.    */
        real **psi_m = NULL;      /* only under -alpha */
        int  **ix_psi = NULL;
        real  *mu_m  = vector(1, nser);
        int   *ix_mu = ivector(1, nser);
        int    nmu   = 0;
        real ***F_m  = tensor(1, nf_ix, 1, nser, 1, nser);
        for (int a = 1; a <= nser; a++) { mu_m[a] = 0.0; ix_mu[a] = 0; }
        for (int k = 1; k <= nf_ix; k++)
            for (int a = 1; a <= nser; a++)
                for (int b = 1; b <= nser; b++) F_m[k][a][b] = 0.0;

        if (global_case == 2) {
            for (int j = 1; j <= r; j++) {
                mu_m[++nmu] = x[ii]; ix_mu[nmu] = ii; ii++;
            }
        } else if (global_case == 3) {
            for (int i = 1; i <= s; i++) {
                if (y2mu_free(i)) { mu_m[++nmu] = x[ii]; ix_mu[nmu] = ii; ii++; }
                else              { mu_m[++nmu] = 0.0;   ix_mu[nmu] = 0; }
            }
            for (int j = 1; j <= r; j++) {
                mu_m[++nmu] = x[ii]; ix_mu[nmu] = ii; ii++;
            }
        }
        /*  -trend: the r slopes of W, after the means (par_blocks).       */
        real tr_m[64]; int ix_tr[64];
        for (int j = 1; j <= r && j < 64; j++) {
            tr_m[j] = global_trend ? x[ii] : 0.0;
            ix_tr[j] = global_trend ? ii : 0;
            if (global_trend) ii++;
        }

        if (global_alpha) {
            /* With alpha = A*psi what x[] carries is psi (sa x r), not Lambda, and
               the walk has to consume sa*r and not M*r -- the printer is the
               other place where the vector's layout can fall out of step, and
               the suite's structural block exists for that.
               Both are printed: psi is what is estimated, Lambda = A*psi is what
               is interpretable, and the standard errors exist only for psi.   */
            /*  Under alpha = A*psi what x[] carries is psi (sa x r), not
             *  Lambda: psi is what is estimated and has standard errors,
             *  Lambda = A*psi is what is interpretable and does not.  Both are
             *  reported, in their own places.                                */
            psi_m = matrix(1, alpha_sa, 1, (r > 0 ? r : 1));
            ix_psi = imatrix(1, alpha_sa, 1, (r > 0 ? r : 1));
            for (int i = 1; i <= alpha_sa; i++)
                for (int j = 1; j <= r; j++) {
                    psi_m[i][j] = x[ii]; ix_psi[i][j] = ii; ii++;
                }
            for (int i = 1; i <= nser; i++)
                for (int j = 1; j <= r; j++) {
                    real acc = 0.0;
                    for (int kk = 1; kk <= alpha_sa; kk++)
                        acc += alpha_A[lam2inp(i)][kk] * psi_m[kk][j];
                    Lam_m[i][j] = acc;
                }
        } else {
            for (int i = 1; i <= nser; i++)
                for (int j = 1; j <= r; j++) {
                    Lam_m[i][j] = x[ii];
                    ix_lam[i][j] = ii;
                    ii++;
                }
        }
        /* Under -diagar / -diagma only the M diagonal entries are carried in
           x[] (see calc_nparametrs and vec_shootx), so the walk must consume
           M, not M*M, while still displaying the full matrix.                */
        int nf = (global_p > 1) ? global_p - 1 : 0;
        for (int k = 1; k <= nf; k++)
            for (int i = 1; i <= nser; i++)
                for (int j = 1; j <= nser; j++)
                    if (f_free(i, j)) { ix_F[(k-1)*nser + i][j] = ii;
                                        F_m[k][i][j] = x[ii++]; }
        /*  With -mawarma the free block is only the r*r entries at the top
         *  left; the upper-right block is determined by B2 (which this walk has
         *  not read yet) and the lower s rows are zero.  They are stored here
         *  and printed after B2.  This is the FOURTH walk of the same vector,
         *  and it is exactly where the previous version of this block fell out
         *  of step and published a Theta nobody had estimated.               */
        real ***Th_m = tensor(1, (global_q > 0 ? global_q : 1), 1, nser, 1, nser);
        for (int k = 1; k <= (global_q > 0 ? global_q : 1); k++)
            for (int i = 1; i <= nser; i++)
                for (int j = 1; j <= nser; j++) Th_m[k][i][j] = 0.0;
        for (int k = 1; k <= global_q; k++) {
            if (mawarma_on()) {
                for (int i = 1; i <= r; i++)
                    for (int j = 1; j <= r; j++) {
                        ix_Th[(k-1)*nser + i][j] = ii; Th_m[k][i][j] = x[ii++]; }
            } else if (marow_on()) {
                for (int i = 1; i <= r; i++)
                    for (int j = 1; j <= nser; j++) {
                        ix_Th[(k-1)*nser + i][j] = ii; Th_m[k][i][j] = x[ii++]; }
            } else if (global_matri) {
                for (int i = 1; i <= r; i++)
                    for (int j = 1; j <= nser; j++) {
                        ix_Th[(k-1)*nser + i][j] = ii; Th_m[k][i][j] = x[ii++]; }
                for (int i = r + 1; i <= nser; i++)
                    for (int j = r + 1; j <= nser; j++) {
                        ix_Th[(k-1)*nser + i][j] = ii; Th_m[k][i][j] = x[ii++]; }
            } else if (global_diag_ma) {
                for (int i = 1; i <= nser; i++) {
                    ix_Th[(k-1)*nser + i][i] = ii; Th_m[k][i][i] = x[ii++]; }
            } else {
                for (int i = 1; i <= nser; i++)
                    for (int j = 1; j <= nser; j++) {
                        ix_Th[(k-1)*nser + i][j] = ii; Th_m[k][i][j] = x[ii++]; }
            }
        }
        /* The engine concentrates the covariance scale: it calls elf with
           sigma2 = 1, so the block carried in x[] is identified only up to a
           positive constant.  Report it as Q (what is estimated) and the
           innovation covariance separately as sigma2 * Q, as drvarma does.   */
        real **Qm = matrix(1, nser, 1, nser);
        for (int i = 1; i <= nser; i++)
            for (int j = 1; j <= nser; j++) Qm[i][j] = 0.0;

        /* Same layout as vec_shootx: Q[1][1] = 1, then the variance ratios on
           the diagonal (i >= 2), then the off-diagonals by rows.              */
        Qm[1][1] = 1.0;
        int ix_q0 = ii;                 /* where Q's free block starts in x[] */
        for (int i = 2; i <= nser; i++) Qm[i][i] = x[ii++];
        if (!global_diag_cov)
            for (int i = 2; i <= nser; i++)
                for (int j = 1; j < i; j++) Qm[i][j] = Qm[j][i] = x[ii++];


        /* B2 is stored COLUMN-major in x[] (vec_shootx and init_guess both
           write `for j in 1..r { for i in 1..s }`), so it must be read back in
           that order before being displayed row by row.  Both printers below
           use this one copy, so they cannot disagree.                        */
        real **B2m = matrix(1, s, 1, (r > 0 ? r : 1));
        for (int j = 1; j <= r; j++)
            for (int i = 1; i <= s; i++) {
                if (!b2_held(i)) ix_B2[i][j] = ii;
                B2m[i][j] = b2_held(i) ? B2_fixed[i][j] : x[ii++];
            }
        /*  -xlink: the transfer block, last in x[] */
        int ix_xl0 = ii;
        ii += xl_npar();


        /*  -mawarma: now that B2 has been read, Theta is completed with the
         *  structure it inherits: [T11  T11*B2' ; 0  0].                     */
        if (mawarma_on())
            for (int k = 1; k <= global_q; k++) {
                for (int i = 1; i <= r; i++)
                    for (int jj2 = 1; jj2 <= s; jj2++) {
                        real acc = 0.0;
                        for (int i2 = 1; i2 <= r; i2++)
                            acc += Th_m[k][i][i2] * B2m[jj2][i2];
                        Th_m[k][i][r + jj2] = acc;
                    }
            }

        /*  THE WALK, CHECKED AGAINST npar.  This is the structural guard the
         *  suite's block exists for; a replacement of the printing block on
         *  2026-08-24 left it without a body, so the banner below hung off it
         *  and the error was never emitted.  Found by running the IPC case
         *  side by side with drvarma.                                        */
        if (ii != npar + 1)
            fprintf(stderr, "ERROR output: consumed %d of %d parameters\n",
                    ii - 1, npar);

        banner("Estimated Parameters and Standard Deviations");
        /*  THE MAP TO THE READER'S NOTATION, once, at the head of the table.
         *  Someone who knows VECM knows Johansen's
         *      nabla Y_t = alpha beta' Y_{t-1} + Gamma_1 nabla Y_{t-1} + ...
         *  and this program writes the same model in Mauricio's symbols.  Two
         *  of them coincide (Gamma_i = F_i, beta = B) and one does NOT: the
         *  equation carries -Lambda, so alpha = -Lambda and a POSITIVE Lambda
         *  is error-correcting.  A reader who takes Lambda for alpha reads the
         *  adjustment backwards, which is the one misreading this table has to
         *  make impossible.                                                  */
        fprintf(outputv, "%s",
            global_case == 3
              ? "\n  Gamma(L) (nabla Y_t - delta) = alpha (beta'Y_{t-1} - E[W])"
                " + Theta(L) A_t,  delta = E[nabla Y]\n"
              : "\n  nabla Y_t = alpha (beta'Y_{t-1} - E[W]) + Gamma_1 nabla Y_{t-1}"
                " + ... + Theta(L) A_t\n");
        fprintf(outputv,
            "  with Pi = alpha beta' of rank r, and beta = [I_r ; beta_2] "
            "normalised on Y1.\n"
            "  Johansen's notation throughout: a NEGATIVE alpha is "
            "error-correcting.\n");
        fprintf(outputv, "  Rows read `equation <- regressor`; D.x is nabla x,"
                         " A.x its innovation,\n"
                         "  and ec%s the equilibrium error%s beta'Y_{t-1}.\n",
                (r == 1) ? " is" : "1..ec%d are", (r == 1) ? "" : "s");
        fprintf(outputv, "\n%-28s %13s %12s %8s %7s\n",
                "Parameter", "Estimate", "Std.Error", "t-stat", "p-val");
        fprintf(outputv, "%s\n", DASHBAR);
        {
            char lb[64];
            int a2, b2i, k2;

            if (nmu > 0) {
                fprintf(outputv, "\n%s\n", (global_case == 3)
                        ? "Means: of the drift in nabla Y2, and of the "
                          "equilibrium error"
                        : "Mean of the equilibrium error (the constant "
                          "restricted to the relation)");
                for (a2 = 1; a2 <= nmu; a2++) {
                    if (global_case == 3 && a2 <= s)
                        snprintf(lb, sizeof lb, "  E[D.%s]", sname(a2));
                    else
                        snprintf(lb, sizeof lb, "  E[ec%d]",
                                 (global_case == 3) ? a2 - s : a2);
                    if (ix_mu[a2] > 0) par_row(lb, mu_m[a2], dev[ix_mu[a2]]);
                    else fprintf(outputv, "%-28s %13.6f   held: no drift (%s)\n",
                                 lb, 0.0, case_pre ? "-case pre" : "-nodrift");
                }
                if (global_trend) {
                    fprintf(outputv, "Trend of the equilibrium error (case 4), per "
                                     "period, t centred at mid-sample\n");
                    for (a2 = 1; a2 <= r; a2++) {
                        snprintf(lb, sizeof lb, "  trend[ec%d]", a2);
                        par_row(lb, tr_m[a2], dev[ix_tr[a2]]);
                    }
                }
            }
            if (r > 0) {
                fprintf(outputv, "\nAdjustment coefficients, alpha (M x r)"
                                 "   [negative = error-correcting]\n");
                if (psi_m) {
                    fprintf(outputv, "  free part psi of alpha = A*psi; "
                                     "Lambda = A*psi is in the model block\n");
                    for (a2 = 1; a2 <= alpha_sa; a2++)
                        for (b2i = 1; b2i <= r; b2i++) {
                            snprintf(lb, sizeof lb, "  psi[%d <- ec%d]", a2, b2i);
                            par_row(lb, psi_m[a2][b2i], dev[ix_psi[a2][b2i]]);
                        }
                } else {
                    for (a2 = 1; a2 <= nser; a2++)
                        for (b2i = 1; b2i <= r; b2i++)
                            /*  a2 walks the .inp's order; inp2lam takes it to
                             *  the row of Lambda that belongs to that series. */
                            if (ix_lam[inp2lam(a2)][b2i]) {
                                snprintf(lb, sizeof lb, "  D.%s <- ec%d",
                                         sname(a2), b2i);
                                /*  REPORTED AS alpha, WHICH IS -Lambda.  The
                                 *  parameter vector carries Mauricio's Lambda
                                 *  and the model equation carries -Lambda; a
                                 *  reader who knows VECM knows alpha, and the
                                 *  cast to a VARMA is an internal matter.
                                 *  Only the sign moves: the standard error is
                                 *  the same and so is |t|.                   */
                                par_row(lb, -Lam_m[inp2lam(a2)][b2i],
                                        dev[ix_lam[inp2lam(a2)][b2i]]);
                            }
                }
            }
            if (r > 0 && s > 0) {
                fprintf(outputv, "\nCointegrating vectors, beta = [I_r ; "
                                 "beta_2], normalised on the Y1 block\n");
                if (global_fixb2)
                    fprintf(outputv, "  beta_2 held fixed: imposed, not "
                                     "estimated\n");
                else if (b2row_on)
                    for (a2 = 1; a2 <= s; a2++)
                        if (b2_held(a2))
                            fprintf(outputv, "  %s in ec*: held at %g by "
                                    "-fixb2row (imposed, not estimated)\n",
                                    sname(a2), b2row_val[a2]);
                for (b2i = 1; b2i <= r; b2i++)
                    for (a2 = 1; a2 <= s; a2++)
                        if (ix_B2[a2][b2i]) {
                            snprintf(lb, sizeof lb, "  %s in ec%d",
                                     sname(a2), b2i);
                            par_row(lb, B2m[a2][b2i], dev[ix_B2[a2][b2i]]);
                        }
            }
            if (xl_on) {
                /*  -xlink: omega and delta per link, its gain nu(1) with a
                 *  delta-method s.e., and the Wald of the whole link.  The
                 *  transfer relates the LEVELS: nu(1) is the long-run effect
                 *  of a unit of X (w units) on the level of the output.     */
                int c0 = ix_xl0, kk, jj;
                fprintf(outputv, "\nTransfer-function inputs, on the LEVELS of "
                                 "the output:\n  nu(B) = omega(B)/delta(B) B^b,  "
                                 "omega(B) = w0 - w1 B - ...,  delta(B) = 1 - "
                                 "d1 B - ...\n");
                for (kk = 1; kk <= xl_nlnk; kk++) {
                    struct xl_link *L = &xl_lnk[kk];
                    int np_k = L->s + 1 + L->r, *ixw = ivector(1, np_k);
                    real om1 = 0.0, de1 = 1.0, g, gse = 0.0, *gr = vector(1, np_k);
                    for (jj = 0; jj <= L->s; jj++) {
                        snprintf(lb, sizeof lb, "  %s <- %s  w%d",
                                 sname(L->out), xl_inname[L->in], jj);
                        par_row(lb, x[c0 + jj], dev[c0 + jj]);
                        om1 += (jj == 0) ? x[c0] : -x[c0 + jj];
                        ixw[jj + 1] = c0 + jj;
                    }
                    for (jj = 1; jj <= L->r; jj++) {
                        snprintf(lb, sizeof lb, "  %s <- %s  d%d",
                                 sname(L->out), xl_inname[L->in], jj);
                        par_row(lb, x[c0 + L->s + jj], dev[c0 + L->s + jj]);
                        de1 -= x[c0 + L->s + jj];
                        ixw[L->s + 1 + jj] = c0 + L->s + jj;
                    }
                    g = om1 / de1;
                    for (jj = 0; jj <= L->s; jj++) gr[jj + 1] = ((jj == 0) ? 1.0 : -1.0) / de1;
                    for (jj = 1; jj <= L->r; jj++) gr[L->s + 1 + jj] = om1 / (de1 * de1);
                    if (cov) {
                        int a, b;
                        for (a = 1; a <= np_k; a++)
                            for (b = 1; b <= np_k; b++)
                                gse += gr[a] * cov[ixw[a]][ixw[b]] * gr[b];
                        gse = (gse > 0.0) ? sqrt(gse) : 0.0;
                    }
                    fprintf(outputv, "  gain nu(1) of %s <- %s: %12.6f  (s.e. %.6f, "
                            "delta method)  (b, r, s) = (%d, %d, %d)\n",
                            sname(L->out), xl_inname[L->in], g, gse,
                            L->b, L->r, L->s);
                    {
                        char title[160];
                        snprintf(title, sizeof title, "Link %s <- %s, whole:",
                                 sname(L->out), xl_inname[L->in]);
                        emit_wald(x, cov, ixw, np_k, title,
                                  "H0: every omega and delta of the link = 0",
                                  "REJECT H0 -> the input enters the output.",
                                  "Cannot reject H0 -> the link can be dropped.");
                    }
                    free_vector(gr, 1, np_k);
                    free_ivector(ixw, 1, np_k);
                    c0 += np_k;
                }
            }
            if (nf > 0) {
                fprintf(outputv, "\nShort-run dynamics, Gamma(k) on "
                                 "nabla Y_{t-k}\n");
                /*  BUG-23: F is internal in rows AND columns (vec_shootx builds
                 *  Phi* = Cbar F Cinv), so the table walks the .inp's order and
                 *  reads the internal entry, as the Lambda rows already did. */
                for (k2 = 1; k2 <= nf; k2++)
                    for (a2 = 1; a2 <= nser; a2++)
                        for (b2i = 1; b2i <= nser; b2i++) {
                            int A2 = inp2lam(a2), B2c = inp2lam(b2i);
                            if (ix_F[(k2-1)*nser + A2][B2c]) {
                                snprintf(lb, sizeof lb, "  D.%s <- D.%s(-%d)",
                                         sname(a2), sname(b2i), k2);
                                par_row(lb, F_m[k2][A2][B2c],
                                        dev[ix_F[(k2-1)*nser + A2][B2c]]);
                            }
                        }
            }
            if (global_q > 0) {
                /*  BUG-49: at an MA root on the unit circle the fit is a
                 *  constrained point, not an interior maximum, and the MA's
                 *  standard errors are not defined (Theorem 10's Omega > 0
                 *  fails; VILL printed 0.000001 and t = -3e5).  They are not
                 *  printed; the roots and the diagnosis below say why.      */
                real mm_ma = 1.0e12;
                report_operator_roots("", vp->theta, vp->m, vp->q, &mm_ma, 1);
                ma_on_boundary = (mm_ma < 1.0001);
                fprintf(outputv, "\nMoving average on the innovations, "
                                 "Theta(k)%s\n", ma_on_boundary
                        ? "   [an MA root is ON the unit circle: s.e. not defined, "
                          "see the roots below]" : "");
                for (k2 = 1; k2 <= global_q; k2++)
                    for (a2 = 1; a2 <= nser; a2++)
                        for (b2i = 1; b2i <= nser; b2i++) {
                            int A2 = inp2lam(a2), B2c = inp2lam(b2i);
                            if (ix_Th[(k2-1)*nser + A2][B2c]) {
                                snprintf(lb, sizeof lb, "  D.%s <- A.%s(-%d)",
                                         sname(a2), sname(b2i), k2);
                                par_row(lb, Th_m[k2][A2][B2c],
                                        ma_on_boundary ? 0.0
                                        : dev[ix_Th[(k2-1)*nser + A2][B2c]]);
                            }
                        }
            }
            /*  Q, not Sigma: the engine concentrates the scale, so what is
             *  estimated is the covariance up to a positive constant, with
             *  Q[1,1] = 1.  Sigma = sigma2 * Q is in the model block.        */
            {
                int qi = ix_q0;
                /*  BUG-18: the Q block of the vector is internal -- its [1][1] is
                 *  Y1's variance -- so its rows keep that order (each row IS a
                 *  parameter, with its standard error) and are NAMED through
                 *  lam2inp.                                                  */
                fprintf(outputv, "\nInnovation covariance up to scale, Q "
                                 "(Sigma = sigma2 * Q, var %s = 1)\n",
                        sname(lam2inp(1)));
                for (a2 = 2; a2 <= nser; a2++) {
                    snprintf(lb, sizeof lb, "  var %s", sname(lam2inp(a2)));
                    par_row(lb, Qm[a2][a2], dev[qi++]);
                }
                if (!global_diag_cov)
                    for (a2 = 2; a2 <= nser; a2++)
                        for (b2i = 1; b2i < a2; b2i++) {
                            snprintf(lb, sizeof lb, "  cov %s, %s",
                                     sname(lam2inp(a2)), sname(lam2inp(b2i)));
                            par_row(lb, Qm[a2][b2i], dev[qi++]);
                        }
            }
        }
        fprintf(outputv, "%s\n", DASHBAR);
        fprintf(outputv, "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 "
                         "'.' 0.1 ' ' 1\n");
        if (!se_from_fdhess)
            fprintf(outputv, "  ! standard errors from the BFGS-accumulated "
                             "factor%s\n",
                    global_fdhess ? " (-fdhess could not be used, see the header)"
                                  : "; -fdhess uses the Hessian at the optimum");


        /* ================= THE MODEL, IN VECM FORM ======================= */
        /*  Reported in Johansen's notation, which is the one a reader of this
         *  class of model already has: alpha = -Lambda, beta = B, Gamma_k =
         *  F_k.  The internal parameterisation is Mauricio's -- it is what
         *  makes the exact likelihood computable -- and it stays in the
         *  parameter vector, in the source and in docs/MODEL.md.  A report
         *  that speaks the algorithm's notation charges the reader for an
         *  internal decision.                                                */
        banner("Vector Error Correction Model");
        /*  BUG-45: in case 3 the equation omitted the drift.  The model is
         *  mean-corrected in nabla Y: Gamma(L)(nabla Y_t - delta), with
         *  delta = E[nabla Y] -- E[nabla Y2] on the Y2 block and, since W has
         *  no drift, -B2' E[nabla Y2] on the Y1 block.  The estimation was
         *  right; only the printed form was not.                            */
        if (global_case == 3)
            fprintf(outputv, "\n  Gamma(L) (nabla Y_t - delta) = alpha (beta'Y_{t-1} - E[W])"
                             " + Theta(L) A_t\n"
                             "  Gamma(L) = I - Gamma_1 L - ...,  Theta(L) = I - Theta1 L"
                             " - ... - Theta_q L^q,   Pi = alpha beta'\n"
                             "  delta = E[nabla Y], the drift: E[nabla Y2] on the Y2 block,"
                             " -beta_2' E[nabla Y2] on Y1\n\n");
        else
            fprintf(outputv, "\n  nabla Y_t = alpha (beta'Y_{t-1} - E[W])"
                             " + Gamma_1 nabla Y_{t-1} + ... + Theta(L) A_t\n"
                             "  Theta(L) = I - Theta1 L - ... - Theta_q L^q,"
                             "   Pi = alpha beta'\n\n");
        if (nmu > 0) {
            fprintf(outputv, "%s vector:\n",
                    global_case == 3 ? "E[nablaY2] and E[W]" : "E[W]");
            for (int a2 = 1; a2 <= nmu; a2++)
                fprintf(outputv, "  %12.6f\n", mu_m[a2]);
        }
        if (global_case == 3 && nmu >= s) {
            fprintf(outputv, "delta = E[nabla Y] (M), rows in the .inp's order:\n");
            for (int a2 = 1; a2 <= nser; a2++) {
                real d = 0.0;
                if (a2 <= s) d = mu_m[a2];
                else {
                    for (int i3 = 1; i3 <= s; i3++) d -= B2m[i3][a2 - s] * mu_m[i3];
                    if (global_trend) d += tr_m[a2 - s];
                }
                fprintf(outputv, "  %12.6f\n", d);
            }
        }
        fprintf(outputv, "alpha matrix (M x r), rows in the .inp's order:\n");
        for (int a2 = 1; a2 <= nser; a2++) {
            fprintf(outputv, "  ");
            for (int b2i = 1; b2i <= r; b2i++)
                fprintf(outputv, "%12.6f", -Lam_m[inp2lam(a2)][b2i]);
            fprintf(outputv, "\n");
        }
        for (int k2 = 1; k2 <= nf; k2++) {
            fprintf(outputv, "Gamma(%d) matrix, rows and columns in the .inp's order:\n", k2);
            for (int a2 = 1; a2 <= nser; a2++) {
                fprintf(outputv, "  ");
                for (int b2i = 1; b2i <= nser; b2i++)
                    fprintf(outputv, "%12.6f", F_m[k2][inp2lam(a2)][inp2lam(b2i)]);
                fprintf(outputv, "\n");
            }
        }
        for (int k2 = 1; k2 <= global_q; k2++) {
            fprintf(outputv, "theta(%d) matrix%s:\n", k2,
                    mawarma_on()  ? "   (inherited: the nabla Y2 block has no MA"
                                    " and the cross block is beta-determined)"
                  : global_matri  ? "   (the equilibrium errors do not enter the"
                                    " nabla Y2 equations with a lag)"
                  : marow_on()    ? "   (the nabla Y2 block carries no MA of "
                                    "its own)" : "");
            for (int a2 = 1; a2 <= nser; a2++) {
                fprintf(outputv, "  ");
                for (int b2i = 1; b2i <= nser; b2i++)
                    fprintf(outputv, "%12.6f", Th_m[k2][inp2lam(a2)][inp2lam(b2i)]);
                fprintf(outputv, "\n");
            }
        }
        free_tensor(Th_m, 1, (global_q > 0 ? global_q : 1), 1, nser, 1, nser);
        fprintf(outputv, "beta_2 matrix (s x r)%s:\n",
                b2row_on ? "  [rows held by -fixb2row at the values given]" :
                global_fixb2 ? (global_fixb2_given
                    ? "  [FIXED at the value given]"
                    : "  [FIXED at its static-OLS estimate: data-chosen, so an"
                      " LR against the free model is not valid]") : "");
        for (int a2 = 1; a2 <= s; a2++) {
            fprintf(outputv, "  ");
            for (int b2i = 1; b2i <= r; b2i++)
                fprintf(outputv, "%12.6f", B2m[a2][b2i]);
            fprintf(outputv, "\n");
        }

        /*  beta IN THE .inp's ROW ORDER.  B = [I_r ; B2] is written in the
         *  INTERNAL order [Y1 ; Y2]; every other matrix in this report -- alpha,
         *  Gamma, theta, Pi, the responses -- has its rows in the .inp's order,
         *  [Y2 block ; Y1 block].  Printing beta in the internal one paired it
         *  row by row with the wrong series.  Found on 2026-08-24 by the
         *  beta' gain = 0 certificate, which is the only thing in the program
         *  that multiplies beta by something in the .inp's order.            */
        fprintf(outputv, "beta matrix (M x r), rows in the .inp's order:\n");
        for (int row = 1; row <= nser; row++) {
            fprintf(outputv, "  ");
            for (int c = 1; c <= r; c++)
                fprintf(outputv, "%12.6f",
                        (row <= s) ? B2m[row][c]
                                   : ((row - s == c) ? 1.0 : 0.0));
            fprintf(outputv, "\n");
        }

        /* ---- Pi = Lambda * B', y sus autovalores ------------------------- */
        /* Pi is the long-run matrix and, unlike Lambda and B, is INVARIANT to the
           normalisation: any reparameterisation Lambda -> Lambda*G,
           B -> B*G^-T leaves Pi unchanged.  That is why it is what to look at
           when comparing fits, and why it is printed here.                   */
        if (r > 0) {
            real **Pi = matrix(1, nser, 1, nser);
            real *wr = vector(1, nser), *wi = vector(1, nser);
            int a, b, j;
            for (a = 1; a <= nser; a++)
                for (b = 1; b <= nser; b++) {
                    real acc = 0.0;
                    for (j = 1; j <= r; j++) {
                        /*  Both indices in the .inp's order, as everywhere
                         *  else: Lambda's rows already are, and beta's have to
                         *  be put in it (see the beta matrix above).         */
                        real Bbj = (b <= s) ? B2m[b][j]
                                            : ((b - s == j) ? 1.0 : 0.0);
                        /*  alpha = -Lambda, so Pi = alpha beta' = -Lambda B'.
                         *  Reported with Johansen's sign, which is the one
                         *  whose eigenvalues read as adjustment speeds.      */
                        acc -= Lam_m[inp2lam(a)][j] * Bbj;
                    }
                    Pi[a][b] = acc;
                }
            fprintf(outputv, "Pi = alpha beta' matrix (M x M), the long run:\n");
            for (a = 1; a <= nser; a++) {
                fprintf(outputv, "  ");
                for (b = 1; b <= nser; b++) fprintf(outputv, "%12.6f", Pi[a][b]);
                fprintf(outputv, "\n");
            }
            {   /* eigenvalues, on a copy: eigenqr destroys its argument */
                real **Pc = matrix(1, nser, 1, nser);
                for (a = 1; a <= nser; a++) for (b = 1; b <= nser; b++)
                    Pc[a][b] = Pi[a][b];
                eigenqr(Pc, nser, wr, wi);
                fprintf(outputv, "  eigenvalues of Pi:");
                for (a = 1; a <= nser; a++) {
                    if (fabs(wi[a]) < 1.0e-12) fprintf(outputv, "  %.6f", wr[a]);
                    else fprintf(outputv, "  %.6f%+.6fi", wr[a], wi[a]);
                }
                fprintf(outputv, "\n");
                free_matrix(Pc, 1, nser, 1, nser);
            }
            /*  Pi has rank r BY CONSTRUCTION here, so its M-r zero
             *  eigenvalues prove nothing about the rank; the instrument is
             *  -lrtest.  Said in one line because it changes how the number is
             *  read; argued in docs/USAGE.md 4.                              */
            fprintf(outputv, "  ! Pi has rank r by construction: the %d zero "
                             "eigenvalue%s prove%s nothing about r (use -lrtest)\n",
                    nser - r, (nser - r == 1) ? "" : "s",
                    (nser - r == 1) ? "s" : "");
            free_vector(wi, 1, nser);
            free_vector(wr, 1, nser);
            free_matrix(Pi, 1, nser, 1, nser);
        }
        /*  BUG-18: Qm is internal ([Y1 ; Y2]); printed in the .inp's order,
         *  like every other matrix of the report.  The normalisation is on
         *  Y1's variance, which is no longer the [1][1] entry of this print. */
#define QIN(a, b) (((a) >= (b)) ? Qm[(a)][(b)] : Qm[(b)][(a)])
        fprintf(outputv, "Q matrix (lower triangle, rows and columns in the "
                         ".inp's order; var %s = 1 by normalisation):\n",
                sname(lam2inp(1)));
        for (int a2 = 1; a2 <= nser; a2++) {
            fprintf(outputv, "  ");
            for (int b2i = 1; b2i <= a2; b2i++)
                fprintf(outputv, "%12.6f", QIN(inp2lam(a2), inp2lam(b2i)));
            fprintf(outputv, "\n");
        }
        fprintf(outputv, "Sigma = sigma2 * Q, in the .inp's order:\n");
        for (int a2 = 1; a2 <= nser; a2++) {
            fprintf(outputv, "  ");
            for (int b2i = 1; b2i <= a2; b2i++)
                fprintf(outputv, "%12.6f", vp->sigma2 * QIN(inp2lam(a2), inp2lam(b2i)));
            fprintf(outputv, "\n");
        }
        /*  P2 — |Sigma| IS THIS PROGRAM'S HOMOLOGATION CRITERION and until
         *  2026-08-20 it did not print it: it had to be worked out by hand from
         *  the matrix above, rounded to six decimals, and that manual
         *  arithmetic is precisely where the register lost track of three of
         *  the four rows of its beta exit criterion.  A program that declares a
         *  criterion and does not emit it forces somebody else to compute it,
         *  and whoever computes it gets it wrong.  See
         *  docs/PLAN_PRODUCCION.md P2.
         *
         *  log|Sigma| is given too, which is what enters the likelihood and the
         *  only readable thing when |Sigma| goes to 1e-30 with a large M.     */
        {
            real **Sm = matrix(1, nser, 1, nser);
            real d1 = 0.0, d2 = 0.0; int ifd = 0;
            for (int i = 1; i <= nser; i++)
                for (int j = 1; j <= nser; j++)
                    Sm[i][j] = vp->sigma2 * (j <= i ? Qm[i][j] : Qm[j][i]);
            choldcp(Sm, nser, &d1, &d2, &ifd);
            if (ifd == 0) {
                /*  choldcp leaves the determinant as d1 * 2^d2 -- mantissa and
                 *  exponent apart, so as not to overflow -- and ALREADY
                 *  accumulates the squares of the factor's diagonal (nlatools.c:
                 *  `*d1 *= mat[j][j] * mat[j][j]`), so that IS det(Sigma) and
                 *  not its factor's.  Squaring it again gave 6.06e-06 where the
                 *  register has 0.002461, which is its exact square: the first
                 *  version of this line did just that.                       */
                real logdet = log(fabs(d1)) + d2 * log(2.0);
                fprintf(outputv, "  |Sigma| = %.10g      log|Sigma| = %.6f\n",
                        exp(logdet), logdet);
                if (!quiet_mode)
                    printf("  |Sigma| = %.10g\n", exp(logdet));
            } else {
                fprintf(outputv, "  |Sigma|: not computed (Sigma is not "
                                 "positive definite here)\n");
            }
            free_matrix(Sm, 1, nser, 1, nser);
        }

        /* ---- Triangularizacion Sigma = P D P' ----------------------------- */
        /* LDL' decomposition of the innovation covariance: P unit lower
           triangular, D diagonal.  With A_t = P A*_t, the innovations A*_t are
           UNCORRELATED (cov = D), so the system premultiplied by P^-1 can be
           read equation by equation: that is what allows one to speak of a
           single equation without dragging in the contemporaneous correlation
           of the others.  BVECM section 4; the legacy did it for the bivariate
           case only.

           IMPORTANT, and that is why it is said in the output: the ordering is
           that of the .inp's COLUMNS.  Another order gives another P.  It is
           the same class of silent decision as the choice of the Y1 block.    */
        {
            real **Sg = matrix(1, nser, 1, nser);
            real **P  = matrix(1, nser, 1, nser);
            real  *D  = vector(1, nser);
            int a, b, k, ok_ldl = 1;
            for (a = 1; a <= nser; a++)
                for (b = 1; b <= nser; b++)
                    Sg[a][b] = vp->sigma2 * QIN(inp2lam(a), inp2lam(b));   /* BUG-18 */
            for (a = 1; a <= nser; a++)
                for (b = 1; b <= nser; b++) P[a][b] = (a == b) ? 1.0 : 0.0;
            for (b = 1; b <= nser; b++) {
                real acc = Sg[b][b];
                for (k = 1; k < b; k++) acc -= P[b][k] * P[b][k] * D[k];
                D[b] = acc;
                if (D[b] <= 0.0) { ok_ldl = 0; break; }
                for (a = b + 1; a <= nser; a++) {
                    real s2 = Sg[a][b];
                    for (k = 1; k < b; k++) s2 -= P[a][k] * P[b][k] * D[k];
                    P[a][b] = s2 / D[b];
                }
            }
            if (ok_ldl) {
                fprintf(outputv, "P matrix (Sigma = P D P', P unit lower "
                                 "triangular):\n");
                for (a = 1; a <= nser; a++) {
                    fprintf(outputv, "  ");
                    for (b = 1; b <= a; b++) fprintf(outputv, "%12.6f", P[a][b]);
                    fprintf(outputv, "\n");
                }
                fprintf(outputv, "D vector:\n  ");
                for (a = 1; a <= nser; a++) fprintf(outputv, "%12.6f", D[a]);
                fprintf(outputv, "\n");
                fprintf(outputv, "  own share D_i/Sigma_ii:\n  ");
                for (a = 1; a <= nser; a++)
                    fprintf(outputv, "%11.1f%%", 100.0 * D[a] / Sg[a][a]);
                fprintf(outputv, "\n");
                fprintf(outputv, "  ! the ordering is the .inp's column order: "
                             "another order gives another P\n");
            } else {
                fprintf(outputv, "P, D: not computed (Sigma is not positive "
                                 "definite at the optimum)\n");
            }
            free_vector(D, 1, nser);
            free_matrix(P, 1, nser, 1, nser);
            free_matrix(Sg, 1, nser, 1, nser);
        }
        free_matrix(Qm, 1, nser, 1, nser);


        banner("Cointegration Diagnostics");
        /*  The rank condition, beside the roots and for the same reason: it
         *  says whether the point it stopped at is a model of the rank that was
         *  asked for or of another.  It is recomputed at the last evaluation,
         *  which is the one vec_shootx left just before.                     */
        if (global_r > 0 && global_q > 0 && granger_sv >= 0.0) {
            fprintf(outputv,
                "\nRank condition (Granger): sigma_s(alpha_perp' Theta(1)) "
                "= %.3e\n", granger_sv);
            if (granger_sv < global_rankadm_tol) {
                /*  AND TO THE TERMINAL AS WELL.  A fit that denies its own rank
                 *  is not a worse fit: it is the fit of another model, and
                 *  whoever runs the program has to find out without opening the
                 *  .out.  It is the decision of step 4 of the plan: the default
                 *  CALCULATION does not move -- no recorded result moves -- but
                 *  the PRESENTATION stops offering as an answer something the
                 *  theory does not license (docs/THEORY.md, corollary 5.1).  */
                if (!quiet_mode)
                    printf("\n  *** WARNING: sigma_s(alpha_perp' Theta(1)) "
                           "= %.3e < %.1e\n"
                           "%s"
                           "      Its standard errors and any LR against it "
                           "need not have\n"
                           "      their usual distribution.\n"
                           "%s"
                           "      See the ladder:  drvec <file> %d %d %d "
                           "-specs\n", granger_sv, global_rankadm_tol,
                           /*  Two tiers (BUG-46).  Only G = 0 is a denial of
                            *  the rank; below the floor it is a distance, and
                            *  the floor is a convention.                    */
                           (granger_sv < GRANGER_ZERO)
                             ? "      This fit DENIES THE RANK it was estimated "
                               "at: it is not a\n"
                               "      worse fit, it is the fit of another model.\n"
                             : "      This fit is NEAR the set where it would deny "
                               "its rank (a\n"
                               "      common factor at z = 1); the floor is a "
                               "convention, -rankadm sets it.\n",
                           /*  Where it comes from: the free class CONTAINS points
                            *  the model does not admit, and -marow only removes
                            *  one route to them (BUG-48).                  */
                           (!global_q || global_marow || global_mawarma || global_warma)
                             ? ""
                             : "      The free MA class CONTAINS points the model "
                               "does not admit;\n"
                               "      -marow removes one route to them (BUG-48).\n",
                           global_p, global_q, global_r);
                fprintf(outputv,
                  "  *** %s the floor %.2f (a convention; -rankadm sets it).\n"
                  "  That matrix is what makes the long-run impact\n"
                  "  C(1) = B_perp (Lambda_perp' Gamma B_perp)^-1 Lambda_perp'\n"
                  "  Theta(1) have rank M-r.  Where it degenerates the FITTED\n"
                  "  model denies the rank it was estimated at -- it says r and\n"
                  "  its parameters leave no stochastic trend.  The estimate is\n"
                  "  then on the edge of the region the model class allows, so\n"
                  "  standard errors and LR statistics do not have their usual\n"
                  "  distributions there.  -rankadm refuses such points; -mawarma\n"
                  "  makes them unreachable by construction.  See\n"
                  "  docs/HOMOLOGATION.md 4h.\n",
                  (granger_sv < GRANGER_ZERO) ? "ZERO to working precision, under"
                                              : "Not zero, but below", global_rankadm_tol);
            } else
                fprintf(outputv,
                  "  admissible (tolerance %.2f)\n", global_rankadm_tol);
        }
        gate_contract(vp);
        /* ---- Diagnostic on the normalisation ------------------------------ */
        /* B = [I_r ; B2] assumes the Y1 block genuinely appears in every
           cointegrating relation.  If it does not, B2 blows up and the model
           becomes a silent trap: the fit "works" and describes something else.
           Mauricio warns about it (p. 3648) and refers to Luukkonen et al.
           (1999) and Kurozumi (2005); Melard, Roy and Saidi avoid it by using
           the null space of Phi(1) instead of a normalisation.
           The measure used here is unit-free: in W = Y1 + B2'Y2 each series
           weighs |coefficient| * sd(series), so the Y1 block's share of that
           total weight is reported.  A tiny share says the relation is not
           really about Y1 and that the normalisation is forced.              */
        if (r > 0 && s > 0) {
            real *sdY2 = vector(1, s);
            int a, t, j;
            for (a = 1; a <= s; a++) {
                real m1 = 0.0, v = 0.0;
                for (t = 1; t <= nobs; t++) m1 += Y2_levels[t][a];
                m1 /= nobs;
                for (t = 1; t <= nobs; t++) v += (Y2_levels[t][a] - m1)
                                               * (Y2_levels[t][a] - m1);
                sdY2[a] = sqrt(v / (nobs > 1 ? nobs - 1 : 1));
            }
            fprintf(outputv, "\nNormalisation, share of the weight carried by "
                             "the Y1 block:\n");
            for (j = 1; j <= r; j++) {
                real m1 = 0.0, v = 0.0, w1, w2 = 0.0, share;
                for (t = 1; t <= nobs; t++) m1 += datamat[t][s + j];
                m1 /= nobs;
                for (t = 1; t <= nobs; t++) v += (datamat[t][s + j] - m1)
                                               * (datamat[t][s + j] - m1);
                w1 = sqrt(v / (nobs > 1 ? nobs - 1 : 1));      /* coef = 1 */
                for (a = 1; a <= s; a++) w2 += fabs(B2m[a][j]) * sdY2[a];
                share = (w1 + w2 > 0.0) ? w1 / (w1 + w2) : 1.0;
                fprintf(outputv, "  relation %d: the Y1 block carries %5.1f%% "
                                 "of the weight", j, 100.0 * share);
                if (share < 0.05) {
                    fprintf(outputv, "   <-- DUBIOUS\n");
                    fprintf(stderr,
                        "WARNING: cointegrating relation %d barely involves the "
                        "Y1 block (%.1f%%).\n"
                        "         beta = [I_r; beta_2] normalises on Y1, so this fit "
                        "may be describing\n"
                        "         a relation among the other series with an "
                        "inflated B2.  Consider\n"
                        "         reordering the columns of the .inp.  See "
                        "docs/MODEL.md 5.4.\n", j, 100.0 * share);
                } else fprintf(outputv, "\n");
            }
            free_vector(sdY2, 1, s);
        }

        /*  P6.8 — the hypotheses about the relations, here: the walk above has
         *  just been checked against npar, so the indices it noted are the good
         *  ones.                                                             */
        hypothesis_block(x, cov, ix_lam, ix_B2, ix_F, ix_Th);
        free_imatrix(ix_Th,  1, nq_ix * nser, 1, nser);
        free_imatrix(ix_F,   1, nf_ix * nser, 1, nser);
        free_imatrix(ix_B2,  1, (s > 0 ? s : 1), 1, (r > 0 ? r : 1));
        free_imatrix(ix_lam, 1, nser, 1, (r > 0 ? r : 1));
        /*  P10 — what the reading pass allocated.  It is not decoration: the
         *  suite's valgrind block caught all five the moment they existed.   */
        free_tensor(F_m, 1, nf_ix, 1, nser, 1, nser);
        free_ivector(ix_mu, 1, nser);
        free_vector(mu_m, 1, nser);
        if (psi_m)  free_matrix(psi_m, 1, alpha_sa, 1, (r > 0 ? r : 1));
        if (ix_psi) free_imatrix(ix_psi, 1, alpha_sa, 1, (r > 0 ? r : 1));

        /*  The diagnosis of the residuals and the roots close the report on
         *  the FIT; the forecast is a product and comes after them.          */
        /*  P11 — the impulse responses and the decomposition, in drvarma's
         *  place in the report: after the hypotheses and before the residual
         *  diagnosis.  The horizon is drvarma's rule too -- 10 on a short
         *  sample, 20 otherwise -- so that the two reports are read the same
         *  way.                                                              */
        level_irf_fevd(vp, B2m, (nobs < 40) ? 10 : 20);

        residual_diagnostics(vp);
        operator_roots(vp);
        /*  AND THE PER-SERIES DIAGNOSIS, which is the suite's and which drvec
         *  simply did not have: the residual moments with their dates, the
         *  standardized plot, the histogram, and the ACF and PACF with their
         *  bands and Ljung-Box.  Vendored whole in src/diagnose.c; drvec only
         *  calls it.  Measured against drvarma on the three-CPI case, this is
         *  most of the 1665 lines drvec was missing.                         */
        diagnose_ybar(vp);

        /*  P5 — the forecast, here: it is the last place where B2m is still
         *  alive and where the fit is already made and diagnosed.            */
        if (global_fcast > 0) forecast_vec(vp, B2m, global_fcast, 0.95);
        if (global_estwin > 0) {
            if (global_fcast < 1)
                fprintf(outputv, "\n(-estwin needs a horizon: give -f H)\n");
            else if (rolling_eval(x, global_estwin, global_fcast) != 0)
                fprintf(outputv, "\n(-estwin %d leaves no room for %d steps)\n",
                        global_estwin, global_fcast);
        }

        free_matrix(Lam_m, 1, nser, 1, (r > 0 ? r : 1));
        free_matrix(B2m, 1, s, 1, (r > 0 ? r : 1));
        }   /* !warma_done */

        /*  -warma prints its own block and skips the one above, but the
         *  residuals and the operators are the same in both parameterisations,
         *  so its report gets them here.  (Inside the branch, the -warma path
         *  lost them: 145 lines to 69.)                                      */
        if (warma_done) {
            residual_diagnostics(vp); operator_roots(vp);
            diagnose_ybar(vp);
        }
        if (warma_done) free_matrix(Lam_m, 1, nser, 1, (r > 0 ? r : 1));

    } else {
        /*  P1 — THE FAILURE IS VISIBLE FROM OUTSIDE.  Until 2026-08-20 this
         *  branch wrote the diagnosis into the .out and returned 0 to the
         *  shell, so a script chaining fits could not tell an estimated model
         *  from one that was not.  It exits with 2, which in the port's
         *  convention is "recognised but cannot be honoured", and 1 is reserved
         *  for a USAGE error.  Mind the distinction that matters: termcode 3 --
         *  "last global step failed to locate a lower point" -- is NOT this.
         *  It is an explained stop, with its convergence note, and it still
         *  exits with 0: turning it into a failure would mark as an error most
         *  of the fits this program publishes.                               */
        estimation_failed = 1;
        fprintf(outputv, "\nESTIMATION FAILED: ifault = %d\n", ifault);
        switch (ifault) {
            case 1: fprintf(outputv, "  Q not positive definite\n"); break;
            case 2: fprintf(outputv, "  AR operator has unit root\n"); break;
            case 3: fprintf(outputv, "  AR operator non-stationary\n"); break;
            case 4: fprintf(outputv, "  MA operator non-invertible\n"); break;
            case 5: fprintf(outputv, "  Numerical problem\n"); break;
            case 6: fprintf(outputv, "  Error in vec_shootx()\n"); break;
        }
        fprintf(stderr, "drvec: estimation failed (ifault = %d); "
                        "see %s\n", ifault, outname);
    }
}

/*****************************************************************************/
/*  P8 — run_rungs: the -rungs mode, out of main().  A mode is a program in   */
/*  its own right -- it estimates, prints and exits -- so it belongs in a     */
/*  function of its own and not at the same indentation as the argument       */
/*  parsing.  Returns the process exit code.                                  */
/*****************************************************************************/
static int run_rungs(void)
{
        const int NR = 3;
        const int DAR[3] = {1, 1, 0}, DMA[3] = {1, 1, 0}, DCOV[3] = {1, 0, 0};
        const char *NAME[3] = { "0  F, Theta, Sigma diagonal",
                                "1  Sigma free",
                                "2  F and Theta free" };
        real *ll  = vector(0, NR - 1);
        int  *npr = ivector(0, NR - 1);
        int  *good = ivector(0, NR - 1);
        int k, M = nser, nf = (global_p > 1) ? global_p - 1 : 0, q = global_q;
        /*  BUG-45: USAGE.md says each rung starts from the optimum of the one
         *  below, and the code cold-started every rung.  It now carries F,
         *  Theta, Sigma and mu up, as the ladder does (fit_r0_ladder), and
         *  fits with the same search.                                        */
        real ***Fc = tensor(1, (nf > 0 ? nf : 1), 1, M, 1, M);
        real ***Tc = tensor(1, (q > 0 ? q : 1), 1, M, 1, M);
        real **Sc = matrix(1, M, 1, M), *muc = vector(1, M);
        int have_below = 0;

        macheps = cmacheps();
        global_r = 0;

        fprintf(outputv, "\n=== The ladder: rungs at r = 0 ===\n");
        printf("\nThe ladder, rungs at r = 0:\n");

        for (k = 0; k < NR; k++) {
            struct Tvarma vr;
            real *xr, *devr, **covr;
            int np, ifr = 0;

            global_diag_ar = DAR[k]; global_diag_ma = DMA[k];
            global_diag_cov = DCOV[k];
            build_y2_levels();
            np = calc_nparametrs();
            xr = vector(1, np); devr = vector(1, np);
            covr = matrix(1, np, 1, np);
            vr.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
            if (have_below) pack_r0(xr, Fc, Tc, Sc, muc);
            else            init_guess(xr, np);
            {
                real llk = 0.0, s2k;
                ifr = fit_search(xr, np, devr, covr, &llk, &s2k, 1, 0, NULL);
                vr.logelf = llk;
            }
            good[k] = (ifr == 0);
            npr[k]  = np;
            ll[k]   = good[k] ? vr.logelf : 0.0;
            have_below = 0;
            {   /*  reserved here, whatever happened, and released once below */
                real llk = vr.logelf;
                int ifs = 0;
                vec_shootx(xr, &vr, &ifs, 1, 0);
                vr.logelf = llk;
            }
            if (good[k]) {
                int i2, j2, k2;
                {
                    /*  at r = 0 Phi* = F, Theta* = Theta, Sigma* = Sigma      */
                    for (k2 = 1; k2 <= nf; k2++) for (i2 = 1; i2 <= M; i2++) for (j2 = 1; j2 <= M; j2++) Fc[k2][i2][j2] = vr.phi[k2][i2][j2];
                    for (k2 = 1; k2 <= q; k2++)  for (i2 = 1; i2 <= M; i2++) for (j2 = 1; j2 <= M; j2++) Tc[k2][i2][j2] = vr.theta[k2][i2][j2];
                    for (i2 = 1; i2 <= M; i2++) { muc[i2] = vr.mu[i2]; for (j2 = 1; j2 <= M; j2++) Sc[i2][j2] = vr.qq[i2][j2]; }
                    have_below = 1;
                }
            }
            printf("  rung %s : %s (ifault=%d)\n", NAME[k],
                   good[k] ? "ok" : "estimation failed", ifr);
            /*  Rung 0 is the base: its contract is demanded right here, which is
             *  where it is being built upon.                                 */
            if (k == 0 && good[k]) gate_contract(&vr);
            vec_shootx(xr, &vr, &ifr, 0, 1);           /* liberar */
            free_matrix(covr, 1, np, 1, np);
            free_vector(devr, 1, np);
            free_vector(xr, 1, np);
        }

        fprintf(outputv, "\n  rung                          npar         logL"
                         "         AIC         BIC\n");
        fprintf(outputv, "  --------------------------------------------------"
                         "-------------------\n");
        for (k = 0; k < NR; k++) {
            if (!good[k]) {
                fprintf(outputv, "  %-28s  --   estimation failed\n", NAME[k]);
                continue;
            }
            fprintf(outputv, "  %-28s %4d %12.4f %11.4f %11.4f\n", NAME[k],
                    npr[k], ll[k],
                    (-2.0 * ll[k] + 2.0 * npr[k]) / nobs,
                    (-2.0 * ll[k] + npr[k] * log((real) nobs)) / nobs);
        }

        fprintf(outputv, "\n  step        LR = 2*[L(k+1) - L(k)]    df    "
                         "p-value\n");
        fprintf(outputv, "  ------------------------------------------------"
                         "-----\n");
        for (k = 0; k + 1 < NR; k++) {
            real lr;
            int df;
            if (!good[k] || !good[k + 1]) {
                fprintf(outputv, "  %d -> %d      not available\n", k, k + 1);
                continue;
            }
            lr = 2.0 * (ll[k + 1] - ll[k]);
            df = npr[k + 1] - npr[k];
            fprintf(outputv, "  %d -> %d   %14.4f       %3d   %9.4f%s\n",
                    k, k + 1, lr, df,
                    (df > 0 && lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0,
                    (lr < -1.0e-6) ? "   *** NEGATIVE: the wider fit is worse,"
                                     " so it did not converge" : "");
        }

        fprintf(outputv,
            "\n  These are ordinary nested comparisons: the restricted model is\n"
            "  an INTERIOR point of the wider one, so the log-likelihood cannot\n"
            "  fall and the statistic is chi2 on the stated degrees of freedom.\n"
            "  The next rung -- adding the VEC matrix, r = 1 -- is not ordinary\n"
            "  in either respect: the null sits ON the boundary of the alternative\n"
            "  and B2 is unidentified under it.  It is reported by -lrtest, with\n"
            "  a parametric bootstrap available for its distribution.\n");

        free_ivector(good, 0, NR - 1);
        free_ivector(npr, 0, NR - 1);
        free_vector(ll, 0, NR - 1);
        free_tensor(Fc, 1, (nf > 0 ? nf : 1), 1, M, 1, M);
        free_tensor(Tc, 1, (q > 0 ? q : 1), 1, M, 1, M);
        free_matrix(Sc, 1, M, 1, M); free_vector(muc, 1, M);
        fclose(outputv);
        printf("Done. Output written to %s\n", outputf);
        cleanup_names(outputf, inputf, base_name);
        return 0;
}

/*****************************************************************************/
/*  P8 — run_specs: the -specs mode, out of main().  Returns the exit code.   */
/*****************************************************************************/
static int run_specs(void)
{
        const int NS = 5;
        const char *NM[5] = { "warma  ", "mawarma", "marow  ", "matri  ", "free   " };
        real ll[5], gg[5], mam[5], b2v[5];
        int  npv[5], okv[5], tcv[5], adm[5];
        int  k, M = nser;

        macheps = cmacheps();
        if (global_r < 1) {
            fprintf(stderr, "ERROR: -specs needs r >= 1\n");
            exit(1);
        }
        fprintf(outputv, "\n=== The specification ladder ===\n");
        printf("\nSpecification ladder:\n");

        for (k = 0; k < NS; k++) {
            struct Tvarma vs;
            real *xs, *devs, **covs;
            int np, ifs = 0, s2 = M - global_r;

            global_warma = (k == 0); global_mawarma = (k == 1);
            global_marow = (k == 2);  global_matri   = (k == 3);
            build_y2_levels();
            np = calc_nparametrs();
            xs = vector(1, np); devs = vector(1, np);
            covs = matrix(1, np, 1, np);
            vs.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
            init_guess(xs, np);
            if (global_warma) {
                /*  the same admissible shrinkage the normal route uses         */
                static const real shr[6] = { 1.0, 0.8, 0.5, 0.3, 0.1, 0.0 };
                int nm2, nl2, nmid2, nt2, nfw = (global_p > 1) ? global_p - 1 : 0;
                int nar, i2, mi;
                real *ar0;
                par_blocks(&nm2, &nl2, &nmid2, &nt2);
                nar = nl2 + nfw * M * global_r;
                ar0 = vector(1, (nar > 0 ? nar : 1));
                for (i2 = 1; i2 <= nar; i2++) ar0[i2] = xs[nm2 + i2];
                vec_shootx(xs, &vs, &ifs, 1, 0);
                for (mi = 0; mi < 6; mi++) {
                    real p1, p2, p3; int ife = 0, ifc = 0;
                    for (i2 = 1; i2 <= nar; i2++) xs[nm2 + i2] = shr[mi] * ar0[i2];
                    vec_shootx(xs, &vs, &ifc, 0, 0);
                    if (ifc != 0) continue;
                    elf(vs.m, vs.n, vs.p, vs.q, vs.mu, vs.phi, vs.theta, vs.qq,
                        vs.w, 1.0, vs.xitol, TRUE, vs.a, &p1, &p2, &p3, &ife);
                    if (ife == 0) break;
                }
                free_vector(ar0, 1, (nar > 0 ? nar : 1));
                vec_shootx(xs, &vs, &ifs, 0, 1);
            }
            vec_shootx(xs, &vs, &ifs, 1, 0);
            est(&vec_shootx, np, xs, devs, covs, 500, 200, 1e-5, 1e-7,
                vs.xitol, vs.a, &vs.sigma2, &vs.logelf, &ifs);
            okv[k] = (ifs == 0);
            npv[k] = np;
            ll[k]  = okv[k] ? vs.logelf : 0.0;
            tcv[k] = termcode_from_out(outputf);
            gg[k] = -1.0; mam[k] = -1.0; b2v[k] = 0.0;
            if (okv[k]) {
                real mm = 1.0e30;
                vec_shootx(xs, &vs, &ifs, 0, 0);      /* recover the fit  */
                report_operator_roots("MA", vs.theta, vs.m, vs.q, &mm, 1);
                mam[k] = (mm < 1.0e29) ? mm : -1.0;
                for (int j2 = 1; j2 <= global_r; j2++)
                    b2v[k] = global_fixb2 ? B2_fixed[1][j2]
                           : xs[np - s2 * global_r + (j2 - 1) * s2 + 1];
                if (global_warma) {
                    /*  in Ybar coordinates one has to come back first        */
                    real **B2w = matrix(1, s2, 1, global_r);
                    real **Lw = matrix(1, M, 1, global_r);
                    real ***Fw = tensor(1, (global_p > 1 ? global_p - 1 : 1), 1, M, 1, M);
                    real ***Tw = tensor(1, (global_q > 0 ? global_q : 1), 1, M, 1, M);
                    real **Sw = matrix(1, M, 1, M);
                    for (int j2 = 1; j2 <= global_r; j2++)
                        for (int i2 = 1; i2 <= s2; i2++)
                            B2w[i2][j2] = global_fixb2 ? B2_fixed[i2][j2]
                                        : xs[np - s2*global_r + (j2-1)*s2 + i2];
                    for (int k2 = 1; k2 <= (global_q > 0 ? global_q : 1); k2++)
                        for (int i2 = 1; i2 <= M; i2++)
                            for (int j2 = 1; j2 <= M; j2++) Tw[k2][i2][j2] = 0.0;
                    warma_inverse(&vs, B2w, Lw, Fw, Tw, Sw);
                    gg[k] = granger_smin(Lw, B2w, Tw, M, global_r, global_q);
                    free_matrix(Sw, 1, M, 1, M);
                    free_tensor(Tw, 1, (global_q > 0 ? global_q : 1), 1, M, 1, M);
                    free_tensor(Fw, 1, (global_p > 1 ? global_p - 1 : 1), 1, M, 1, M);
                    free_matrix(Lw, 1, M, 1, global_r);
                    free_matrix(B2w, 1, s2, 1, global_r);
                } else gg[k] = granger_sv;
            }
            /*  gg = -1 is "does not apply" (q = 0: Theta(1) = I, the condition
             *  holds trivially), not a failure: with q = 0 every non-warma
             *  rung was marked NO (BUG-45).                                */
            adm[k] = (okv[k] && (gg[k] < 0.0 || gg[k] >= global_rankadm_tol));
            printf("  %s : %s\n", NM[k], okv[k] ? "ok" : "fallo");
            vec_shootx(xs, &vs, &ifs, 0, 1);
            free_matrix(covs, 1, np, 1, np);
            free_vector(devs, 1, np);
            free_vector(xs, 1, np);
        }
        global_warma = global_mawarma = global_marow = global_matri = 0;

        fprintf(outputv,
          "\n  spec      npar          logL   term        G     MAmin       B2  adm\n"
          "  ------------------------------------------------------------------------\n");
        for (k = 0; k < NS; k++) {
            const char *tn = (tcv[k] == 1) ? "grad" : (tcv[k] == 2) ? "step"
                           : (tcv[k] == 3) ? "lower" : (tcv[k] == 0) ? "--" : "gave up";
            if (!okv[k]) {
                fprintf(outputv, "  %s  %4d   estimation failed\n", NM[k], npv[k]);
                continue;
            }
            fprintf(outputv, "  %s  %4d %13.4f  %-6s ", NM[k], npv[k], ll[k], tn);
            if (gg[k] >= 0.0) fprintf(outputv, "%8.3e ", gg[k]);
            else              fprintf(outputv, "%8s ", "n/a");
            if (mam[k] >= 0.0) fprintf(outputv, "%8.3f ", mam[k]);
            else               fprintf(outputv, "%8s ", "n/a");
            fprintf(outputv, "%8.4f  %s\n", b2v[k], adm[k] ? "yes" : "NO");
        }

        fprintf(outputv,
          "\n  step               LR     df   verdict\n"
          "  ----------------------------------------------------------------------\n");
        for (k = 0; k + 1 < NS; k++) {
            real lr;
            int df = npv[k+1] - npv[k];
            if (!okv[k] || !okv[k+1]) {
                fprintf(outputv, "  %s -> %s   not available\n", NM[k], NM[k+1]);
                continue;
            }
            lr = 2.0 * (ll[k+1] - ll[k]);
            fprintf(outputv, "  %s -> %s %8.3f %4d   ", NM[k], NM[k+1], lr, df);
            if (df == 0 && fabs(lr) < 1.0e-6)
                fprintf(outputv, "the same model (no MA left to restrict)\n");
            else if (lr < -1.0e-6)
                fprintf(outputv, "NEGATIVE: the wider fit is worse, so it did "
                                 "not converge\n");
            else if (!adm[k] || !adm[k+1])
                fprintf(outputv, "no p-value: %s is not admissible, so the "
                                 "statistic is not chi2 (-matest)\n",
                        adm[k] ? NM[k+1] : NM[k]);
            /*  An MA root ON the unit circle is the other boundary (BUG-49):
             *  the same threshold -lrtest uses.  Once BUG-46 made the rank
             *  condition right, Milan's matri and free rungs passed it with
             *  MAmin = 1.000, and a chi2 p = 0.0000 came out of a boundary.  */
            else if ((mam[k] >= 0.0 && mam[k] < 1.0001) ||
                     (mam[k+1] >= 0.0 && mam[k+1] < 1.0001))
                fprintf(outputv, "no p-value: %s has an MA root on the unit "
                                 "circle, so the statistic is not chi2 (-matest)\n",
                        (mam[k] >= 0.0 && mam[k] < 1.0001) ? NM[k] : NM[k+1]);
            else
                fprintf(outputv, "chi2 p = %.4f\n",
                        (df > 0) ? gsl_cdf_chisq_Q(lr, df) : 1.0);
        }

        fprintf(outputv,
          "\n  READ THE LAST COLUMN FIRST.  By Theorem 3 of docs/THEORY.md the\n"
          "  process has cointegrating rank r if and only if\n"
          "  rank(Lambda_perp' Theta(1)) = M - r, and by Theorem 4 the set where\n"
          "  that fails lies INSIDE the one the optimiser searches.  A rung with\n"
          "  G = 0 is not a worse fit of this model: it is a fit of another\n"
          "  one, whose rank is not the rank it was estimated at, and a small G\n"
          "  is near that.  Corollary 5.1\n"
          "  then removes the usual distributions, which is why no chi2 p-value\n"
          "  is printed for a comparison involving it.  The floor used here is\n"
          "  %.1e, a convention (-rankadm sets it): only G = 0 denies the rank.\n"
          "  Nor is one printed where a rung's MA has a root on the unit circle\n"
          "  (MAmin = 1.000): that is the other boundary (BUG-49).\n",
          global_rankadm_tol);

        fclose(outputv);
        printf("Done. Output written to %s\n", outputf);
        cleanup_names(outputf, inputf, base_name);
        return 0;
}

/*****************************************************************************/
/*  P8 — run_ma_ar_test: the -matest and -artest modes, out of main().  They  */
/*  are one function because they are one procedure with two pairs: the same  */
/*  bootstrap under the restricted model, with a different H0 and H1.         */
/*  Returns the exit code.                                                    */
/*****************************************************************************/
static int run_ma_ar_test(void)
{
        /*  Two tests with the same machinery and a different pair:
         *    -matest   H0 = mawarma  against  H1 = free    (half the MA)
         *    -artest   H0 = warma    against  H1 = mawarma (half the AR)
         *  The second is the one step 5 of the plan asked for, and its reason
         *  for simulating is NOT the same: here the unrestricted model IS
         *  admissible (G between 0.90 and 1.00 over the whole bank), so it is
         *  not 4i's boundary problem.  It is that the restriction
         *  Gamma_i = m_i alpha' is of REDUCED RANK on F_i, and the LR of a rank
         *  restriction is not chi2 when the true rank may be below the one the
         *  restriction allows -- and in five of the eight pairs F1 comes out
         *  nearly of rank one, which leaves m_i nearly unidentified.         */
        int use_ar = (global_artest > 0);
        int reps = use_ar ? global_artest : global_matest;
        int k0 = use_ar ? 0 : 1, k1 = use_ar ? 1 : 4;
        int np0, np1, ok0 = 0, ok1 = 0, nb, df;
        real l0, l1, lr, cv[3] = {0,0,0}, pv = -1.0;
        real *x0, *dev0, **cov0;
        struct Tvarma v0;
        int ifr = 0;

        macheps = cmacheps();
        if (global_r < 1 || global_q < 1) {
            fprintf(stderr, "ERROR: -matest/-artest need r >= 1 and q >= 1\n");
            exit(1);
        }
        /* [1] the restricted model, which is H0 -- and it is kept, because the simulation comes from it */
        set_spec(k0);
        build_y2_levels();
        np0 = calc_nparametrs();
        x0 = vector(1, np0); dev0 = vector(1, np0); cov0 = matrix(1, np0, 1, np0);
        v0.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
        init_guess(x0, np0);
        /*  P12: the search makes the start admissible first -- the ladder this
         *  fit lacked (BUG-35) -- and keeps the best of its starts.          */
        ifr = fit_search(x0, np0, dev0, cov0, &v0.logelf, &v0.sigma2, 1, 0, NULL);
        ok0 = (ifr == 0); l0 = v0.logelf;

        /* [2] the unrestricted one */
        set_spec(k1);
        l1 = fit_ll(global_r, &ok1);
        np1 = calc_nparametrs();
        df = np1 - np0;

        if (!ok0 || !ok1) {
            fprintf(outputv, "\n%s: one of the two fits failed "
                             "(restricted %s, unrestricted %s); no test.\n",
                    use_ar ? "-artest" : "-matest",
                    ok0 ? "ok" : "failed", ok1 ? "ok" : "failed");
        } else {
            lr = 2.0 * (l1 - l0);
            /*  The header and the numbers go in TWO calls and not in one with a
             *  conditional format string: with the ternary, the literals that
             *  followed concatenate onto only one of the two branches and the
             *  %f are left without a format in the other.  It compiles, and the
             *  numbers are lost in silence -- which is how they were lost the
             *  first time this was written.                                  */
            fprintf(outputv, "%s", use_ar
                ? "\n=== The triangular short-run dynamics against free F ===\n\n"
                  "  H0: Gamma_i = M_i alpha' -- every lag enters through W,\n"
                  "      which is what the triangular class implies\n"
                  "  H1: F_i free (the moving-average structure held in both)\n\n"
                : "\n=== The inherited moving average against the free one ===\n\n"
                  "  H0: Theta = [T11  T11*B2' ; 0  0], the structure a WARMA\n"
                  "      process implies for its VEC representation\n"
                  "  H1: Theta free\n\n");
            fprintf(outputv,
                "  restricted     logL = %15.10f   (%d parameters)\n"
                "  unrestricted   logL = %15.10f   (%d parameters)\n"
                "  LR = 2*[L(H1) - L(H0)] = %.4f   on %d df\n"
                "  chi2 p-value, FOR REFERENCE ONLY          = %.4f\n",
                l0, np0, l1, np1, lr, df,
                (lr > 0.0) ? gsl_cdf_chisq_Q(lr, df) : 1.0);
            printf("  restricted %.6f   free %.6f   LR %.4f (%d df)\n",
                   l0, l1, lr, df);

            nb = bootstrap_ma(x0, np0, reps, cv, &pv, lr, k0, k1);
            if (nb >= 10) {
                fprintf(outputv,
                  "\n  Parametric bootstrap under H0, %d of %d replications "
                  "usable:\n"
                  "    critical values   10%%: %8.4f   5%%: %8.4f   1%%: %8.4f\n"
                  "    bootstrap p-value = %.4f\n", nb, reps,
                  cv[0], cv[1], cv[2], pv);
                fprintf(outputv, "    verdict: %s\n",
                    (lr > cv[2]) ? "reject H0 at 1%" :
                    (lr > cv[1]) ? "reject H0 at 5%" :
                    (lr > cv[0]) ? "reject H0 at 10%" : "H0 not rejected");
                /*  The verdict is read off the CRITICAL VALUES and the p-value
                 *  also counts the observed statistic, so with few
                 *  replications the two can end up on different sides of a
                 *  threshold.  Saying so is cheaper than picking one and
                 *  keeping quiet.                                            */
                if ((lr > cv[0] && pv > 0.10) || (lr > cv[1] && pv > 0.05) ||
                    (lr > cv[2] && pv > 0.01))
                    fprintf(outputv,
                      "    (the p-value counts the observed statistic itself, so\n"
                      "     with B = %d it can sit on the other side of the same\n"
                      "     threshold as the critical value; both are printed)\n",
                      reps);
                printf("  bootstrap: p = %.4f  (%d/%d replicas)\n", pv, nb,
                       reps);
            } else {
                fprintf(outputv, "\n  Parametric bootstrap: only %d usable "
                                 "replications; no critical values.\n", nb);
            }
            fprintf(outputv, use_ar ?
              "\n  WHY THE BOOTSTRAP AND NOT THE chi2, and the reason is NOT\n"
              "  the one in -matest.  Here the unrestricted model IS admissible\n"
              "  -- the rank condition reads 0.90 to 1.00 across the bank -- so\n"
              "  this is not a boundary of the model class.  It is that\n"
              "  Gamma_i = M_i alpha' is a REDUCED-RANK restriction on F_i, and\n"
              "  the likelihood ratio for a rank restriction is not chi2 when\n"
              "  the true rank may be below the one the restriction allows: in\n"
              "  five of the eight pairs F1 comes out near rank one already, and\n"
              "  M_i is then close to unidentified.  The bootstrap is simulated\n"
              "  FROM THE RESTRICTED FIT, which is H0.\n"
              :
              "\n  WHY THE BOOTSTRAP AND NOT THE chi2.  The unrestricted\n"
              "  optimum on this kind of data stops in the neighbourhood where\n"
              "  the rank condition degenerates -- sigma_min(Lambda_perp'\n"
              "  Theta(1) B_perp) of order 0.01-0.1 against ~1 for a correctly\n"
              "  specified model, with Theta(1) singular to working precision.\n"
              "  An LR whose unrestricted estimate sits on the edge of the\n"
              "  admissible region does not have its asymptotic distribution,\n"
              "  so the chi2 column above is a reference and not a test.  The\n"
              "  bootstrap distribution is simulated FROM THE RESTRICTED FIT,\n"
              "  which is H0, and it inherits the sample size, the\n"
              "  deterministic case and the moving-average structure.\n"
              "  Read it with its floor of 1/(B+1) = %.4f and its Monte Carlo\n"
              "  error sqrt(p(1-p)/B) = %.4f at p = 0.05.\n"
              "  See docs/HOMOLOGATION.md 4g and 4h.\n",
              1.0 / (real) (reps + 1),
              sqrt(0.05 * 0.95 / (real) reps));
        }
        free_matrix(cov0, 1, np0, 1, np0);
        free_vector(dev0, 1, np0);
        free_vector(x0, 1, np0);
        fclose(outputv);
        printf("Done. Output written to %s\n", outputf);
        cleanup_names(outputf, inputf, base_name);
        return 0;
}

/*****************************************************************************/
/*  P8 — run_lrtest: the -lrtest mode, out of main().  It estimates every     */
/*  rank from 0 to M-1, so it is the one mode that runs the whole estimation  */
/*  M times over; it prints the sequential table and, with -bootstrap, the    */
/*  critical values.  Returns the exit code.                                  */
/*****************************************************************************/
static int run_lrtest(void)
{
        int M = nser, ok;
        macheps = cmacheps();           /* the engine needs it; [3] is skipped */
        /* Ranks 0..M-1.  r = 0 is the no-cointegration null: Pi = 0, so the
           model is a plain VARMA on nabla Y.  It is the first and most
           important comparison of the sequence.                              */
        real *ll  = vector(0, M - 1);
        int  *npr = ivector(0, M - 1);
        int  *good = ivector(0, M - 1);
        int  *tcr = ivector(0, M - 1);   /* how each rank's winning fit stopped */
        int  *bnd = ivector(0, M - 1);   /* 1 if its MA sits on the boundary (BUG-49) */

        fprintf(outputv, "\n=== Sequential LR test for the cointegration rank ===\n");
        printf("\nSequential LR test for the cointegration rank:\n");

        /*  BUG-43: THE SCALE.  On rao7 in raw units (the s.d. of the first
         *  differences spans 0.0074 .. 2702) the optimiser stopped where it
         *  started and the rank sequence read 38.57 / 29.96 / 6.14; the same
         *  data with each column rescaled gave 80.09 / 39.09 / 11.84.  The
         *  model is equivariant to Y -> D Y with D diagonal, and the jacobian
         *  n * sum log d_i is the same at every rank, so the LR does not move:
         *  the fits are made on rescaled data (a power of ten per series, so
         *  no digit is lost) and the logLs reported in the original units.
         *  Not done when something on the command line is in the original
         *  units and would have to be rescaled too (a given B2, an alpha file,
         *  a seed).                                                          */
        real *dsc = vector(1, M), jac_sc = 0.0;
        int scaled = 0;
        for (int i2 = 1; i2 <= M; i2++) dsc[i2] = 1.0;
        if (!global_fixb2_given && !b2row_on && !alpha_file && !global_seed
            && !global_seedb2
            && nobs_raw > 2) {
            for (int i2 = 1; i2 <= M; i2++) {
                real m1 = 0.0, v = 0.0, sd;
                int T = nobs_raw - 1;
                for (int t = 2; t <= nobs_raw; t++) m1 += rawmat[t][i2] - rawmat[t-1][i2];
                m1 /= T;
                for (int t = 2; t <= nobs_raw; t++) {
                    real d = rawmat[t][i2] - rawmat[t-1][i2] - m1; v += d * d;
                }
                sd = sqrt(v / (T > 1 ? T - 1 : 1));
                /*  Only a GROSS mis-scaling, two orders or more: equivariance
                 *  makes the optimum the same, but not the optimiser's path,
                 *  and on UKconsumption (changes ~0.01) rescaling landed rank
                 *  1 on an optimum 0.40 lower.  Same threshold as the SCALE
                 *  warning of a single fit.                                 */
                if (sd > 0.0 && fabs(log10(sd)) > 2.0) {
                    dsc[i2] = pow(10.0, -floor(log10(sd) + 0.5));
                    scaled = 1;
                }
            }
            if (scaled) {
                for (int t = 1; t <= nobs_raw; t++)
                    for (int i2 = 1; i2 <= M; i2++) rawmat[t][i2] *= dsc[i2];
                fprintf(outputv, "  Internal scaling (the LR does not depend on it; "
                                 "logLs below are in the\n  original units, the "
                                 "search tables in the scaled ones):");
                for (int i2 = 1; i2 <= M; i2++)
                    fprintf(outputv, " %s x %g", series_names ? series_names[i2] : "y",
                            dsc[i2]);
                fprintf(outputv, "\n");
            }
        }

        /* With -bootstrap one has to be able to SIMULATE from each rank's fit,
           so its parameter vector is kept instead of freed.                   */
        real **xkeep = NULL; int *npkeep = NULL;
        if (global_boot > 0) {
            xkeep  = (real **) malloc((size_t)(M + 1) * sizeof(real *));
            npkeep = ivector(0, M - 1);
            for (int rr = 0; rr <= M - 1; rr++) { xkeep[rr] = NULL; npkeep[rr] = 0; }
        }

        for (int rr = 0; rr <= M - 1; rr++) {
            global_r = rr;
            build_y2_levels();
            int np = calc_nparametrs();
            real *xr   = vector(1, np);
            real *devr = vector(1, np);
            real **covr = matrix(1, np, 1, np);
            real llr = 0.0, s2r = 0.0;
            struct search_out so;
            int ifr;
            /*  P12: every rank is the best of several starts, and -multistart
             *  is honoured here too (BUG-25).                               */
            init_guess(xr, np);
            /*  P12: the ladder as the start of every rank >= 1; the rungs
             *  below the rank are computed once and cached.               */
            if (rr > 0 && ladder_wanted())
                gate_profile_seed(xr, np);
            else if (rr == 0 && ladder_wanted())
                ladder_seed_r0(xr, np);
            fprintf(outputv, "\n--- rank r = %d ---\n", rr);
            ifr = fit_search(xr, np, devr, covr, &llr, &s2r,
                             (global_multistart > 1) ? global_multistart : 1,
                             1, &so);
            ok = (ifr == 0);
            good[rr] = ok;
            npr[rr]  = np;
            ll[rr]   = ok ? llr : 0.0;
            tcr[rr]  = ok ? so.tc : 0;
            bnd[rr]  = 0;
            if (ok && global_q > 0) {
                struct Tvarma vb; int ifb = 0; real mm = 1.0e12;
                vb.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
                vec_shootx(xr, &vb, &ifb, 1, 0);
                if (ifb == 0) report_operator_roots("", vb.theta, vb.m, vb.q, &mm, 1);
                vec_shootx(xr, &vb, &ifb, 0, 1);
                bnd[rr] = (mm < 1.0001);
            }
            printf("  r = %d : %s (ifault=%d; %d starts, best `%s')\n", rr,
                   ok ? "ok" : "estimation failed", ifr, so.ntried, so.label);
            if (global_boot > 0 && ok) {
                xkeep[rr] = vector(1, np); npkeep[rr] = np;
                for (int i2 = 1; i2 <= np; i2++) xkeep[rr][i2] = xr[i2];
            }
            free_matrix(covr, 1, np, 1, np);
            free_vector(devr, 1, np);
            free_vector(xr, 1, np);
        }

        /*  back to the original units: p_Y(y) = p_{DY}(Dy) |D|^n            */
        if (scaled) {
            for (int i2 = 1; i2 <= M; i2++) jac_sc += log(dsc[i2]);
            jac_sc *= nobs;
            for (int rr = 0; rr <= M - 1; rr++) if (good[rr]) ll[rr] += jac_sc;
        }
        fprintf(outputv, "\n  r    npar        logL         AIC         BIC\n");
        fprintf(outputv, "  ---------------------------------------------------\n");
        for (int rr = 0; rr <= M - 1; rr++) {
            if (!good[rr]) { fprintf(outputv, "  %-4d  --   estimation failed\n", rr); continue; }
            real aic = (-2.0 * ll[rr] + 2.0 * npr[rr]) / nobs;
            real bic = (-2.0 * ll[rr] + npr[rr] * log((real) nobs)) / nobs;
            fprintf(outputv, "  %-4d %4d %12.4f %11.4f %11.4f\n",
                    rr, npr[rr], ll[rr], aic, bic);
        }

        fprintf(outputv,
            "\n  H0: P = r   vs   H1: P = r+1        LR = 2*[L(r+1) - L(r)]\n");
        if (global_alpha)
            fprintf(outputv,
                "  (every rank estimated UNDER the restriction alpha = A*psi,\n"
                "   so this is the rank sequence within H1(r) and NOT the usual\n"
                "   one.  The tabulated critical values do not apply and are not\n"
                "   printed: they are for alpha free.)\n");
        else
            fprintf(outputv,
                "  (asymptotic critical values: %s)\n",
                (global_case == 1) ? "case 1, no constant" :
                (global_case == 2) ? "case 2, restricted constant" :
                                     "case 3 — NOT TABULATED HERE");
        fprintf(outputv, "\n  r    M-r        LR      10%%      5%%      1%%\n");
        fprintf(outputv, "  ---------------------------------------------------\n");
        for (int rr = 0; rr <= M - 2; rr++) {
            if (!good[rr] || !good[rr+1]) {
                fprintf(outputv, "  %-4d  --   (a model in the pair failed)\n", rr);
                continue;
            }
            real lr = 2.0 * (ll[rr+1] - ll[rr]);
            int  g  = M - rr;                    /* common trends under H0 */
            /* Rank r is nested in rank r+1 in the PARAMETERS, but not in the
               exact likelihood: at the boundary Lambda -> 0 of rank r+1 the
               stationary initial-state term of W diverges, so sup L(r+1) can
               sit below L(r) with both fits on their optimum (BUG-26).  A
               negative LR is then a statistic like any other -- below every
               critical value -- not a proof of a failed fit; a fit that did
               stop early is flagged below from its own stop code.            */
            fprintf(outputv, "  %-4d %4d %10.4f", rr, g, lr);
            if (global_alpha) {
                /* Under alpha = A*psi the statistic has ANOTHER distribution: the
                   tables are for a free alpha.  Printing them here would be
                   giving wrong critical values that look right, which is worse
                   than giving none.                                          */
                fprintf(outputv, "        -        -        -   (restricted:"
                                 " tabulated values do not apply)");
            } else if (global_q > 0 && (global_marow || global_mawarma
                                        || global_matri || global_warma)) {
                /*  BUG-27.  The restricted MA classes are written on the blocks
                 *  [nabla Y2 ; W], which do not exist at r = 0, so rank 0 has a
                 *  free Theta and rank r >= 1 the restricted one: the pair is
                 *  not nested, and the table is for a nested pair.  The
                 *  statistic is printed; -bootstrap calibrates it.          */
                fprintf(outputv, "        -        -        -   (MA class restricted"
                                 " at r >= 1 only: not nested; -bootstrap)");
            } else if (global_case != 3 && g >= 1 && g <= LR_MAXTRENDS) {
                const real *cv = (global_case == 1) ? lr_cval_none[g-1]
                                                    : lr_cval_const[g-1];
                fprintf(outputv, " %8.2f %8.2f %8.2f", cv[0], cv[1], cv[2]);
                if      (lr > cv[2]) fprintf(outputv, "   reject H0 at 1%%");
                else if (lr > cv[1]) fprintf(outputv, "   reject H0 at 5%%");
                else if (lr > cv[0]) fprintf(outputv, "   reject H0 at 10%%");
                else                 fprintf(outputv, "   H0 not rejected");
            } else {
                fprintf(outputv, "        -        -        -   (no values)");
            }
            /*  A statistic built on a fit that did not stop on the gradient is
             *  printed, but not without saying so (BUG-25).  tc 3 is often a
             *  true optimum on a flat surface (CONVERGENCE.md): read the search
             *  table of that rank before trusting the verdict.             */
            if (bnd[rr] || bnd[rr+1])
                fprintf(outputv, "   [MA on the boundary at rank %d: no known distribution]",
                        bnd[rr] ? rr : rr + 1);
            if ((tcr[rr] > 2) || (tcr[rr+1] > 2))
                fprintf(outputv, "   [check: rank %d stopped by criterion %d]",
                        (tcr[rr] > 2) ? rr : rr + 1,
                        (tcr[rr] > 2) ? tcr[rr] : tcr[rr+1]);
            /*  Only when both fits stopped on the gradient is a negative LR the
             *  exact likelihood's non-nesting (BUG-26); otherwise a fit simply
             *  did not reach its optimum (rao6, M = 8: -464.7).             */
            if (lr < 0.0)
                fprintf(outputv, "%s", ((tcr[rr] > 2) || (tcr[rr+1] > 2))
                    ? "   [LR < 0: a fit did not reach its optimum]"
                    : "   [LR < 0: exact likelihood, not nested at Lambda = 0]");
            fprintf(outputv, "\n");
        }
        fprintf(outputv,
            "\n  Distribution is the non-standard Johansen one.  That MA terms do\n"
            "  not affect it (Yap and Reinsel 1995, Thm. 3; Mauricio 2006, Remark 5)\n"
            "  is proven for the CONDITIONAL likelihood, a trace test and a strictly\n"
            "  invertible MA -- not for this exact lambda-max, nor at an MA root on\n"
            "  the unit circle (BUG-27).\n"
            "  Values: case 1, MacKinnon-Haug-Michelis (1999) with no deterministic\n"
            "  term; case 2, urca 1.3.4 ca.jo(ecdet=\"const\"), Osterwald-Lenum (1992).\n"
            "  CAUTION: the tables are for Johansen's CONDITIONAL LR.  This one is\n"
            "  the EXACT likelihood, which carries W's stationary initial state and\n"
            "  is not nested at Lambda = 0: under H0 it sits about log T below\n"
            "  Johansen's and can be negative, so against these tables the test is\n"
            "  UNDERSIZED (case 1, T = 500: 3.5%% at a nominal 5%%).  -bootstrap N\n"
            "  calibrates the statistic that is actually computed (BUG-26).\n"
            "  r = 0 is the no-cointegration null (Pi = 0, a plain VARMA on\n"
            "  nabla Y); r = M would be a stationary process in levels and is not\n"
            "  expressible here, so the sequence ends at r = M-1.  Read it with\n"
            "  AIC/BIC, and check the optimizer banner of every rank before\n"
            "  trusting a statistic.\n");
        if (global_alpha)
            fprintf(outputv,
                "  NOTE: with alpha = A*psi imposed, what is being tested at each\n"
                "  step is the rank WITHIN the restricted model.  That is a\n"
                "  legitimate question and a different one; its distribution is\n"
                "  not the tabulated Johansen one.  For the usual rank test, drop\n"
                "  the restriction.\n");

        /* ---- critical values by parametric bootstrap, if asked for --------- */
        if (global_boot > 0) {
            fprintf(outputv,
                "\n  === Parametric bootstrap under H0 (%d replications) ===\n"
                "  The asymptotic values above are known to be optimistic at these\n"
                "  sample sizes: measured, the sequential test over-rejects about\n"
                "  SIX times its nominal level at n = 120, and these percentiles cut\n"
                "  that to four without closing it.  They are empirical\n"
                "  percentiles of the statistic simulated FROM THE FITTED MODEL at\n"
                "  rank r, so they carry the sample size, the deterministic case and\n"
                "  the MA component that the tables cannot.\n", global_boot);
            fprintf(outputv, "\n  r    M-r        LR      10%%      5%%      1%%"
                             "   p-value  reps\n");
            fprintf(outputv, "  ------------------------------------------------"
                             "-----------------\n");
            printf("\nBootstrap under H0 (%d replications per comparison):\n",
                   global_boot);
            {
                char fname[600];
                snprintf(fname, sizeof fname, "%s.boot", out_base);
                boot_dump = fopen(fname, "w");
                if (boot_dump)
                    fprintf(boot_dump, "# drvec %s -bootstrap %d: the rank test, one "
                            "line per replication\nh0,rep,status,lr\n",
                            DRVEC_VERSION, global_boot);
                if (boot_dump) fflush(boot_dump);
            }
            for (int rr = 0; rr <= M - 2; rr++) {
                real cv[3] = {0.0, 0.0, 0.0}, pv = -1.0, lr;
                int reps;
                if (!good[rr] || !good[rr+1] || xkeep == NULL || xkeep[rr] == NULL) {
                    fprintf(outputv, "  %-4d  --   (a model in the pair failed)\n", rr);
                    continue;
                }
                lr = 2.0 * (ll[rr+1] - ll[rr]);
                printf("  r = %d -> %d ...", rr, rr+1); fflush(stdout);
                reps = bootstrap_rank(rr, xkeep[rr], npkeep[rr], global_boot,
                                      cv, &pv, lr);
                if (reps < 10) {
                    fprintf(outputv, "  %-4d %4d %10.4f   only %d usable "
                                     "replications: not reported\n",
                            rr, M - rr, lr, reps);
                    printf(" only %d usable replications\n", reps);
                    continue;
                }
                fprintf(outputv, "  %-4d %4d %10.4f %8.2f %8.2f %8.2f  %7.4f  %4d",
                        rr, M - rr, lr, cv[0], cv[1], cv[2], pv, reps);
                /* The verdict is read off the CRITICAL VALUES, not the p-value,
                   because the p-value has a floor of 1/(reps+1): with 100
                   replications it cannot go below 0.0099, so "rejects at 1%"
                   would be unreachable by construction even if the statistic
                   exceeded the 99th percentile.                              */
                if      (lr > cv[2]) fprintf(outputv, "   reject H0 at 1%%\n");
                else if (lr > cv[1]) fprintf(outputv, "   reject H0 at 5%%\n");
                else if (lr > cv[0]) fprintf(outputv, "   reject H0 at 10%%\n");
                else                 fprintf(outputv, "   H0 not rejected\n");
                printf(" p = %.4f (%d replicas)\n", pv, reps);
            }
            if (boot_dump) {
                fclose(boot_dump);
                boot_dump = NULL;
                fprintf(outputv, "\n  Every replication, with its LR*, is in %s.boot.\n",
                        out_base);
            }
            /* The Monte Carlo error, stated instead of hidden: the plan's
               contingency asked for it to be reported if N had to be small.   */
            fprintf(outputv,
                "\n  Monte Carlo error, said rather than hidden:\n"
                "   - a bootstrap p-value from B replications has standard error\n"
                "     sqrt(p(1-p)/B); at p = 0.05 and B = %d that is %.4f, so read a\n"
                "     p-value near a threshold as undecided, not as a decision;\n"
                "   - and it has a FLOOR of 1/(B+1) = %.4f -- with this B the p-value\n"
                "     cannot go below that however extreme the statistic is, which is\n"
                "     why the verdict above is read from the critical values.  For a\n"
                "     p-value that can resolve 1%%, B >= 999.\n"
                "   - replications where either fit failed are discarded and\n"
                "     counted in the reps column; a NEGATIVE LR is kept, since the\n"
                "     observed statistic can be negative too (BUG-26).\n",
                global_boot, sqrt(0.05 * 0.95 / (real) global_boot),
                1.0 / (real) (global_boot + 1));
            for (int rr = 0; rr <= M - 1; rr++)
                if (xkeep && xkeep[rr]) free_vector(xkeep[rr], 1, npkeep[rr]);
            if (xkeep) free(xkeep);
            if (npkeep) free_ivector(npkeep, 0, M - 1);
        }

        if (scaled)
            for (int t = 1; t <= nobs_raw; t++)
                for (int i2 = 1; i2 <= M; i2++) rawmat[t][i2] /= dsc[i2];
        free_vector(dsc, 1, M);
        free_ivector(good, 0, M - 1);
        free_ivector(tcr, 0, M - 1);
        free_ivector(bnd, 0, M - 1);
        free_ivector(npr, 0, M - 1);
        free_vector(ll, 0, M - 1);
        /*  BUG-20: the .out, not the base name (on the .pre route that is the
         *  first input file, and read as a model just overwritten).        */
        printf("Done. Output written to %s\n", outputf);
        fclose(outputv);
        cleanup_names(outputf, inputf, base_name);
        return 0;
}


/*****************************************************************************/
/*  P1 — the command line is validated BEFORE estimating                      */
/*                                                                           */
/*  WHY.  Until 2026-08-20 the option loop was a chain of strcmp with NO    */
/*  else branch: an unknown option -- a typo like -diagcv, or an option of  */
/*  another program of the suite -- was ignored IN SILENCE and drvec        */
/*  estimated a different model without saying so.  That is exactly what    */
/*  the program's own thesis does not admit: the specification IS the       */
/*  result, and the fit of another specification is not a worse fit but the */
/*  fit of another model (SPECIFICATION_PLAN.md 8).  A program that shouts  */
/*  when sigma_min falls below 0.2 and keeps quiet when handed an option    */
/*  that does not exist has its alarms badly distributed.  And p, q, r were */
/*  read with atoi, which silently returns 0 on text: `drvec file x y z`    */
/*  died by SIGSEGV inside init_guess, as did p = 0, p = -1 and p = 200.    */
/*                                                                           */
/*  THE CONVENTION IS NOT INVENTED HERE, it is the suite's.  drtran in C    */
/*  uses getopt with `default: usage(argv[0]); return 1;`, and the port     */
/*  como principio -- "Refusing rather than ignoring the option" -- con tres  */
/*  exit codes: 0 success (and -h), 1 a malformed command line, 2 the       */
/*  option is recognised but cannot be honoured.  drvec cannot use getopt   */
/*  as it stands because its vocabulary is of WORDS and not letters, so the */
/*  minimum-delta form is this: a TABLE that declares which options exist   */
/*  and what argument each takes, a validation pass that walks it before    */
/*  anything is touched, and the usual chain of strcmp intact behind it.    */
/*  The validation ASSIGNS NOTHING: that way it cannot accidentally change  */
/*  what is estimated, and no golden value moves.                          */
/*                                                                           */
/*  The table is also the source of the full listing usage() prints, so     */
/*  that the ACCEPTED list and the DOCUMENTED list cannot come apart --     */
/*  usage() used to cover 22 of 33 options -- and the suite compares them.  */
/*  See docs/PLAN_PRODUCCION.md P1.                                         */
/*****************************************************************************/


enum opt_arg {
    A_NONE,      /* bandera                                                  */
    A_STR,       /* file or prefix, mandatory                                */
    A_INT_POS,   /* entero >= 1, obligatorio                                 */
    A_CASE,      /* 1, 2 o 3                                                 */
    A_METHOD,    /* 1 o 2                                                    */
    A_REAL,      /* real, obligatorio                                        */
    A_REAL_OPT,  /* OPTIONAL real: consumed if it parses whole (-fixb2)      */
    A_INT_REAL,  /* an integer >= 1 and a real, both mandatory (-fixb2row)   */
    A_INT_INT,   /* two integers >= 1, both mandatory (-gzero)               */
    A_TOL_OPT    /* OPTIONAL real > 0, if it does not start with '-' (-rankadm) */
};

static const struct opt_spec {
    const char   *name;
    enum opt_arg  arg;
    const char   *val;     /* what the value is called in the listing         */
} OPT_TABLE[] = {
    { "-mean",        A_NONE,     NULL   },
    { "-case",        A_CASE,     "1|2|3|pre"},
    { "-nodrift",     A_INT_POS,  "i"    },
    { "-m",           A_METHOD,   "1|2"  },
    { "-diagar",      A_NONE,     NULL   },
    { "-trend",       A_NONE,     NULL   },
    { "-diagma",      A_NONE,     NULL   },
    { "-diagcov",     A_NONE,     NULL   },
    { "-levels",      A_NONE,     NULL   },
    { "-differenced", A_NONE,     NULL   },
    { "-fixb2",       A_REAL_OPT, "[v]"  },
    { "-fixb2row",    A_INT_REAL, "i v"  },
    { "-xpre",        A_STR,      "FILE" },
    { "-xlink",       A_STR,      "FILE" },
    { "-xsys",        A_INT_POS,  "i"    },
    { "-gzero",       A_INT_INT,  "i j"  },
    { "-lrtest",      A_NONE,     NULL   },
    { "-bootstrap",   A_INT_POS,  "N"    },
    { "-irfboot",     A_INT_POS,  "N"    },
    { "-rungs",       A_NONE,     NULL   },
    { "-specs",       A_NONE,     NULL   },
    { "-warma",       A_NONE,     NULL   },
    { "-mawarma",     A_NONE,     NULL   },
    { "-marow",       A_NONE,     NULL   },
    { "-mafree",      A_NONE,     NULL   },
    { "-f",           A_INT_POS,  "H"    },
    { "-estwin",      A_INT_POS,  "E"    },
    { "-C",           A_STR,      "FILE" },
    { "-matri",       A_NONE,     NULL   },
    { "-rankadm",     A_TOL_OPT,  "[tol]"},
    { "-matest",      A_INT_POS,  "N"    },
    { "-artest",      A_INT_POS,  "N"    },
    { "-alpha",       A_STR,      "FILE" },
    { "-weakex",      A_INT_POS,  "i"    },
    { "-multistart",  A_INT_POS,  "n"    },
    { "-eval",        A_NONE,     NULL   },
    { "-fdhess",      A_NONE,     NULL   },
    { "-seed",        A_STR,      "pfx"  },
    { "-seedybar",    A_STR,      "pfx"  },
    { "-seedgate",    A_NONE,     NULL   },
    { "-noladder",    A_NONE,     NULL   },
    { "-seedjoh",     A_NONE,     NULL   },
    { "-seedb2",      A_REAL,     "v"    },
    { "-interv",      A_STR,      "pfx"  },
    { "-writeres",    A_STR,      "pfx"  },
    { "-writeinp",    A_STR,      "pfx"  },
    { "-name",        A_STR,      "NAME" },
    { NULL,           A_NONE,     NULL   }
};

/*  The full listing, generated from the table.  It is what makes it       */
/*  impossible for usage() and the parser to diverge again.                */
static void usage_option_list(FILE *o)
{
    int i, col = 0;
    fprintf(o, "\nEvery option drvec accepts (docs/USAGE.md documents them all):\n ");
    for (i = 0; OPT_TABLE[i].name; i++) {
        char item[48];
        if (OPT_TABLE[i].val)
            snprintf(item, sizeof item, "%s %s", OPT_TABLE[i].name, OPT_TABLE[i].val);
        else
            snprintf(item, sizeof item, "%s", OPT_TABLE[i].name);
        if (col && col + (int) strlen(item) + 2 > 72) { fprintf(o, "\n "); col = 0; }
        fprintf(o, " %-*s", (int) strlen(item) + 1, item);
        col += (int) strlen(item) + 2;
    }
    fprintf(o, "\n");
}

static void usage(FILE *o)
{
    fprintf(o, "\ndrvec %s — VEC model EML estimation (Mauricio 2006)\n",
            DRVEC_VERSION);
    fprintf(o, "\nTwo ways in, and they estimate the same model:\n\n");
    fprintf(o, "  drvec file p q r [options]\n");
    fprintf(o, "        one .inp with every series in it (file.inp -> file.out)\n\n");
    fprintf(o, "  drvec s1.pre s2.pre [... sM.pre] p q r [options]\n");
    fprintf(o, "        ONE UNIVARIATE MODEL PER SERIES, as in drtran: each .pre\n");
    fprintf(o, "        brings its series, its Box-Cox and rescaling, and its\n");
    fprintf(o, "        deterministic terms, which are subtracted here.  The files\n");
    fprintf(o, "        are lined up by DATE and the common sample is used.  The\n");
    fprintf(o, "        column order is the .inp's: the first M-r files are the\n");
    fprintf(o, "        nabla Y2 block and the last r the Y1 block.  Products are\n");
    fprintf(o, "        named <stem1>_<stem2>...; -name NAME overrides that (-m is\n");
    fprintf(o, "        taken here: it is the estimation method)\n\n");
    fprintf(o, "  file  : data file name (without .inp extension)\n");
    fprintf(o, "  p     : AR order of stationary VARMA on Ȳ_t\n");
    fprintf(o, "  q     : MA order\n");
    fprintf(o, "  r     : cointegration rank (0 <= r < M; 0 is the diagonal rung;\n"
               "          ignored with -lrtest)\n\n");
    fprintf(o, "Deterministic cases (Mauricio 2006, Remark 6):\n");
    fprintf(o, "  -case 1 : E[∇Y₂]=0, E[W]=0     (default)\n");
    fprintf(o, "  -case 2 : E[∇Y₂]=0, E[W]≠0     (-mean alone also selects it)\n");
    fprintf(o, "  -case 3 : E[∇Y₂]≠0, E[W]≠0\n");
    fprintf(o, "  -case pre : case 3 with each drift as its .pre says (free iff the\n");
    fprintf(o, "              univariate frees its mean): the ladder's bottom rung\n");
    fprintf(o, "              is then exactly the univariates\n");
    fprintf(o, "  -nodrift i : hold E[∇Y₂ᵢ] = 0 inside case 3 (repeatable)\n");
    fprintf(o, "  -trend  : case 4, a linear trend in W (restricted to beta) on top\n");
    fprintf(o, "            of case 3 / -case pre; a single fit (no -lrtest, -f)\n\n");
    fprintf(o, "Data layout (cols 1..s are the Y₂ block, cols s+1..M the Y₁ block):\n");
    fprintf(o, "  default        every series in LEVELS; ∇Y₂ is formed internally\n");
    fprintf(o, "                 (one observation is consumed)\n");
    fprintf(o, "  -differenced   legacy: cols 1..s already hold ∇Y₂.  The Y₂ levels\n");
    fprintf(o, "                 are then unknown and get cumulated from zero, which\n");
    fprintf(o, "                 breaks -case 1 and makes E[W] incomparable\n\n");
    fprintf(o, "  -fixb2 [v]     hold B2 fixed instead of estimating it; npar\n");
    fprintf(o, "                 drops by s*r.  With a value, every entry of B2 is\n");
    fprintf(o, "                 pinned at v -- an a-priori restriction, so 2*[L(free)\n");
    fprintf(o, "                 - L(fixed)] IS a valid LR test, chi2 with s*r df\n");
    fprintf(o, "                 (Mauricio 2006 Table 5 tests B = [1,0]', i.e. -fixb2 0).\n");
    fprintf(o, "                 Without a value B2 is held at its static-OLS estimate:\n");
    fprintf(o, "                 useful as a warm start or a conditioning check, but\n");
    fprintf(o, "                 the restriction is then data-chosen, so the LR\n");
    fprintf(o, "                 statistic is NOT a valid test.\n\n");
    fprintf(o, "  -fixb2row i v  hold ROW i of B2 (variable i of the nabla Y2 block,\n");
    fprintf(o, "                 in the input's order) at v in every relation and\n");
    fprintf(o, "                 estimate the other rows; repeatable.  The free model\n");
    fprintf(o, "                 is fitted too and the LR against it is reported,\n");
    fprintf(o, "                 chi2 with (rows held)*r df -- with -alpha/-weakex\n");
    fprintf(o, "                 the joint test.  Unlike the Wald on a normalised\n");
    fprintf(o, "                 coefficient, the LR does not depend on which series\n");
    fprintf(o, "                 beta is normalised on\n\n");
    fprintf(o, "  -xpre FILE     an exogenous input (a fue .pre; repeatable), used\n");
    fprintf(o, "                 with -xlink.  X is FROZEN at its .pre: BoxCox minus\n");
    fprintf(o, "                 its deterministics (as drtran); its ARMA backcasts\n");
    fprintf(o, "                 the presample not covered by real data.  I(0) or I(1)\n");
    fprintf(o, "  -xlink FILE    the links, one per line `OUT <- IN b r s' (drtran):\n");
    fprintf(o, "                 Y_out - omega(B)/delta(B) B^b X_in is the VEC process\n");
    fprintf(o, "                 (\"by subtraction\", on the LEVELS; r <= 2).  The\n");
    fprintf(o, "                 VEC without inputs is fitted first, omega = 0 must\n");
    fprintf(o, "                 reproduce it (reported), and the LR against it is\n");
    fprintf(o, "                 the test.  .pre route, levels layout, case 2 or 3,\n");
    fprintf(o, "                 a single fit\n\n");
    fprintf(o, "  -xsys i        series i (position in the input, in the nabla Y2\n");
    fprintf(o, "                 block) is an input EMBEDDED in the system (drtran's\n");
    fprintf(o, "                 embedded cast, for an input with the outputs'\n");
    fprintf(o, "                 operator, e.g. I(1)): alpha_i = 0 and no other\n");
    fprintf(o, "                 series in its short run; repeatable.  The free VEC\n");
    fprintf(o, "                 is fitted too and the LR of strong exogeneity is\n");
    fprintf(o, "                 reported.  The transfer is read in the outputs'\n");
    fprintf(o, "                 rows: beta (gain), Gamma (lags), Q (omega_0).  q = 0\n\n");
    fprintf(o, "  -gzero i j     Gamma_k[i][j] = 0 for every k: series j (position in\n");
    fprintf(o, "                 the input) is not in the short run of equation i;\n");
    fprintf(o, "                 repeatable, i != j, p >= 2.  With -weakex/-alpha it\n");
    fprintf(o, "                 is the conditional model's restriction (weak\n");
    fprintf(o, "                 exogeneity and no feedback).  The free VEC is fitted\n");
    fprintf(o, "                 too and the LR reported.  A single fit, q = 0\n\n");
    fprintf(o, "  -irfboot N     bands for the level responses and the FEVD by\n");
    fprintf(o, "                 parametric bootstrap: N samples simulated from the\n");
    fprintf(o, "                 fit and REFITTED as the data were (same rank and\n");
    fprintf(o, "                 restrictions); percentile bands 90 %% and 95 %%, in\n");
    fprintf(o, "                 the .out and whole in <base>.irfboot.  Deterministic\n\n");
    fprintf(o, "  -lrtest        sequential LR test for the cointegration rank:\n");
    fprintf(o, "                 estimates r = 0..M-1 and reports 2*[L(r+1) - L(r)];\n");
    fprintf(o, "                 incompatible with -differenced\n\n");
    fprintf(o, "  -specs         the specification ladder: warma, mawarma, marow,\n");
    fprintf(o, "                 matri and free, in one run, with npar, logL,\n");
    fprintf(o, "                 termination, the rank condition G, the smallest\n");
    fprintf(o, "                 MA root, B2 and an ADMISSIBLE column -- and no\n");
    fprintf(o, "                 chi2 p-value where the theory does not give one\n\n");
    fprintf(o, "  -artest N      test the triangular short-run dynamics\n");
    fprintf(o, "                 (Gamma_i = M_i alpha': every lag enters through\n");
    fprintf(o, "                 W) against free F, with N bootstrap replications\n\n");
    fprintf(o, "  -matest N      test the inherited moving average against the\n");
    fprintf(o, "                 free one with N parametric bootstrap replications\n");
    fprintf(o, "                 under the restricted model.  The chi2 reference\n");
    fprintf(o, "                 is printed too, and it is NOT a test: the\n");
    fprintf(o, "                 unrestricted optimum sits on the edge of the\n");
    fprintf(o, "                 admissible region\n\n");
    fprintf(o, "  -rankadm [tol] refuse parameter points where the fitted model\n");
    fprintf(o, "                 denies its own rank: sigma_min of\n");
    fprintf(o, "                 Lambda_perp' Theta(1) B_perp below tol (default\n");
    fprintf(o, "                 0.2, which is the empty gap between the 0.016-\n");
    fprintf(o, "                 0.133 the degenerate fits give and the 0.52-1.00\n");
    fprintf(o, "                 the admissible ones do).  REPORTED always\n\n");
    fprintf(o, "  -warma         parameterise the TRANSFORMED system directly, in\n");
    fprintf(o, "                 the coordinates the triangular model is stated in:\n");
    fprintf(o, "                 Phi*_k = [0 Psi_k ; 0 Phi_k], Theta*_k diagonal\n");
    fprintf(o, "                 in the W block, and B2 entering ONLY through the\n");
    fprintf(o, "                 data.  This is the class the BVECM theorems cover\n\n");
    fprintf(o, "  -marow         Theta = [T11 T12 ; 0 0]: the differenced block\n");
    fprintf(o, "                 carries no moving average of its own, but the\n");
    fprintf(o, "                 cross block stays free.  q*s*M parameters fewer\n");
    fprintf(o, "                 than free\n\n");
    fprintf(o, "  -matri         the moving average is BLOCK-TRIANGULAR:\n");
    fprintf(o, "                 Theta = [T11 T12 ; 0 T22].  Only the lower-left\n");
    fprintf(o, "                 block is zeroed -- the differenced block keeps\n");
    fprintf(o, "                 its own moving average.  q*s*r parameters fewer\n");
    fprintf(o, "                 than free\n\n");
    fprintf(o, "  -mawarma       the moving average INHERITS its structure instead\n");
    fprintf(o, "                 of being free: Theta = [T11  T11*B2' ; 0  0], the\n");
    fprintf(o, "                 form a WARMA process implies for its VEC\n");
    fprintf(o, "                 representation (BVECM corollary 2).  q*r*r\n");
    fprintf(o, "                 parameters instead of q*M*M\n\n");
    fprintf(o, "  -seedjoh       seed B2 with the canonical reduced-rank solution\n");
    fprintf(o, "                 (Johansen's eigenvalue problem, closed form)\n");
    fprintf(o, "                 instead of the static OLS regression\n\n");
    fprintf(o, "  -seedb2 v      start B2 at v and estimate it FREE (not -fixb2,\n");
    fprintf(o, "                 which holds it).  A measuring instrument: it is\n");
    fprintf(o, "                 how you ask whether the answer depends on where\n");
    fprintf(o, "                 B2 starts\n\n");
    fprintf(o, "  -seedgate      seed the r >= 1 fit by profiling: estimate the\n");
    fprintf(o, "                 r = 0 rung, hold F, Theta and Sigma there, and fit\n");
    fprintf(o, "                 Lambda and B2 on it before releasing everything.\n");
    fprintf(o, "                 Not the default: the recorded results rest on the\n");
    fprintf(o, "                 cold start, and this moves only when measured to\n");
    fprintf(o, "                 be at least as good\n\n");
    fprintf(o, "  -rungs         the ladder below the rank: rungs 0 (F, Theta and\n");
    fprintf(o, "                 Sigma diagonal), 1 (Sigma free) and 2 (F and Theta\n");
    fprintf(o, "                 free), all at r = 0, with their chi2 LRs.  These are\n");
    fprintf(o, "                 ordinary nested comparisons; adding the VEC matrix\n");
    fprintf(o, "                 is not, and lives in -lrtest\n\n");
    fprintf(o, "Forecasting, and the files it writes (P5, P6.7):\n");
    fprintf(o, "  -f H           forecast H steps in LEVELS, with bands.  The\n");
    fprintf(o, "                 table goes to the .out and the dated forecast to\n");
    fprintf(o, "                 <file>.forecast, which is the file the rest of\n");
    fprintf(o, "                 the suite writes too\n\n");
    fprintf(o, "  -estwin E      estimate ONCE on observations 1..E, hold the\n");
    fprintf(o, "                 parameters fixed and roll the origin forward,\n");
    fprintf(o, "                 scoring each forecast against what happened.\n");
    fprintf(o, "                 Needs -f H.  The summary goes to the .out and the\n");
    fprintf(o, "                 per-origin errors to <file>.recursive, which is\n");
    fprintf(o, "                 what a test of equal predictive ability needs\n\n");
    fprintf(o, "  -C FILE        write those per-origin errors to FILE instead of\n");
    fprintf(o, "                 <file>.recursive.  A redirection, not a condition\n\n");
    fprintf(o, "Every run also reports, without being asked, Wald tests of weak\n");
    fprintf(o, "exogeneity (row of Lambda = 0) and of exclusion from the\n");
    fprintf(o, "cointegrating relations (row of B2 = 0), read off the covariance\n");
    fprintf(o, "the estimation already produced.  Use -fdhess before quoting one.\n\n");
    /*  The program has more options than fit here, and a list duplicated in
     *  two places diverges.  Where the full one is, is stated.              */
    fprintf(o, "The options above are the ones that need a paragraph; the full\n"
               "list follows, and docs/USAGE.md documents every one of them.\n"
               "docs/GETTING_STARTED.md has the order to use them in -- select\n"
               "the rank at q = 0, look at -specs at the selected rank, then\n"
               "estimate.  Every fit reports the rank condition, and one that\n"
               "denies the rank it was estimated at says so.\n");
    usage_option_list(o);
}

/*  A USAGE error exits with 1 and through stderr.  The whole usage is not
 *  printed on every misplaced value: that buries the message.  It is printed
 *  when the problem is that the option does not exist, which is when the list
 *  helps.                                                                    */
static void bad_cli(const char *fmt, ...)
{
    va_list ap;
    fprintf(stderr, "drvec: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n       `drvec -h' lists every option; docs/USAGE.md documents them.\n");
    exit(1);
}

/*  strtol/strtod with a check on the end of the text.  atoi("x") returns 0
 *  without saying anything, and a silent 0 in p is a SIGSEGV in init_guess.  */
static int arg_int(const char *s, long *out)
{
    char *end;
    long v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE) return 0;
    *out = v;
    return 1;
}

static int arg_real(const char *s, double *out)
{
    char *end;
    double v;
    if (!s || !*s) return 0;
    errno = 0;
    v = strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE) return 0;
    *out = v;
    return 1;
}

static const struct opt_spec *opt_lookup(const char *name)
{
    int i;
    for (i = 0; OPT_TABLE[i].name; i++)
        if (strcmp(OPT_TABLE[i].name, name) == 0) return &OPT_TABLE[i];
    return NULL;
}

/*  The suggestion for a typo: the longest common prefix.  With -diagcv it
 *  returns -diagcov and with -multistar it returns -multistart, which is 90 %
 *  of the real typos.                                                        */
static const char *opt_nearest(const char *name)
{
    int i, best = 0; const char *hit = NULL;
    for (i = 0; OPT_TABLE[i].name; i++) {
        int k = 0;
        while (name[k] && OPT_TABLE[i].name[k] && name[k] == OPT_TABLE[i].name[k]) k++;
        if (k > best) { best = k; hit = OPT_TABLE[i].name; }
    }
    return (best >= 4) ? hit : NULL;
}

/*  validate_cli — the preliminary pass over the options.  `first` is where
 *  they start, which is 5 on the .inp route and M+4 on the .pre one.  It stops
 *  at the
 *  first problem and assigns nothing.                                        */
static void validate_cli(int argc, char *argv[], int first)
{
    int i;
    for (i = first; i < argc; i++) {
        const struct opt_spec *o;
        long iv; double rv;

        if (argv[i][0] != '-') {
            bad_cli("unexpected argument `%s' — options start with `-'", argv[i]);
        }
        o = opt_lookup(argv[i]);
        if (!o) {
            const char *near = opt_nearest(argv[i]);
            fprintf(stderr, "drvec: unknown option `%s'\n", argv[i]);
            if (near) fprintf(stderr, "       did you mean `%s'?\n", near);
            usage(stderr);
            exit(1);
        }
        switch (o->arg) {
        case A_NONE:
            break;
        case A_STR:
            if (i + 1 >= argc) bad_cli("%s needs %s", o->name, o->val);
            i++;
            break;
        case A_INT_POS:
            if (i + 1 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (!arg_int(argv[i+1], &iv) || iv < 1)
                bad_cli("%s needs an integer >= 1, got `%s'", o->name, argv[i+1]);
            i++;
            break;
        case A_CASE:
            if (i + 1 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (strcmp(argv[i+1], "pre") != 0
                && (!arg_int(argv[i+1], &iv) || iv < 1 || iv > 3))
                bad_cli("%s must be 1, 2, 3 (Mauricio 2006, Remark 6) or pre (case 4 is -trend), got `%s'",
                        o->name, argv[i+1]);
            i++;
            break;
        case A_METHOD:
            if (i + 1 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (!arg_int(argv[i+1], &iv) || iv < 1 || iv > 2)
                bad_cli("%s must be 1 (exact ML) or 2 (approximate), got `%s'",
                        o->name, argv[i+1]);
            i++;
            break;
        case A_REAL:
            if (i + 1 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (!arg_real(argv[i+1], &rv))
                bad_cli("%s needs a number, got `%s'", o->name, argv[i+1]);
            i++;
            break;
        case A_INT_INT:
            if (i + 2 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (!arg_int(argv[i+1], &iv) || iv < 1 || iv > GZ_MAX)
                bad_cli("%s needs an equation 1..%d, got `%s'", o->name, GZ_MAX, argv[i+1]);
            if (!arg_int(argv[i+2], &iv) || iv < 1 || iv > GZ_MAX)
                bad_cli("%s needs a regressor 1..%d, got `%s'", o->name, GZ_MAX, argv[i+2]);
            i += 2;
            break;
        case A_INT_REAL:
            if (i + 2 >= argc) bad_cli("%s needs %s", o->name, o->val);
            if (!arg_int(argv[i+1], &iv) || iv < 1 || iv > B2ROW_MAX)
                bad_cli("%s needs a row number >= 1, got `%s'", o->name, argv[i+1]);
            if (!arg_real(argv[i+2], &rv))
                bad_cli("%s needs a value for the row, got `%s'", o->name, argv[i+2]);
            i += 2;
            break;
        case A_REAL_OPT:
            /*  -fixb2: the value is optional and is recognised by parsing whole,
                which is exactly the criterion the assigner uses.             */
            if (i + 1 < argc && arg_real(argv[i+1], &rv)) i++;
            break;
        case A_TOL_OPT:
            /*  -rankadm: the value is optional and is recognised by NOT starting
                with '-', which is the assigner's criterion.  If present it has
                to be a positive number: a tolerance <= 0 switches the warning
                off without saying so, and that is the alarm the program must
                not lose.                                                     */
            if (i + 1 < argc && argv[i+1][0] != '-') {
                if (!arg_real(argv[i+1], &rv) || rv <= 0.0)
                    bad_cli("%s needs a tolerance > 0, got `%s'",
                            o->name, argv[i+1]);
                i++;
            } else if (i + 1 < argc && arg_real(argv[i+1], &rv)) {
                /*  A NEGATIVE number after -rankadm: the assigner would not take
                    it as a value (it looks at the leading '-') and it would
                    fall through as an unknown option, with a message that says
                    nothing.                                                  */
                bad_cli("%s needs a tolerance > 0, got `%s'", o->name, argv[i+1]);
            }
            break;
        }
    }
}

/*****************************************************************************/
/*  main                                                                      */
/*****************************************************************************/
/*****************************************************************************/
/*  P8 — parse_cli: the command line, out of main().                         */
/*                                                                           */
/*  267 lines that read arguments and set flags, at the same indentation as   */
/*  the estimation they configure.  Everything it touches is already at file  */
/*  scope -- the option globals, the two route variables, the three file      */
/*  names -- so the cut costs no parameters at all, which is the sign that it */
/*  was one block pretending to be part of another.                          */
/*                                                                           */
/*  It returns -1 to say "carry on", or a process exit code: 0 for the        */
/*  queries (-h, --version) and 1 for a malformed line.  The rules those      */
/*  codes follow are P1's, and validate_cli is where they are enforced.       */
/*****************************************************************************/
static int parse_cli(int argc, char *argv[])
{
    /*  -h/--help and --version, before anything else: they are queries, not
        runs, and they exit with 0 through stdout (the port's convention).
        With no useful argument, the usage goes to stderr and exits 1, which
        is a usage error.                                                    */
    if (argc >= 2 && (strcmp(argv[1], "-h") == 0 ||
                      strcmp(argv[1], "--help") == 0)) {
        usage(stdout);
        return 0;
    }
    if (argc >= 2 && (strcmp(argv[1], "--version") == 0 ||
                      strcmp(argv[1], "-version") == 0)) {
        printf("drvec %s\n", DRVEC_VERSION);
        printf("VEC model EML estimation (Mauricio 2006), on the drvarma engine\n");
        printf("GPL v2 or later\n");
        return 0;
    }
    if (argc < 5) {
        usage(stderr);
        return 1;
    }

    /*  P9 — WHICH ROUTE.  Leading arguments that end in `.pre` are series, one
     *  univariate model each, and then p q r follow: that is drtran's
     *  interface, and drvec belongs on drtran's side of the line (see
     *  read_pre_inputs).  Anything else is the .inp route, untouched -- every
     *  figure in the register was measured through it.
     *
     *  The test is the SUFFIX and not the existence of the file, so that a
     *  mistyped path is reported as a missing .pre and not as a nonsensical
     *  AR order.                                                            */
    {
        int k = 1;
        while (k < argc && ends_with(argv[k], ".pre")) k++;
        n_pre = k - 1;
        if (n_pre > 0) {
            pre_route = 1;
            pre_files = argv;                 /* 1..n_pre, borrowed          */
            if (argc < n_pre + 4) {
                fprintf(stderr, "ERROR: after the .pre files come p, q and r\n");
                usage(stderr);
                return 1;
            }
            i_pqr    = n_pre + 1;
            first_opt = n_pre + 4;
        }
    }

    /* Size these from the actual argument, not a fixed 80: a longer path used
       to overflow them through the strcpy/strcat below and abort the run.
       +8 covers the ".inp"/".out" suffix and the terminator.                 */
    {
        int need = (int) strlen(argv[1]) + 8;
        inputf    = NEW_STR(need);
        outputf   = NEW_STR(need);
        base_name = NEW_STR(need);
        if (!inputf || !outputf || !base_name) {
            fprintf(stderr, "ERROR: out of memory for file names\n");
            exit(1);
        }
    }

    strcpy(base_name, argv[1]);
    snprintf(out_base, sizeof out_base, "%s", base_name);
    strcpy(inputf, argv[1]);

    /*  P9 — the products' name on the .pre route.  Default: the file stems
     *  joined, which is drtran's <output>_<input> generalised to M series and
     *  says at a glance what was fitted.  -name overrides it, and is read here
     *  rather than in the option loop because the name has to exist before the
     *  files are opened.                                                     */
    {   /*  BUG-45: -name was read on the .pre route only, and silently
         *  ignored on the .inp route.                                        */
        int k;
        for (k = first_opt; k + 1 < argc; k++)
            if (strcmp(argv[k], "-name") == 0) { model_name = argv[k+1]; break; }
    }
    if (model_name)
        snprintf(out_base, sizeof out_base, "%s", model_name);
    else if (pre_route) {
        int k;
        {
            char stem[128];
            size_t used = 0;
            out_base[0] = '\0';
            for (k = 1; k <= n_pre; k++) {
                path_stem(argv[k], stem, sizeof stem);
                used += (size_t) snprintf(out_base + used, sizeof out_base - used,
                                          "%s%s", (k > 1) ? "_" : "", stem);
                if (used >= sizeof out_base - 1) break;
            }
        }
    }

    /*  p, q, r with strtol and a check on the end.  With atoi, `drvec f x y z`
        gave p = q = r = 0 and the process died inside init_guess without a
        word; p = 0 did the same by the legitimate route.  The lower bound on p
        is 1: the AR order of the stationary VARMA on Ybar, of which the
        effective order on nabla Y is p - 1 (MODEL.md 5.2).  The UPPER bound is
        not set here but with the degrees of freedom, once the file has been
        read: it is the sample that fixes it, not a number.                  */
    {
        long lp, lq, lr;
        if (!arg_int(argv[i_pqr], &lp) || lp < 1)
            bad_cli("p (the AR order) must be an integer >= 1, got `%s'",
                    argv[i_pqr]);
        if (!arg_int(argv[i_pqr+1], &lq) || lq < 0)
            bad_cli("q (the MA order) must be an integer >= 0, got `%s'",
                    argv[i_pqr+1]);
        if (!arg_int(argv[i_pqr+2], &lr) || lr < 0)
            bad_cli("r (the cointegration rank) must be an integer >= 0, got `%s'",
                    argv[i_pqr+2]);
        global_p = (int) lp;
        global_q = (int) lq;
        global_r = (int) lr;
    }

    /*  The validation pass, before anything is assigned.  See P1.           */
    validate_cli(argc, argv, first_opt);

    /* Parse options */
    for (int i = first_opt; i < argc; i++) {
        if      (strcmp(argv[i], "-mean") == 0)    global_include_mean = 1;
        else if (strcmp(argv[i], "-case") == 0 && i+1 < argc) {
            if (strcmp(argv[i+1], "pre") == 0) { global_case = 3; case_pre = 1; i++; }
            else global_case = atoi(argv[++i]);
            case_given = 1;
        }
        else if (strcmp(argv[i], "-diagar") == 0)  global_diag_ar = 1;
        else if (strcmp(argv[i], "-trend") == 0)   global_trend = 1;
        else if (strcmp(argv[i], "-diagma") == 0)  global_diag_ma = 1;
        else if (strcmp(argv[i], "-diagcov") == 0) global_diag_cov = 1;
        else if (strcmp(argv[i], "-m") == 0 && i+1 < argc)
            met = atoi(argv[++i]);
        else if (strcmp(argv[i], "-lrtest") == 0)  global_lrtest = 1;
        else if (strcmp(argv[i], "-rungs") == 0)   global_rungs = 1;
        else if (strcmp(argv[i], "-seedgate") == 0) global_seedgate = 1;
        else if (strcmp(argv[i], "-noladder") == 0) global_noladder = 1;
        else if (strcmp(argv[i], "-seedjoh") == 0)  global_seedjoh = 1;
        else if (strcmp(argv[i], "-mawarma") == 0)  global_mawarma = 1;
        else if (strcmp(argv[i], "-matri") == 0)    global_matri = 1;
        else if (strcmp(argv[i], "-marow") == 0)    global_marow = 1;
        else if (strcmp(argv[i], "-mafree") == 0)   global_mafree = 1;
        else if (strcmp(argv[i], "-f") == 0 && i+1 < argc)
            global_fcast = atoi(argv[++i]);
        else if (strcmp(argv[i], "-estwin") == 0 && i+1 < argc)
            global_estwin = atoi(argv[++i]);
        else if (strcmp(argv[i], "-C") == 0 && i+1 < argc)
            fc_csv = argv[++i];
        else if (strcmp(argv[i], "-warma") == 0)    global_warma = 1;
        else if (strcmp(argv[i], "-specs") == 0)    global_specs = 1;
        else if (strcmp(argv[i], "-artest") == 0 && i+1 < argc)
            global_artest = atoi(argv[++i]);
        else if (strcmp(argv[i], "-matest") == 0 && i+1 < argc)
            global_matest = atoi(argv[++i]);
        else if (strcmp(argv[i], "-rankadm") == 0) {
            global_rankadm = 1;
            if (i+1 < argc && argv[i+1][0] != '-') global_rankadm_tol = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-seedb2") == 0 && i+1 < argc) {
            global_seedb2 = 1; global_seedb2_value = atof(argv[++i]);
        }
        else if (strcmp(argv[i], "-levels") == 0)  global_levels = 1;  /* default */
        else if (strcmp(argv[i], "-differenced") == 0) global_levels = 0;
        else if (strcmp(argv[i], "-name") == 0 && i+1 < argc) i++;  /* read above */
        else if (strcmp(argv[i], "-writeinp") == 0 && i+1 < argc) {
            global_writeinp = 1; inp_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-writeres") == 0 && i+1 < argc) {
            global_writeres = 1; res_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-eval") == 0) global_eval = 1;
        else if (strcmp(argv[i], "-fdhess") == 0) global_fdhess = 1;
        else if (strcmp(argv[i], "-bootstrap") == 0 && i+1 < argc)
            global_boot = atoi(argv[++i]);
        else if (strcmp(argv[i], "-irfboot") == 0 && i+1 < argc)
            global_irfboot = atoi(argv[++i]);
        else if (strcmp(argv[i], "-interv") == 0 && i+1 < argc) {
            global_interv = 1; interv_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-multistart") == 0 && i+1 < argc)
            global_multistart = atoi(argv[++i]);
        else if (strcmp(argv[i], "-alpha") == 0 && i+1 < argc) {
            global_alpha = 1; alpha_file = argv[++i];
        }
        else if (strcmp(argv[i], "-weakex") == 0 && i+1 < argc) {
            global_alpha = 1; alpha_weakex = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-seed") == 0 && i+1 < argc) {
            global_seed = 1; seed_route = SEED_RESID; pre_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-seedybar") == 0 && i+1 < argc) {
            global_seed = 1; seed_route = SEED_YBAR;  pre_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "-xpre") == 0 && i + 1 < argc) {
            if (xl_nin >= XL_MAXIN) {
                fprintf(stderr, "ERROR: at most %d -xpre inputs\n", XL_MAXIN);
                exit(1);
            }
            xl_infile[++xl_nin] = argv[++i];
        }
        else if (strcmp(argv[i], "-xlink") == 0 && i + 1 < argc) {
            xl_netfile = argv[++i];
            xl_on = 1;
        }
        else if (strcmp(argv[i], "-nodrift") == 0 && i + 1 < argc) {
            int ix = atoi(argv[++i]);
            if (ix < 1 || ix > ND_MAX) {
                fprintf(stderr, "ERROR: -nodrift %d: the series is named by its "
                                "position in the input, 1..%d\n", ix, ND_MAX);
                exit(2);
            }
            nodrift[ix] = 1; nodrift_on = 1;
        }
        else if (strcmp(argv[i], "-xsys") == 0 && i + 1 < argc) {
            int ix = atoi(argv[++i]);
            if (ix < 1 || ix > XS_MAX) {
                fprintf(stderr, "ERROR: -xsys %d: the series is named by its "
                                "position in the input, 1..%d\n", ix, XS_MAX);
                exit(2);
            }
            if (!xs_inp[ix]) xs_nx++;
            xs_inp[ix] = 1;
            xs_on = 1;
        }
        else if (strcmp(argv[i], "-gzero") == 0 && i + 2 < argc) {
            int ig = atoi(argv[i+1]), jg = atoi(argv[i+2]);   /* range: validate_cli */
            i += 2;
            if (ig == jg) {
                fprintf(stderr, "ERROR: -gzero %d %d: the diagonal of Gamma is the "
                                "series' own model\n       (the ladder's rung), and is "
                                "never held.\n", ig, jg);
                exit(2);
            }
            if (!gz[ig][jg]) gz_n++;
            gz[ig][jg] = 1;
            gz_on = 1;
        }
        else if (strcmp(argv[i], "-fixb2row") == 0 && i + 2 < argc) {
            int ir = atoi(argv[i+1]);
            b2row_fix[ir] = 1;                   /* range checked by validate_cli */
            b2row_val[ir] = atof(argv[i+2]);
            b2row_on = 1;
            i += 2;
        }
        else if (strcmp(argv[i], "-fixb2") == 0) {
            global_fixb2 = 1;
            /* An optional numeric argument pins B2 at a value chosen a priori,
               which is what makes the LR test against the free model valid.  */
            if (i + 1 < argc) {
                char *end;
                double v = strtod(argv[i+1], &end);
                if (end != argv[i+1] && *end == '\0') {
                    global_fixb2_value = v;
                    global_fixb2_given = 1;
                    i++;
                }
            }
        }
    }
    /* -mean implies case 2 (E[W]≠0) unless a case was given explicitly.
       BUG-45: an explicit -case 1 was overridden all the same.  Case 1 has no
       mean, so the two contradict each other: refused, not resolved.      */
    if (global_include_mean && global_case == 1) {
        if (case_given) {
            fprintf(stderr, "ERROR: -case 1 has no deterministic term and -mean "
                            "asks for one.  Drop one of them\n"
                            "       (-mean alone means case 2).\n");
            exit(2);
        }
        global_case = 2;
    }

    /*  THE MOVING-AVERAGE DEFAULT, decided again on 2026-09-23: Theta FREE,
     *  which is Mauricio (2006)'s model.
     *
     *  From 2026-08-20 to 2026-09-23 the default was -marow (the s rows of
     *  nabla Y2 in every Theta_k zero), on two legs.  The theoretical one --
     *  Corollary 6.3, "the admissible region IS the whole space of that class
     *  and the engine's gate imposes it" -- is false in both directions, and
     *  the source it was attributed to (the BVECM article) does not estimate
     *  that class: it and its accompanying program estimate the full model,
     *  the program with an MA on the nabla Y2 equation -- exactly the entry
     *  -marow zeroes.  The empirical leg was estimability in short samples
     *  (HOMOLOGATION 4q/4r), and that is what the search of P12 addresses:
     *  the free fit now starts, among other places, from the -marow and -matri
     *  optima embedded, so it cannot end below them.  On the wheat pairs the
     *  data want that entry (-matri beats -marow by LR >= 5.96 in 37 of 40),
     *  and the free class forecast better in 8 of 9 cases (HOMOLOGATION 4t).
     *  The free class also keeps -lrtest nested (BUG-27) and is what the rank
     *  test's theory assumes.  -marow, -matri, -mawarma and -warma remain,
     *  as restrictions to test.  See docs/ESTUDIO_MAROW_2026-09-23.md and
     *  BUGS.md BUG-48.                                                       */

    /*  P9 — -interv is the .inp route's way of getting the deterministic terms
     *  out of a .pre.  On the .pre route they are already out, subtracted by
     *  read_pre_inputs from the same models, so accepting the option would mean
     *  subtracting them twice -- silently, and with the same signature as
     *  BUG-15.  It is refused rather than ignored: a user who typed it was
     *  asking for something, and being told it already happened is the answer. */
    if (pre_route && global_interv) {
        fprintf(stderr,
            "ERROR: -interv does nothing on the .pre route -- the deterministic\n"
            "       terms are already subtracted, from those same models.\n"
            "       Accepting it would subtract them twice.\n");
        exit(1);
    }
    if (pre_route && !global_levels) {
        fprintf(stderr,
            "ERROR: -differenced does not apply to the .pre route.  A .pre\n"
            "       carries the series in levels and says how it is differenced;\n"
            "       drvec forms nabla Y2 itself.\n");
        exit(1);
    }

    /*  BUG-30.  -warma's branch of vec_shootx reads a full M x r Lambda, while
     *  par_blocks counts only the free alpha_sa x r of Lambda = A psi: the
     *  vector fell out of step, B2 was read past its end, and the fit claimed
     *  a restriction it did not impose.  Refused until -warma implements it. */
    /*  -fixb2row: what it can be combined with is what has been checked.
     *  -fixb2 already holds every row; the modes re-fit at other ranks or in
     *  other coordinates (-warma estimates the transformed system, where B2
     *  is not a parameter), and none of them carries the row restriction.  */
    /*  -xlink: the transfer is subtracted from a single fit on the .pre
     *  route, in the levels layout.  The modes re-fit at other ranks or
     *  in other coordinates and do not carry it; the forecast and the
     *  rolling evaluation would need the future of X; -writeinp/-writeres
     *  write the data without it.                                         */
    if (xl_on || xl_nin > 0) {
        if (!xl_on || xl_nin == 0) {
            fprintf(stderr, "ERROR: -xpre and -xlink go together: the inputs and the\n"
                            "       network of links that says where they enter.\n");
            exit(2);
        }
        if (!pre_route || !global_levels || global_warma || global_lrtest
            || global_specs || global_rungs || global_matest || global_artest
            || global_eval || global_fcast > 0 || global_estwin > 0
            || global_writeinp || global_writeres) {
            fprintf(stderr,
                "ERROR: -xlink needs the .pre route in the levels layout, and a\n"
                "       single fit: it cannot be combined with -differenced, -warma,\n"
                "       -lrtest, -specs, -rungs, -matest, -artest, -eval, -f, -estwin,\n"
                "       -writeinp or -writeres.\n");
            exit(2);
        }
        if (global_case == 1) {
            /*  X keeps its mean (only its deterministics are subtracted,
             *  as in drtran), so nu(1)*E[X] lands in the mean of W, which
             *  case 1 holds at zero: the inputs would be a restriction.   */
            fprintf(stderr,
                "ERROR: -xlink needs a free mean of W (-case 2 or 3): the input\n"
                "       keeps its mean, and nu(1) times it goes to E[W].\n");
            exit(2);
        }
    }
    /*  -irfboot: one fit, in the levels layout (simulate_h0 builds levels),
     *  and a model it can simulate: -xlink would need the future of X and a
     *  model for it, -trend a trend the generator does not carry, -warma its
     *  own walk of B2.                                                     */
    if (global_irfboot > 0 && (!global_levels || xl_on || global_trend
                               || global_warma || global_lrtest || global_specs
                               || global_rungs || global_matest || global_artest
                               || global_eval || global_estwin > 0
                               || global_writeinp || global_writeres)) {
        fprintf(stderr,
            "ERROR: -irfboot bootstraps the responses of a single fit in the levels\n"
            "       layout: it cannot be combined with -differenced, -xlink, -trend,\n"
            "       -warma, -estwin, -writeinp, -writeres or the modes -lrtest,\n"
            "       -specs, -rungs, -matest, -artest and -eval (not yet).\n");
        exit(2);
    }
    /*  -trend (case 4) builds on case 3's drifts.  The rank test would need
     *  case 4's tables (or a bootstrap that carries the trend); the forecast
     *  and the rolling evaluation would have to extend it; -writeinp writes
     *  Ybar without it; -warma walks its own mean block.                   */
    if (global_trend && (global_case != 3 || global_warma || global_lrtest
                         || global_specs || global_rungs || global_matest
                         || global_artest || global_eval || global_fcast > 0
                         || global_estwin > 0 || global_writeinp
                         || global_writeres)) {
        fprintf(stderr,
            "ERROR: -trend (case 4) goes on case 3's drifts (-case 3 or -case pre)\n"
            "       for a single fit: it cannot be combined with -warma, -f,\n"
            "       -estwin, -writeinp, -writeres or the modes -lrtest, -specs,\n"
            "       -rungs, -matest, -artest and -eval (not yet).\n");
        exit(2);
    }
    /*  -nodrift / -case pre: a mask on case 3's drifts.  -warma walks its
     *  own mean block and does not carry it; -nodrift outside case 3 holds
     *  what is already held.                                              */
    if ((nodrift_on || case_pre) && global_warma) {
        fprintf(stderr, "ERROR: -nodrift and -case pre cannot be combined with "
                        "-warma.\n");
        exit(2);
    }
    if ((nodrift_on || case_pre) && global_lrtest)
        fprintf(stderr,
            "WARNING: with some drifts held (-nodrift / -case pre) the rank test's\n"
            "         asymptotic tables are case 3's, which do not apply to the\n"
            "         mixed case: read the bootstrap (-bootstrap N), not the table.\n");
    if (nodrift_on && global_case != 3) {
        fprintf(stderr,
            "ERROR: -nodrift holds a drift of case 3 (-case 3 or -case pre); in\n"
            "       cases 1 and 2 no series has one.\n");
        exit(2);
    }
    if (case_pre && !pre_route) {
        fprintf(stderr, "ERROR: -case pre reads each series' mean from its .pre: "
                        "it needs the .pre route.\n");
        exit(2);
    }
    /*  -xsys: one fit with the mask, against the free VEC.  The rank test
     *  would need Harbo et al.'s tables (or a bootstrap conditional on X);
     *  the modes refit in other settings and do not carry the mask; -warma
     *  has no F to mask; the MA walks do not carry it yet (q = 0); and alpha
     *  is set by the mask itself, so -alpha/-weakex would be a second A.    */
    if (xs_on && (global_warma || global_lrtest || global_specs || global_rungs
                  || global_matest || global_artest || global_eval
                  || global_alpha || xl_on || global_q > 0)) {
        fprintf(stderr,
            "ERROR: -xsys embeds an input in the system for a single fit with\n"
            "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
            "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
            "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
            "       and -eval.\n");
        exit(2);
    }
    /*  -gzero: one restricted fit against the free VEC, like -xsys; -alpha and
     *  -weakex are welcome (the joint restriction), the other masks are not. */
    if (gz_on && (global_warma || global_lrtest || global_specs || global_rungs
                  || global_matest || global_artest || global_eval || xl_on
                  || xs_on || b2row_on || global_fixb2 || global_diag_ar
                  || global_q > 0 || global_p < 2)) {
        fprintf(stderr,
            "ERROR: -gzero holds entries of Gamma(k) for a single fit with q = 0\n"
            "       and p >= 2 (at p = 1 there is no Gamma).  It combines with\n"
            "       -alpha/-weakex, not with -xsys, -fixb2row, -fixb2, -xlink,\n"
            "       -diagar, -warma, a moving average, or the modes -lrtest,\n"
            "       -specs, -rungs, -matest, -artest and -eval.\n");
        exit(2);
    }
    if (b2row_on && (global_fixb2 || global_warma || global_lrtest
                     || global_specs || global_rungs || global_matest
                     || global_artest || global_eval)) {
        fprintf(stderr,
            "ERROR: -fixb2row holds rows of B2 at a single fit.  It cannot be\n"
            "       combined with -fixb2 (which holds every row), -warma, or the\n"
            "       modes -lrtest, -specs, -rungs, -matest, -artest and -eval.\n");
        exit(2);
    }
    if (global_warma && global_alpha) {
        fprintf(stderr,
            "ERROR: -warma cannot be combined with -alpha or -weakex: the WARMA\n"
            "       parametrisation does not impose alpha = A psi.  Use -mawarma\n"
            "       (the same MA structure, VEC coordinates) with the restriction.\n");
        exit(2);
    }
    /*  BUG-36.  With -differenced the Y2 block arrives differenced and its
     *  levels are cumulated from an arbitrary zero, so a LEVEL forecast of
     *  Y2 and a MAPE on it mean nothing (muskrat 1.41 for ~13.6, MAPE 107.9 %
     *  for 2.98 %).  Refused rather than printed.                          */
    if (!global_levels && (global_fcast > 0 || global_estwin > 0)) {
        fprintf(stderr,
            "ERROR: -f and -estwin are incompatible with -differenced: the Y2\n"
            "       levels would be cumulated from an arbitrary zero.  Supply\n"
            "       every series in levels (the default layout).\n");
        exit(2);
    }
    /*  BUG-45.  The modes are dispatched in turn and the first one returns,
     *  so a second mode, a forecast asked of a mode, or a bootstrap with
     *  nothing to bootstrap was dropped without a word (`-lrtest -specs'
     *  ran only -specs).  Refused: exit 2, recognised but not honoured.   */
    {
        int nmodes = (global_rungs > 0) + (global_specs > 0)
                   + (global_matest > 0 || global_artest > 0)
                   + (global_lrtest > 0) + (global_eval > 0);
        if (nmodes > 1) {
            fprintf(stderr, "ERROR: one mode per run: -lrtest, -specs, -rungs, "
                            "-matest/-artest and -eval\n"
                            "       cannot be combined.\n");
            exit(2);
        }
        if (nmodes == 1 && (global_fcast > 0 || global_estwin > 0)) {
            fprintf(stderr, "ERROR: -f and -estwin act on a single fit; the mode "
                            "asked for does not\n       produce one.  Run them "
                            "separately.\n");
            exit(2);
        }
        if (global_warma && (global_diag_ar || global_diag_ma)) {
            fprintf(stderr, "ERROR: -warma has its own AR and MA structure and does "
                            "not impose\n       -diagar/-diagma (the header said "
                            "\"F diagonal\" all the same).\n");
            exit(2);
        }
        if (global_writeinp && global_writeres) {
            fprintf(stderr, "ERROR: -writeinp and -writeres each write their files "
                            "and stop;\n       run them separately.\n");
            exit(2);
        }
        if (global_boot > 0 && !global_lrtest) {
            fprintf(stderr, "ERROR: -bootstrap calibrates -lrtest and does "
                            "nothing without it.\n");
            exit(2);
        }
    }
    /*  BUG-31, case 1.  -matest and -artest simulate under H0 in LEVELS, into a
     *  sample laid out for the levels route; with -differenced the columns
     *  1..s arrive already differenced, so the simulation wrote a row past the
     *  end and, before that, the wrong thing.  -lrtest refuses it already.  */
    if (!global_levels && (global_matest > 0 || global_artest > 0)) {
        fprintf(stderr,
            "ERROR: -matest and -artest are incompatible with -differenced: the\n"
            "       bootstrap simulates the series in levels.  Supply every\n"
            "       series in levels (the default layout).\n");
        exit(2);
    }

    if (global_lrtest) {
        /* The column split of the .inp is s = M - r, so varying r only makes
           sense when every series is supplied in levels.                     */
        if (!global_levels) {
            fprintf(stderr,
                "ERROR: -lrtest is incompatible with -differenced.  The column\n"
                "       split depends on r (cols 1..M-r are Y_2), so a file that\n"
                "       is already differenced cannot be re-read at another rank.\n"
                "       Supply every series in levels (the default layout).\n");
            exit(1);
        }
        if (global_r < 1) global_r = 1;   /* r on the command line is ignored */
    } else if (global_r < 0) {
        fprintf(stderr, "ERROR: cointegration rank r must be >= 0\n");
        exit(1);
    } else if (global_r == 0 && !quiet_mode) {
        /* r = 0 is no longer an error.  It is the ladder's DIAGONAL RUNG: with
           r = 0 one has Cbar = I and Hbar = 0, the exact likelihood factorises
           with the diagonal flags, and that is where the suite's two contracts
           live (LADDER_AS_OPTIMISATION.md 2.1 and 3): the crossing identity and
           the optimality certificate.  It used to be reachable only from inside
           -lrtest, which is precisely where it cannot be inspected.  See
           docs/PLAN_BETA.md F2.8.                                            */
        printf("r = 0: no cointegration, VARMA(%d,%d) on nabla Y "
               "(the diagonal rung)\n", global_p, global_q);
    }
    return -1;
}

/*****************************************************************************/
/*  P8 — run_eval: the -eval mode, out of main().  It evaluates the           */
/*  likelihood AT THE STARTING POINT and stops, which is the diagnostic that  */
/*  separates a bad seed from a bad path.  Returns the exit code.             */
/*****************************************************************************/
static int run_eval(real *x, int npar, struct Tvarma *vp)
{
    int ifault = 0;
        const real LOG2PI = 1.837877066;
        real pi1, pi2, pi3, ll;
        int ifev = 0;
        elf(vp->m, vp->n, vp->p, vp->q, vp->mu, vp->phi,
            vp->theta, vp->qq, vp->w, 1.0, vp->xitol,
            TRUE, vp->a, &pi1, &pi2, &pi3, &ifev);
        if (ifev > 0) {
            printf("eval: elf returns ifault = %d at the starting point\n", ifev);
            fprintf(outputv, "eval: ifault = %d\n", ifev);
        } else {
            ll = -0.5 * vp->m * vp->n * (LOG2PI - log((real) vp->m)
                 - log((real) vp->n) + 1.0)
                 - 0.5 * vp->n * (vp->m * log(pi1) + log(pi2));
            printf("eval: logelf at the starting point = %15.10f  "
                   "(sigma2 = %.10f)\n", ll, pi1 / (vp->n * vp->m));
            fprintf(outputv, "eval logelf : %15.10f\n", ll);
            if (seed_have_uv) {
                /* The ladder's crossing identity: with r = 0 and diagonal structure
                   the exact likelihood factorises, so this must coincide with
                   the sum of the univariate ones from the .pre files.  What is
                   left over is the transformation's gap, not the fit's.      */
                fprintf(outputv, "sum univariate : %15.10f\n", seed_logl_sum);
                printf("      sum univariate = %15.10f   difference = %.3e\n",
                       seed_logl_sum, ll - seed_logl_sum);
            }
        }
        vec_shootx(x, vp, &ifault, 0, 1);   /* deallocate */
        fclose(outputv);
        cleanup_names(outputf, inputf, base_name);
        return 0;
}

int main(int argc, char *argv[])
{
    /*  P12.  GSL's default handler calls abort() on any error.  Every GSL call
     *  in this program checks its return code -- canonical_b2's generalised
     *  eigenproblem, granger_smin's SVD -- and falls back when it fails, but
     *  the default handler never lets them return: on collinear data the
     *  Johansen seed, which the search now always tries, killed the program
     *  with signal 6 instead of letting it discard that start.              */
    gsl_set_error_handler_off();


    {
        int rc = parse_cli(argc, argv);
        if (rc >= 0) return rc;
    }

    /*  The names of the products.  outputf has to be sized from out_base and
     *  not from argv[1], which on the .pre route is one file of several.     */
    {
        int need = (int) strlen(out_base) + 8;
        char *o = NEW_STR(need);
        if (!o) { fprintf(stderr, "ERROR: out of memory for file names\n"); exit(1); }
        FREE_STR(outputf);
        outputf = o;
    }
    strcpy(outputf, out_base);
    strcat(outputf, ".out");
    if (pre_route) snprintf(inputf, (size_t) strlen(argv[1]) + 8, "%s", "the .pre files");
    else           strcat(inputf, ".inp");

    printf("\nDRVEC %s — VEC model EML estimation (Mauricio 2006)\n",
           DRVEC_VERSION);
    printf("Input  : %s\n", inputf);
    printf("Output : %s\n", outputf);
    printf("Model  : VEC(%d) with stationary VARMA(%d,%d) on Ȳ_t\n",
           global_r, global_p, global_q);
    printf("Case   : %d\n", global_case);

    /* [1] Read the data ---------------------------------------------------- */
    if (pre_route) {
        /*  P9 — one univariate model per series, as in drtran.  Everything the
         *  .inp reader below sets, this sets too, so nothing downstream can
         *  tell the two routes apart.                                        */
        if (read_pre_inputs(pre_files, n_pre) != 0) exit(1);
        if (xl_on && xl_load() != 0) exit(1);
        if (global_r >= nser) {
            fprintf(stderr, "ERROR: r=%d must be < M=%d\n", global_r, nser);
            exit(1);
        }
    } else {
        read_inp_input(inputf);
    }

    /* The data must contain, in column order [Y_2 block ; Y_1 block]:
       - cols 1..s (s = M - r): Y_{2t} in levels  (or ∇Y_{2t} with -differenced)
       - cols s+1..M (r):       Y_{1t} in levels
    */
    build_y2_levels();          /* once: never inside the likelihood loop */

    if (!global_levels && global_case == 1)
        fprintf(stderr,
            "WARNING: -case 1 with -differenced.  Y_2 levels are reconstructed by\n"
            "         cumulating from an arbitrary zero origin, which shifts W_t\n"
            "         by B2'c.  With E[W] = 0 (case 1) nothing absorbs that shift\n"
            "         and it contaminates B2.  Supply every series in levels (the\n"
            "         default layout), or use -case 2 / -case 3.\n");

    printf("Series: %d, Obs: %d, Rank: r=%d  (%s)\n", nser, nobs, global_r,
           global_levels ? "levels" : "legacy pre-differenced layout");

    /* -alpha / -weakex: load the A of the alpha = A*psi restriction.  Done
       here because it needs nser, and before npar is computed.               */
    if (nodrift_on) {
        int i_;
        for (i_ = nser + 1; i_ <= ND_MAX; i_++)
            if (nodrift[i_]) {
                fprintf(stderr, "ERROR: -nodrift %d: the system has %d series\n",
                        i_, nser);
                exit(2);
            }
    }
    if (gz_on) {
        int i_, j_;
        for (i_ = 1; i_ <= GZ_MAX; i_++)
            for (j_ = 1; j_ <= GZ_MAX; j_++)
                if (gz[i_][j_] && (i_ > nser || j_ > nser)) {
                    fprintf(stderr, "ERROR: -gzero %d %d: the system has %d series\n",
                            i_, j_, nser);
                    exit(2);
                }
        printf("Gamma(k) held at 0 for k = 1..%d (-gzero):\n", global_p - 1);
        for (i_ = 1; i_ <= nser; i_++)
            for (j_ = 1; j_ <= nser; j_++)
                if (gz[i_][j_])
                    printf("  D.%s <- D.%s\n", series_names[i_], series_names[j_]);
    }
    if (xs_on) {
        int i_, c_;
        if (xs_nx >= nser) {
            fprintf(stderr, "ERROR: -xsys declares every series an input: "
                            "there is nothing left to explain.\n");
            exit(2);
        }
        for (i_ = nser + 1; i_ <= XS_MAX; i_++)
            if (xs_inp[i_]) {
                fprintf(stderr, "ERROR: -xsys %d: the system has %d series\n",
                        i_, nser);
                exit(2);
            }
        for (i_ = nser - global_r + 1; i_ <= nser; i_++)
            if (xs_inp[i_]) {
                fprintf(stderr,
                    "ERROR: -xsys %d: series %d is in the Y1 block (the last %d of\n"
                    "       the input), on which beta is normalised.  An input does\n"
                    "       not adjust to the equilibrium, so it cannot be the\n"
                    "       series a relation is solved for: put it in the nabla Y2\n"
                    "       block (among the first %d).\n",
                    i_, i_, global_r, nser - global_r);
                exit(2);
            }
        /*  alpha_X = 0 for every input: A is the identity without their
         *  columns, kept in the .inp's order like -weakex's (BUG-28/29).
         *  At r = 0 there is no alpha and the mask is only on F.          */
        if (global_r >= 1) {
            alpha_sa = nser - xs_nx;
            alpha_A  = matrix(1, nser, 1, alpha_sa);
            for (i_ = 1; i_ <= nser; i_++)
                for (c_ = 1; c_ <= alpha_sa; c_++) alpha_A[i_][c_] = 0.0;
            for (i_ = 1, c_ = 0; i_ <= nser; i_++)
                if (!xs_inp[i_]) alpha_A[i_][++c_] = 1.0;
            global_alpha = 1;
        }
        printf("Inputs embedded in the system (-xsys), strongly exogenous:\n");
        for (i_ = 1; i_ <= nser; i_++)
            if (xs_inp[i_])
                printf("  %-12s alpha = 0%s, no other series in its short run\n",
                       series_names[i_], global_r >= 1 ? "" : " (r = 0: none)");
    }
    else if (global_alpha) {
        int bad = alpha_weakex ? build_weakex_A(alpha_weakex)
                               : load_alpha_A(alpha_file);
        if (bad) exit(1);
        if (global_r < 1) {
            fprintf(stderr, "ERROR: alpha = A*psi means nothing with r = 0 "
                            "(there is no error-correction term)\n");
            exit(1);
        }
        printf("Restriction H1(r): alpha = A*psi, with A of %d x %d%s\n",
               nser, alpha_sa,
               alpha_weakex ? " (weak exogeneity)" : "");
    }

    /* -writeinp: emit one .inp per component of Ȳ and stop.  It is a mode, not
       an addition to the estimation: the next step of the ladder is fue's.    */
    if (global_writeinp) {
        int bad;
        printf("Writing one .inp per component of Ȳ (for ART/fue):\n");
        bad = write_component_inps(inp_prefix);
        if (bad) { fprintf(stderr, "ERROR: not all of them could be written\n"); exit(1); }
        printf("Now: for each file, 'python -m fue %s.<i> eml' (or ART),\n"
               "and then drvec ... -seed %s\n", inp_prefix, inp_prefix);
        exit(0);
    }

    /* -seed: read the .pre files and seed the MA block (the only thing the
       .pre can seed; see load_seed_pre).                                      */
    if (global_seed) {
        /* The univariate information reaches the DIAGONAL RUNG and no higher.
           With r >= 1 the marginal of a component of Ybar is not the diagonal
           block of the joint model, Cbar and Lambda couple, and PhiBar_p is
           determined by F_{p-1}, so the AR is overdetermined.  It is measured:
           in -case 2 the seed starts 17 units below the cold start.  It warns
           instead of forbidding, because the measurement has to be
           reproducible.  See docs/PLAN_BETA.md F2.8.                         */
        if (global_r > 0 && seed_route == SEED_YBAR)
            fprintf(stderr,
                "WARNING: -seedybar with r = %d.  Univariate information carries an\n"
                "         optimum only on the diagonal rung (r = 0); with r >= 1\n"
                "         it makes the starting point worse.  Measured in\n"
                "         docs/PLAN_BETA.md F2.8.\n", global_r);
        if (load_seed_pre(pre_prefix) != 0)
            fprintf(stderr, "WARNING: no seed; starting cold\n");
        else
            printf("MA seed read from %s.<1..%d>.pre (route %s)\n",
                   pre_prefix, nser,
                   seed_route == SEED_RESID ? "residuals" : "components of Ybar");
    }

    /* [2] Open output ------------------------------------------------------ */
    outputv = fopen(outputf, "w");
    if (!outputv) { fprintf(stderr, "ERROR: cannot write %s\n", outputf); exit(1); }

    /*  P10 — THE HEADER, in the suite's `key : value` shape (drvarma.c:415 and
     *  report.py:_header_block).  It says with what the file was produced,
     *  which is the one thing a results file cannot omit, and it says it the
     *  way the other programs of the suite say it.                           */
    /*  WHAT THE MODEL IS AND HOW IT IS ESTIMATED, named properly.  The
     *  program is not "Mauricio 2006": it is a compendium, and citing one
     *  paper beside the name is not what a results file does.  What a results
     *  file states is the MODEL and the ALGORITHM, which is what makes the
     *  numbers reproducible: a VARMA-VEC estimated by exact unconditional
     *  maximum likelihood, with Mauricio's algorithm for the transformation
     *  and AS 311 for the exact likelihood of the resulting stationary
     *  system.  The references are in docs/REFERENCES.md.                    */
    fprintf(outputv, "Program          : DRVEC %s\n", DRVEC_VERSION);
    fprintf(outputv, "Input Data File  : %s\n", inputf);
    fprintf(outputv, "Output File      : %s\n", outputf);
    if (global_lrtest)   /* BUG-45: r is varied, not the one on the line */
        fprintf(outputv, "Model            : VARMA-VECM(%d,%d), M = %d series, "
                         "cointegration rank r = 0..%d (sequential test)\n",
                global_p, global_q, nser, nser - 1);
    else
        fprintf(outputv, "Model            : VARMA-VECM(%d,%d), M = %d series, "
                         "cointegration rank r = %d\n",
                global_p, global_q, nser, global_r);
    /*  THE ALGORITHM HAS A NAME, and it is not the author's surname: the
     *  exact likelihood of the transformed stationary system is evaluated by
     *  ALGORITHM AS 311 (Mauricio 1997), and the transformation that turns the
     *  VECM into that stationary system is Mauricio (2006).  Naming the
     *  algorithm and citing it in parentheses is what a results file does; the
     *  full references are in docs/REFERENCES.md.                            */
    fprintf(outputv, "Estimation       : %s\n", (met == 2)
            ? "Exact Unconditional Maximum Likelihood, Algorithm AS 311 "
              "(Mauricio 1997), xi sequence NOT truncated (-m 2)"
            : "Exact Unconditional Maximum Likelihood, Algorithm AS 311 "
              "(Mauricio 1997), xi sequence truncated at 1e-3");
    fprintf(outputv, "Transformation   : VECM to stationary VARMA "
                     "(Mauricio 2006)\n");
    fprintf(outputv, "Deterministic    : case %d -- %s%s%s\n",
            global_trend ? 4 : global_case,
            global_case == 1 ? "E[nabla Y2] = 0, E[W] = 0"
          : global_case == 2 ? "E[nabla Y2] = 0, E[W] free"
                             : "E[nabla Y2] free, E[W] free",
            nodrift_on ? (case_pre ? " (drifts as in the .pre files)"
                                   : " (some drifts held by -nodrift)") : "",
            global_trend ? ", and a trend in W (restricted to beta)" : "");
    /* The deterministic terms come off the LEVELS, before nabla Y2 and W are
       formed -- which is where fue's cast removes them too, its block [6] comes
       before [7] -- and that is why the levels have to be rebuilt afterwards.
       It goes here, and not earlier, because it leaves a record in the .out and
       that file is not open yet further up: writing there blew up with outputv
       at NULL.                                                                */
    if (global_interv) {
        printf("Subtracting the deterministic terms declared in the .pre files:\n");
        subtract_interventions(interv_prefix);
        build_y2_levels();
    }

    /*  The structure, in ONE line: three "Diagonal X : no" lines said less
     *  than this does, and the MA class -- which is what actually moved on
     *  2026-08-20 -- gets named rather than deduced.                         */
    fprintf(outputv, "Structure        : F %s, Theta %s, Sigma %s\n",
            global_diag_ar ? "diagonal"
          : gz_on ? "with entries held at 0 (-gzero, below)" : "free",
            global_q == 0 ? "absent (q = 0)"
          : global_warma  ? "diagonal in the W block (-warma)"
          : mawarma_on()  ? "= [T11  T11*B2' ; 0  0] (-mawarma)"
          : global_matri  ? "= [T11 T12 ; 0 T22] (-matri)"
          : global_diag_ma ? "diagonal (-diagma)"
          : marow_on()    ? "= [T11 T12 ; 0 0] (-marow)"
                          : "free (the default; -mafree)",
            global_diag_cov ? "diagonal" : "free");
    fprintf(outputv, "Series           :");
    { int j; for (j = 1; j <= nser; j++)
        fprintf(outputv, " %s", series_names ? series_names[j] : "y"); }
    fprintf(outputv, "   (%d in the nabla Y2 block, %d in Y1)\n",
            nser - global_r, global_r);
    {   /*  BUG-37: the first ESTIMATED observation, not the first raw one  */
        int ey = data_start_year, et = data_start_sub;
        ObsToDate(data_start_year, data_start_sub, nobs_raw - nobs + 1,
                  data_freq, &ey, &et);
        fprintf(outputv, "Sample           : %d observations from %d",
                nobs, ey);
        if (data_freq > 1) fprintf(outputv, "/%d", et);
    }
    fprintf(outputv, ", %s (raw %d)\n",
            data_freq == 1 ? "annual" : data_freq == 4 ? "quarterly"
          : data_freq == 12 ? "monthly" : "irregular", nobs_raw);
    if (!global_levels)
        fprintf(outputv, "  ! legacy layout: cols 1..s arrive already "
                         "differenced (-differenced)\n");
    /*  BUG-43: a series whose changes are orders of magnitude from 1 leaves
     *  the optimiser (finite-difference steps of ~6e-6) stopping where it
     *  started.  -lrtest rescales internally; a single fit reports in the
     *  data's units, so here it is said, with the factor to use.          */
    if (!global_lrtest && nobs_raw > 2) {
        int warned = 0;
        for (int i2 = 1; i2 <= nser; i2++) {
            real m1 = 0.0, v = 0.0, sd;
            int T = nobs_raw - 1;
            for (int t = 2; t <= nobs_raw; t++) m1 += rawmat[t][i2] - rawmat[t-1][i2];
            m1 /= T;
            for (int t = 2; t <= nobs_raw; t++) {
                real d = rawmat[t][i2] - rawmat[t-1][i2] - m1; v += d * d;
            }
            sd = sqrt(v / (T > 1 ? T - 1 : 1));
            if (sd > 0.0 && fabs(log10(sd)) > 2.0) {
                if (!warned)
                    fprintf(outputv, "  ! SCALE: the optimiser works on changes of "
                                     "order 1, and these are not (BUG-43):\n");
                fprintf(outputv, "      %-12s s.d. of its changes %.3g: multiply it "
                                 "by %g\n", series_names ? series_names[i2] : "y",
                        sd, pow(10.0, -floor(log10(sd) + 0.5)));
                if (!warned && !quiet_mode)
                    fprintf(stderr, "WARNING: series on very different scales; "
                                    "see SCALE in the .out (BUG-43)\n");
                warned = 1;
            }
        }
    }
    /*  A Box-Cox or a differencing order declared in the .inp is NOT applied
     *  on this route -- drvec forms nabla Y2 itself and takes the series as
     *  they come.  Reading a directive and ignoring it in silence is the one
     *  thing this program refuses to do everywhere else.                     */
    if (!pre_route && (fabs(trans_lambda - 1.0) > 1.0e-12 || trans_d || trans_D))
        fprintf(outputv, "  ! the .inp declares lambda = %g, d = %d, D = %d, and "
                         "this route applies NONE of them:\n"
                         "    supply the series already transformed, or use the "
                         ".pre route, which does apply them\n",
                trans_lambda, trans_d, trans_D);

    /*  P4 — WHICH CLASS IS BEING ESTIMATED, said in the header and not
     *  deduced from the flags.  The default moved on 2026-08-20 and an .out
     *  without this line is ambiguous with respect to the whole earlier
     *  register.                                                             */
    if (global_q > 0) {
        const char *cls =
            global_warma   ? "triangular (WARMA), estimated in Ybar coordinates"
          : mawarma_on()   ? "Theta = [T11  T11*B2' ; 0  0]  (-mawarma)"
          : global_matri   ? "Theta = [T11  T12 ; 0  T22]  (-matri)"
          : global_diag_ma ? "Theta diagonal  (-diagma)"
          : marow_on()     ? "Theta = [T11  T12 ; 0  0]  (-marow)"
          :                  "Theta FREE  (the default, Mauricio 2006; -marow was the default 2026-08-20 to 2026-09-23)";
        if (!quiet_mode) printf("MA     : %s\n", cls);
    }

    /*  P1 — THE UPPER BOUND ON p AND q IS SET BY THE SAMPLE, not by a number.
     *  Until 2026-08-20 `drvec file 60 1 1` on 61 observations was attempted:
     *  it failed with ifault = 3 after a while, and returned 0 to the shell.
     *  And `q = 60` hung.  A model with as many parameters as data is not an
     *  ill-conditioned model, it is a model that is not identified, and
     *  estimating it produces not a result but a number.  The limit is set
     *  where it can be measured -- with the file already read -- and on npar,
     *  which is the quantity that really governs: it covers large p, large q
     *  and large M alike.
     *
     *  The threshold is npar < nobs * M, i.e. at least one datum per parameter.
     *  It is deliberately generous: it is not a statistical criterion -- the
     *  AIC and BIC the program already prints are there for that -- but the
     *  boundary below which the fit means nothing at all.                    */
    {
        int npar_check = calc_nparametrs();
        if (npar_check >= nobs * nser) {
            fprintf(stderr,
                "drvec: the model has %d parameters and the sample has %d data "
                "points\n"
                "       (%d observations x %d series).  There is nothing to "
                "estimate:\n"
                "       lower p or q, or use a longer sample.\n",
                npar_check, nobs * nser, nobs, nser);
            fprintf(outputv,
                "\nREFUSED: %d parameters against %d data points.  Not estimated.\n",
                npar_check, nobs * nser);
            fclose(outputv);
            return 1;
        }
    }

    /* [3a] Sequential LR test for the cointegration rank (Mauricio 2006,
            Remark 5 and Table 3): estimate r = 1..M-1 and report
            2*[L(r+1) - L(r)] against the non-standard asymptotic values.    */
    /*  -rungs — THE LADDER, emitted by the program and not assembled by the
     *  user.
     *
     *  WHY.  The suite's construction goes from OPTIMA TO OPTIMA: each rung is
     *  estimated, certified and handed to the one above.  Until now drvec knew
     *  how to certify ITS BASE (the diagonal gate) and how to test the rank
     *  (-lrtest), but the intermediate rungs -- the ones from the base to free
     *  cross dynamics -- had to be assembled by hand, running the program three
     *  times and subtracting.  A user doing that by hand gets the degrees of
     *  freedom wrong, and above all leaves no record.
     *
     *  WHAT EACH RUNG IS.  All with r = 0, i.e. with no VEC matrix yet: what is
     *  added is correlation structure, not cointegration.
     *
     *    0   F, Theta and Sigma diagonal     the certified base; the
     *                                        likelihood factorises
     *    1   Sigma free                      contemporaneous correlation
     *    2   F and Theta free                cross dynamics
     *
     *  All three are ORDINARY nested comparisons -- the restricted model is an
     *  interior point of the wide one -- so the logL cannot fall and the
     *  statistic is chi2 with the degrees of freedom printed.  The next rung,
     *  r = 1, is NOT ordinary either in the seeding or in the distribution, and
     *  that is why it lives in -lrtest and not here: see
     *  docs/VEC_EMBEDDING_PLAN.md.                                           */
    if (global_rungs) return run_rungs();

    /*  -specs — THE SPECIFICATION LADDER, emitted by the program.
     *
     *  Five NESTED models of the moving average and the short-run dynamics,
     *  from the most restricted to the free one, in a single command:
     *
     *    warma    Phi*_k = [0 Psi_k ; 0 Phi_k] and Theta* diagonal -- the class
     *             the theorems cover (docs/THEORY.md, definition 3)
     *    mawarma  Theta = [T11 T11B2' ; 0 0]   -- the same MA structure with F
     *             free
     *    marow    Theta = [T11 T12 ; 0 0]      -- the cross block free
     *    matri    Theta = [T11 T12 ; 0 T22]    -- and the differenced one with
     *             an MA
     *    free     Theta free
     *
     *  Why in one command: because the question a user has in front of them is
     *  not "how well does this specification fit" but "which of them, and is it
     *  admissible", and those two are not answered by one run.
     *
     *  AND WHY THE ADMISSIBILITY COLUMN IS THE FIRST ONE TO READ.  By theorem 3
     *  of docs/THEORY.md the process has rank r if and only if
     *  rank(Lambda_perp' Theta(1)) = M - r, and by theorem 4 the set where that
     *  fails is INSIDE the one the optimiser walks.  A rung with a small G is
     *  not a worse fit: it is the fit of another model.  By corollary 5.1,
     *  moreover, neither its standard errors nor an LR against it have their
     *  distribution, so the chi2 p-value is printed ONLY when both rungs
     *  compared are admissible, and where they are not, that is said and the
     *  reader is sent to -matest.                                            */
    if (global_specs) return run_specs();

    /*  -matest N — the test of the inherited MA against the free one, with its
     *  distribution SIMULATED rather than assumed.  It is a mode and ends
     *  here.                                                                 */
    if (global_matest > 0 || global_artest > 0) return run_ma_ar_test();

    if (global_lrtest) return run_lrtest();

    /* [3] Estimation ------------------------------------------------------- */
    int npar = calc_nparametrs();
    real *x   = vector(1, npar);
    real *dev = vector(1, npar);
    real **cov = matrix(1, npar, 1, npar);

    struct Tvarma varma1;
    macheps = cmacheps();
    varma1.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;

    /* With alpha = A*psi the FREE model is needed for the LR, so H(r) is
       estimated first and H1(r) afterwards.  The degrees of freedom are
       (M - sa)*r, explicit in Johansen and Swensen (2024).                    */
    /*  -xlink: the VEC rung below -- the same model without the transfer --
     *  fitted first, with its own ladder.  Its optimum with omega = 0 is the
     *  BRIDGE (the start of the hybrid) and its logL the base of the LR.   */
    real *xl_xbase = NULL; int xl_npb = 0;
    if (xl_on) {
        int iff; real s2b = 0.0, llb = 0.0;
        real *devb, **covb;
        fprintf(outputv, "\n=== -xlink: the VEC rung below, without the transfer "
                         "inputs ===\n");
        printf("  -xlink: fitting the VEC without the inputs (the rung below)\n");
        xl_on = 0;
        xl_npb = calc_nparametrs();
        xl_xbase = vector(1, xl_npb);
        devb = vector(1, xl_npb);
        covb = matrix(1, xl_npb, 1, xl_npb);
        init_guess(xl_xbase, xl_npb);
        if (ladder_wanted() && global_r > 0) gate_profile_seed(xl_xbase, xl_npb);
        else if (ladder_wanted() && global_r == 0) ladder_seed_r0(xl_xbase, xl_npb);
        iff = fit_search(xl_xbase, xl_npb, devb, covb, &llb, &s2b, 1, 0, NULL);
        xl_on = 1;
        xl_base_ok = (iff == 0);
        xl_base_ll = llb;
        gate_seed_ok = 0;            /* that profile was the rung below's */
        printf("  VEC without inputs : logL = %15.10f%s\n", llb,
               xl_base_ok ? "" : "  (the estimation failed)");
        fprintf(outputv, "logL of the VEC without the inputs: %18.10f\n", llb);
        free_matrix(covb, 1, xl_npb, 1, xl_npb);
        free_vector(devb, 1, xl_npb);
    }

    real lr_free = 0.0; int lr_free_ok = 0;
    if (b2row_on) {
        /*  -fixb2row (alone or with -alpha/-weakex): the row must exist, and
         *  the FREE model -- every restriction off -- is fitted with the same
         *  search as the main fit, so that the LR compares two optima.        */
        int s_ = nser - global_r, i_, sa_ = global_alpha, bo_ = b2row_on;
        if (global_r < 1) {
            fprintf(stderr, "ERROR: -fixb2row needs r >= 1: at r = 0 there is no "
                            "beta to restrict.\n");
            exit(2);
        }
        for (i_ = s_ + 1; i_ <= B2ROW_MAX; i_++)
            if (b2row_fix[i_]) {
                fprintf(stderr, "ERROR: -fixb2row %d: B2 has %d row%s (the nabla Y2 "
                        "block, the first %d series in the input's order).\n",
                        i_, s_, (s_ == 1) ? "" : "s", s_);
                exit(2);
            }
        /*  With -xsys the inputs stay embedded in the free model too: the LR
         *  is then about beta inside the conditional VEC.                  */
        global_alpha = xs_on ? sa_ : 0; b2row_on = 0;
        {
            int npf = calc_nparametrs(), iff;
            real *xf = vector(1, npf), *devf = vector(1, npf);
            real **covf = matrix(1, npf, 1, npf);
            real llf = 0.0, s2f = 0.0;
            FILE *o_save = outputv, *nul = tmpfile();
            int q_save = quiet_mode;
            if (nul) outputv = nul;
            quiet_mode = 1;
            init_guess(xf, npf);
            iff = fit_search(xf, npf, devf, covf, &llf, &s2f, 1, 0, NULL);
            outputv = o_save; quiet_mode = q_save;
            if (nul) fclose(nul);
            lr_free_ok = (iff == 0);
            lr_free    = llf;
            printf("  H(r)  free         : logL = %15.10f%s\n", lr_free,
                   lr_free_ok ? "" : "  (the estimation failed)");
            free_matrix(covf, 1, npf, 1, npf);
            free_vector(devf, 1, npf);
            free_vector(xf, 1, npf);
        }
        global_alpha = sa_; b2row_on = bo_;
    }
    else if (xs_on) {
        /*  -xsys: the FREE VEC -- no mask on F, no alpha = 0 -- with the same
         *  search, so that the LR of strong exogeneity compares two optima. */
        int sa_ = global_alpha, npr = calc_nparametrs();
        xs_on = 0; global_alpha = 0;
        {
            int npf = calc_nparametrs(), iff;
            real *xf = vector(1, npf), *devf = vector(1, npf);
            real **covf = matrix(1, npf, 1, npf);
            real llf = 0.0, s2f = 0.0;
            FILE *o_save = outputv, *nul = tmpfile();
            int q_save = quiet_mode;
            xs_df = npf - npr;
            if (nul) outputv = nul;
            quiet_mode = 1;
            init_guess(xf, npf);
            if (ladder_wanted() && global_r > 0) gate_profile_seed(xf, npf);
            else if (ladder_wanted() && global_r == 0) ladder_seed_r0(xf, npf);
            iff = fit_search(xf, npf, devf, covf, &llf, &s2f, 1, 0, NULL);
            outputv = o_save; quiet_mode = q_save;
            if (nul) fclose(nul);
            lr_free_ok = (iff == 0);
            lr_free    = llf;
            gate_seed_ok = 0;            /* that profile was the free model's */
            printf("  VEC free (no -xsys): logL = %15.10f%s\n", lr_free,
                   lr_free_ok ? "" : "  (the estimation failed)");
            free_matrix(covf, 1, npf, 1, npf);
            free_vector(devf, 1, npf);
            free_vector(xf, 1, npf);
        }
        xs_on = 1; global_alpha = sa_;
    }
    else if (gz_on) {
        /*  -gzero (and -alpha/-weakex if given): the FREE VEC -- no mask on F,
         *  alpha free -- with the same search, as -xsys's.                   */
        int sa_ = global_alpha, npr = calc_nparametrs();
        gz_on = 0; global_alpha = 0;
        {
            int npf = calc_nparametrs(), iff;
            real *xf = vector(1, npf), *devf = vector(1, npf);
            real **covf = matrix(1, npf, 1, npf);
            real llf = 0.0, s2f = 0.0;
            FILE *o_save = outputv, *nul = tmpfile();
            int q_save = quiet_mode;
            gz_df = npf - npr;
            if (nul) outputv = nul;
            quiet_mode = 1;
            init_guess(xf, npf);
            if (ladder_wanted() && global_r > 0) gate_profile_seed(xf, npf);
            else if (ladder_wanted() && global_r == 0) ladder_seed_r0(xf, npf);
            iff = fit_search(xf, npf, devf, covf, &llf, &s2f, 1, 0, NULL);
            outputv = o_save; quiet_mode = q_save;
            if (nul) fclose(nul);
            lr_free_ok = (iff == 0);
            lr_free    = llf;
            gate_seed_ok = 0;            /* that profile was the free model's */
            printf("  VEC free (no -gzero%s): logL = %15.10f%s\n",
                   sa_ ? ", alpha free" : "", lr_free,
                   lr_free_ok ? "" : "  (the estimation failed)");
            free_matrix(covf, 1, npf, 1, npf);
            free_vector(devf, 1, npf);
            free_vector(xf, 1, npf);
        }
        gz_on = 1; global_alpha = sa_;
    }
    else if (global_alpha) {
        int save = global_alpha;
        global_alpha = 0;
        {
            int npf = calc_nparametrs();
            real *xf = vector(1, npf), *devf = vector(1, npf);
            real **covf = matrix(1, npf, 1, npf);
            struct Tvarma vf;
            int iff;
            vf.xitol = varma1.xitol;
            init_guess(xf, npf);
            vec_shootx(xf, &vf, &iff, 1, 0);
            est(&vec_shootx, npf, xf, devf, covf, 500, 200, 1e-5, 1e-7,
                vf.xitol, vf.a, &vf.sigma2, &vf.logelf, &iff);
            lr_free_ok = (iff == 0);
            lr_free    = vf.logelf;
            printf("  H(r)  free         : logL = %15.10f%s\n", lr_free,
                   lr_free_ok ? "" : "  (the estimation failed)");
            vec_shootx(xf, &vf, &iff, 0, 1);
            free_matrix(covf, 1, npf, 1, npf);
            free_vector(devf, 1, npf);
            free_vector(xf, 1, npf);
        }
        global_alpha = save;
    }

    init_guess(x, npar);

    /*  -seedgate: route (B).  It goes AFTER init_guess and not in its place,
     *  for two reasons.  The head and tail of the vector -- the mean, Lambda
     *  and B2 -- need a starting point for the conditional step, and the
     *  conditional regression's is the one there is.  And if the profiling does
     *  not work out, what is left is exactly route (C), with no invented
     *  intermediate route: it either crosses whole or it does not cross.     */
    /*  -warma: an ADMISSIBLE start, by shrinking the autoregressive block.
     *
     *  The same wall as -seedgate from another side: the regression that seeds
     *  the coefficients of W_{t-k} can give a non-stationary Phi*, and then est
     *  does not even start ("bad initial estimates") and there is no fit, which
     *  is what happened on mink_muskrat.  A ladder of factors on THAT block is
     *  walked -- the direction given by the data, the scale by admissibility --
     *  and the start is the first one the engine accepts.  With factor 0 the
     *  system is Ybar_t = A*_t, trivially stationary, so the ladder always
     *  terminates.                                                           */
    if (global_warma) {
        static const real shr[6] = { 1.0, 0.8, 0.5, 0.3, 0.1, 0.0 };
        int nmean_, nlam_, nmid_, ntail_, nf_w = (global_p > 1) ? global_p - 1 : 0;
        int nar, i2, mi;
        real *ar0;
        struct Tvarma vt;
        int ift = 0;

        par_blocks(&nmean_, &nlam_, &nmid_, &ntail_);
        nar = nlam_ + nf_w * nser * global_r;
        ar0 = vector(1, (nar > 0 ? nar : 1));
        for (i2 = 1; i2 <= nar; i2++) ar0[i2] = x[nmean_ + i2];
        vt.xitol = (met == 2) ? -1.0e-3 : 1.0e-3;
        vec_shootx(x, &vt, &ift, 1, 0);
        for (mi = 0; mi < 6; mi++) {
            real pi1, pi2, pi3;
            int ifev = 0, ifc = 0;
            for (i2 = 1; i2 <= nar; i2++) x[nmean_ + i2] = shr[mi] * ar0[i2];
            vec_shootx(x, &vt, &ifc, 0, 0);
            if (ifc != 0) continue;
            elf(vt.m, vt.n, vt.p, vt.q, vt.mu, vt.phi, vt.theta, vt.qq, vt.w,
                1.0, vt.xitol, TRUE, vt.a, &pi1, &pi2, &pi3, &ifev);
            if (ifev == 0) break;
        }
        if (mi > 0 && mi < 6 && !quiet_mode)
            printf("  -warma: start shrunk to x%.1f to make it admissible\n",
                   shr[mi]);
        if (mi >= 6) {
            fprintf(outputv, "\n-warma: no admissible starting point was found "
                             "even with the autoregressive block at zero.\n");
            for (i2 = 1; i2 <= nar; i2++) x[nmean_ + i2] = 0.0;
        }
        vec_shootx(x, &vt, &ift, 0, 1);
        free_vector(ar0, 1, (nar > 0 ? nar : 1));
    }

    if (global_seedb2) {
        int nmean_, nlam_, nmid_, ntail_, i2;
        par_blocks(&nmean_, &nlam_, &nmid_, &ntail_);
        for (i2 = 1; i2 <= ntail_; i2++)
            x[nmean_ + nlam_ + nmid_ + i2] = global_seedb2_value;
        if (ntail_ == 0)
            fprintf(stderr, "WARNING: -seedb2 does nothing with -fixb2 or r = 0\n");
    }
    /*  P12: the ladder is the default start (the suite's convention: from
     *  the certified gate, rung by rung, then route (B) across the rank).   */
    if (!xl_on && ladder_wanted() && global_r > 0)
        gate_profile_seed(x, npar);
    else if (!xl_on && ladder_wanted() && global_r == 0)
        ladder_seed_r0(x, npar);
    /*  -xlink: the bridge.  The rung below's optimum, the transfer at zero:
     *  the block is last in x[], so the first xl_npb entries line up.      */
    if (xl_on && xl_xbase) {
        int ib;
        for (ib = 1; ib <= xl_npb; ib++) x[ib] = xl_xbase[ib];
        for (ib = xl_npb + 1; ib <= npar; ib++) x[ib] = 0.0;
        xl_bridge_ll = xl_eval_ll(x, varma1.xitol, &xl_bridge_ok);
        printf("  bridge (VEC optimum, omega = 0): logL = %15.10f%s\n",
               xl_bridge_ll, xl_bridge_ok ? "" : "  (could not be evaluated)");
        free_vector(xl_xbase, 1, xl_npb);
        xl_xbase = NULL;
    }

    /* -writeres: the residuals of the conditional regression, which is what
       init_guess has just published.  It is a mode and ends here.             */
    if (global_writeres) {
        printf("Writing one .inp per residual of the conditional regression:\n");
        if (write_resid_inps(res_prefix) != 0) exit(1);
        printf("Now: 'python -m fue %s.<i> eml' and then drvec ... -seed %s\n",
               res_prefix, res_prefix);
        fclose(outputv);
        cleanup_names(outputf, inputf, base_name);
        return 0;
    }

    int ifault = 0;      /* fit_search() sets it */
    vec_shootx(x, &varma1, &ifault, 1, 0);  /* allocate */

    /* -eval: the likelihood AT THE STARTING POINT, without optimising.
       It is the diagnostic that separates two things easily confused: a bad
       seed (it starts worse) from an optimiser that from a better seed ends up
       worse (the surface).  Without this, comparing only the final logLs does
       not distinguish a sign error from a path problem.
       The formula is drvmlest.c:133's, with sigma2 = 1 and atf = TRUE.       */
    if (global_eval) return run_eval(x, npar, &varma1);


    /* -multistart n: estimate from n starting points and keep the best.
     *
     * This is NOT touching the optimiser -- which is not touched --, it is
     * running it several times.  And it is justified by two measurements of our
     * own, not by custom:
     *
     *   - F2 measured that the point where it stops depends strongly on the
     *     point where it starts: in case 1, moving Theta by hundredths moves
     *     the answer by 14.45 units.  That is exactly the condition in which
     *     multi-start pays.
     *   - The global search that serves as the reference for |Sigma| (0.002311)
     *     WAS DONE THIS WAY, by multi-start, and it ends up glued to chekma's
     *     invertibility barrier with max|lambda(Theta1)| = 1.00005.  drvec's
     *     fit ends at the SAME barrier -- 1.000050 measured -- but at another
     *     point of it, with |Sigma| 0.002461.  That is: same edge, worse spot.
     *
     * The perturbations are deterministic (own generator with a fixed seed) so
     * that a result can be reproduced: a multi-start that cannot be repeated is
     * no use as evidence.
     */
    /*  THE OPTIMALITY CERTIFICATE, and there is ONE window in which to take it.
     *
     *  The suite's ladder says that a `.pre` is an OPTIMUM in re-runnable form,
     *  and that convention is CHECKABLE: re-estimating an optimum does not move
     *  the numbers, while a specification does.  The difference between the two
     *  likelihoods -- the fit's and that of the values brought in -- is >= 0 by
     *  construction and equals zero if and only if what came in were the
     *  univariate optima.
     *
     *  It has to be evaluated HERE because est() overwrites the structure: once
     *  it has run, the question can no longer be answered.  It is the same
     *  protocol as drtran's gate (LADDER_AS_OPTIMISATION.md 2.1 and 7.1), and
     *  it is inherited whole instead of reinvented.                          */
    {
        const real LOG2PI = 1.837877066;
        real pi1, pi2, pi3;
        int ifs = 0;
        vec_shootx(x, &varma1, &ifs, 0, 0);
        if (ifs == 0) {
            elf(varma1.m, varma1.n, varma1.p, varma1.q, varma1.mu, varma1.phi,
                varma1.theta, varma1.qq, varma1.w, 1.0, varma1.xitol,
                FALSE, varma1.a, &pi1, &pi2, &pi3, &ifs);
            if (ifs == 0) {
                gate_ll_start = -0.5 * varma1.m * varma1.n
                    * (LOG2PI - log((real) varma1.m) - log((real) varma1.n) + 1.0)
                    - 0.5 * varma1.n * (varma1.m * log(pi1) + log(pi2));
                gate_have_start = 1;
            }
        }
    }

    /*  P12: the fit is the best of several starts (fit_search), and
     *  -multistart adds its perturbations to them.  run_multistart was the
     *  first version of this, for one flag and one caller.                 */
    {
        struct search_out so;
        ifault = fit_search(x, npar, dev, cov, &varma1.logelf, &varma1.sigma2,
                            (global_multistart > 1) ? global_multistart : 1,
                            1, &so);
        if (!quiet_mode && so.ntried > 1)
            printf("  Search: %d starts, %d converged; best `%s' (logL %.6f)\n",
                   so.ntried, so.nok, so.label, varma1.logelf);
    }

    report_fit(x, dev, cov, npar, &varma1, ifault, outputf,
               lr_free, lr_free_ok);
    /*  -irfboot: after the report, which it appends to; the main fit is
     *  read from x and nothing of it is changed.                          */
    if (global_irfboot > 0 && !estimation_failed)
        irf_bootstrap(x, npar, (nobs < 40) ? 10 : 20, global_irfboot);

    /* [4] Cleanup ---------------------------------------------------------- */
    vec_shootx(x, &varma1, &ifault, 0, 1);  /* deallocate */
    free_seed_pre();
    free_matrix(cov, 1, npar, 1, npar);
    free_vector(dev, 1, npar);
    free_vector(x, 1, npar);
    /*  datamat and Y2_levels are released by free_case_data(), from
     *  cleanup_names, which is where all seven exits pass.  Here they were
     *  freed with `nobs` and not with the ALLOCATION's dimension, which with
     *  -estwin no longer coincide: the window trims nobs and the matrices stay
     *  whole.                                                                */
    /*  BUG-20: the file WRITTEN, not argv[1] -- on the .pre route argv[1] is
     *  the first input model, and the message read as an announcement that
     *  it had just been overwritten.  Said before cleanup_names frees it.  */
    if (!estimation_failed)
        printf("Done. Output written to %s\n", outputf);
    fclose(outputv);
    cleanup_names(outputf, inputf, base_name);

    if (estimation_failed) return 2;
    return 0;
}
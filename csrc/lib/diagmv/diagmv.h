/* diagmv.h -- the multivariate residual diagnosis shared by drvarma, drtran
 * and drvec: Hosking's portmanteau and the multivariate Jarque-Bera.
 *
 * The host provides diagmvhost.h: `real`, matrix()/vector() and their free_*,
 * matrix_multiply/transpose/inverse, svdcp() and chisq().
 *
 * Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 * GPL v2 or later; see COPYING.
 */
#ifndef DIAGMV_H
#define DIAGMV_H

#include <stdio.h>
#include "diagmvhost.h"

/* Lags of Hosking's Q: freq + 2 for seasonal data, 10 otherwise, at most
   nobs - 2.  The rule of drvarma's Python port, which every engine follows. */
int  diagmv_lags(int freq, int nobs);

/* Hosking (1980) on residuals res[1..nobs][1..m] over s lags.  k is the
   number of DYNAMIC coefficients estimated to get them (AR, MA, adjustment,
   transfer weights: never means, Sigma or a cointegrating vector), and
   df = m^2 s - k.  With df < 1 the p-value is NaN. */
void diagmv_hosking(real **res, int nobs, int m, int s, int k,
                    real *Q, int *df, real *pval);

/* Multivariate Jarque-Bera (Lutkepohl 2005, 4.5) on the residuals
   standardised by the SYMMETRIC square root of their covariance, so that
   the statistic does not depend on the order of the series.  JB = skewness
   part + kurtosis part, each chi2(m); JB ~ chi2(2m).  NaN when the
   covariance is singular. */
void diagmv_jb(real **res, int nobs, int m, real *JB, real *skew,
               real *kurt, real *pval);

/* The block of the .out: both tests, with their degrees of freedom said. */
void multivariate_diagnostics(real **res, int nobs, int m, int freq, int k,
                              FILE *out);

#endif

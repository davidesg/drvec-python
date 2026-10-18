/*
 * diagmv.c -- the multivariate residual diagnosis of drvarma, drtran and
 * drvec: Hosking's portmanteau and the multivariate Jarque-Bera.
 *
 * It lived in each engine's diagnose.c, three copies of the same two tests,
 * and all three had the same two defects (drvec BUG-42, items 3 and 4):
 *
 *   - Hosking's Q was referred to chi2(m^2 s), as if the residuals were a
 *     series and not the output of a fitted model.  Hosking (1980) gives
 *     m^2 (s - p - q) for a VARMA(p, q); for a VECM, Bruggemann, Lutkepohl
 *     and Saikkonen (2006) give m^2 (s - p + 1) - m r, which is the same
 *     rule: subtract the dynamic coefficients estimated (Gamma and alpha,
 *     not beta, which is superconsistent).  Each engine passes its count k,
 *     restrictions applied.  The test without it hardly ever rejected.
 *   - The "multivariate" Jarque-Bera was a sum of univariate ones, which is
 *     chi2(2m) only when the residuals are uncorrelated: with correlated
 *     residuals it over-rejected (7-10 % at a nominal 5 %, simulated).  Here
 *     it is Lutkepohl's, on residuals standardised by the symmetric square
 *     root of their covariance -- not the Cholesky factor, which would make
 *     the statistic depend on the order of the series.
 *
 * And the lags of Hosking's Q are freq + 2 (10 for non-seasonal data), the
 * rule of drvarma's Python port; the C took floor(sqrt(n)).
 *
 * The host provides diagmvhost.h.
 *
 * Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 * GPL v2 or later; see COPYING.
 */

#include "diagmv.h"
#include <math.h>
#include <stdlib.h>

int diagmv_lags(int freq, int nobs)
{
    int s = (freq > 1) ? freq + 2 : 10;
    if (s > nobs - 2) s = nobs - 2;
    if (s < 1) s = 1;
    return s;
}

/* Residual means, and the residuals centred on them: res[1..n][1..m]. */
static real **centred(real **res, int nobs, int m)
{
    int t, j;
    real **x = matrix(1, nobs, 1, m);
    for (j = 1; j <= m; j++) {
        real mu = 0.0;
        for (t = 1; t <= nobs; t++) mu += res[t][j];
        mu /= nobs;
        for (t = 1; t <= nobs; t++) x[t][j] = res[t][j] - mu;
    }
    return x;
}

void diagmv_hosking(real **res, int nobs, int m, int s, int k,
                    real *Q, int *df, real *pval)
{
    int r, i, j, t;
    real **x = centred(res, nobs, m);
    real **C0 = matrix(1, m, 1, m), **C0inv = matrix(1, m, 1, m);
    real **Cr = matrix(1, m, 1, m), **Crt = matrix(1, m, 1, m);
    real **t1 = matrix(1, m, 1, m), **t2 = matrix(1, m, 1, m);
    real **t3 = matrix(1, m, 1, m);

    /* C(r)[i][j] = (1/n) sum_t x[t][i] x[t+r][j] */
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            real sm = 0.0;
            for (t = 1; t <= nobs; t++) sm += x[t][i] * x[t][j];
            C0[i][j] = sm / nobs;
        }
    matrix_inverse(C0, C0inv, m);

    *Q = 0.0;
    for (r = 1; r <= s; r++) {
        real tr = 0.0;
        for (i = 1; i <= m; i++)
            for (j = 1; j <= m; j++) {
                real sm = 0.0;
                for (t = 1; t <= nobs - r; t++) sm += x[t][i] * x[t + r][j];
                Cr[i][j] = sm / nobs;
            }
        matrix_transpose(Cr, Crt, m, m);
        matrix_multiply(Crt, C0inv, t1, m, m, m);
        matrix_multiply(t1, Cr, t2, m, m, m);
        matrix_multiply(t2, C0inv, t3, m, m, m);
        for (i = 1; i <= m; i++) tr += t3[i][i];
        *Q += tr;
    }
    *Q *= nobs;

    *df = m * m * s - k;
    *pval = (*df >= 1) ? 1.0 - chisq(*Q, *df) : NAN;

    free_matrix(t3, 1, m, 1, m);
    free_matrix(t2, 1, m, 1, m);
    free_matrix(t1, 1, m, 1, m);
    free_matrix(Crt, 1, m, 1, m);
    free_matrix(Cr, 1, m, 1, m);
    free_matrix(C0inv, 1, m, 1, m);
    free_matrix(C0, 1, m, 1, m);
    free_matrix(x, 1, nobs, 1, m);
}

void diagmv_jb(real **res, int nobs, int m, real *JB, real *skew,
               real *kurt, real *pval)
{
    int i, j, t;
    real **x = centred(res, nobs, m);
    real **S = matrix(1, m, 1, m), **V = matrix(1, m, 1, m);
    real **R = matrix(1, m, 1, m);              /* S^(-1/2) */
    real *w = vector(1, m), *v = vector(1, m);
    real *b1 = vector(1, m), *b2 = vector(1, m);
    real wmax = 0.0;

    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            real sm = 0.0;
            for (t = 1; t <= nobs; t++) sm += x[t][i] * x[t][j];
            S[i][j] = sm / nobs;
        }

    /* S symmetric positive definite: S = V diag(w) V', and
       S^(-1/2) = V diag(w^(-1/2)) V'.                                      */
    svdcp(S, m, m, w, V);
    for (j = 1; j <= m; j++) if (w[j] > wmax) wmax = w[j];
    for (j = 1; j <= m; j++)
        if (!(w[j] > 1e-12 * wmax)) {
            *JB = *skew = *kurt = *pval = NAN;
            goto done;
        }
    for (i = 1; i <= m; i++)
        for (j = 1; j <= m; j++) {
            real sm = 0.0;
            int l;
            for (l = 1; l <= m; l++) sm += V[i][l] * V[j][l] / sqrt(w[l]);
            R[i][j] = sm;
        }

    for (j = 1; j <= m; j++) b1[j] = b2[j] = 0.0;
    for (t = 1; t <= nobs; t++) {
        for (i = 1; i <= m; i++) {
            real sm = 0.0;
            for (j = 1; j <= m; j++) sm += R[i][j] * x[t][j];
            v[i] = sm;
        }
        for (i = 1; i <= m; i++) {
            real v2 = v[i] * v[i];
            b1[i] += v2 * v[i];
            b2[i] += v2 * v2;
        }
    }
    *skew = *kurt = 0.0;
    for (i = 1; i <= m; i++) {
        real s3 = b1[i] / nobs, s4 = b2[i] / nobs - 3.0;
        *skew += s3 * s3;
        *kurt += s4 * s4;
    }
    *skew *= nobs / 6.0;
    *kurt *= nobs / 24.0;
    *JB = *skew + *kurt;
    *pval = 1.0 - chisq(*JB, 2 * m);

done:
    free_vector(b2, 1, m);
    free_vector(b1, 1, m);
    free_vector(v, 1, m);
    free_vector(w, 1, m);
    free_matrix(R, 1, m, 1, m);
    free_matrix(V, 1, m, 1, m);
    free_matrix(S, 1, m, 1, m);
    free_matrix(x, 1, nobs, 1, m);
}

void multivariate_diagnostics(real **res, int nobs, int m, int freq, int k,
                              FILE *out)
{
    int s = diagmv_lags(freq, nobs), df;
    real Q, pQ, JB, sk, ku, pJB;

    diagmv_hosking(res, nobs, m, s, k, &Q, &df, &pQ);
    diagmv_jb(res, nobs, m, &JB, &sk, &ku, &pJB);

    fprintf(out, "\n\n");
    fprintf(out, "=============================================================\n");
    fprintf(out, "           MULTIVARIATE RESIDUAL DIAGNOSTICS                \n");
    fprintf(out, "=============================================================\n");

    fprintf(out, "\nHosking's Multivariate Portmanteau Test (lag %d):\n", s);
    if (df >= 1) {
        fprintf(out, "  Q(%d) = %.4f, p-value = %.4f\n", df, Q, pQ);
        fprintf(out, "  df = m^2 s - k = %d - %d, k = dynamic coefficients "
                     "estimated\n", m * m * s, k);
        if (pQ < 0.05)
            fprintf(out, "  *** REJECT H0: residuals are not white noise.\n");
        else
            fprintf(out, "  Cannot reject H0: residuals appear white noise.\n");
    } else {
        fprintf(out, "  Q = %.4f; df = m^2 s - k = %d - %d < 1: too few lags "
                     "for the parameters, no p-value.\n", Q, m * m * s, k);
    }

    fprintf(out, "\nMultivariate Jarque-Bera Test (normality; Lutkepohl, "
                 "symmetric square root of Sigma):\n");
    if (isfinite(JB)) {
        fprintf(out, "  JB(%d) = %.4f, p-value = %.4f\n", 2 * m, JB, pJB);
        fprintf(out, "  skewness %.4f, kurtosis %.4f, each chi2(%d)\n",
                sk, ku, m);
        if (pJB < 0.05)
            fprintf(out, "  *** REJECT H0: residuals are not normally "
                         "distributed.\n");
        else
            fprintf(out, "  Cannot reject H0: residuals appear normal.\n");
    } else {
        fprintf(out, "  not computed: the residual covariance is singular.\n");
    }

    fprintf(out, "=============================================================\n");
}

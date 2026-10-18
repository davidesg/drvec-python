/*****************************************************************************/
/*  main.h -- part of drvarma (multivariate VARMA modelling).
 *
 *  Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.  Distributed WITHOUT ANY WARRANTY; see the
 *  GNU General Public License (file COPYING) for details.
 *****************************************************************************/

/***************************************************************************/
/*  DRV.H                                                                  */
/*  Header file for maximum-likelihood estimation modules.                 */
/*  Copyright (C) Jos? Alberto Mauricio, 1995.                             */
/*  Revised for DRVARMA (2025).                                            */
/***************************************************************************/
#ifndef MAIN_H
#define MAIN_H

#ifdef __APPLE__
#include <stdlib.h>
#elif defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#define TRUE    1
#define FALSE   0
#define MAXSTR 80
#define OK      1
#define WRONG   0
#define FREE_ARG char*

/* Constante matem?tica PI si no est? definida */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef char * STRING;
typedef double real;

/* Estructura para modelo VARMA (normalizado) */
struct Tvarma
    {
    int  m; int n; int p; int q;
    real *mu; real ***phi; real ***theta;
    real **qq; real **w; real **a;
    real sigma2; real logelf; real xitol;
    };

/* Estructura para series temporales univariantes */
struct Tseries
    {
    char *name;                  /* Nombre de la serie */
    int  nobs;                   /* N?mero de observaciones */
    int  freq;                   /* Frecuencia (observaciones por a?o) */
    int  begtime;                /* Per?odo inicial */
    int  begyear;                /* A?o inicial */
    int  endtime;                /* Per?odo final */
    int  endyear;                /* A?o final */
    int  tmornsop;               /* tiempo muerto (no usado) */
    real mean;                   /* Media muestral */
    real var;                    /* Varianza muestral */
    real skew;                   /* Coeficiente de asimetr?a */
    real kurt;                   /* Coeficiente de curtosis */
    int  max;                    /* ?ndice del m?ximo */
    int  min;                    /* ?ndice del m?nimo */
    real *data;                  /* Vector de datos */
    int  numbering;              /* 1 = numeracion simple, 0 = fechas       */
    real refactor;               /* factor de reescalado (se anade para FUE) */
    };

/* Estructura de modelo univariante de FUE.  Copiada byte a byte de
   drtran/include/main.h (2026-08-17); la necesita fue_pre_reader.c, que
   tambien viene de drtran.  Ver PLAN_BETA.md F2 (procedencia).          */

/* -------------------------------------------------------------------------- */
/* Estructura para modelo univariante completo de FUE (nueva)                */
/* -------------------------------------------------------------------------- */
struct Tusmodel
{
    int NdetVar;                 /* numero de variables deterministas */
    char **detspec;              /* especificacion de cada detvar tal como viene
                                    en el .pre ("cos 2", "step 6 2008", ...).
                                    Se guarda para poder REGENERARLAS en fechas
                                    futuras al prever: son funciones del tiempo. */
    int *Nomega, *Ndelta;        /* ordenes de omega y delta para cada detvar */
    real **Omega, **Delta;       /* coeficientes omega y delta */
    int **Imega, **Ielta;        /* indicadores: 1=estimado, 0=fijo */

    int NumAr1, NumAr2;          /* numero de factores AR regulares y anuales */
    int *p1, *p2;                /* ordenes de cada factor AR */
    real **Ar1, **Ar2;           /* coeficientes AR */
    int **Ia1, **Ia2;            /* flags estimacion */

    int NumMa1, NumMa2;          /* numero de factores MA regulares y anuales */
    int *q1, *q2;                /* ordenes de cada factor MA */
    real **Ma1, **Ma2;           /* coeficientes MA */
    int **Im1, **Im2;            /* flags estimacion */

    int NumAr1f, NumMa1f;        /* numero de factores AR/MA de frecuencia fija */
    int *pfre1, *qfre1;          /* frecuencias de los factores fijos */
    real **Ar1f, **Ma1f;         /* coeficientes AR/MA de frecuencia fija */
    int *Ia1f, *Im1f;            /* flags estimacion */

    int Imu;                     /* flag para la media */
    real mu;                     /* valor de la media */

    real boxlam;                 /* parametro lambda de Box-Cox */
    int nrdiff;                  /* diferencias regulares */
    int nadiff;                  /* diferencias anuales completas */
    int *ifadf;                  /* factores irreducibles de la dif. anual */
    int sper;                    /* periodo estacional (freq) */
    int ornsop;                  /* orden del operador no estacionario */
    real *rnsop;                 /* coeficientes del operador no estacionario */
    char *residuals;             /* nombre de la serie de residuos */
    real cbands;                 /* bandas de confianza para ACF */
};

/*****************************************************************************/
/*  Funciones de estimaci?n (de elfvarma.c y drvmlest.c)                     */
/*****************************************************************************/

void est( void (*cast)( real *, struct Tvarma *, int *, int, int ),
          int npar, real *par, real *dev, real **cov, int maxits, int nrits,
          real grtol, real sptol, real xitol, real **a, real *sigma2,
          real *logelf, int *ifault );

void elf( int m, int n, int p, int q, real *mu, real ***phi, real ***theta,
          real **qq, real **w, real sigma2, real delta, int atf, real **a,
          real *f1, real *f2, real *logelf, int *ifault );

void chekma( int m, int q, real ***theta, real *wr, real *wi,
             real *wmod, int *ifault );

/*****************************************************************************/
/*  Funciones de ?lgebra lineal y gesti?n de memoria (de nlatools.c)         */
/*****************************************************************************/

void ludcp( real **a, int n, int *ip );
void lusol( real **a, real *b, int n, int *ip );
void choldcp( real **mat, int n, real *d1, real *d2, int *ifault );
void cholfor( real **matl, int n, real *rhsol );
void cholbak( real **matl, int n, real *rhsol );
void cholsol( real **matl, int n, real *rhsol );
void eigenqr( real **a, int n, real *wr, real *wi );
void svdcp( real **a, int m, int n, real *w, real **v );
void svsol( real **u, real *w, real **v, int n, real *x );

void nrerror( char error_text[] );
real *vector( long nl, long nh );
int  *ivector( long nl, long nh );
real **matrix( long nrl, long nrh, long ncl, long nch );
int  **imatrix( long nrl, long nrh, long ncl, long nch );
real ***tensor( long nrl, long nrh, long ncl, long nch, long ndl, long ndh );
void free_vector( real *v, long nl, long nh );
void free_ivector( int *v, long nl, long nh );
void free_matrix( real **m, long nrl, long nrh, long ncl, long nch );
void free_imatrix( int **m, long nrl, long nrh, long ncl, long nch );
void free_tensor( real ***t, long nrl, long nrh, long ncl, long nch,
                  long ndl, long ndh );

/*---------------------------------------------------------------------------*/
/*  Funciones auxiliares matriciales                                         */
/*---------------------------------------------------------------------------*/
 void matrix_multiply(real **A, real **B, real **C, int n, int m, int p);
 void matrix_transpose(real **A, real **B, int n, int m);
 void matrix_inverse(real **A, real **invA, int n);

real rmax( real a, real b );
real rmin( real a, real b );
real cmacheps( void );
int  round_local( real num );

/*****************************************************************************/
/*  Funciones de manejo de cadenas (de nlatools.c)                           */
/*****************************************************************************/

STRING NEW_STR( int size );
void   FREE_STR( STRING s );
int    COPY_STR( STRING source, int i, int n, STRING dest );

/*****************************************************************************/
/*  Funciones de an?lisis de series (diagnose.c)  */
/*****************************************************************************/

void ObsToDate( int beg_per, int beg_sub, int obs_no, int freq,
                int *per, int *sub );
void DateToObs( int beg_per, int beg_sub, int per, int sub, int freq,
                int *obs_no );
/* Easter esta en nlatools.c:679 desde siempre, pero no estaba declarada aqui;
   la usa fue_pre_reader.c para la deterministica "easter" (solo mensual).     */
void Easter( int *day, int *month, int year );
real Mean( real *data, int nobs );
real Stdev( real *data, int nobs );
int  MaxVal( real *data, int nobs );
int  MinVal( real *data, int nobs );
real Skew( real *data, int nobs );
real Kurt( real *data, int nobs );
//void Acf( struct Tseries *ser, int lags, real *corr );
void Acf(real *data, int nobs, int lags, real *corr, real mean, real var);
void Pacf( int lags, real *pcorr );
real ChiTest( real *corr, int lags, int nobs );

void File_StatSer( struct Tseries *ser );
void File_PlotSer( struct Tseries *ser );
void File_HistSer( struct Tseries *ser );
void File_CorrSer( struct Tseries *ser, int npar );
void PlotCor( real *corr, int lags, int isacf, struct Tseries *ser, int npar );
void PlotCCF( real *corr, int lags, struct Tseries *ser );
void ccf( struct Tseries *ser1, struct Tseries *ser2, int lags, real *corr );
real ChiTestC( real *corr, int lags, int nobs );
/* Hosking and the multivariate Jarque-Bera: lib/diagmv/diagmv.h */
void diagnose(struct Tvarma *varma, const int *npar_ser);
/* Funciones de diagn?stico adicionales (declaradas en diagnose.c) */
void test_last_lag_significance(real *x, real **cov, int npar,
                                int m, int p, int q,
                                int include_mean,
                                int diag_ar, int diag_ma, int diag_cov,
                                real *chi2, int *df, real *pval);
void test_last_ar_significance(real *x, real **cov, int npar,
                               int m, int p, int q,
                               int include_mean,
                               int diag_ar, int diag_ma, int diag_cov,
                               real *chi2, int *df, real *pval);
void test_last_ma_significance(real *x, real **cov, int npar,
                               int m, int p, int q,
                               int include_mean,
                               int diag_ar, int diag_ma, int diag_cov,
                               real *chi2, int *df, real *pval);
void test_cross_effects_significance(real *x, real **cov, int npar,
                                     int m, int p, int q,
                                     int include_mean,
                                     int diag_ar, int diag_ma, int diag_cov,
                                     real *chi2, int *df, real *pval);
void all_hypothesis_tests(real *x, real **cov, int npar,
                          int m, int p, int q,
                          int include_mean,
                          int diag_ar, int diag_ma, int diag_cov,
                          FILE *outputv);

void impulse_response(struct Tvarma *varma, int horizon, FILE *outputv);
void variance_decomposition(struct Tvarma *varma, int horizon, FILE *outputv);

void AnalyzeOneSeries(real *res, int nobs, int idx, int freq);

/*****************************************************************************/
/*  Funciones de distribuci?n estad?stica (de nlatools.c)                    */
/*****************************************************************************/

real chisq( real x, int df );          /* Distribuci?n Chi?cuadrado acumulada */
real tdist( real t, int df );          /* Distribuci?n t?Student acumulada   */
real normal_cdf( real z );             /* Distribuci?n normal est?ndar acumulada */

/*****************************************************************************/
/*  Funciones propias de drvarma.c (declaradas para completitud)             */
/*****************************************************************************/

//void preestimar_parametros( real *x, int npar );
//void print_parameters( real *x, real *dev, real **cov, int npar, int m );
//void print_matrices( struct Tvarma *varma );
//void print_roots( struct Tvarma *varma );

/*****************************************************************************/
/**
 * @brief Compute conditional covariance matrices using exponential weighting.
 *
 * This function implements the rational?inattention volatility model.
 * Steps:
 *   1. Computes Mahalanobis distances d_t = ?_t' ??? ?_t for all residuals,
 *      where ? = ?? Q is the unconditional covariance matrix.
 *   2. Estimates ? as the proportion of distances exceeding the empirical
 *      (1??) quantile.
 *   3. Builds exponentially decaying weights ?_k = ?(1??)^k over a finite window
 *      and normalises them to sum to 1.
 *   4. For each time t, constructs H_t = ?_{k=0}^{min(window-1, t-1)} ?_k ?_{t-k} ?_{t-k}'.
 *
 * The output file (basename .volexp) contains one line per observation:
 *   t  var1 var2 ... varm  cov12 cov13 ... cov(m-1)m
 *
 * @param varma    Pointer to an estimated VARMA model (must contain residuals,
 *                 ?? and the normalised covariance matrix Q).
 * @param alpha    Significance level for the Mahalanobis threshold (0 < alpha < 1).
 *                 Typical value: 0.05. Determines the cutoff for extreme events.
 * @param window   Number of lagged residuals to include in the weighted sum.
 *                 Must be at least 1. Larger windows give smoother estimates.
 * @param basename Base name of the input file (used to construct the output file name).
 */
void compute_exponential_volatility(struct Tvarma *varma, real alpha,
                                    int window, const char *basename);

/**
 * @brief Compute conditional covariance matrices using a simple moving window.
 *
 * For each time t >= window, computes the unbiased sample covariance matrix
 * of the residuals in the window [t?window+1, t] (equal weights).
 * Results are written to a file named basename .volmov with the same format
 * as described above. Observations before t = window are omitted.
 *
 * @param varma    Pointer to an estimated VARMA model.
 * @param window   Window length (number of observations). Must be at least 2.
 * @param basename Base name of the input file.
 */
void compute_moving_window_volatility(struct Tvarma *varma, int window,
                                      const char *basename);
#endif /* MAIN_H */

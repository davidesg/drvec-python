/*****************************************************************************/
/*  elfvarma.c -- part of drvarma (multivariate VARMA modelling).
 *
 *  Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.  Distributed WITHOUT ANY WARRANTY; see the
 *  GNU General Public License (file COPYING) for details.
 *****************************************************************************/

/*****************************************************************************/
/*  ELFVARMA.C                                                               */
/*  Computation of the log-likelihood function of a vector ARMA(p,q) model.  */
/*  Source: Mauricio, J.A. (1995) JASA 90, 282-291.                          */
/*  Copyright (C) Jos‚ Alberto Mauricio, 1995.                               */
/*****************************************************************************/

#include "main.h"                     /* Header file (prototype declarations) */

/*****************************************************************************/
/*****************************************************************************/

void elf( int m, int n, int p, int q, real *mu, real ***phi, real ***theta,
          real **qq, real **w, real sigma2, real delta, int atf, real **a,
          real *f1, real *f2, real *logelf, int *ifault )

{
   const real LOG2PI = 1.837877066;

/* Declaration of variables:                                                 */

   int  i, ii, j, jj, jl, k, kk, g, nlim;
   real **q1, **q1inv, ***gamwa, *gamma, ***rxi;
   real **mtmp0, **mtmp1, **mtmp2, **mtmp3, *vtmp1, *vtmp2, *vechh;
   real d1, d2, s1, s2, detq, detom;

/* Declaration of functions:                                                 */

   void cgamma( int m, int p, int q, real ***phi, real ***theta, real **qq,
                real ***gamwa, real *gamma, int *ifault );
   void cxi( int m, int n, int q, real ***theta, real **q1inv, real xitol,
             int *r, real ***rxi );
   void cres( int m, int n, int g, int nlim, real ***rxi, real **q1,
              real **matm, real **matl, real *lambda, real **res );

/* Initialization:                                                           */

   *logelf = 0.0;                                /* Log-likelihood value.    */
   *f1     = 0.0;                                /* Quadratic form.          */
   *f2     = 0.0;                                /* Determinant (see [11]).  */
   *ifault =   0;                                /* Error flag.              */
   for ( i = 1; i <= n; i++ )
       for ( j = 1; j <= m; j++ ) a[i][j] = 0.0; /* Matrix of residuals.     */
   g = ( p > q ) ? p : q;                        /* Max(p,q).                */

/* [0]: Copy lower triangle of qq into upper triangle of qq and              */
/*      into upper triangle of q1 (qq is nowhere else modified):             */

   q1 = matrix( 1, m, 1, m );

   for ( i = 1; i <= m; i++ )
       for ( j = i; j <= m; j++ )
           {
           qq[i][j] = qq[j][i];
           q1[j][i] = 0.0;
           q1[i][j] = qq[i][j];
           }

/* [1]: Calculate the inverse of the Cholesky factor of qq (store as q1inv)  */
/*      and the determinant of qq (store as detq):                           */

   choldcp( q1, m, &d1, &d2, ifault );       /* Carry out choldcp of q1.     */

   if ( *ifault > 0 )
      {
      *ifault = 1;                           /* qq is NOT positive definite: */
      goto e1;                               /* set flag and return.         */
      }

   q1inv = matrix( 1, m, 1, m );
   vtmp1 = vector( 1, m );

   for ( i = 1; i <= m; i++ )
       {
       for ( j = 1; j <= m; j++ )
           {
           vtmp1[j]    = 0.0;
           q1inv[j][i] = 0.0;
           }
       vtmp1[i] = 1.0;
       cholfor( q1, m, vtmp1 );
       for ( j = i; j <= m; j++ ) q1inv[j][i] = vtmp1[j];
       }

   detq = d1 * pow( 2.0, d2 );
   free_vector( vtmp1, 1, m );

/* [2]: Compute the autocovariances of w(t) (store as gamma) and the         */
/*      cross-covariances between w(t) and a(t) (store as gamwa)             */

   if ( p > 0 )
      {
      gamma = vector( 1, m * (m + 1) / 2 + m * m * (p - 1) );
      if ( q > 0 ) gamwa = tensor( -q+1, 0, 1, m, 1, m );
      cgamma( m, p, q, phi, theta, qq, gamwa, gamma, ifault );

      if ( *ifault > 0 )            /* Autoregressive unit root(s) detected: */
         {                          /* set flag and return.                  */
         *ifault = 2;
         goto e2;
         }
      }

/* [3]: Compute M (Cholesky factor of v1 * omega * v1') and store as mtmp0:  */

   mtmp0 = matrix( 1, m * g, 1, m * g );

   for ( i = 1; i <= m * g; i++ )
       for ( j = 1; j <= m * g; j++ ) mtmp0[i][j] = 0.0;

/* [3.1]: Compute omega * v1' and store as mtmp1:                            */

   mtmp1 = matrix( 1, m * (p + q), 1, m * g );

   for ( i = 1; i <= m * (p + q); i++ )
       for ( j = 1; j <= m * g; j++ ) mtmp1[i][j] = 0.0;

   for (i = 1; i <= p; i++ )
       for (j = 1; j <= g; j++ )
           {
           for ( k = j - i; k <= p - i; k++ )
               for ( ii = 1; ii <= m; ii++ )
                   for ( jj = 1; jj <= m; jj++ )
                       {
                       s1 = 0.0;
                       for ( kk = 1; kk <= m; kk++ )
                           {
                           if ( k > 0 )
                              jl = m * (m + 1) / 2 + m * m * (k - 1) +
                                   m * (kk - 1) + ii;
                           else if (k < 0 )
                              jl = m * (m + 1) / 2 - m * m * (k + 1) +
                                   m * (ii - 1) + kk;
                           else
                              {
                              if ( kk >= ii )
                                 jl = kk * (kk - 1) / 2 + ii;
                              else
                                 jl = ii * (ii - 1) / 2 + kk;
                              }
                           s1 += gamma[jl] * phi[p-k-i+j][jj][kk];
                           }
                       mtmp1[ii+(i-1)*m][jj+(j-1)*m] += s1;
                       }
           for ( k = j - i; k <= q - i; k++ )
               if ( p + k <= q )
                  for ( ii = 1; ii <= m; ii++ )
                      for ( jj = 1; jj <= m; jj++ )
                          {
                          s1 = 0.0;
                          for ( kk = 1; kk <= m; kk++ )
                              s1 += gamwa[-q+p+k][ii][kk] *
                                    theta[q-k-i+j][jj][kk];
                          mtmp1[ii+(i-1)*m][jj+(j-1)*m] -= s1;
                          }
           }

   for ( i = p + 1; i <= p + q; i++ )
       for ( j = 1; j <= g; j++ )
           {
           for ( k = p + j - i; k <= p + p - i; k++ )
               if ( p - k <= q )
                  for ( ii = 1; ii <= m; ii++ )
                      for ( jj = 1; jj <= m; jj++ )
                          {
                          s1 = 0.0;
                          for ( kk = 1; kk <= m; kk++ )
                              s1 += gamwa[-q+p-k][kk][ii] *
                                    phi[p+p-k-i+j][jj][kk];
                          mtmp1[ii+(i-1)*m][jj+(j-1)*m] += s1;
                          }
           if ( p - i + j <= 0 )
              for ( ii = 1; ii <= m; ii++ )
                  for ( jj = 1; jj <= m; jj++ )
                      {
                      s1 = 0.0;
                      for ( kk = 1; kk <= m; kk++ )
                          s1 += qq[ii][kk] * theta[q+p-i+j][jj][kk];
                      mtmp1[ii+(i-1)*m][jj+(j-1)*m] -= s1;
                      }
           }

/* [3.2]: Compute v1 * omega * v1' and store as mtmp0:                       */

   for ( i = 1; i <= g; i++ )
       for ( j = i; j <= g; j++ )
           {
           for ( k = 0; k <= p - i; k++ )
              for ( ii = 1; ii <= m; ii++ )
                  {
                  jl = ( i == j ) ? ii : 1;
                  for ( jj = jl; jj <= m; jj++ )
                      {
                      s1 = 0.0;
                      for ( kk = 1; kk <= m; kk++ )
                          s1 += phi[p-k][ii][kk] *
                                mtmp1[kk+(k+i-1)*m][jj+(j-1)*m];
                      mtmp0[ii+(i-1)*m][jj+(j-1)*m] += s1;
                      }
                  }

           for ( k = 0; k <= q - i; k++ )
              for ( ii = 1; ii <= m; ii++ )
                  {
                  jl = ( i == j ) ? ii : 1;
                  for ( jj = jl; jj <= m; jj++ )
                      {
                      s1 = 0.0;
                      for ( kk = 1; kk <= m; kk++ )
                          s1 += theta[q-k][ii][kk] *
                                mtmp1[kk+(k+p+i-1)*m][jj+(j-1)*m];
                      mtmp0[ii+(i-1)*m][jj+(j-1)*m] -= s1;
                      }
                  }
           }
   free_matrix( mtmp1, 1, m * (p + q), 1, m * g );

/* [3.3]: Compute M and overwrite mtmp0 (carry out choldcp of mtmp0):        */

   choldcp( mtmp0, m * g, &d1, &d2, ifault );

   if ( *ifault > 0 )                   /* Strict non-stationarity detected: */
      {                                 /* set flag and return.              */
      *ifault = 3;
      goto e3;
      }

/* [4]: Compute matrix polynomial q1inv * xi(k) and store as rxi:            */

/* [4.1]: Check for strict non-invertibility through chekma:                 */

   if ( q > 0 )
      {
      vtmp1 = vector( 1, m * q );
      vtmp2 = vector( 1, m * q );
      vechh = vector( 1, m * q );
      chekma( m, q, theta, vtmp1, vtmp2, vechh, ifault );
      free_vector( vechh, 1, m * q );
      free_vector( vtmp2, 1, m * q );
      free_vector( vtmp1, 1, m * q );
      }

   if ( *ifault > 0 )                  /* Strict non-invertibility detected: */
      {                                /* set flag and return.               */
      *ifault = 4;
      goto e3;
      }

/* [4.2]: Compute rxi:                                                       */

   rxi = tensor( 0, n-1, 1, m, 1, m );
   cxi( m, n, q, theta, q1inv, delta, &nlim, rxi );

/* [5]: Compute vector eta and store as a:                                   */

   for ( i = 1; i <= n; i++ )
       for ( j = 1; j <= m; j++ ) a[i][j] = 0.0;

   vtmp1 = vector( 1, m );
   vtmp2 = vector( 1, m );

/* [5.1]: Calculate conditional residuals recursively and store as a:        */

   for ( i = 1; i <= n; i++ )
       {
       for ( j = 1; j <= m; j++ ) vtmp1[j] = 0.0;
       for ( j = 1; j <= p; j++ )
           if ( i - j >= 1 )
              for ( ii = 1; ii <= m; ii++ )
                  {
                  s1 = 0.0;
                  for ( k = 1; k <= m; k++ )
                      s1 += phi[j][ii][k] * (w[i-j][k] - mu[k]);
                  vtmp1[ii] += s1;
                  }

       for ( j = 1; j <= m; j++ ) vtmp2[j] = 0.0;
       for ( j = 1; j <= q; j++ )
           if ( i - j >= 1 )
              for ( ii = 1; ii <= m; ii++ )
                  {
                  s1 = 0.0;
                  for ( k = 1; k <= m; k++ )
                      s1 += theta[j][ii][k] * a[i-j][k];
                  vtmp2[ii] += s1;
                  }

       for ( ii = 1; ii <= m; ii++ )
           a[i][ii] = (w[i][ii] - mu[ii]) - vtmp1[ii] + vtmp2[ii];
       }

/* [5.2]: Premultiply each m-block of a by q1inv and overwrite a:            */

   for ( i = 1; i <= n; i++ )
       {
       for ( j = 1; j <= m; j++ )
           {
           s1 = 0.0;
           for ( k = 1; k <= j; k++ ) s1 += q1inv[j][k] * a[i][k];
           vtmp1[j] = s1;
           }
       for ( j = 1; j <= m; j++ ) a[i][j] = vtmp1[j];
       }

   free_vector( vtmp2, 1, m );
   free_vector( vtmp1, 1, m );

/* [6]: Compute vector M'h and store as vechh:                               */

   vechh = vector( 1, m * g );

   for ( i = 1; i <= m * g; i++ ) vechh[i] = 0.0;

/* [6.1]: Compute vector h and store as vechh:                               */

   for ( j = 1; j <= g; j++ )
       for ( i = 0; i <= n - j; i++ )
           if ( i <= nlim )
              for ( jj = 1; jj <= m; jj++ )
                  {
                  s1 = 0.0;
                  for ( k = 1; k <= m; k++ )
                      s1 = s1 + rxi[i][k][jj] * a[i+j][k];
                  vechh[jj+(j-1)*m] += s1;
                  }

/* [6.2]: Premultiply vechh by M' (M stored as mtmp0) and overwrite vechh:   */

   for ( i = 1; i <= m * g; i++ )
       {
       s1 = 0.0;
       for ( k = i; k <= m * g; k++ ) s1 += mtmp0[k][i] * vechh[k];
       vechh[i] = s1;
       }

/* [...]: Save matrix M as mtmp3 if residuals are requested (atf = TRUE):    */

   if ( atf )
      {
      mtmp3 = matrix(1, m * g, 1, m * g );

      for ( i = 1; i <= m * g; i++ )
          for ( j = 1; j <= i; j++ ) mtmp3[i][j] = mtmp0[i][j];
      }

/* [7]: Compute H'H and store as mtmp2:                                      */

   mtmp1 = matrix(1, m * g, 1, m * g );
   mtmp2 = matrix(1, m * g, 1, m * g );

   for ( i = 1; i <= m * g; i++ )
       for ( j = 1; j <= m * g; j++ ) mtmp2[i][j] = 0.0;

   for ( i = 1; i <= g; i++ )
       for ( k = 0; k <= n - i; k++ )
           if ( k + i - 1 <= nlim )
              for ( ii = 1; ii <= m; ii++ )
                  {
                  jl = ( i == 1 ) ? ii : m;
                  for ( jj = 1; jj <= jl; jj++ )
                      {
                      s1 = 0.0;
                      for ( kk = 1; kk <= m; kk++ )
                          s1 += rxi[k][kk][ii] * rxi[k+i-1][kk][jj];
                      mtmp2[ii+(i-1)*m][jj] += s1;
                      }
                  }

   for ( i = 2; i <= g; i++ )
       for ( j = 2; j <= i; j++ )
           for ( ii = 1; ii <= m; ii++ )
               {
               jl = ( i == j ) ? ii : m;
               for ( jj = 1; jj <= jl; jj++ )
                   {
                   s1 = 0.0;
                   if ( (n - i + 1 <= nlim) && (n - j + 1 <= nlim) )
                      for ( kk = 1; kk <= m; kk++ )
                          s1 += rxi[n-i+1][kk][ii] * rxi[n-j+1][kk][jj];
                   mtmp2[ii+(i-1)*m][jj+(j-1)*m] =
                          mtmp2[ii+(i-2)*m][jj+(j-2)*m] - s1;
                   }
               }

   for ( i = 1; i <= m * g; i++ )
       for ( j = i + 1; j <= m * g; j++ ) mtmp2[i][j] = mtmp2[j][i];

/* [8]: Compute I+M'H'HM and its Cholesky factor L (overwrite mtmp0):        */

/* [8.1]: Compute M'H'H and store as mtmp1:                                  */

   for ( i = 1; i <= m * g; i++ )
       for ( j = 1; j <= m * g; j++ )
           {
           s1 = 0.0;
           for ( k = i; k <= m * g; k++ )
               s1 += mtmp0[k][i] * mtmp2[k][j];
           mtmp1[i][j] = s1;
           }

/* [8.2]: Store M as mtmp2 (overwrite) and initialize mtmp0:                 */

   for ( i = 1; i <= m * g; i++ )
       for ( j = 1; j <= m * g; j++ )
           {
           mtmp2[i][j] = mtmp0[i][j];
           mtmp0[i][j] = 0.0;
           }

/* [8.3]: Calculate I + M'H'HM and store as mtmp0:                           */

   for ( i = 1; i <= m * g; i++ )
       {
       for ( j = i; j <= m * g; j++ )
           {
           s1 = 0.0;
           for ( k = j; k <= m * g; k++ )
               s1 += mtmp1[i][k] * mtmp2[k][j];
           mtmp0[i][j] = s1;
           }
       mtmp0[i][i] += 1.0;
       }

   free_matrix( mtmp2, 1, m * g, 1, m * g );
   free_matrix( mtmp1, 1, m * g, 1, m * g );

/* [8.4]: Compute the Cholesky factor L of I+M'H'HM (overwrite mtmp0) and    */
/*        its determinant (store as detom) through choldcp of mtmp0:         */

   choldcp( mtmp0, m * g, &d1, &d2, ifault );

   if ( *ifault > 0 )                         /* This should never happen !! */
      {
      *ifault = 5;
      goto e4;
      }

   detom = d1 * pow( 2.0, d2 );

/* [9]: Calculate lambda using forward substitution and overwrite vechh:     */

   cholfor( mtmp0, m * g, vechh );

/* [10]: Calculate the quadratic form and return as f1:                      */

   s1 = 0.0;
   for ( i = 1; i <= n; i++ )
       for ( j = 1; j <= m; j++ ) s1 += a[i][j] * a[i][j];

   s2 = 0.0;
   for ( i = 1; i <= m * g; i++ ) s2 += vechh[i] * vechh[i];

   *f1 = s1 - s2;

/* [11]: Calculate the determinant = detom * pow( detq,n ) (note that f2 is  */
/*       returned as the appropriate factor in the objective function):      */

   *f2 = log( detom ) / n;
   *f2 = detq * exp( *f2 );

/* [12]: Compute the value of the log-likelihood and return as logelf:       */

   *logelf = -0.5 * ( n * m * ( LOG2PI + log( sigma2 ) ) + n * log( detq ) +
             log( detom ) + *f1 / sigma2 );

/* [13]: If requested, compute residuals and return as a:                    */

   if ( atf ) cres( m, n, g, nlim, rxi, q1, mtmp3, mtmp0, vechh, a );

/* [14]: That's all, folks !! (deallocate dynamic memory and return):        */

e4:if ( atf ) free_matrix( mtmp3, 1, m * g, 1, m * g );
   free_vector( vechh, 1, m * g );
   free_tensor( rxi, 0, n-1, 1, m, 1, m );
e3:free_matrix( mtmp0, 1, m * g, 1, m * g );
e2:if ( p > 0 )
      {
      if ( q > 0 ) free_tensor( gamwa, -q+1, 0, 1, m, 1, m );
      free_vector( gamma, 1, m * (m + 1) / 2 + m * m * (p - 1) );
      }
   free_matrix( q1inv, 1, m, 1, m );
e1:free_matrix( q1, 1, m, 1, m );
}

/*****************************************************************************/
/*****************************************************************************/

void chekma( int m, int q, real ***theta, real *wr, real *wi,
             real *wmod, int *ifault )

{
   int  i, j, k, n;
   real **a;

   *ifault  = 0;
   n        = m * q;
   a        = matrix( 1, n, 1, n );

   for ( i = 1; i <= n; i ++ )
       {
       for ( j = 1; j <= n; j ++ ) a[i][j] = 0.0;
       wr[i] = 0.0;
       wi[i] = 0.0;
       }

   for ( k = 1; k <= q; k ++ )
       for ( i = 1; i <= m; i++ )
           for ( j = 1; j <= m; j++ ) a[i][j+(k-1)*m] = theta[k][i][j];
   for ( k = 1; k <= q-1; k++ )
       for ( j = 1; j <= m; j++ ) a[j+k*m][j+(k-1)*m] = 1.0;

   eigenqr( a, n, wr, wi );           /* Compute eigenvalues through hqr:    */

   for ( i = 1; i <= n; i++ )
       {
       wmod[i] = a[i][i];
       if ( a[i][i] >= 1.00005 ) *ifault = 1;
       }

   free_matrix( a, 1, n, 1, n );
}

/*****************************************************************************/

void cgamma( int m, int p, int q, real ***phi, real ***theta, real **qq,
             real ***gamwa, real *gamma, int *ifault )

{
   int  big, row, col, h, i, ii, j, jj, k, l, r, s, *indx;
   real **wzero, **mzero, **mat, sum;

   *ifault = 0;

/* [1]: Compute the q-1 cross-covariance matrices and return as gamwa:       */

   if ( q > 0 )
      for ( i = 1; i <= m; i++ )
          for ( j = 1; j <= m; j++ ) gamwa[0][i][j] = qq[i][j];

   for ( k = 1; k <= q - 1; k++ )
       for ( i = 1; i <= m; i++ )
           for ( j = 1; j <= m; j++ )
               {
               sum = 0.0;
               for ( h = 1; h <= m; h++ )
                   sum -= theta[k][i][h] * qq[h][j];
               for ( l = 1; l <= k; l++ )
                   if ( l <= p )
                      for (h = 1; h <= m; h++ )
                          sum += phi[l][i][h] * gamwa[l-k][h][j];
               gamwa[-k][i][j] = sum;
               }

/* [2]: Compute diagonal and upper triangle of w(0) and store as wzero:      */

   wzero = matrix( 1, m, 1, m );
   mzero = matrix( 1, m, 1, m );

   for ( i = 1; i <= m; i++ )
       for ( j = 1; j <= m; j++ ) wzero[i][j] = 0.0;

   for ( i = 1; i <= p; i++ )
       for ( j = i; j <= q; j++ )
           if ( j >= i )
              {
              for ( ii = 1; ii <= m; ii++ )
                  for ( jj = 1; jj <= m; jj++ )
                      {
                      sum = 0.0;
                      for ( k = 1; k <= m; k++ )
                          sum += phi[i][ii][k] * gamwa[i-j][k][jj];
                      mzero[ii][jj] = sum;
                      }
              for ( ii = 1; ii <= m; ii++ )
                  for ( jj = 1; jj <= m; jj++ )
                      {
                      sum = 0.0;
                      for ( k = 1; k <= m; k++ )
                          sum += mzero[ii][k] * theta[j][jj][k];
                      wzero[ii][jj] += sum;
                      }
              }

   for ( i = 1; i <= m; i++ )
       for ( j = i; j <= m; j++ )
           wzero[i][j] = qq[i][j] - wzero[i][j] - wzero[j][i];

   for ( j = 1; j <= q; j++ )
       {
       for ( ii = 1; ii <= m; ii++ )
           for ( jj = 1; jj <= m; jj++ )
               {
               sum = 0.0;
               for ( k = 1; k <= m; k++ )
                   sum += theta[j][ii][k] * qq[k][jj];
               mzero[ii][jj] = sum;
               }
       for ( ii = 1; ii <= m; ii++ )
           for ( jj = ii; jj <= m; jj++ )
               {
               sum = 0.0;
               for ( k = 1; k <= m; k++ )
                   sum += mzero[ii][k] * theta[j][jj][k];
               wzero[ii][jj] += sum;
               }
       }
   free_matrix( mzero, 1, m, 1, m );

/* [3]: Set up system of equations and store as mat and gamma:               */

   big = m * (m + 1) / 2 + m * m * (p - 1);
   mat = matrix( 1, big, 1, big );

   for ( i = 1; i <= big; i++ )
       {
       for ( j = 1; j <= big; j++ ) mat[i][j] = 0.0;
       gamma[i] = 0.0;
       }

/* [3.1]: Compute the first m * (m + 1) / 2 rows:                            */

   for ( j = 1; j <= m; j++ )
       for ( i = 1; i <= j; i++ )
           {
           row = j * (j - 1) / 2 + i;

/* [3.1.1]: Compute the first m * (m + 1) / 2 columns:                       */

           for ( l = 1; l <= m; l++ )
               for ( k = 1; k <= l; k++ )
                   {
                   col = l * (l - 1) / 2 + k;
                   sum = 0.0;
                   if ( k == l )
                      for ( r = 1; r <= p; r++ )
                          sum -= phi[r][i][k] * phi[r][j][l];
                   else
                      for ( r = 1; r <= p; r++ )
                          sum -= phi[r][i][k] * phi[r][j][l] +
                                 phi[r][i][l] * phi[r][j][k];
                   mat[row][col] = sum;
                   }

/* [3.1.2]: Compute the remaining m * m * (p - 1) columns:                   */

           for ( s = 1; s <= p - 1; s++ )
               for ( l = 1; l <= m; l++ )
                   for ( k = 1; k <= m; k++ )
                       {
                       col = m * (m + 1) / 2 + m * m * (s - 1) +
                             m * (l - 1) + k;
                       sum = 0.0;
                       for ( r = 1; r <= p - s; r++ )
                           sum -= phi[r+s][i][k] * phi[r][j][l]
                                + phi[r+s][j][k] * phi[r][i][l];
                       mat[row][col] = sum;
                       }

/* [3.1.3]: Set up right-hand-side (gamma) and diagonal of mat:              */

           mat[row][row] += 1.0;
           gamma[row] = wzero[i][j];
           }

/* [3.2]: Compute the remaining m * m * (p - 1) rows:                        */

   for ( s = 1; s <= p - 1; s++ )
      for ( i = 1; i <= m; i++ )
          for ( j = 1; j <= m; j++ )
              {
              row = m * (m + 1) / 2 + m * m * (s - 1) + m * (i - 1) + j;

/* [3.2.1]: Compute the first m * (m + 1) / 2 columns:                       */

              for ( l = 1; l <= m; l++ )
                  {
                  if ( l <= j )
                      col = j * (j - 1) / 2 + l;
                  else
                      col = l * (l - 1) / 2 + j;
                  mat[row][col] = -phi[s][i][l];
                  }

/* [3.2.2]: Compute the remaining m * m * (p - 1) columns:                   */

              for ( r = 1; r <= p - 1; r++ )
                  for ( l = 1; l <= m; l++ )
                      {
                      col = m * (m + 1) / 2 + m * m * (r - 1) +
                            m * (j - 1) + l;
                      if ( r + s <= p )
                         mat[row][col] = -phi[r+s][i][l];
                      if ( s > r )
                         {
                         col = m * (m + 1) / 2 + m * m * (r - 1) +
                               m * (l - 1) + j;
                         mat[row][col] -= phi[s-r][i][l];
                         }
                      }

/* [3.2.3]: Set up right-hand-side (gamma) and diagonal of mat:              */

              mat[row][row] += 1.0;
              gamma[row] = 0.0;
              for ( h = s; h <= q; h++ ) for ( k = 1; k <= m; k++ )
                  gamma[row] -= gamwa[s-h][j][k] * theta[h][i][k];
              }

/* [4]: Solve for gamma (autocovariance matrices) and return:                */

   indx = ivector( 1, big );
   ludcp( mat, big, indx );
   if ( indx[big] != 0 )
      lusol( mat, gamma, big, indx );
   else
      *ifault = 1;

   free_ivector( indx, 1, big );
   free_matrix( mat, 1, big, 1, big );
   free_matrix( wzero, 1, m, 1, m );
}

/*****************************************************************************/

void cxi( int m, int n, int q, real ***theta, real **q1inv, real xitol,
          int *r, real ***rxi )

{
   int  h, i, ii, j, jj, k, delta, nq;
   real s1, s2, **mtmp;

   delta = FALSE;

   for ( k = 0; k <= n - 1; k++ )
       for ( i = 1; i <= m; i++ )
           for ( j = 1; j <= m; j++ ) rxi[k][i][j] = 0.0;
   for ( i = 1; i <= m; i++ ) rxi[0][i][i] = 1.0;

/* [1]: Update index r and compute matrix sequence (store as rxi):           */

   *r = 0;

   do
   {
   *r = *r + 1;
   for ( j = 1; j <= q; j++ )
       if ( *r >= j )
          for ( ii = 1; ii <= m; ii++ )
              for ( jj = 1; jj <= m; jj++ )
                  {
                  s1 = 0.0;
                  for ( h = 1; h <= m; h++ )
                      s1 += theta[j][ii][h] * rxi[*r-j][h][jj];
                  rxi[*r][ii][jj] += s1;
                  }
   s2 = 0.0;
   for ( ii = 1; ii <= m; ii++ )
       for ( jj = 1; jj <= m; jj++ )
           s2 += fabs( rxi[*r][ii][jj] );

   if ( s2 < xitol )
      {
      nq    = 1;
      delta = TRUE;
      while ( (nq <= q) && (*r < n - 1) && (delta) )
            {
            nq = nq + 1;
            *r = *r + 1;
            for ( j = 1; j <= q; j++ )
                if ( *r >= j )
                   for ( ii = 1; ii <= m; ii++ )
                       for ( jj = 1; jj <= m; jj++ )
                           {
                           s1 = 0.0;
                           for ( h = 1; h <= m; h++ )
                               s1 += theta[j][ii][h] * rxi[*r-j][h][jj];
                           rxi[*r][ii][jj] += s1;
                           }
            s2 = 0.0;
            for ( ii = 1; ii <= m; ii++ )
                for ( jj = 1; jj <= m; jj++ )
                    s2 += fabs( rxi[*r][ii][jj] );

            if ( s2 > xitol ) delta = FALSE;
            }
      if ( delta ) *r -= nq;
      }
   }
   while ( (!delta) && (*r < n - 1) );

/* [2]: Premultiply every rxi (general) by q1inv (lower triangular):         */

   mtmp = matrix( 1, m, 1, m );

   for ( k = 0; k <= *r; k++ )
       {
       for ( i = 1; i <= m; i++ )
           for ( j = 1; j <= m; j++ )
               {
               s1 = 0.0;
               for ( h = 1; h <= i; h++ )
                   s1 += q1inv[i][h] * rxi[k][h][j];
               mtmp[i][j] = s1;
               }
       for ( i = 1; i <= m; i++ )
           for ( j = 1; j <= m; j++ )
               rxi[k][i][j] = mtmp[i][j];
       }
   free_matrix( mtmp, 1, m, 1, m );
}

/*****************************************************************************/

void cres( int m, int n, int g, int nlim, real ***rxi, real **q1,
           real **matm, real **matl, real *lambda, real **res )

{
   int  h, i, j, jj;
   real sum;

/* [1]: Solve for c in L'c = lambda and store as lambda (overwrite):         */

   cholbak( matl, m * g, lambda );

/* [2]: Compute d = Mc and store as lambda (overwrite):                      */

   for ( i = m * g; i >= 1; i-- )
       {
       sum = 0.0;
       for ( j = 1; j <= i; j++ ) sum += matm[i][j] * lambda[j];
       lambda[i] = sum;
       }

/* [3]: Compute residuals and return as res (overwrite):                     */

   for ( i = 1; i <= n; i++ )
       for ( j = 1; j <= i; j++ )
           if ( (i - j <= nlim) && (j <= g) )
              for ( jj = 1; jj <= m; jj++ )
                  {
                  sum = 0.0;
                  for ( h = 1; h <= m; h++ )
                      sum += rxi[i-j][jj][h] * lambda[h + (j - 1) * m];
                  res[i][jj] -= sum;
                  }
   for ( j = 1; j <= n; j++ )
       for ( i = m; i >= 1; i-- )
           {
           sum = 0.0;
           for ( h = 1; h <= i; h++ ) sum += q1[i][h] * res[j][h];
           res[j][i] = sum;
           }
}

/*****************************************************************************/

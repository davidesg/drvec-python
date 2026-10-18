/*****************************************************************************/
/*  qnewtopt.c -- part of drvarma (multivariate VARMA modelling).
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
/*  QNEWTOPT.C                                                               */
/*  Numerical optimization routines: Factorized BFGS Quasi-Newton method.    */
/*  Line search procedures: backtracking and line minimization.              */
/*  Source: Dennis & Schnabel (1983).                                        */
/*  Copyright (C) Jos� Alberto Mauricio, 1995.                               */
/*****************************************************************************/

#include "main.h"              /* Header file (prototype declarations)        */
extern real macheps;          /* Machine epsilon (global: declared in DRV.C) */
extern FILE *outputv;         /* Output file (global: declared in DRV.C)     */
extern int quiet_mode;   /* added */

/*****************************************************************************/
/*****************************************************************************/

void raxopt( real (*func)(real *), real *fk, int n, real *xk, real **b,
             int maxits, int nrits, real gradtol, real steptol )

{
   real fkp1, *xkp1, *gk, *gkp1, *dk, lambda, maxstep, eta;
   int  i, j, k, retcode, maxcmax, consecmax, maxtaken, termcode;

/* Declaration of functions used within raxopt:                              */

   void report( int n, int k, real *x, real *g, real f, int termcode );
   int umstop0( int n, real *x, real f, real *g, real gradtol,
                int maxits, int *consecmax );
   int umstop( int n, real *xk, real *xkp1, real fkp1, real *gkp1,
               int retcode, real gradtol, real steptol, int k, int maxits,
               int maxcmax, int maxtaken, int *consecmax );
   real lnsrch( int n, real *xk, real fk, real *gk, real *dk, real *xkp1,
                real *fkp1, real maxstep, real steptol, int *retcode,
                int *maxtaken, real (*func)(real *) );
   void bfgsfac( int n, real *xk, real *xkp1, real *gk, real *gkp1,
                 real eta, real **B );
   void cdgrad( real (*func)(real *), int n, real *x, real eta, real *g );

/* Allocate temporary storage:                                               */

   xkp1 = vector( 1, n );
   gk   = vector( 1, n );
   gkp1 = vector( 1, n );
   dk   = vector( 1, n );

/* Dennis-Schnabel parameters:                                               */

   for ( maxstep = 0.0, i = 1; i <= n; i++ ) maxstep += xk[i] * xk[i];
   maxstep = sqrt( maxstep );
   maxstep = rmax( maxstep, 1.0 );
   maxstep = 1.0e+3 * maxstep;
   eta     = macheps;
   maxcmax = 5;

/* Initialization:                                                           */

   k   =   0;                                 /* Iteration counter.          */
   *fk = 1.0;                                 /* Objective function at k=0.  */
   cdgrad( func, n, xk, eta, gk );            /* Gradient vector at k=0.     */
   for ( i = 1; i <= n; i++ )                 /* Search-direction matrix.    */
       {
       for ( j = 1; j <= n; j++ ) b[i][j] = 0.0;
       b[i][i] = 1.0;
       }

   termcode = umstop0( n, xk, *fk, gk, gradtol, maxits, &consecmax );

/* Main iteration loop:                                                      */

   (void) nrits;            /* per-nrits reporting removed (kept for ABI) */

   while ( termcode == 0 )
      {
      /* Per-iteration progress goes to stdout only (keeps the .out report
         clean: only the final convergence summary is written there). */
      if (!quiet_mode) printf( "%4d  F: %.10f", k, *fk );

      for ( i = 1; i <= n; i++ ) dk[i] = -gk[i];
      cholsol( b, n, dk );

      lambda = lnsrch( n, xk, *fk, gk, dk, xkp1, &fkp1,
               maxstep, steptol, &retcode, &maxtaken, func );

      if (!quiet_mode) printf( "   step: %.6f\n", lambda );

      cdgrad( func, n, xkp1, eta, gkp1 );     /* Gradient at new iterate.    */

      termcode = umstop( n, xk, xkp1, fkp1, gkp1, retcode, gradtol, steptol,
                         k + 1, maxits, maxcmax, maxtaken, &consecmax );

      bfgsfac( n, xk, xkp1, gk, gkp1, eta, b );

      k += 1;                                 /* Update itaration counter.   */

      for ( i = 1; i <= n; i++ )              /* Update iterate, gradient    */
          {                                   /* and objective function.     */
          xk[i] = xkp1[i];
          gk[i] = gkp1[i];
          }
      *fk = fkp1;
      }                                       /* Back for another iteration. */

   report( n, k, xk, gk, *fk, termcode );     /* Report on convergence.      */
   if (!quiet_mode) printf( "%4d F: %0.10f\n", k, *fk );

/* Deallocate temporary storage and return:                                  */

   free_vector( dk, 1, n );
   free_vector( gkp1, 1, n );
   free_vector( gk, 1, n );
   free_vector( xkp1, 1, n );
}

/*****************************************************************************/

void report( int n, int k, real *x, real *g, real f, int termcode )

{
   const char *crit;

   (void) n; (void) x; (void) g;   /* per-parameter x/g dump removed:        */
                                    /* the estimates already appear in the    */
                                    /* "Estimated Parameters" table.          */
   if ( quiet_mode ) return;

   switch ( termcode )
       {
       case 1:  crit = "norm of scaled gradient <= gradtol";                break;
       case 2:  crit = "scaled distance between last two steps <= steptol"; break;
       case 3:  crit = "last global step failed to locate a lower point";   break;
       case 4:  crit = "iteration limit reached";                           break;
       case 5:  crit = "five consecutive steps of maximum length taken";    break;
       default: crit = NULL;                                                break;
       }

   fprintf( outputv, "\n=============================================================\n" );
   fprintf( outputv, "  OPTIMIZER %s after %d iterations\n",
            ( termcode == 1 || termcode == 2 ) ? "CONVERGED" : "STOPPED", k );
   fprintf( outputv, "  Objective function = %.12f\n", f );
   if ( crit ) fprintf( outputv, "  Convergence criterion: %s\n", crit );
   fprintf( outputv, "=============================================================\n\n" );
}

/*****************************************************************************/

int umstop0( int n, real *x, real f, real *g, real gradtol,
             int maxits, int *consecmax )

{
   int i;
   real max1, tmp;

   if ( maxits == 0 ) return( 4 );

   *consecmax = 0;

   max1 = fabs( g[1] ) * (fabs( x[1] ) + 1.0) / (fabs( f ) + 1.0);
   for (i = 2; i <= n; i++ )
       {
       tmp = fabs( g[i] ) * (fabs( x[i] ) + 1.0) / (fabs( f ) + 1.0);
       if ( tmp > max1 ) max1 = tmp;
       }

   if ( max1 <= gradtol )
      return( 1 );
   else
      return( 0 );
}

/*****************************************************************************/

int umstop( int n, real *xk, real *xkp1, real fkp1, real *gkp1,
            int retcode, real gradtol, real steptol, int k, int maxits,
            int maxcmax, int maxtaken, int *consecmax )

{
   real max1, max2, tmp;
   int i;

   max1 = fabs( gkp1[1] ) * (fabs( xkp1[1] ) + 1.0) / (fabs( fkp1 ) + 1.0);
   for ( i = 2; i <= n; i++ )
       {
       tmp = fabs( gkp1[i] ) * (fabs( xkp1[i] ) + 1.0) / (fabs( fkp1 ) + 1.0);
       if ( tmp > max1 ) max1 = tmp;
       }

   max2 = fabs( xkp1[1] - xk[1] ) / (fabs( xkp1[1] ) + 1.0);
   for ( i = 2; i <= n; i++ )
       {
       tmp = fabs( xkp1[i] - xk[i] ) / (fabs( xkp1[i] ) + 1.0);
       if ( tmp > max2 ) max2 = tmp;
       }

   if ( max1 <= gradtol )
      return( 1 );
   else if ( retcode == 1 )
      return( 3 );
   else if ( max2 <= steptol )
      return( 2 );
   else if ( k >= maxits )
      return( 4 );
   else if ( maxtaken )
      {
      *consecmax += 1;
      if ( *consecmax == maxcmax )
         return( 5 );
      else
         return( 0 );
      }
   else
      {
      *consecmax = 0;
      return( 0 );
      }
}

/*****************************************************************************/

void bfgsfac( int n, real *xk, real *xkp1, real *gk, real *gkp1,
              real eta, real **B )

{
   real *s, *y, *u, *t, tmp1, tmp2, tmp3, tol, alpha;
   int  i, j, skpupd;
   void qrupdate( int n, real *u, real *v, real **M );

   s = vector( 1, n );
   y = vector( 1, n );
   u = vector( 1, n );
   t = vector( 1, n );

   for ( i = 1; i <= n; i++ )
       {
       s[i] = xkp1[i] - xk[i];
       y[i] = gkp1[i] - gk[i];
       }

   tmp1 = 0.0;
   tmp2 = 0.0;
   tmp3 = 0.0;

   for ( i = 1; i <= n; i++ )
       {
       tmp1 += y[i] * s[i];
       tmp2 += s[i] * s[i];
       tmp3 += y[i] * y[i];
       }

   if ( tmp1 > sqrt( macheps * tmp2 * tmp3 ) )
      {
      tmp2 = 0.0;
      for ( i = 1; i <= n; i++ )
          {
          t[i] = 0.0;
          for ( j = i; j <= n; j++ ) t[i] += B[j][i] * s[j];
          tmp2 += t[i] * t[i];
          }

      alpha  = sqrt( tmp1 / tmp2 );
      tol    = sqrt( eta );
      skpupd = 1;

      for ( i = 1; i <= n; i++ )
          {
          tmp3 = 0.0;
          for ( j = 1; j <= i; j++ ) tmp3 += B[i][j] * t[j];
          if ( fabs( y[i] - tmp3 ) >= tol * rmax(fabs( gk[i] ), fabs(gkp1[i])))
             skpupd = 0;
          u[i] = y[i] - alpha * tmp3;
          }

      if ( skpupd == 0 )
         {
         tmp3 = sqrt( tmp1 * tmp2 );
         for ( i = 1; i <= n; i++ ) t[i] /= tmp3;

         for ( i = 2; i <= n; i++ )
             for ( j = 1; j <= i - 1; j++ )
                 {
                 B[j][i] = B[i][j];
                 B[i][j] = 0.0;
                 }

         qrupdate( n, t, u, B );

         for ( i = 2; i <= n; i++ )
             for ( j = 1; j <= i - 1; j++ )
                 {
                 B[i][j] = B[j][i];
                 B[j][i] = 0.0;
                 }
         }
      }

   free_vector( t, 1, n );
   free_vector( u, 1, n );
   free_vector( y, 1, n );
   free_vector( s, 1, n );
}

/*****************************************************************************/

void qrupdate( int n, real *u, real *v, real **M )

{
   int  i, j, k;
   void jacrot( int n, int i, real a, real b, real **M );

   for ( k = n; k >= 1; k-- )
       {
       if ( u[k] ) break;
       }
   if ( k < 1 ) k = 1;

   for ( i = k - 1; i >= 1; i-- )
       {
       jacrot( n, i, u[i], -u[i + 1], M );
       if ( u[i] == 0.0 )
          u[i] = fabs( u[i + 1] );
       else
          u[i] = sqrt( u[i] * u[i] + u[i + 1] * u[i + 1] );
       }

   for ( j = 1; j <= n; j++ ) M[1][j] += u[1] * v[j];

   for ( i = 1; i <= k - 1; i++ )
       jacrot( n, i, M[i][i], - M[i + 1][i], M );
}

/*****************************************************************************/

void jacrot( int n, int i, real a, real b, real **M )

{
   int  j;
   real den, c, s, y, w;

   if ( a == 0.0 )
      {
      c = 0.0;
      s = ( b >= 0.0 ? 1.0 : -1.0 );
      }
   else
      {
      den = sqrt( a * a + b * b );
      c   = a / den;
      s   = b / den;
      }

   for ( j = i; j <= n; j++ )
       {
       y           = M[i][j];
       w           = M[i + 1][j];
       M[i][j]     = c * y - s * w;
       M[i + 1][j] = s * y + c * w;
       }
}

/*****************************************************************************/

void cdgrad( real (*func)(real *), int n, real *x, real eta, real *g )

{
   real third, stepi, tempi, fpls, fmns;
   int  i;

   third = 1.0 / 3.0;
   third = pow( eta, third );

   for ( i = 1; i <= n; i++ )
       {
       if ( x[i] == fabs( x[i] ) )
          stepi = third * rmax( x[i], 1.0 );
       else
          stepi = third * rmin( x[i], -1.0 );

       tempi = x[i];

       x[i]   = x[i] + stepi;
       stepi  = x[i] - tempi;    /* Reduces finite precision errors slightly */
       fpls   = (*func)( x );
       x[i]   = tempi - stepi;
       fmns   = (*func)( x );

       g[i]   = (fpls - fmns) / (2.0 * stepi);
       x[i]   = tempi;
       }
}

/*****************************************************************************/

void fdhess( real (*func)(real *), int n, real *x, real f, real eta, real **H )

{
   real cubreta, tempi, tempj, *fneigh, *step, fii, fij;
   int  i, j;

   fneigh  = vector( 1, n );
   step    = vector( 1, n );
   cubreta = 1.0 / 3.0;
   cubreta = pow( eta, cubreta );

   for ( i = 1; i <= n; i++ )
       {
       if ( x[i] == fabs( x[i] ) )
          step[i] = cubreta * rmax( x[i], 1.0 );
       else
          step[i] = cubreta * rmin( x[i], -1.0 );

       tempi     = x[i];
       x[i]      = x[i] + step[i];
       step[i]   = x[i] - tempi;
       fneigh[i] = (*func)( x );
       x[i]      = tempi;
       }

   for ( i = 1; i <= n; i++ )
       {
       tempi   = x[i];
       x[i]    = x[i] + 2.0 * step[i];
       fii     = (*func)( x );
       H[i][i] = (f + fii - 2.0 * fneigh[i]) / (step[i] * step[i]);
       x[i]    = tempi + step[i];

       for ( j = i + 1; j <= n; j++ )
           {
           tempj   = x[j];
           x[j]    = x[j] + step[j];
           fij     = (*func)( x );
           H[i][j] = (f - fneigh[i] + fij - fneigh[j]) / (step[i] * step[j]);
           H[j][i] = H[i][j];
           x[j]    = tempj;
           }

       x[i] = tempi;
       }

   free_vector( step, 1, n );
   free_vector( fneigh, 1, n );
}

/*****************************************************************************/

/* lnsrch lives in lib/optim/lnsrch.c: one source for the five engines. drvec's
   copy entered the monorepo on 2026-09-27 without the fix for a non-finite
   objective (the line search spun for ever on a NaN), which the other four
   engines have carried since 2026-09-26.                                     */

/*****************************************************************************/
/*****************************************************************************/

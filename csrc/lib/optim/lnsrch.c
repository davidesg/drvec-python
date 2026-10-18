/*
 * lnsrch.c -- the backtracking line search of the quasi-Newton optimiser
 * (Dennis & Schnabel, algorithm A6.3.1), shared by drvarma, drtran, fue and
 * fuf.
 *
 * It lived in each engine's qnewtopt.c, four copies of one function that
 * differed in a single "a == 0" against "a == 0.0". The defect below was
 * fixed in one of them (drvarma 5.0) and hung the other three; one source
 * fixes all of them. The rest of qnewtopt.c (raxopt, its report) stays in
 * each engine: those copies have diverged in what they print.
 *
 * A NON-FINITE objective is an inadmissible point. With a NaN every
 * comparison in the loop is false: the step was neither accepted nor
 * abandoned, lambda became NaN and the search never returned (drvarma
 * BUGS.md; drtran met it on real data and ran for an hour and a half).
 * Such a point shrinks the step without interpolating, and the search
 * gives up like any other failed one once lambda is below minlam.
 *
 * On a finite trajectory the arithmetic is the original's, operation for
 * operation: drvarma's .inp bench and fue/fuf's golden files are
 * byte-identical before and after.
 *
 * The host provides optimhost.h: `real`, rmax() and nrerror().
 *
 * Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 * GPL v2 or later; see COPYING.
 */

#include "optimhost.h"
#include <math.h>

real lnsrch( int n, real *xk, real fk, real *gk, real *dk, real *xkp1,
             real *fkp1, real maxstep, real steptol, int *retcode,
             int *maxtaken, real (*func)(real *) )

{
   real alpha, newtlen, tmp, initslp, rellen, minlam, lambda, tlambda;
   real prelam, pfkp1, t1, t2, t3, a, b, disc;
   int  haveprev = 0;               /* a previous backtrack to interpolate */
   int  i;

   *maxtaken = 0;
   *retcode  = 2;
   alpha     = 1.0e-4;

   for ( newtlen = 0.0, i = 1; i <= n; i++ ) newtlen += dk[i] * dk[i];
   newtlen = sqrt( newtlen );
   if ( newtlen > maxstep )         /* Scale if attempted step is too large: */
      {
      tmp = maxstep / newtlen;
      for ( i = 1; i <= n; i++ ) dk[i] *= tmp;
      newtlen = maxstep;
      }

   for ( initslp = 0.0, i = 1; i <= n; i++ ) initslp += gk[i] * dk[i];

   rellen = 0.0;                    /* Compute minlam:                       */
   for ( i = 1; i <= n; i++ )
       {
       tmp = fabs( dk[i] ) / rmax( fabs( xk[i] ), 1.0 );
       if ( tmp > rellen ) rellen = tmp;
       }
   minlam = steptol / rellen;
   lambda = 1.0;                    /* Always try full step first.           */

   do                               /* Start of iteration loop:              */
      {
      for ( i = 1; i <= n; i++ ) xkp1[i] = xk[i] + lambda * dk[i];
      *fkp1 = (*func)( xkp1 );

   /* A non-finite objective is an inadmissible point. With a NaN every      */
   /* comparison below is false: the step was neither accepted nor           */
   /* abandoned, lambda became NaN and the loop spun for ever (BUGS.md, the  */
   /* line search that never returns). Shrink without interpolating, and     */
   /* give up like any other failed search once lambda is below minlam.      */
      if ( !isfinite( *fkp1 ) )
         {
         if ( lambda < minlam )
            {
            *retcode = 1;
            for ( i = 1; i <= n; i++ ) xkp1[i] = xk[i];
            *fkp1 = fk;
            }
         else
            lambda = 0.1 * lambda;
         continue;
         }

      if ( *fkp1 <=  fk + alpha * lambda * initslp )
         {
      /* Sufficient function decrease:                                       */
         *retcode = 0;
         if ( (lambda == 1.0) && (newtlen > 0.99 * maxstep) ) *maxtaken = 1;
         }
      else if ( lambda < minlam )
         {
      /* Convergence on x values:                                            */
      /* The point goes back to xk, and so must its value: *fkp1 still held  */
      /* the rejected trial's, and raxopt returned it as the optimum's       */
      /* (drvec BUG-41, item 2; fue's Python port already did this).         */
         *retcode = 1;
         for ( i = 1; i <= n; i++ ) xkp1[i] = xk[i];
         *fkp1 = fk;
         }
      else
         {
      /* Backtrack:                                                          */
      /* First time (not "lambda == 1.0": after a non-finite point lambda     */
      /* is already below 1 with nothing to interpolate from):                */
         if ( !haveprev )
            tlambda = -initslp / (2.0 * (*fkp1 - fk - initslp));
      /* Subsequent backtracks:                                              */
         else
            {
            t1 = *fkp1 - fk - lambda * initslp;
            t2 = pfkp1 - fk - prelam * initslp;
            t3 = 1.0 / (lambda - prelam);
            a  = (t1 / (lambda * lambda) - t2 / (prelam * prelam)) * t3;
            b  = (t2 * lambda / (prelam * prelam) -
                 t1 * prelam / (lambda * lambda)) * t3;
            if ( a == 0.0 )
               tlambda = -initslp / (2.0 * b);
            else
               {
               disc = b * b - 3.0 * a * initslp;
               if ( disc < 0.0 )
               /* Probably, dk is not a direction of descent (initslp > 0):  */
                  nrerror( "ROUNDOFF PROBLEM IN LINE SEARCH" );
               else
                  tlambda = (-b + sqrt( disc )) / (3.0 * a);
               }
            if ( tlambda > 0.5 * lambda ) tlambda = 0.5 * lambda;
            }
         prelam = lambda;
         pfkp1  = *fkp1;
         haveprev = 1;
         if ( tlambda <= 0.1 * lambda )
            lambda = 0.1 * lambda;
         else
            lambda = tlambda;
         }
      }
   while ( *retcode ==  2 );    /* Try again */
   return( lambda );
}

/*****************************************************************************/
/*  nlatools.c -- part of drvarma (multivariate VARMA modelling).
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
/*  NLATOOLS.C                                                               */
/*  Numerical linear algebra and dynamic memory allocation routines.         */
/*                                                                           */
/*  Copyright (C) Jose Alberto Mauricio, 1995 (LU/Cholesky, string utils).   */
/*  Eigenvalues (eigenqr) and SVD (svdcp/svsol) via the GNU Scientific       */
/*  Library (GSL).  Dynamic memory routines (C) Arthur B. Treadway &         */
/*  David E. Guerrero, 2009.                                                 */
/*                                                                           */
/*  Free of Numerical Recipes code.  Free software under the GNU General     */
/*  Public License (see COPYING), v2 or, at your option, any later version.  */
/*****************************************************************************/

#include "main.h"            /* Header file (prototype declarations)        */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
extern real macheps;          /* Machine epsilon (global: declared in DRV.C) */
extern FILE *outputv;         /* Output file (global: declared in DRV.C)     */

/*****************************************************************************/

static  int iminarg1, iminarg2;
#define IMIN(a,b) (iminarg1=(a),iminarg2=(b),(iminarg1) < (iminarg2) ?\
        (iminarg1) : (iminarg2))
#define RADIX 2.0
#define SWAP(g,h) {y=(g);(g)=(h);(h)=y;}
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

/*****************************************************************************/
/*****************************************************************************/

void ludcp( real **a, int n, int *ip )

{
   int  i, j, k, kp1, m;
   real tmp;

   ip[n] = 1;
   for ( k = 1; k <= n; k++ )
       {
       if ( k != n )
          {
          kp1 = k + 1;
          m = k;
          for ( i = kp1; i <= n; i++ )
              if ( fabs( a[i][k] ) > fabs( a[m][k] ) ) m = i;
          ip[k] = m;
          if ( m != k ) ip[n] = -ip[n];
          tmp = a[m][k];
          a[m][k] = a[k][k];
          a[k][k] = tmp;
          if ( tmp != 0.0 )
             {
             for ( i = kp1; i <= n; i++ ) a[i][k] /= -tmp;
             for ( j = kp1; j <= n; j++ )
                 {
                 tmp = a[m][j];
                 a[m][j] = a[k][j];
                 a[k][j] = tmp;
                 if ( tmp != 0.0 )
                    for ( i = kp1; i <= n; i++ ) a[i][j] += a[i][k] * tmp;
                 }
             }
          }
       if ( a[k][k] == 0.0 ) ip[n] = 0;
       }
}

/****************************************************************************/

void lusol( real **a, real *b, int n, int *ip )

{
   int  i, k, km1, kp1, k1, m, nm1;
   real tmp;

   if ( n != 1 )
      {
      nm1 = n - 1;
      for ( k = 1; k <= nm1; k++ )
          {
          kp1 = k + 1;
          m = ip[k];
          tmp = b[m];
          b[m] = b[k];
          b[k] = tmp;
          for ( i = kp1; i <= n; i++ ) b[i] += a[i][k] * tmp;
          }
      for ( k1 = 1; k1 <= nm1; k1++ )
          {
          km1 = n - k1;
          k = km1 + 1;
          b[k] /= a[k][k];
          tmp = -b[k];
          for ( i = 1; i <= km1; i++ ) b[i] += a[i][k] * tmp;
          }
      }
   b[1] /= a[1][1];
}

/****************************************************************************/
/****************************************************************************/

void choldcp( real **mat, int n, real *d1, real *d2, int *ifault )

{
   int   i, j, k;
   real  sum1, minl, maxoffl, minl2, maxadd, minljj, sqrteps;

   *ifault = 0;
   *d1     = 1.0;
   *d2     = 0.0;
   sqrteps = sqrt( macheps );

/* [1]: check wether mat is numerically different from zero (redundancy):   */

   for ( maxoffl = 0.0, j = 1; j <= n; j++ )
       if ( sqrt( fabs( mat[j][j] ) ) > maxoffl )
          maxoffl = sqrt( fabs( mat[j][j] ) );
   if ( maxoffl * maxoffl <= sqrteps )
      {
      for ( i = 1; i <= n; i++ ) for ( j = 1; j <= n; j++ ) mat[i][j] = 0.0;
      return;
      }

/* [2]: initialize finite-arithmetic constants:                             */

   minl   = 0.0;
   minl2  = sqrteps * maxoffl;
   maxadd = 0.0;

/* [3]: form the j-th column of the Cholesky factor of mat:                 */

   for ( j = 1; j <= n; j++ )
       {
       sum1 = mat[j][j];
       for ( i = 1; i <= j - 1; i++ ) sum1 -= mat[j][i] * mat[j][i];

       if ( (sum1 != fabs( sum1 )) && (fabs( sum1 ) > minl2) )
          {
          *ifault = 1;
          return;
          }
       else
          mat[j][j] = sum1;

       minljj = 0.0;
       for ( i = j + 1; i <= n; i++ )
           {
           sum1 = mat[j][i];
           for ( k = 1; k <= j - 1; k++ ) sum1 -= mat[i][k] * mat[j][k];
           mat[i][j] = sum1;
           if ( fabs( mat[i][j] ) > minljj ) minljj = fabs( mat[i][j] );
           }

       if ( (minljj / maxoffl) > minl )
          minljj /= maxoffl;
       else
          minljj = minl;

       if ( mat[j][j] > (minljj * minljj) )
          mat[j][j] = sqrt( mat[j][j] );
       else
          {
          if ( minljj < minl2 )
             minljj = minl2;
          if ( maxadd < (minljj * minljj - mat[j][j]) )
             maxadd = minljj * minljj - mat[j][j];
          mat[j][j] = minljj;
          }

       *d1 *= mat[j][j] * mat[j][j];
       while ( fabs( *d1 ) >= 1.0 )
             {
             *d1 *= 0.0625;
             *d2 += 4.0;
             }
       while ( fabs( *d1 ) < 0.0625 )
             {
             *d1 *= 16.0;
             *d2 -=  4.0;
             }

       for ( i = j + 1; i <= n; i++ ) mat[i][j] /= mat[j][j];
       }

   for ( j = 2; j <= n; j++ )
       for ( i = 1; i <= j - 1; i++ ) mat[i][j] = 0.0;

}

/****************************************************************************/

void cholfor( real **matl, int n, real *rhsol )

{
   int   i, j;
   real  tmp;

   rhsol[1] /= matl[1][1];

   for ( i = 2; i <= n; i++ )
       {
       tmp = 0.0;
       for ( j = 1; j <= i - 1; j++ )
           tmp += matl[i][j] * rhsol[j];
       rhsol[i] = (rhsol[i] - tmp) / matl[i][i];
       }
}

/****************************************************************************/

void cholbak( real **matl, int n, real *rhsol )

{
   int   i, j;
   real  tmp;

   rhsol[n] /= matl[n][n];

   for ( i = n - 1; i >= 1; i-- )
       {
       tmp = 0.0;
       for ( j = i + 1; j <= n; j++ )
           tmp += matl[j][i] * rhsol[j];
       rhsol[i] = (rhsol[i] - tmp) / matl[i][i];
       }
}

/****************************************************************************/

void cholsol( real **matl, int n, real *rhsol )

{
   cholfor( matl, n, rhsol );
   cholbak( matl, n, rhsol );
}

/****************************************************************************/
/****************************************************************************/

real pythag( real a, real b );      /* forward decl (defined below) */

/****************************************************************************/
/*  Eigenvalues of a real general matrix via GSL (replaces NR hqr/balanc/  */
/*  elmhes/tred2).  On exit wr/wi hold the real/imag parts (1-based) and    */
/*  a[i][i] the modulus of the i-th eigenvalue (as the former NR routine).  */
/****************************************************************************/

void eigenqr( real **a, int n, real *wr, real *wi )
{
   int i, j;
   gsl_matrix *A = gsl_matrix_alloc( (size_t)n, (size_t)n );
   gsl_vector_complex *eval = gsl_vector_complex_alloc( (size_t)n );
   gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc( (size_t)n );
   gsl_complex ev;

   for ( i = 0; i < n; i++ )
      for ( j = 0; j < n; j++ )
         gsl_matrix_set( A, i, j, a[i+1][j+1] );

   gsl_eigen_nonsymm_params( 0, 1, w );          /* balance before solving */
   gsl_eigen_nonsymm( A, eval, w );

   for ( i = 0; i < n; i++ )
      {
      ev = gsl_vector_complex_get( eval, i );
      wr[i+1] = GSL_REAL( ev );
      wi[i+1] = GSL_IMAG( ev );
      }
   for ( i = 1; i <= n; i++ ) a[i][i] = pythag( wr[i], wi[i] );

   gsl_eigen_nonsymm_free( w );
   gsl_vector_complex_free( eval );
   gsl_matrix_free( A );
}

/****************************************************************************/
/*  Singular value decomposition via GSL: a = u w v^T (1-based).            */
/*  On exit a holds u (m x n), w the singular values (n), v the n x n V.    */
/****************************************************************************/

void svdcp( real **a, int m, int n, real *w, real **v )
{
   int i, j;
   gsl_matrix *A = gsl_matrix_alloc( (size_t)m, (size_t)n );
   gsl_matrix *V = gsl_matrix_alloc( (size_t)n, (size_t)n );
   gsl_vector *S = gsl_vector_alloc( (size_t)n );
   gsl_vector *work = gsl_vector_alloc( (size_t)n );

   for ( i = 0; i < m; i++ )
      for ( j = 0; j < n; j++ )
         gsl_matrix_set( A, i, j, a[i+1][j+1] );

   gsl_linalg_SV_decomp( A, V, S, work );

   for ( i = 0; i < m; i++ )
      for ( j = 0; j < n; j++ )
         a[i+1][j+1] = gsl_matrix_get( A, i, j );
   for ( j = 0; j < n; j++ ) w[j+1] = gsl_vector_get( S, j );
   for ( i = 0; i < n; i++ )
      for ( j = 0; j < n; j++ )
         v[i+1][j+1] = gsl_matrix_get( V, i, j );

   gsl_vector_free( work ); gsl_vector_free( S );
   gsl_matrix_free( V ); gsl_matrix_free( A );
}

/****************************************************************************/
/*  Solve (u w v^T) y = x using the SVD factors; x is overwritten with y.  */
/****************************************************************************/

void svsol( real **u, real *w, real **v, int n, real *x )
{
   int i, j;
   gsl_matrix *U = gsl_matrix_alloc( (size_t)n, (size_t)n );
   gsl_matrix *V = gsl_matrix_alloc( (size_t)n, (size_t)n );
   gsl_vector *S = gsl_vector_alloc( (size_t)n );
   gsl_vector *b = gsl_vector_alloc( (size_t)n );
   gsl_vector *sol = gsl_vector_alloc( (size_t)n );

   for ( i = 0; i < n; i++ )
      {
      for ( j = 0; j < n; j++ )
         {
         gsl_matrix_set( U, i, j, u[i+1][j+1] );
         gsl_matrix_set( V, i, j, v[i+1][j+1] );
         }
      gsl_vector_set( S, i, w[i+1] );
      gsl_vector_set( b, i, x[i+1] );
      }

   gsl_linalg_SV_solve( U, V, S, b, sol );

   for ( i = 0; i < n; i++ ) x[i+1] = gsl_vector_get( sol, i );

   gsl_vector_free( sol ); gsl_vector_free( b ); gsl_vector_free( S );
   gsl_matrix_free( V ); gsl_matrix_free( U );
}

/****************************************************************************/
/*  Arthur B. Treadway & David E. Guerrero (2009) -- dynamic memory.        */
/*  1-based (nl-based) indexing; allocations sized [0..nh] (calloc) so any   */
/*  nl >= 0 is valid without undefined pointer arithmetic.                   */
/****************************************************************************/

void nrerror( char error_text[] )
{
   fprintf( stderr, "Unrecoverable run-time error:\n%s\n... Exiting to system ...\n",
            error_text );
   exit( 1 );
}

/* DELTA drvec, 2026-08-19: adopted from the suite's shared copy.  Same defect
   and same history as tensor() -- the Numerical-Recipes cleanup dropped the base
   offset -- so v[nl..nh] was addressable only for nl >= 0.  No call site in
   drvec uses a negative lower bound today, which is why this one was latent
   rather than live, but the suite corrected it after the transfer-function
   program was bitten by exactly that (its identification allocates
   vector(-nlags, nlags)).  Aligning removes the divergence.  Note the contract
   this establishes: the release MUST be given the same nl as the allocation,
   where the previous form ignored it.  Verified mechanically over the whole
   test suite; see docs/SUITE_INTEGRATION.md §5.                              */

real *vector( long nl, long nh )
{
   real *v = (real *)calloc( (size_t)(nh - nl + 1), sizeof(real) );
   if ( !v ) nrerror( "ALLOCATION FAILURE in vector()" );
   return( v - nl );
}

int *ivector( long nl, long nh )                    /* see vector() above */
{
   int *v = (int *)calloc( (size_t)(nh - nl + 1), sizeof(int) );
   if ( !v ) nrerror( "ALLOCATION FAILURE in ivector()" );
   return( v - nl );
}

real **matrix( long nrl, long nrh, long ncl, long nch )
{
   long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
   real **m = (real **)calloc( (size_t)nrow, sizeof(real *) );
   real *data;
   if ( !m ) nrerror( "ALLOCATION FAILURE 1 in matrix()" );
   m -= nrl;
   data = (real *)calloc( (size_t)(nrow * ncol), sizeof(real) );
   if ( !data ) nrerror( "ALLOCATION FAILURE 2 in matrix()" );
   for ( i = nrl; i <= nrh; i++ ) m[i] = data + (i - nrl) * ncol - ncl;
   return( m );
}

int **imatrix( long nrl, long nrh, long ncl, long nch )
{
   long i, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
   int **m = (int **)calloc( (size_t)nrow, sizeof(int *) );
   int *data;
   if ( !m ) nrerror( "ALLOCATION FAILURE 1 in imatrix()" );
   m -= nrl;
   data = (int *)calloc( (size_t)(nrow * ncol), sizeof(int) );
   if ( !data ) nrerror( "ALLOCATION FAILURE 2 in imatrix()" );
   for ( i = nrl; i <= nrh; i++ ) m[i] = data + (i - nrl) * ncol - ncl;
   return( m );
}

/* DELTA drvec, 2026-08-19: adopted from the suite's own correction, not devised
   here.  The Numerical-Recipes cleanup that reimplemented these allocators
   dropped the base offset and allocated (nrh + 1) row pointers indexed from
   zero.  That is sound while nrl >= 0 -- it merely over-allocates -- but elf()
   allocates gamwa as tensor( -q+1, 0, 1, m, 1, m ), whose lower bound is
   NEGATIVE as soon as q >= 2; the initialisation loop then wrote at t[-1] and
   corrupted the heap, so no model with two or more MA lags could be estimated.

   The defect was diagnosed and fixed elsewhere in the suite on 2026-06-15 and
   is recorded in the univariate program's defect register; drvec had continued
   to carry the pre-fix copy, and reached the fault by the ordinary route of a
   user specifying q >= 2.  The code below is the suite's fix verbatim, so this
   function is now identical to the copy shared by the transfer-function and
   VARMA programs.  See docs/SUITE_INTEGRATION.md §5.

   It changes no computed value: for nrl >= 0 -- every call site in drvec
   itself -- the slots addressed are the same as before.                      */
real ***tensor( long nrl, long nrh, long ncl, long nch, long ndl, long ndh )
{
   long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
   real ***t = (real ***)calloc( (size_t)nrow, sizeof(real **) );
   real **planes; real *data;
   if ( !t ) nrerror( "ALLOCATION FAILURE 1 in tensor()" );
   t -= nrl;   /* Offset the base so t[nrl..nrh] is addressable, matching the
                  (i-nrl) offsets used for planes/data below. Without this a
                  tensor with a negative lower row index (e.g. gamwa's -q+1,
                  q>0) writes t[nrl<0] out of bounds -> heap corruption and a
                  double free in free_tensor. Undone there via free(t + nrl).
                  (Restores the behaviour of the original NR t -= nrl.)       */
   planes = (real **)calloc( (size_t)(nrow * (nch + 1)), sizeof(real *) );
   if ( !planes ) nrerror( "ALLOCATION FAILURE 2 in tensor()" );
   data = (real *)calloc( (size_t)(nrow * ncol * (ndh + 1)), sizeof(real) );
   if ( !data ) nrerror( "ALLOCATION FAILURE 3 in tensor()" );
   for ( i = nrl; i <= nrh; i++ )
      {
      t[i] = planes + (i - nrl) * (nch + 1);
      for ( j = ncl; j <= nch; j++ )
         t[i][j] = data + ( (i - nrl) * ncol + (j - ncl) ) * (ndh + 1);
      }
   return( t );
}

void free_vector( real *v, long nl, long nh ) { if ( v ) free( v + nl ); }
void free_ivector( int *v, long nl, long nh ) { if ( v ) free( v + nl ); }
void free_matrix( real **m, long nrl, long nrh, long ncl, long nch )
   { if ( m ) { free( m[nrl] + ncl ); free( m + nrl ); } }
void free_imatrix( int **m, long nrl, long nrh, long ncl, long nch )
   { if ( m ) { free( m[nrl] + ncl ); free( m + nrl ); } }
void free_tensor( real ***t, long nrl, long nrh, long ncl, long nch,
                  long ndl, long ndh )
   { if ( t ) { free( t[nrl][ncl] ); free( t[nrl] ); free( t + nrl ); } }

real rmax( real a, real b )

{
   return( ( a > b ) ? a : b );
}

/****************************************************************************/

real rmin( real a, real b )

{
   return( ( a < b ) ? a : b );
}

/****************************************************************************/

real cmacheps( void )

{
   real e;

   e = 1.0;
   do {
      e /= 2.0;
   } while ( (1.0 + e ) > 1.0 );
   return( 2.0 * e );
}

/****************************************************************************/

int round_local( real num )

{
   real tmp1, tmp2;
   int  itmp;

   tmp1 = fabs( num );
   tmp2 = floor( tmp1 );
   if ( tmp1 - tmp2 >= 0.5 ) tmp2 = ceil( tmp1 );
   itmp = (int)tmp2;

   if ( num >= 0.0 )
      return( itmp );
   else
      return( -itmp );
}

/****************************************************************************/

real pythag( real a, real b )

{
   real at, bt;

   at = fabs( a );
   bt = fabs( b );
   at *= at;
   bt *= bt;
   return( sqrt( at + bt ) );
}

/****************************************************************************/
/****************************************************************************/

STRING NEW_STR( int size )

// Returns the starting address of a (size + 1)-char string, allocates space
// and initializes the string to an empty string.

{
    STRING s;

    if ( (s = (STRING)malloc( (size_t)(size + 1) * sizeof( char ) )) == NULL )
       return( NULL );
    else
       {
       *s = '\0';
       return( s );
       }
}

/****************************************************************************/

void FREE_STR( STRING s )

// Deallocates previously allocated space for string s.

{
    free( (char *)s );
}

/****************************************************************************/

int DELETE_STR( STRING s, int i, int n )

// Removes n chars from string s starting from the i-th position of s.

{
    if ( (i >= 0) && (n > 0) && ((i + n) <= strlen( s )) )
       {
       s[i] = '\0';
       strcat( s, s + i + n );
       return( OK );
       }
    else
       return( WRONG );
}

/****************************************************************************/

int COPY_STR( STRING source, int i, int n, STRING dest )

// Fills string dest with a copy of a substring of source, starting from
// the i-th position of source and consisting of n chars.

{
    register int j;

    if ( (i >= 0) && (n > 0) && ((i + n) <= strlen( source )) )
       {
       for ( j = 0; j <= n - 1; j++ )
           dest[j] = source[i + j ];
       dest[n] = '\0';
       return( OK );
       }
    else
       return( WRONG );
}

/****************************************************************************/

int INSERT_STR( STRING s1, STRING s, int i )

// Inserts the string s1 into s at the i-th position of s;
// if i == strlen( s ), then s1 is added to s.

{
    register int j, k;

    j = strlen( s1 );
    if ( (i >= 0) && (i <= strlen( s )) && (j > 0)  )
       {
       for ( k = strlen( s ) - i; k >= 0; k-- )
           s[i + j + k] = s[i + k];
       s += i;
       while ( *s1 ) *s++ = *s1++;
       return( OK );
       }
    else
       return( WRONG );
}

/****************************************************************************/

int POS_STR( STRING s1, STRING s2 )

// Returns the position (in s2) of the first occurrence of s1 in s2,
// or -1 if s1 does not exist in s2.

{
    STRING aux1, aux2, prevs2 = s2;

    for ( aux1 = s1, aux2 = s2; *aux2; aux2++ )
        if ( *aux1 == *aux2 )
           {
           aux1++;
           if ( !*aux1 )
              return( aux2 - s2 - (aux1 - s1) + 1 );
           }
        else
          {
          aux1 = s1;
          aux2 = prevs2++;
          }
    return( -1 );
}

/****************************************************************************/

int CHANGE_STR( STRING s1, int i, STRING s2 )

// Replaces strlen( s2 ) chars in s1, starting from the ith position of s1,
// with the chars from the string s2.

{
    int j, s1l, s2l;

    s1l = strlen( s1 );
    s2l = strlen( s2 );
    if ( (s1l == 0) || (i < 0) || (i > s1l) || ((i + s2l) > s1l) )
       return( WRONG );
    if ( s2l == 0 )
       return( OK );
    if ( s2l > (s1l - i + 1) )
       {
       s1[i] = '\0';
       strcat( s1, s2 );
       }
    else
       {
       s1 += i;
       for ( j = 0; j < s2l; j++ )
           *s1++ = *s2++;
       }
    return( OK );
}

/****************************************************************************/

void BLANKS_STR( STRING s )

{
   int k, len;

   len = strlen( s );
   while ( (len > 0) && (s[--len] == ' ') )     // Remove trailing blanks
      s[len] = '\0';
   k = 0;                                       // Remove leading blanks
   while ( (k <= len) && (s[k] == ' ') )
      k++;
   COPY_STR( s, k, len+1-k, s );
}

/****************************************************************************/

void UPCASE_STR( STRING s )

{
   int i, len;

   len = strlen( s );
   for ( i = 0; i < len; i++ )
       s[i] = toupper( s[i] );
}

/****************************************************************************/
/****************************************************************************/

void Easter( int *day, int *month, int year )

{
   div_t a, b, c, d, e;

   a = div( year, 19 );
   b = div( year,  4 );
   c = div( year,  7 );
   d = div( 19 * a.rem + 24, 30 );
   e = div( 2 * b.rem + 4 * c.rem + 6 * d.rem + 5, 7 );

   *day = 22 + d.rem + e.rem;

   if ( *day <= 31 )
      *month = 3;
   else
      {
      *day  -= 31;
      *month =  4;
      }
}

/****************************************************************************/
/****************************************************************************/

/* calcnu (escalar) quitada 2026-10-16: nadie la llamaba. El filtro
   determinista de la suite es lib/fuedet (BUG-40 de drvec).              */

/****************************************************************************/
/*****************************************************************************/

/*****************************************************************************/
/* DISTRIBUCI�N CHI-CUADRADO - FUNCI�N DE DISTRIBUCI�N ACUMULADA (CDF)      */
/*****************************************************************************/

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Funci�n gamma (necesaria para chi-cuadrado) */
static real gamma_func(real x) {
    /* Aproximaci�n de Stirling para gamma(x) */
    real coef[6] = {
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    };

    real ser = 1.000000000190015;
    real tmp = x + 5.5;
    tmp -= (x + 0.5) * log(tmp);

    for (int j = 0; j < 6; j++) {
        ser += coef[j] / (x + j + 1);
    }

    return exp(-tmp + log(2.5066282746310005 * ser / x));
}

/* Funci�n gamma incompleta regularizada P(a,x) */
static real gammap(real a, real x) {
    /* Serie para gamma incompleta regularizada */
    if (x < a + 1.0) {
        /* Usar desarrollo en serie */
        real ap = a;
        real del = 1.0 / a;
        real sum = del;

        for (int n = 1; n <= 100; n++) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (fabs(del) < fabs(sum) * 1e-10) break;
        }

        return sum * exp(-x + a * log(x) - log(gamma_func(a)));
    } else {
        /* Usar fracci�n continua */
        real b = x + 1.0 - a;
        real c = 1.0 / 1e-30;
        real d = 1.0 / b;
        real h = d;

        for (int i = 1; i <= 100; i++) {
            real an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = b + an / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            real del = d * c;
            h *= del;
            if (fabs(del - 1.0) < 1e-10) break;
        }

        return 1.0 - exp(-x + a * log(x) - log(gamma_func(a))) * h;
    }
}

/* Distribuci�n chi-cuadrado acumulada - P(?� < x | df) */
real chisq(real x, int df) {
    if (df <= 0) return 0.0;
    if (x <= 0.0) return 0.0;
    /*  BUG-42(2): here was `if (x > 1000.0) return 1.0;`, which put every
     *  p-value at 0 once Hosking's df = m^2*s neared 1000 (M = 5 from
     *  n = 1681). No cap is needed: Wilson-Hilferty below holds at any x
     *  for df >= 30, and gammap() handles df < 30.                         */

    /* Para df peque�os, c�lculo exacto */
    if (df < 30) {
        return gammap(df / 2.0, x / 2.0);
    } else {
        /* Para df grandes, usar aproximaci�n normal (Wilson-Hilferty) */
        real z = (pow(x / df, 1.0/3.0) - (1.0 - 2.0/(9.0 * df))) / sqrt(2.0/(9.0 * df));

        /* Distribuci�n normal acumulada */
        real t = 1.0 / (1.0 + 0.2316419 * fabs(z));
        real d = 0.3989423 * exp(-z * z / 2.0);
        real prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.821256 + t * 1.330274))));

        /*  BUG-13, arreglado el 2026-08-21 en las cuatro copias a la vez.
         *
         *  El polinomio de Abramowitz-Stegun evaluado en |z| da la cola
         *  SUPERIOR de |z|.  Para z > 0 la CDF es 1 - Q(z), y por eso la linea
         *  de abajo es correcta.  Para z < 0 la cola superior de |z| YA ES la
         *  cola inferior de z, o sea la CDF, y no hay que voltear nada: aqui
         *  habia un segundo `if (z < 0) prob = 1.0 - prob;` que la invertia.
         *
         *  Como z < 0 es el caso de un estadistico POR DEBAJO de su media --
         *  es decir, aquel en que el modelo esta bien --, esta funcion
         *  devolvia el complemento justo cuando no debia, y todo p-valor
         *  escrito como `1.0 - chisq(...)` salia complementado.  El veredicto
         *  de la diagnosis se invertia: residuos limpios declarados no
         *  blancos.  Medido: Q = 23.4777 con 40 g.l. tiene p = 0.9825 y se
         *  imprimia 0.0175 con "REJECT H0".  Solo se disparaba en los casos
         *  BUENOS, que es lo que lo hacia dificil de ver.
         *
         *  Encontrado desde drvec el 2026-08-19; declarado BUG-13 en el
         *  registro de defectos de la suite.  Comprobado contra
         *  gsl_cdf_chisq_P: con el arreglo coincide a ~1e-4 en todo el rango,
         *  que es el error de Wilson-Hilferty y no un defecto.  Ningun
         *  llamante se habia adaptado al fallo -- los cuatro programas
         *  escriben `1.0 - chisq(...)` --, asi que corregirlo no rompe nada.
         *
         *  ESTE FICHERO ES COMPARTIDO: si se toca aqui, hay que tocarlo en
         *  drvarma, drtran, drvec y la copia del paquete de Python.          */
        if (z > 0) prob = 1.0 - prob;

        return prob;
    }
}

/* Versi�n alternativa m�s simple (menos precisa pero m�s r�pida) */
real chisq_simple(real x, int df) {
    if (df <= 0 || x <= 0.0) return 0.0;
    if (x > 1000.0) return 1.0;

    /* Aproximaci�n de Wilson-Hilferty (buena para df > 30) */
    if (df > 30) {
        real z = (pow(x / df, 1.0/3.0) - (1.0 - 2.0/(9.0 * df))) / sqrt(2.0/(9.0 * df));

        /* Distribuci�n normal est�ndar acumulada usando erf */
        return 0.5 * (1.0 + erf(z / sqrt(2.0)));
    }

    /* Para df peque�os, usar serie m�s simple */
    real sum = 0.0;
    real term = exp(-x/2.0);

    if (df % 2 == 0) {
        /* df par */
        int k = df / 2;
        real running_sum = term;
        for (int i = 1; i < k; i++) {
            term *= x / (2.0 * i);
            running_sum += term;
        }
        sum = 1.0 - running_sum;
    } else {
        /* df impar - usar f�rmula m�s compleja */
        real sqrt_x = sqrt(x);
        real t = sqrt_x / sqrt(df);
        sum = 2.0 * (1.0 - 0.5 * (1.0 + erf(t/sqrt(2.0))));

        /* Correcci�n */
        if (df > 1) {
            term = exp(-x/2.0) * sqrt(x/2.0) / sqrt(M_PI);
            sum += term;
            for (int i = 1; i < (df-1)/2; i++) {
                term *= x / (2.0 * i + 1.0);
                sum += term;
            }
        }
    }

    return (sum < 0.0) ? 0.0 : (sum > 1.0) ? 1.0 : sum;
}

/*****************************************************************************/
/* DISTRIBUCI�N t-STUDENT - FUNCI�N DE DISTRIBUCI�N ACUMULADA (CDF)         */
/*****************************************************************************/

real tdist(real t, int df) {
    if (df <= 0) return 0.5;  /* Distribuci�n indefinida */

    real x = df / (df + t * t);

    if (df % 2 == 0) {
        /* Grados de libertad par */
        real prob = 0.5 * (1.0 + t / sqrt(df + t * t));
        real term = 1.0;

        for (int i = 2; i <= df - 2; i += 2) {
            term *= (i - 1.0) * x / i;
            prob += term;
        }

        return prob;
    } else {
        /* Grados de libertad impar */
        real prob = 0.5 + atan(t / sqrt(df)) / M_PI;
        if (df == 1) return prob;

        real term = t * sqrt(x) / sqrt(df * M_PI);
        prob += term;

        for (int i = 3; i <= df - 2; i += 2) {
            term *= (i - 2.0) * x / (i - 1.0);
            prob += term;
        }

        return prob;
    }
}

/* Versi�n alternativa usando beta incompleta */
real tdist_beta(real t, int df) {
    real x = df / (df + t * t);

    /* Beta incompleta regularizada I_x(a,b) */
    /* Para la t-student: P(T < t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2) si t > 0 */

    if (t == 0.0) return 0.5;

    real a = df / 2.0;
    real b = 0.5;

    /* Aproximaci�n de la beta incompleta */
    real bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x));

    if (x < (a + 1.0) / (a + b + 2.0)) {
        /* Usar desarrollo en serie */
        real apb = a + b;
        real ap1 = a + 1.0;
        real ns = floor(b + x * apb);
        real tx, sum;
        real w = 1.0;

        for (int i = 0; i < ns; i++) {
            w *= (apb + i) * x / (ap1 + i);
        }

        tx = w;
        sum = w;

        for (int n = ns + 1; n <= 100; n++) {
            w *= (apb + n - 1.0) * x / (ap1 + n - 1.0);
            sum += w;
            if (fabs(w) < fabs(sum) * 1e-10) break;
        }

        real result = bt * sum / (a * (1.0 + (b - ns) / (ap1 + ns - 1.0)));

        if (t > 0) {
            return 1.0 - 0.5 * result;
        } else {
            return 0.5 * result;
        }
    } else {
        /* Usar fracci�n continua */
        real c = 1.0;
        real d = 1.0 - (a + b) * x / (a + 1.0);
        if (fabs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        real h = d;

        for (int i = 1; i <= 100; i++) {
            int m2 = 2 * i;
            real aa = i * (b - i) * x / ((a + m2 - 1.0) * (a + m2));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            h *= d * c;

            aa = -(a + i) * (a + b + i) * x / ((a + m2) * (a + m2 + 1.0));
            d = 1.0 + aa * d;
            if (fabs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c;
            if (fabs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            real del = d * c;
            h *= del;

            if (fabs(del - 1.0) < 1e-10) break;
        }

        real result = 1.0 - bt * h / a;

        if (t > 0) {
            return 1.0 - 0.5 * result;
        } else {
            return 0.5 * result;
        }
    }
}

/* Funci�n de distribuci�n normal est�ndar acumulada (para referencia) */
real normal_cdf(real z) {
    /* Aproximaci�n de Abramowitz y Stegun (precisi�n 7.5e-8) */
    real t = 1.0 / (1.0 + 0.2316419 * fabs(z));
    real d = 0.3989423 * exp(-z * z / 2.0);
    real prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.821256 + t * 1.330274))));

    if (z > 0) prob = 1.0 - prob;
    return prob;
}

/*---------------------------------------------------------------------------*/
/*  Funciones auxiliares matriciales                                         */
/*---------------------------------------------------------------------------*/
 void matrix_multiply(real **A, real **B, real **C, int n, int m, int p) {
    int i, j, k;
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= p; j++) {
            real sum = 0.0;
            for (k = 1; k <= m; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

void matrix_transpose(real **A, real **B, int n, int m) {
    int i, j;
    for (i = 1; i <= n; i++)
        for (j = 1; j <= m; j++)
            B[j][i] = A[i][j];
}

 void matrix_inverse(real **A, real **invA, int n) {
    /* Calcula la inversa de A usando LU (ludcp y lusol) */
    real **Acopy = matrix(1, n, 1, n);
    int *ip = ivector(1, n);
    real *col = vector(1, n);
    int i, j;

    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            Acopy[i][j] = A[i][j];

    ludcp(Acopy, n, ip);
    for (j = 1; j <= n; j++) {
        for (i = 1; i <= n; i++) col[i] = 0.0;
        col[j] = 1.0;
        lusol(Acopy, col, n, ip);
        for (i = 1; i <= n; i++) invA[i][j] = col[i];
    }

    free_vector(col, 1, n);
    free_ivector(ip, 1, n);
    free_matrix(Acopy, 1, n, 1, n);
}


/*****************************************************************************/


#undef FREE_ARG
#undef SIGN
#undef SWAP
#undef RADIX
#undef IMIN

/****************************************************************************/

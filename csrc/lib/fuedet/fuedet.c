/*
 * fuedet.c -- ver fuedet.h.
 *
 * El codigo es el de fue.c: calcnu (fuedet_nu) y el paso [5] de cast_us
 * (fuedet_core), con el MISMO orden de bucles y la MISMA acumulacion -- tmp2
 * por variable, tmp1 += tmp2, y una sola resta --. No es una reescritura: un
 * orden distinto da otros bits, y los bits son los que comparan las baterias
 * de fue, fuf, drtran, drvarma y drvec.
 *
 * Lo unico propio es la memoria: malloc en vez de los vector() del anfitrion,
 * para que el fichero no dependa de nadie y se pueda copiar tal cual.
 *
 * ESTE FICHERO TIENE UNA COPIA FUERA: atws/fue/fue/csrc/fuedet.c (el paquete
 * de Python), byte a byte, vigilada por su sha256. Si se cambia aqui, se
 * copia alli y se apunta el sha256 nuevo en sus pruebas.
 *
 * Copyright (C) 1995-2026 J.A. Mauricio, A.B. Treadway & D.E. Guerrero.
 * License: GPL v2 or later.
 */

#include <stdlib.h>

#include "fuedet.h"

int fuedet_nulag( int nomega, int ndelta )
{
   if ( ndelta == 0 )
      return nomega;
   else
      return FUEDET_NULAG_DELTA;
}

/* calcnu de fue.c, sin cambiar una operacion. */
void fuedet_nu( double *omega, int s, double *delta, int r,
                double *nu, int lags )
{
   int  i, j;
   double sum1, sum2;

   nu[0] = omega[0];
   for ( j = 1; j <= lags; j++ )
       {
       sum1 = 0.0;
       if ( r > 0 )
          for ( i = 1; i <= j; i++ )
              if ( i <= r ) sum1 = sum1 + delta[i] * nu[j-i];
       sum2 = 0.0;
       if ( s > 0 )
          if ( j <= s ) sum2 = omega[j];
       nu[j] = sum1 - sum2;
       }
}

/* El paso [5] de fue.c. Con data == NULL devuelve la suma (det); si no,
   data - suma.                                                             */
static int fuedet_core( int nvar, double **omega, int *nomega,
                        double **delta, int *ndelta,
                        double **x, const double *data, int n, double *out )
{
   int  i, j, k, *NuLag;
   double **Nu, tmp1, tmp2;

   if ( nvar <= 0 )
      {
      for ( i = 1; i <= n; i++ ) out[i] = data ? data[i] : 0.0;
      return 0;
      }

   NuLag = (int *)malloc( (size_t)(nvar + 1) * sizeof(int) );
   Nu    = (double **)calloc( (size_t)(nvar + 1), sizeof(double *) );
   if ( !NuLag || !Nu ) { free( NuLag ); free( Nu ); return -1; }

   for ( j = 1; j <= nvar; j++ )
       {
       NuLag[j] = fuedet_nulag( nomega[j], ndelta[j] );
       Nu[j] = (double *)malloc( (size_t)(NuLag[j] + 1) * sizeof(double) );
       if ( !Nu[j] )
          {
          for ( k = 1; k < j; k++ ) free( Nu[k] );
          free( Nu ); free( NuLag );
          return -1;
          }
       fuedet_nu( omega[j], nomega[j], delta[j], ndelta[j], Nu[j], NuLag[j] );
       }

   for ( i = 1; i <= n; i++ )
       {
       tmp1 = 0.0;
       for ( j = 1; j <= nvar; j++ )
           {
           tmp2 = 0.0;
           for ( k = 0; k <= NuLag[j]; k++ ) if ( i-k >= 1 )
               tmp2 += Nu[j][k] * x[j][i-k];
           tmp1 += tmp2;
           }
       if ( data )
          out[i] = data[i] - tmp1;
       else
          out[i] = tmp1;
       }

   for ( j = nvar; j >= 1; j-- ) free( Nu[j] );
   free( Nu );
   free( NuLag );
   return 0;
}

int fuedet_sum( int nvar, double **omega, int *nomega,
                double **delta, int *ndelta,
                double **x, int n, double *det )
{
   return fuedet_core( nvar, omega, nomega, delta, ndelta, x, NULL, n, det );
}

int fuedet_noise( int nvar, double **omega, int *nomega,
                  double **delta, int *ndelta,
                  double **x, const double *data, int n, double *out )
{
   return fuedet_core( nvar, omega, nomega, delta, ndelta, x, data, n, out );
}

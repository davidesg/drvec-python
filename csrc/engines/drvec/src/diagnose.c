/*****************************************************************************/
/*  diagnose.c -- part of drvarma (multivariate VARMA modelling).
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
/*  diagnose.c                                                              */
/*  Diagnostic functions for VARMA residuals: summary statistics, plots,    */
/*  autocorrelations, cross-correlations, histograms, etc.                  */
/*  Adapted from drv.c (original by José Alberto Mauricio, 1995).           */
/*****************************************************************************/

#include "main.h"
#include <math.h>
#include <string.h>

extern FILE *outputv;   /* global output file */
extern real macheps;

/* Parsed data metadata (defined in drvarma.c) used for dating residuals and
   labelling series in the diagnostics. */
extern int data_freq, data_start_year, data_start_sub;
extern int trans_d, trans_D;       /* differencing orders */
extern char **series_names;        /* 1..nser series names */

/* Column/series label: the parsed series name if available, else "<prefix><i>".
   Writes into buf when falling back; the returned pointer is valid until the
   next call that reuses the same buffer. */
static const char *col_label(int i, const char *prefix, char *buf) {
    if (series_names && series_names[i]) return series_names[i];
    sprintf(buf, "%s%d", prefix, i);
    return buf;
}

/*---------------------------------------------------------------------------*/
/*  Basic statistical functions                                              */
/*---------------------------------------------------------------------------*/
real Mean(real *data, int nobs) {
    int i;
    real sum = 0.0;
    for (i = 1; i <= nobs; i++) sum += data[i];
    return sum / nobs;
}

real Stdev(real *data, int nobs) {
    real ave = Mean(data, nobs);
    real sum = 0.0;
    int i;
    for (i = 1; i <= nobs; i++) sum += (data[i] - ave) * (data[i] - ave);
    return sqrt(sum / nobs);
}

real Skew(real *data, int nobs) {
    real ave = Mean(data, nobs);
    real std = Stdev(data, nobs);
    if (std < 1e-20) return 0.0;
    real sum = 0.0;
    int i;
    for (i = 1; i <= nobs; i++) sum += pow((data[i] - ave) / std, 3.0);
    return sum / nobs;
}

real Kurt(real *data, int nobs) {
    real ave = Mean(data, nobs);
    real std = Stdev(data, nobs);
    if (std < 1e-20) return 0.0;
    real sum = 0.0;
    int i;
    for (i = 1; i <= nobs; i++) sum += pow((data[i] - ave) / std, 4.0);
    return sum / nobs - 3.0;
}

int MaxVal(real *data, int nobs) {
    int imax = 1;
    real max = data[1];
    int i;
    for (i = 2; i <= nobs; i++) if (data[i] > max) { max = data[i]; imax = i; }
    return imax;
}

int MinVal(real *data, int nobs) {
    int imin = 1;
    real min = data[1];
    int i;
    for (i = 2; i <= nobs; i++) if (data[i] < min) { min = data[i]; imin = i; }
    return imin;
}

/*---------------------------------------------------------------------------*/
/*  Autocorrelation and related tests                                        */
/*---------------------------------------------------------------------------*/
void Acf(real *data, int nobs, int lags, real *corr, real mean, real var) {
    int i, j;
    for (j = 1; j <= lags; j++) {
        corr[j] = 0.0;
        for (i = 1; i <= nobs - j; i++)
            corr[j] += (data[i] - mean) * (data[i+j] - mean);
        corr[j] /= (nobs * var);
    }
}

void Pacf(int lags, real *pcorr) {
    int i, j;
    real sum1, sum2;
    real **MatPacf = matrix(1, lags, 1, lags);
    real *corr = vector(1, lags);
    for (i = 1; i <= lags; i++) corr[i] = pcorr[i];   /* input is acf */
    MatPacf[1][1] = corr[1];
    for (i = 2; i <= lags; i++) {
        sum1 = sum2 = 0.0;
        for (j = 1; j <= i-1; j++) {
            sum1 += MatPacf[i-1][j] * corr[i-j];
            sum2 += MatPacf[i-1][j] * corr[j];
        }
        MatPacf[i][i] = (corr[i] - sum1) / (1.0 - sum2);
        for (j = 1; j <= i-1; j++)
            MatPacf[i][j] = MatPacf[i-1][j] - MatPacf[i][i] * MatPacf[i-1][i-j];
    }
    for (i = 1; i <= lags; i++) pcorr[i] = MatPacf[i][i];
    free_matrix(MatPacf, 1, lags, 1, lags);
    free_vector(corr, 1, lags);
}

real ChiTest(real *corr, int lags, int nobs) {
    int i;
    real chisqr = 0.0;
    for (i = 1; i <= lags; i++)
        chisqr += corr[i] * corr[i] / (nobs - i);
    chisqr *= nobs * (nobs + 2);
    return chisqr;
}

/*---------------------------------------------------------------------------*/
void ObsToDate(int beg_per, int beg_sub, int obs_no, int freq,
                       int *per, int *sub) {
    div_t cad;
    if (obs_no + beg_sub - 1 <= freq) {
        *per = beg_per;
        *sub = beg_sub + obs_no - 1;
    } else {
        cad = div(obs_no - (freq - beg_sub + 1), freq);
        if (cad.rem > 0) {
            *per = beg_per + cad.quot + 1;
            *sub = cad.rem;
        } else {
            *per = beg_per + cad.quot;
            *sub = freq;
        }
    }
}


/*---------------------------------------------------------------------------*/
/*  Plotting functions (ASCII)                                               */
/*---------------------------------------------------------------------------*/
 void PlotCor( real *corr, int lags, int isacf, struct Tseries *ser, int npar )

{
   int  nobs, freq, i, j, posi, symbol, round_local( real );
   real pos, HorInc;
   STRING Guions, Marcas, TmpStr;

   nobs = ser->nobs;
   freq = ser->freq;

   Guions = NEW_STR( 80 );
   Marcas = NEW_STR( 80 );
   TmpStr = NEW_STR( 80 );

   strcpy( Guions, "-------------+-------------------------+-------------------------+--------------" );
   HorInc = 25.0;
   if ( isacf )
      {
      strcpy( Marcas, "            -1                         0                         1  L-B Q  DF" );
      fprintf( outputv, "Autocorrelation function (acf " );
      }
   else
      {
      strcpy( Marcas, "            -1                         0                         1" );
      fprintf( outputv, "Partial autocorrelation function (pacf " );
      }
   fprintf( outputv, "bands =  %5.3f):\n", 2.0 / sqrt( nobs ) );
   fprintf( outputv, "\n" );
   fprintf( outputv, "%s\n", Marcas );
   fprintf( outputv, "%s\n", Guions );

   for ( i = 1; i <= lags; i++ )
       {
       TmpStr[0] = '\0';
       while ( strlen( TmpStr ) <= 52 ) strcat( TmpStr, " " );
       if ( (freq != 1) && ( i % freq == 0) )
          {
          fprintf( outputv, "%4d %7.3f +", i, corr[i] );
          symbol     = '+';
          TmpStr[51] = '+';
          }
       else
          {
          fprintf( outputv, "%4d %7.3f |", i, corr[i] );
          symbol     = '*';
          TmpStr[51] = '|';
          }
       pos  = corr[i] * HorInc;
       posi = abs( round_local( pos ) );
       if ( pos <= 0.0 )
          for ( j = 25 - posi; j <= 25; j++ ) TmpStr[j] = symbol;
       else
          for ( j = 25; j <= 25 + posi; j++ ) TmpStr[j] = symbol;
       TmpStr[25] = '|';
       pos  = 2.0 / sqrt( nobs ) * HorInc;
       posi = round_local( pos );
       if ( TmpStr[25 + posi] == ' ' )
          TmpStr[25 + posi] = ':';
       if ( TmpStr[25 - posi] == ' ' )
          TmpStr[25 - posi] = ':';
       fprintf( outputv, "%s", TmpStr );

       if ( (freq != 1) && (i % freq == 0) && (isacf) && (i - npar >= 1) )
          {
          fprintf( outputv, "%6.2f ", ChiTest( corr, i, nobs ) );
          fprintf( outputv, "%3d", i - npar );
          }
       if ( (i % freq != 0) && (isacf) && (i == lags) && (i - npar >= 1) )
          {
          fprintf( outputv, "%6.2f ", ChiTest( corr, i, nobs ) );
          fprintf( outputv, "%3d", i - npar );
          }
       if ( (freq == 1) && (isacf) && (i == lags) && (i - npar >= 1) )
          {
          fprintf( outputv, "%6.2f ", ChiTest( corr, i, nobs ) );
          fprintf( outputv, "%3d", i - npar );
          }
       fprintf( outputv, "\n" );

       }

   fprintf( outputv, "%s\n", Guions );
   fprintf( outputv, "%s\n", Marcas );
   fprintf( outputv, "\n" );

   FREE_STR( TmpStr );
   FREE_STR( Marcas );
   FREE_STR( Guions );
}

void PlotCCF(real *corr, int lags, struct Tseries *ser) {
    int nobs = ser->nobs;
    int freq = ser->freq;
    int i, j, posi;
    real pos, HorInc = 25.0;
    char Guions[81], Marcas[81], TmpStr[81];

    strcpy(Guions, "-------------+-------------------------+-------------------------+--------------");
    strcpy(Marcas, "            -1                         0                         1");

    fprintf(outputv, "CCF BANDS  2.0/SQRT(N) =  %2.5f:\n", 2.0 / sqrt(nobs));
    fprintf(outputv, "\n%s\n%s\n", Marcas, Guions);

    for (i = 1; i <= 2*lags+1; i++) {
        memset(TmpStr, ' ', 80); TmpStr[80] = '\0';
        if ((freq != 1) && ((i-lags-1) % freq == 0)) {
            fprintf(outputv, "%4d %7.3f +", i-lags-1, corr[i]);
            TmpStr[51] = '+';
        } else {
            fprintf(outputv, "%4d %7.3f |", i-lags-1, corr[i]);
            TmpStr[51] = '|';
        }
        pos = corr[i] * HorInc;
        posi = (int)(pos + (pos>=0?0.5:-0.5));
        if (pos <= 0.0)
            for (j = 25 - abs(posi); j <= 25; j++) TmpStr[j] = '*';
        else
            for (j = 25; j <= 25 + posi; j++) TmpStr[j] = '*';
        TmpStr[25] = '|';
        pos = 2.0 / sqrt(nobs) * HorInc;
        posi = (int)(pos + 0.5);
        if (TmpStr[25 + posi] == ' ') TmpStr[25 + posi] = ':';
        if (TmpStr[25 - posi] == ' ') TmpStr[25 - posi] = ':';
        fprintf(outputv, "%s\n", TmpStr);
    }
    fprintf(outputv, "%s\n%s\n", Guions, Marcas);
}

/*---------------------------------------------------------------------------*/
/*  Cross-correlation                                                        */
/*---------------------------------------------------------------------------*/
void Ccf(real *data1, real *data2, int nobs, int lags, real *corr,
                real mean1, real mean2, real sd1, real sd2) {
    int i, j;
    for (j = 1; j <= lags+1; j++) {
        corr[j] = 0.0;
        for (i = 1; i <= nobs - j + 1; i++)
            corr[j] += (data1[i] - mean1) * (data2[i+j-1] - mean2);
        corr[j] /= (nobs * sd1 * sd2);
    }
}

real ChiTestC(real *corr, int lags, int nobs) {
    int i;
    real chisqr = 0.0;
    for (i = 1; i <= lags; i++)
        chisqr += corr[i] * corr[i] / (nobs - i + 1);
    chisqr *= nobs * (nobs + 2);
    return chisqr;
}

/*---------------------------------------------------------------------------*/
/*  Residual series analysis                                                 */
/*---------------------------------------------------------------------------*/
void AnalyzeOneSeries(real *res, int nobs, int idx, int freq) {
    struct Tseries ser;
    real mean, sd, var, *corr;
    int lags, npar = 0;   /* npar not used here */
    int i;

    mean = Mean(res, nobs);
    sd = Stdev(res, nobs);
    var = sd * sd;

    ser.name = "residuals";
    ser.nobs = nobs;
    ser.freq = freq;
    ser.begtime = 1;
    ser.begyear = 1;
    ser.endtime = 1;
    ser.endyear = 1;
    ser.mean = mean;
    ser.var = var;
    ser.max = MaxVal(res, nobs);
    ser.min = MinVal(res, nobs);
    ser.skew = Skew(res, nobs);
    ser.kurt = Kurt(res, nobs);
    ser.data = res;

    /* Summary statistics */
    fprintf(outputv, "\n--- Residual series a[%d] ---\n", idx);
    fprintf(outputv, "Mean: %15.8f   Std.Error: %15.8f\n", mean, sd / sqrt(nobs));
    fprintf(outputv, "Variance: %15.8f   Std.Dev.: %15.8f\n", var, sd);
    fprintf(outputv, "Skewness: %15.8f   Kurtosis: %15.8f\n", ser.skew, ser.kurt);
    fprintf(outputv, "Maximum: %15.8f at obs %d\n", res[ser.max], ser.max);
    fprintf(outputv, "Minimum: %15.8f at obs %d\n", res[ser.min], ser.min);

    /* Determine number of lags for ACF */
    if (nobs < 3 * (freq + 1))
        lags = nobs - freq / 2;
    else
        lags = 3 * (freq + 2);
    if (lags > nobs - 2) lags = nobs - 2;

    corr = vector(1, lags);
    Acf(res, nobs, lags, corr, mean, var);
    PlotCor(corr, lags, 1, &ser, npar);
    Pacf(lags, corr);
    PlotCor(corr, lags, 0, &ser, npar);
    free_vector(corr, 1, lags);
}


 /*---------------------------------------------------------------------------*/
/*  Main diagnostic function                                                 */
/*---------------------------------------------------------------------------*/
/*  npar_ser[1..m]: the coefficients of each equation on its own lags, which
 *  the Ljung-Box of that residual subtracts from its degrees of freedom
 *  (BUG-42, item 3).  NULL subtracts nothing.                               */
void diagnose(struct Tvarma *varma, const int *npar_ser) {
    int m = varma->m;
    int n = varma->n;
    int i, j, lags;
    real *res1, *res2, *corr, *corr1, *corr2, *totcorr;
    struct Tseries s1, s2;

    fprintf(outputv, "\n\n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "                RESIDUAL DIAGNOSTICS                         \n");
    fprintf(outputv, "=============================================================\n");

    /* Univariate analysis for each residual series */
    for (i = 1; i <= m; i++) {
        res1 = vector(1, n);
        for (j = 1; j <= n; j++) res1[j] = varma->a[j][i];

        /* Residuals belong to the differenced series, which starts at raw
           observation (d + D*freq + 1); date them from there. */
        int res_offset = trans_d + trans_D * data_freq;
        int rby, rbt;
        ObsToDate(data_start_year, data_start_sub, res_offset + 1, data_freq,
                  &rby, &rbt);

        /* Fill Tseries structure */
        s1.data = res1;
        s1.nobs = n;
        s1.freq = data_freq;
        s1.begtime = rbt;
        s1.begyear = rby;
        s1.name = series_names ? series_names[i] : "residuals";

        if (series_names)
            fprintf(outputv, "\n--- Residual series a[%d] (%s) ---\n", i, series_names[i]);
        else
            fprintf(outputv, "\n--- Residual series a[%d] ---\n", i);
        File_StatSer(&s1);
        File_PlotSer(&s1);
        File_HistSer(&s1);

        /* Determine number of lags for ACF */
        if (n < 3 * (s1.freq + 1))
            lags = n - s1.freq / 2;
        else
            lags = 3 * (s1.freq + 2);
        if (lags > n - 2) lags = n - 2;

        corr = vector(1, lags);
        Acf(res1, n, lags, corr, s1.mean, s1.var);
        PlotCor(corr, lags, 1, &s1, npar_ser ? npar_ser[i] : 0);
        Pacf(lags, corr);
        PlotCor(corr, lags, 0, &s1, npar_ser ? npar_ser[i] : 0);
        free_vector(corr, 1, lags);

        free_vector(res1, 1, n);
    }

    /* Cross-correlations between residuals */
    if (m > 1) {
        fprintf(outputv, "\n--- Cross-correlation functions between residuals ---\n");
        /* determine lags */
        if (n < 3 * (1 + 1))
            lags = n - 1/2;
        else
            lags = 3 * (1 + 2);
        if (lags > n - 2) lags = n - 2;

        corr1 = vector(1, lags + 1);
        corr2 = vector(1, lags + 1);
        totcorr = vector(1, 2 * lags + 1);

        for (i = 1; i < m; i++) {
            res1 = vector(1, n);
            for (j = 1; j <= n; j++) res1[j] = varma->a[j][i];
            s1.data = res1; s1.nobs = n; s1.mean = Mean(res1, n);
            real sd1 = Stdev(res1, n); s1.var = sd1 * sd1;
            s1.freq = 1; s1.begtime = 1; s1.begyear = 1;

            for (j = i+1; j <= m; j++) {
                res2 = vector(1, n);
                for (int t = 1; t <= n; t++) res2[t] = varma->a[t][j];
                s2.data = res2; s2.nobs = n; s2.mean = Mean(res2, n);
                real sd2 = Stdev(res2, n); s2.var = sd2 * sd2;

                if (series_names) {
                    fprintf(outputv, "\nCROSS CORRELATION a[%d] (%s) - a[%d] (%s)\n",
                            j, series_names[j], i, series_names[i]);
                    fprintf(outputv, "      %s --> %s IF k > 0\n", series_names[j], series_names[i]);
                    fprintf(outputv, "      %s --> %s IF k < 0\n", series_names[i], series_names[j]);
                } else {
                    fprintf(outputv, "\nCROSS CORRELATION a[%d] - a[%d]\n", j, i);
                    fprintf(outputv, "      a[%d] --> a[%d] IF k > 0\n", j, i);
                    fprintf(outputv, "      a[%d] --> a[%d] IF k < 0\n", i, j);
                }

                Ccf(res1, res2, n, lags, corr1, s1.mean, s2.mean, sd1, sd2);
                Ccf(res2, res1, n, lags, corr2, s2.mean, s1.mean, sd2, sd1);

                for (int h = 1; h <= lags + 1; h++)
                    totcorr[h] = corr1[lags + 2 - h];
                for (int h = 2; h <= lags + 1; h++)
                    totcorr[lags + h] = corr2[h];

                PlotCCF(totcorr, lags, &s1);
                fprintf(outputv, "Q(%2d) = %6.3f     k >= 0      Q(%2d) = %6.3f      k > 0\n",
                        lags + 1, ChiTestC(corr2, lags + 1, n),
                        lags, ChiTestC(corr2, lags + 1, n) - (corr2[1] * corr2[1] * (n + 2)));
                fprintf(outputv, "Q(%2d) = %6.3f     k <= 0      Q(%2d) = %6.3f      k < 0\n",
                        lags + 1, ChiTestC(corr1, lags + 1, n),
                        lags, ChiTestC(corr1, lags + 1, n) - (corr1[1] * corr1[1] * (n + 2)));
                free_vector(res2, 1, n);
            }
            free_vector(res1, 1, n);
        }
        free_vector(corr1, 1, lags+1);
        free_vector(corr2, 1, lags+1);
        free_vector(totcorr, 1, 2*lags+1);
    }

    fprintf(outputv, "\n=============================================================\n");
}

/*---------------------------------------------------------------------------*/
/*  File_StatSer: detailed statistics for a residual series                  */
/*---------------------------------------------------------------------------*/
 void File_StatSer(struct Tseries *ser) {
    int Maxy, Maxt, Miny, Mint, Aper, Asub;
    real tmp;

    ObsToDate(ser->begyear, ser->begtime, ser->nobs, ser->freq, &Aper, &Asub);
    ser->endtime = Asub;
    ser->endyear = Aper;
    ser->mean = Mean(ser->data, ser->nobs);
    tmp = Stdev(ser->data, ser->nobs);
    ser->var = tmp * tmp;
    ser->skew = Skew(ser->data, ser->nobs);
    ser->kurt = Kurt(ser->data, ser->nobs);
    ser->max = MaxVal(ser->data, ser->nobs);
    ser->min = MinVal(ser->data, ser->nobs);

    fprintf(outputv, "Unconditional residuals (seasonal period: %d)\n", ser->freq);
    fprintf(outputv, "%d observations: ", ser->nobs);
    if (ser->freq > 1)
        fprintf(outputv, "from %d/%d to %d/%d\n",
                ser->begtime, ser->begyear, ser->endtime, ser->endyear);
    else
        fprintf(outputv, "from %d to %d\n", ser->begyear, ser->endyear);
    fprintf(outputv, "\n");
    fprintf(outputv, "                  Mean: %18.6f\n", ser->mean);
    fprintf(outputv, "Standard error of mean: %18.6f\n", tmp / sqrt(ser->nobs));
    fprintf(outputv, "              Variance: %18.6f\n", tmp * tmp);
    fprintf(outputv, "    Standard deviation: %18.6f\n", tmp);
    fprintf(outputv, "              Skewness: %18.6f\n", ser->skew);
    fprintf(outputv, "              Kurtosis: %18.6f\n", ser->kurt);

    ObsToDate(ser->begyear, ser->begtime, ser->max, ser->freq, &Maxy, &Maxt);
    ObsToDate(ser->begyear, ser->begtime, ser->min, ser->freq, &Miny, &Mint);

    if (ser->freq > 1) {
        fprintf(outputv, "               Minimum: %18.6f at %2d/%d (observation %3d)\n",
                ser->data[ser->min], Mint, Miny, ser->min);
        fprintf(outputv, "               Maximum: %18.6f at %2d/%d (observation %3d)\n",
                ser->data[ser->max], Maxt, Maxy, ser->max);
    } else {
        fprintf(outputv, "               Minimum: %18.6f at %d (observation %3d)\n",
                ser->data[ser->min], Miny, ser->min);
        fprintf(outputv, "               Maximum: %18.6f at %d (observation %3d)\n",
                ser->data[ser->max], Maxy, ser->max);
    }
    fprintf(outputv, "\n");
}

/*---------------------------------------------------------------------------*/
/*  File_PlotSer: standardized time series plot                              */
/*---------------------------------------------------------------------------*/
 void File_PlotSer( struct Tseries *ser )

{
   int  i, Aper, Asub, itmp1, itmp2, round_local( real );
   real BandPos1, BandPos2, Pos, AbsMax, HorInc, rtmp1, rtmp2, rtmp3, rtmp4;
   STRING Guions, Marcas, Tmpstr;

   Guions = NEW_STR( 80 );
   Marcas = NEW_STR( 80 );
   Tmpstr = NEW_STR( 80 );

   strcpy( Guions, "-------------+-------------------------+-------------------------+--------------" );
   strcpy( Marcas, "                                       0                          " );

   Aper = 0;
   Asub = 0;

/* Maximum value to plot (if < 2.0, then force 3.0):                         */

   itmp1 = ser->max;
   itmp2 = ser->min;
   rtmp1 = ser->data[itmp1];
   rtmp2 = ser->data[itmp2];
   rtmp3 = ser->mean;
   rtmp4 = sqrt( ser->var );

   /* Degenerate (constant) series: var = 0 -> rtmp4 = 0 would make the scaling
      0/0 = NaN and overflow the plot buffer below.  Skip the plot. */
   if ( !( rtmp4 > 0.0 ) )
      {
      fprintf( outputv, "(residual series is constant; plot skipped)\n\n" );
      goto p1;
      }

   AbsMax = fabs( (rtmp1 - rtmp3) / rtmp4 );
   if ( fabs( (rtmp2 - rtmp3) / rtmp4 ) > AbsMax )
      AbsMax = fabs( (rtmp2 - rtmp3) / rtmp4 );
   if ( AbsMax <= 2.0 ) AbsMax = 3.0;

   /* Use !(<=8) so a non-finite AbsMax (NaN) is also caught. */
   if ( !( AbsMax <= 8.0 ) )
      {
      fprintf( outputv, "Warning: at least one observation above 8 sigmas\n" );
      goto p1;
      }

/* The value of each character + positions of � and 2� bands:                */

   HorInc   = 25.0 / AbsMax;
   BandPos1 = HorInc;
   BandPos2 = 2.0 * HorInc;

   for ( i = 1; i <= 8; i++ ) if ( AbsMax >= i )
       {
       sprintf( Tmpstr, "%d", i);
//       itoa( i, Tmpstr, 10 );
       Guions[39 - round_local( i * HorInc )]     = '+';
       Marcas[39 - round_local( i * HorInc )]     = Tmpstr[0];
       Marcas[39 - round_local( i * HorInc ) - 1] = '-';
       Guions[39 + round_local( i * HorInc )]     = '+';
       Marcas[39 + round_local( i * HorInc )]     = Tmpstr[0];
       Marcas[39 + round_local( i * HorInc ) - 1] = '+';
       }

   fprintf( outputv, "Standardized time series plot " );
   fprintf( outputv, "(original values on right-side column):\n" );
   fprintf( outputv, "\n" );
   fprintf( outputv, "%s\n", Marcas );
   fprintf( outputv, "%s\n", Guions );

/* Standardized time series plot:                                            */

   for ( i = 1; i <= ser->nobs; i++ )
       {
       fprintf( outputv, "%4d", i );
       ObsToDate( ser->begyear, ser->begtime, i, ser->freq, &Aper, &Asub );
       if ( ser->freq == 1 )
          fprintf( outputv, "%7d ", Aper );
       else
          fprintf( outputv, "%3d/%4d", Asub, Aper );
       Tmpstr[0] = '\0';
       while ( strlen( Tmpstr ) <= 54 ) strcat( Tmpstr, " " );
       if ( (ser->freq != 1) && (Asub == ser->freq) )
          {
          Tmpstr[1]  = '+';
          Tmpstr[53] = '+';
          }
       else
          {
          Tmpstr[1]  = '|';
          Tmpstr[53] = '|';
          }
       if ( fabs( (ser->data[i] - rtmp3) / rtmp4 ) >= 2.0 )
          {
          Tmpstr[0]  = '>';
          Tmpstr[54] = '>';
          }
       Pos = (ser->data[i] - rtmp3) / rtmp4 * HorInc;
       Tmpstr[27 + round_local( Pos )] = '*';
       if ( Tmpstr[27] == ' ' )
          Tmpstr[27] = '|';
       if ( Tmpstr[27 + round_local( BandPos1 )] == ' ' )
          Tmpstr[27 + round_local( BandPos1 )] = ':';
       if ( Tmpstr[27 - round_local( BandPos1 )] == ' ' )
          Tmpstr[27 - round_local( BandPos1 )] = ':';
       if ( Tmpstr[27 + round_local( BandPos2 )] == ' ' )
          Tmpstr[27 + round_local( BandPos2 )] = ':';
       if ( Tmpstr[27 - round_local( BandPos2 )] == ' ' )
          Tmpstr[27 - round_local( BandPos2 )] = ':';
       fprintf( outputv, "%s", Tmpstr );
       fprintf( outputv, "%13.10f\n", ser->data[i] );
       }
   fprintf( outputv, "%s\n", Guions );
   fprintf( outputv, "%s\n", Marcas );
   fprintf( outputv, "\n" );

/* Table of outliers:                                                        */

   fprintf( outputv, "                 +------------------------------------------+\n" );
   fprintf( outputv, "                 |       Table of standardized values       |\n" );
   fprintf( outputv, "                 |       greater than or equal to 2.0       |\n" );
   fprintf( outputv, "                 +------------------------------------------+\n" );
   fprintf( outputv, "                 |                                          |\n" );
   fprintf( outputv, "                 | Observation    Date   Standardized value |\n" );
   fprintf( outputv, "                 |                                          |\n" );

   for ( i = 1; i <= ser->nobs; i++ )
       if ( fabs( (ser->data[i] - rtmp3) / rtmp4 ) >= 2.0 )
          {
          fprintf( outputv, "                 |" );
          fprintf( outputv, "%7d", i );
          ObsToDate( ser->begyear, ser->begtime, i, ser->freq, &Aper, &Asub );
          if ( ser->freq == 1 )
             fprintf( outputv, "%13d ", Aper );
          else
             fprintf( outputv, "%9d/%4d", Asub, Aper );
          fprintf( outputv, "%13.2f", (ser->data[i]-rtmp3)/rtmp4 );
          fprintf( outputv, "        |\n" );
          }
   fprintf( outputv, "                 +------------------------------------------+\n" );
   fprintf( outputv, "\n" );

p1:FREE_STR( Tmpstr );
   FREE_STR( Marcas );
   FREE_STR( Guions );
}
/*---------------------------------------------------------------------------*/
/*  File_HistSer: standardized histogram                                     */
/*---------------------------------------------------------------------------*/
 void File_HistSer(struct Tseries *ser) {
    const int NumFil = 17;
    const int NumCol = 64;
    int i, j, NumCat, nphor, fmax, Atip1 = 0, Atip2 = 0;
    real fmax1, rtmp1, rtmp2, xmax, ObsPerFil;
    int *freqs, *chk;
    real *breakk;
    char **shist, **aux;
    char base1[81], base2[81], no[10], yes[10], s1[10], s2[10];

    freqs  = ivector(1, 50);
    chk    = ivector(1, 50);
    breakk = vector(1, 50);
    shist  = (char **)malloc(NumFil * sizeof(char *));
    aux    = (char **)malloc(NumFil * sizeof(char *));
    for (i = 0; i < NumFil; i++) {
        /*  BUG-16.  NumCol + 1 is one byte short.  The rows carry NumCat
         *  categories of nphor characters each, and NumCat*nphor is exactly
         *  NumCol (16*4 with xmax = 4, 32*2 with xmax = 8); then a closing
         *  "|" is appended, so the string is NumCol + 1 characters and needs
         *  NumCol + 2 bytes with its terminator.  strcat wrote the NUL one
         *  past the end of the block on every histogram this routine has ever
         *  drawn.  Found on 2026-08-24 by drvec's valgrind block, the first
         *  time this file was linked into a program that runs valgrind over
         *  its whole output.                                                 */
        shist[i] = (char *)malloc(NumCol + 2);
        aux[i]   = (char *)malloc(NumCol + 2);
        shist[i][0] = '\0';
        aux[i][0]   = '\0';
    }

    /* Find maximum absolute standardized value */
    rtmp1 = ser->mean;
    rtmp2 = sqrt(ser->var);
    /* Degenerate (constant) series: var = 0 -> division by zero / NaN. Skip. */
    if (!(rtmp2 > 0.0)) {
        fprintf(outputv, "(residual series is constant; histogram skipped)\n\n");
        goto h1;
    }
    xmax = fabs((ser->data[ser->max] - rtmp1) / rtmp2);
    if (fabs((ser->data[ser->min] - rtmp1) / rtmp2) > xmax)
        xmax = fabs((ser->data[ser->min] - rtmp1) / rtmp2);
    if (!(xmax <= 8.0)) {
        fprintf(outputv, "Warning: at least one observation above 8 sigmas\n");
        goto h1;
    }
    xmax = (xmax <= 4.0) ? 4.0 : 8.0;

    if (xmax == 4.0) {
        nphor = 4;
        strcpy(no, "    ");
        strcpy(yes, "....");
        strcpy(base1, "        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+");
        strcpy(base2, "       -4      -3      -2      -1       0      +1      +2      +3      +4");
    } else {
        nphor = 2;
        strcpy(no, "  ");
        strcpy(yes, "..");
        strcpy(base1, "        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+");
        strcpy(base2, "       -8  -7  -6  -5  -4  -3  -2  -1   0  +1  +2  +3  +4  +5  +6  +7  +8");
    }

    /* Create breakpoints (bandwidth 0.5) */
    NumCat = 1;
    breakk[NumCat] = -xmax + 0.5;
    while (breakk[NumCat] < xmax) {
        NumCat++;
        breakk[NumCat] = breakk[NumCat-1] + 0.5;
    }

    for (i = 1; i <= 50; i++) freqs[i] = 0;
    for (i = 1; i <= ser->nobs; i++) {
        real num = (ser->data[i] - rtmp1) / rtmp2;
        if (num <= breakk[1])
            freqs[1]++;
        else {
            for (j = 2; j <= NumCat; j++) {
                if (num > breakk[j-1] && num <= breakk[j]) {
                    freqs[j]++;
                    break;
                }
            }
        }
        if (fabs(num) >= 2.0) Atip2++;
        if (fabs(num) >= 1.0) Atip1++;
    }

    fmax = freqs[1];
    for (i = 2; i <= NumCat; i++)
        if (freqs[i] > fmax) fmax = freqs[i];
    fmax1 = fmax;
    ObsPerFil = fmax1 / 16.0;

    /* Fill histogram rows */
    for (j = 2; j <= NumFil; j++) {
        for (i = 1; i <= NumCat; i++) {
            if (freqs[i] > ObsPerFil * (NumFil - j))
                strcat(shist[j-1], yes);
            else
                strcat(shist[j-1], no);
        }
        /* remove trailing spaces? not necessary */
        strcat(shist[j-1], "|");
    }

    for (i = 1; i <= 50; i++) chk[i] = 0;
    for (j = 2; j <= NumFil; j++) {
        for (i = 1; i <= NumCat; i++) {
            if ((freqs[i] > ObsPerFil * (NumFil - j)) && chk[i] == 0) {
                sprintf(s1, "%d", freqs[i]);
                if (nphor == 2) {
                    if (strlen(s1) == 1) strcat(s1, " ");
                } else {
                    if (strlen(s1) == 2) {
                        strcpy(s2, " "); strcat(s2, s1); strcpy(s1, s2);
                    } else if (strlen(s1) == 1) {
                        strcpy(s2, "  "); strcat(s2, s1); strcpy(s1, s2);
                    } else if (strlen(s1) == 3) {
                        strcat(s1, " ");
                    }
                }
                /* A count wider than its cell (100+ with nphor = 2, 10000+
                   with 4) wrote past the row: heap overflow (BUG-42, item 1).
                   The cell keeps its width and says the count did not fit.
                   Same fix as drvarma 5.0.                                  */
                if ((int) strlen(s1) > nphor) {
                    memset(s1, '*', nphor);
                    s1[nphor] = '\0';
                }
                strcat(aux[j-2], s1);
                chk[i] = 1;
            } else {
                strcat(aux[j-2], no);
            }
        }
    }

    strcpy(shist[0], aux[0]);
    shist[0][NumCol-1] = '|';
    shist[0][NumCol] = '\0';
    for (j = 2; j <= NumFil-1; j++) {
        int k;
        for (k = 0; k < (int)strlen(aux[j-1]); k++) {
            if (aux[j-1][k] != ' ') shist[j-1][k] = aux[j-1][k];
        }
    }

    fprintf(outputv, "Standardized time series histogram:\n\n");
    fprintf(outputv, "%s\n", base2);
    fprintf(outputv, "%s\n", base1);
    for (i = 1; i <= NumFil; i++)
        fprintf(outputv, "        |%s\n", shist[i-1]);
    fprintf(outputv, "%s\n", base1);
    fprintf(outputv, "%s\n", base2);
    fprintf(outputv, "\n");

    fprintf(outputv, "%16d values outside (-1,+1): %5.2f %% (31.74 %% expected)\n",
            Atip1, (Atip1 * 100.0) / ser->nobs);
    fprintf(outputv, "%16d values outside (-2,+2): %5.2f %% ( 4.56 %% expected)\n",
            Atip2, (Atip2 * 100.0) / ser->nobs);
    fprintf(outputv, "\n");

h1:
    free_vector(breakk, 1, 50);
    free_ivector(chk, 1, 50);
    free_ivector(freqs, 1, 50);
    for (i = 0; i < NumFil; i++) {
        free(shist[i]);
        free(aux[i]);
    }
    free(shist);
    free(aux);
}



/*  Hosking's portmanteau, the multivariate Jarque-Bera and the block that
 *  prints them live in lib/diagmv, shared with drvarma and drtran (BUG-42,
 *  items 3 and 4).                                                          */

/*---------------------------------------------------------------------------*/
/*  Test de Wald para una hipótesis lineal H0: theta = 0                    */
/*  theta : vector de parámetros bajo H0 (longitud n)                        */
/*  cov   : matriz de covarianza completa (npar × npar)                      */
/*  indices: vector de índices (1..n) que indica qué elementos de cov usar   */
/*  Devuelve chi2, grados de libertad efectivos y p-valor.                   */
/*---------------------------------------------------------------------------*/
static void wald_test(real *theta, real **cov, int *indices, int n,
                      real *chi2, int *df, real *pval) {
    real **Sigma = matrix(1, n, 1, n);
    real *th = vector(1, n);
    int i, j;

    for (i = 1; i <= n; i++) {
        th[i] = theta[indices[i]];
        for (j = 1; j <= n; j++) {
            Sigma[i][j] = cov[indices[i]][indices[j]];
        }
    }

    /* SVD para inversa generalizada */
    real **U = matrix(1, n, 1, n);
    real *w = vector(1, n);
    real **V = matrix(1, n, 1, n);
    for (i = 1; i <= n; i++)
        for (j = 1; j <= n; j++)
            U[i][j] = Sigma[i][j];
    svdcp(U, n, n, w, V);

    real *tmp = vector(1, n);
    for (i = 1; i <= n; i++) tmp[i] = th[i];
    svsol(U, w, V, n, tmp);

    *chi2 = 0.0;
    for (i = 1; i <= n; i++) *chi2 += th[i] * tmp[i];

    /* grados de libertad efectivos (rango) */
    *df = 0;
    real tol = w[1] * sqrt(macheps);
    for (i = 1; i <= n; i++) if (w[i] > tol) (*df)++;
    if (*df == 0) *df = 1;

    /* p-valor usando distribución chi-cuadrado */
    *pval = 1.0 - chisq(*chi2, *df);

    free_matrix(U, 1, n, 1, n);
    free_vector(w, 1, n);
    free_matrix(V, 1, n, 1, n);
    free_vector(tmp, 1, n);
    free_matrix(Sigma, 1, n, 1, n);
    free_vector(th, 1, n);
}

/*---------------------------------------------------------------------------*/
/*  Test de significatividad conjunta del último rezago (AR y MA)            */
/*  H0: todos los coeficientes AR(p) y MA(q) son cero.                       */
/*---------------------------------------------------------------------------*/
void test_last_lag_significance(real *x, real **cov, int npar,
                                int m, int p, int q,
                                int include_mean,
                                int diag_ar, int diag_ma, int diag_cov,
                                real *chi2, int *df, real *pval) {
    int n_ar = diag_ar ? m : m * m;
    int n_ma = diag_ma ? m : m * m;
    int n_h0 = n_ar + (q > 0 ? n_ma : 0);
    if (n_h0 == 0 || p == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }

    /* Calcular índice de inicio de los parámetros */
    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;                     /* primer AR(1) */
    int idx_ar_last = idx_ar_start + (p - 1) * n_ar;  /* primer parámetro de AR(p) */
    int idx_ma_last = -1;
    if (q > 0) {
        int idx_ma_start = idx_ar_start + p * n_ar;
        idx_ma_last = idx_ma_start + (q - 1) * n_ma;
    }

    int *indices = ivector(1, n_h0);
    int k = 1;
    for (int i = 0; i < n_ar; i++) indices[k++] = idx_ar_last + i;
    if (q > 0) {
        for (int i = 0; i < n_ma; i++) indices[k++] = idx_ma_last + i;
    }

    wald_test(x, cov, indices, n_h0, chi2, df, pval);
    free_ivector(indices, 1, n_h0);
}

/*---------------------------------------------------------------------------*/
/*  Test de significatividad conjunta del último rezago AR                   */
/*  H0: todos los coeficientes AR(p) son cero.                               */
/*---------------------------------------------------------------------------*/
void test_last_ar_significance(real *x, real **cov, int npar,
                               int m, int p, int q,
                               int include_mean,
                               int diag_ar, int diag_ma, int diag_cov,
                               real *chi2, int *df, real *pval) {
    if (p == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }
    int n_ar = diag_ar ? m : m * m;
    int n_h0 = n_ar;
    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;
    int idx_ar_last = idx_ar_start + (p - 1) * n_ar;

    int *indices = ivector(1, n_h0);
    for (int i = 0; i < n_h0; i++) indices[i+1] = idx_ar_last + i;
    wald_test(x, cov, indices, n_h0, chi2, df, pval);
    free_ivector(indices, 1, n_h0);
}

/*---------------------------------------------------------------------------*/
/*  Test de significatividad conjunta del último rezago MA                   */
/*  H0: todos los coeficientes MA(q) son cero.                               */
/*---------------------------------------------------------------------------*/
void test_last_ma_significance(real *x, real **cov, int npar,
                               int m, int p, int q,
                               int include_mean,
                               int diag_ar, int diag_ma, int diag_cov,
                               real *chi2, int *df, real *pval) {
    if (q == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }
    int n_ma = diag_ma ? m : m * m;
    int n_h0 = n_ma;
    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;
    int n_ar = diag_ar ? m : m * m;
    int idx_ma_start = idx_ar_start + p * n_ar;
    int idx_ma_last = idx_ma_start + (q - 1) * n_ma;

    int *indices = ivector(1, n_h0);
    for (int i = 0; i < n_h0; i++) indices[i+1] = idx_ma_last + i;
    wald_test(x, cov, indices, n_h0, chi2, df, pval);
    free_ivector(indices, 1, n_h0);
}

/*---------------------------------------------------------------------------*/
/*  Test de todos los efectos cruzados (general)                             */
/*  (Ya existente, pero se corrige el error de índices)                      */
/*---------------------------------------------------------------------------*/
void test_cross_effects_significance(real *x, real **cov, int npar,
                                     int m, int p, int q,
                                     int include_mean,
                                     int diag_ar, int diag_ma, int diag_cov,
                                     real *chi2, int *df, real *pval) {
    if (diag_ar && diag_ma) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }
    int n_ar = diag_ar ? m : m * m;
    int n_ma = diag_ma ? m : m * m;
    int n_cross = 0;
    if (!diag_ar) n_cross += p * m * (m - 1);
    if (!diag_ma && q > 0) n_cross += q * m * (m - 1);
    if (n_cross == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }

    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;
    int *indices = ivector(1, n_cross);
    int k = 1;

    if (!diag_ar) {
        for (int lag = 1; lag <= p; lag++) {
            int base = idx_ar_start + (lag - 1) * n_ar;
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= m; j++) {
                    if (i != j) {
                        indices[k++] = base + (i-1)*m + (j-1);  // CORREGIDO
                    }
                }
            }
        }
    }
    if (!diag_ma && q > 0) {
        int idx_ma_start = idx_ar_start + p * n_ar;
        for (int lag = 1; lag <= q; lag++) {
            int base = idx_ma_start + (lag - 1) * n_ma;
            for (int i = 1; i <= m; i++) {
                for (int j = 1; j <= m; j++) {
                    if (i != j) {
                        indices[k++] = base + (i-1)*m + (j-1);  // CORREGIDO
                    }
                }
            }
        }
    }

    wald_test(x, cov, indices, n_cross, chi2, df, pval);
    free_ivector(indices, 1, n_cross);
}

/*---------------------------------------------------------------------------*/
/*  Test de efectos de otras variables sobre una variable dada               */
/*  H0: todos los coeficientes de la fila 'var' con columnas j≠var son cero  */
/*---------------------------------------------------------------------------*/
void test_effects_on_var(real *x, real **cov, int npar,
                         int m, int p, int q,
                         int include_mean,
                         int diag_ar, int diag_ma, int diag_cov,
                         int var,
                         real *chi2, int *df, real *pval) {
    if (var < 1 || var > m) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }
    int n_ar = diag_ar ? m : m * m;
    int n_ma = diag_ma ? m : m * m;
    int n_cross = 0;
    if (!diag_ar && p > 0) n_cross += p * (m - 1);
    if (!diag_ma && q > 0) n_cross += q * (m - 1);
    if (n_cross == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }

    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;
    int *indices = ivector(1, n_cross);
    int k = 1;

    if (!diag_ar && p > 0) {
        for (int lag = 1; lag <= p; lag++) {
            int base = idx_ar_start + (lag - 1) * n_ar;
            for (int j = 1; j <= m; j++) {
                if (j != var) {
                    indices[k++] = base + (var-1)*m + (j-1);
                }
            }
        }
    }
    if (!diag_ma && q > 0) {
        int idx_ma_start = idx_ar_start + p * n_ar;
        for (int lag = 1; lag <= q; lag++) {
            int base = idx_ma_start + (lag - 1) * n_ma;
            for (int j = 1; j <= m; j++) {
                if (j != var) {
                    indices[k++] = base + (var-1)*m + (j-1);
                }
            }
        }
    }

    wald_test(x, cov, indices, n_cross, chi2, df, pval);
    free_ivector(indices, 1, n_cross);
}

/*---------------------------------------------------------------------------*/
/*  Test de efectos de una variable sobre las otras                          */
/*  H0: todos los coeficientes de la columna 'var' con filas i≠var son cero  */
/*---------------------------------------------------------------------------*/
void test_effects_from_var(real *x, real **cov, int npar,
                           int m, int p, int q,
                           int include_mean,
                           int diag_ar, int diag_ma, int diag_cov,
                           int var,
                           real *chi2, int *df, real *pval) {
    if (var < 1 || var > m) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }
    int n_ar = diag_ar ? m : m * m;
    int n_ma = diag_ma ? m : m * m;
    int n_cross = 0;
    if (!diag_ar && p > 0) n_cross += p * (m - 1);
    if (!diag_ma && q > 0) n_cross += q * (m - 1);
    if (n_cross == 0) {
        *chi2 = 0.0; *df = 0; *pval = 1.0;
        return;
    }

    int idx = 1;
    if (include_mean) idx += m;
    int idx_ar_start = idx;
    int *indices = ivector(1, n_cross);
    int k = 1;

    if (!diag_ar && p > 0) {
        for (int lag = 1; lag <= p; lag++) {
            int base = idx_ar_start + (lag - 1) * n_ar;
            for (int i = 1; i <= m; i++) {
                if (i != var) {
                    indices[k++] = base + (i-1)*m + (var-1);
                }
            }
        }
    }
    if (!diag_ma && q > 0) {
        int idx_ma_start = idx_ar_start + p * n_ar;
        for (int lag = 1; lag <= q; lag++) {
            int base = idx_ma_start + (lag - 1) * n_ma;
            for (int i = 1; i <= m; i++) {
                if (i != var) {
                    indices[k++] = base + (i-1)*m + (var-1);
                }
            }
        }
    }

    wald_test(x, cov, indices, n_cross, chi2, df, pval);
    free_ivector(indices, 1, n_cross);
}


/*---------------------------------------------------------------------------*/
/*  Función que agrupa todos los tests de hipótesis y los imprime con       */
/*  conclusiones breves.                                                     */
/*---------------------------------------------------------------------------*/
void all_hypothesis_tests(real *x, real **cov, int npar,
                          int m, int p, int q,
                          int include_mean,
                          int diag_ar, int diag_ma, int diag_cov,
                          FILE *outputv) {
    real chi2, pval;
    int df;

    fprintf(outputv, "\n\n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "           JOINT HYPOTHESIS TESTS (WALD)                    \n");
    fprintf(outputv, "=============================================================\n");

    // Test de último rezago conjunto (AR y MA)
    if (p > 0 || q > 0) {
        test_last_lag_significance(x, cov, npar, m, p, q, include_mean,
                                   diag_ar, diag_ma, diag_cov,
                                   &chi2, &df, &pval);
        fprintf(outputv, "\nTest of joint significance of last lag (AR(%d) and MA(%d)):\n", p, q);
        fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
        if (pval < 0.05)
            fprintf(outputv, "  Conclusion: REJECT H0 → last lag is statistically significant.\n");
        else
            fprintf(outputv, "  Conclusion: Cannot reject H0 → last lag is not significant.\n");
    }

    // Test de último rezago AR solo
    if (p > 0) {
        test_last_ar_significance(x, cov, npar, m, p, q, include_mean,
                                  diag_ar, diag_ma, diag_cov,
                                  &chi2, &df, &pval);
        fprintf(outputv, "\nTest of joint significance of last AR lag (AR(%d)):\n", p);
        fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
        if (pval < 0.05)
            fprintf(outputv, "  Conclusion: REJECT H0 → last AR lag is significant.\n");
        else
            fprintf(outputv, "  Conclusion: Cannot reject H0 → last AR lag is not significant.\n");
    }

    // Test de último rezago MA solo
    if (q > 0) {
        test_last_ma_significance(x, cov, npar, m, p, q, include_mean,
                                  diag_ar, diag_ma, diag_cov,
                                  &chi2, &df, &pval);
        fprintf(outputv, "\nTest of joint significance of last MA lag (MA(%d)):\n", q);
        fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
        if (pval < 0.05)
            fprintf(outputv, "  Conclusion: REJECT H0 → last MA lag is significant.\n");
        else
            fprintf(outputv, "  Conclusion: Cannot reject H0 → last MA lag is not significant.\n");
    }

    // Test de todos los efectos cruzados
    test_cross_effects_significance(x, cov, npar, m, p, q, include_mean,
                                    diag_ar, diag_ma, diag_cov,
                                    &chi2, &df, &pval);
    if (df > 0) {
        fprintf(outputv, "\nTest of joint significance of all cross effects:\n");
        fprintf(outputv, "  H0: all cross coefficients = 0\n");
        fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
        if (pval < 0.05)
            fprintf(outputv, "  Conclusion: REJECT H0 → cross effects are jointly significant.\n");
        else
            fprintf(outputv, "  Conclusion: Cannot reject H0 → cross effects are not jointly significant.\n");
    } else {
        fprintf(outputv, "\nNo cross effects to test (model already diagonal).\n");
    }

    // Tests direccionales para cada variable
    fprintf(outputv, "\n--- Directional cross-effects tests ---\n");
    for (int var = 1; var <= m; var++) {
        char nb[32];
        const char *vn = col_label(var, "var", nb);
        // Efectos de otras sobre var
        test_effects_on_var(x, cov, npar, m, p, q, include_mean,
                            diag_ar, diag_ma, diag_cov, var,
                            &chi2, &df, &pval);
        if (df > 0) {
            fprintf(outputv, "\n%s: effects of other variables on it:\n", vn);
            fprintf(outputv, "  H0: all coefficients in equation %s from other vars = 0\n", vn);
            fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
            if (pval < 0.05)
                fprintf(outputv, "  Conclusion: REJECT H0 → %s is influenced by others.\n", vn);
            else
                fprintf(outputv, "  Conclusion: Cannot reject H0 → %s is not influenced by others.\n", vn);
        }

        // Efectos de var sobre otras
        test_effects_from_var(x, cov, npar, m, p, q, include_mean,
                              diag_ar, diag_ma, diag_cov, var,
                              &chi2, &df, &pval);
        if (df > 0) {
            fprintf(outputv, "\n%s: effects of it on other variables:\n", vn);
            fprintf(outputv, "  H0: all coefficients of %s in other equations = 0\n", vn);
            fprintf(outputv, "  Wald chi2(%d) = %.4f, p-value = %.4f\n", df, chi2, pval);
            if (pval < 0.05)
                fprintf(outputv, "  Conclusion: REJECT H0 → %s influences others.\n", vn);
            else
                fprintf(outputv, "  Conclusion: Cannot reject H0 → %s does not influence others.\n", vn);
        }
    }
 //   fprintf(outputv, "\n=============================================================\n");
}
/*---------------------------------------------------------------------------*/
/**
 * @brief Calcula y escribe las funciones de respuesta al impulso ortogonalizadas.
 *
 * Dado el modelo VARMA estimado, calcula la representación MA(∞)
 *   \f$ \Psi_h = \sum_{i=1}^{\min(p,h)} \Phi_i \Psi_{h-i} - \Theta_h \f$
 * (con \f$\Psi_0 = \mathbf{I}\f$) y luego la respuesta al impulso ortogonalizada
 *   \f$ \Theta_h^{(o)} = \Psi_h \mathbf{L} \f$,
 * donde \f$\mathbf{L}\f$ es el factor de Cholesky (triangular inferior) de
 * \f$\Sigma = \sigma^2 \mathbf{Q}\f$, de modo que \f$\Sigma = \mathbf{L}\mathbf{L}'\f$.
 *
 * Las respuestas se imprimen en el archivo de salida global `outputv`.
 *
 * @param[in] varma   Puntero a la estructura Tvarma con los parámetros del modelo.
 * @param[in] horizon Número máximo de períodos de la IRF.
 * @param[in] outputv Puntero al archivo de salida donde se escriben los resultados.
 */
void impulse_response(struct Tvarma *varma, int horizon, FILE *outputv) {
    int m = varma->m;
    int p = varma->p;
    int q = varma->q;
    real sigma2 = varma->sigma2;
    real **Q = varma->qq;  /* matriz de covarianza normalizada */

    /* Matriz de covarianza Sigma = sigma2 * Q */
    real **Sigma = matrix(1, m, 1, m);
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++)
            Sigma[i][j] = sigma2 * Q[i][j];

    /* Factor de Cholesky L de Sigma (triangular inferior) */
    real **L = matrix(1, m, 1, m);
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++)
            L[i][j] = Sigma[i][j];

    int ifault;
    real d1, d2;
    choldcp(L, m, &d1, &d2, &ifault);
    if (ifault) {
        fprintf(outputv,
                "Error: descomposición de Cholesky fallida en impulse_response.\n");
        free_matrix(Sigma, 1, m, 1, m);
        free_matrix(L, 1, m, 1, m);
        return;
    }
    /* Ahora L contiene el factor de Cholesky triangular inferior */

    /* Reserva memoria para pesos Psi y respuesta ortogonalizada */
    real ***Psi = (real ***)malloc((horizon + 1) * sizeof(real**));
    real ***OIRF = (real ***)malloc((horizon + 1) * sizeof(real**));
    for (int h = 0; h <= horizon; h++) {
        Psi[h] = matrix(1, m, 1, m);
        OIRF[h] = matrix(1, m, 1, m);
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= m; j++)
                Psi[h][i][j] = 0.0;
    }

    /* Psi[0] = I (identidad) */
    for (int i = 1; i <= m; i++) Psi[0][i][i] = 1.0;

    /* Cálculo de los pesos Psi[h] para h = 1 .. horizon */
    for (int h = 1; h <= horizon; h++) {
        /* Contribución de los términos AR */
        for (int i = 1; i <= p; i++) {
            if (i > h) break;
            real **phi_i = varma->phi[i];
            real **Psi_hi = Psi[h - i];
            real **tmp = matrix(1, m, 1, m);
            matrix_multiply(phi_i, Psi_hi, tmp, m, m, m);
            for (int k = 1; k <= m; k++)
                for (int l = 1; l <= m; l++)
                    Psi[h][k][l] += tmp[k][l];
            free_matrix(tmp, 1, m, 1, m);
        }
        /* Contribución de los términos MA (se RESTAN según el modelo) */
        if (h <= q) {
            real **theta_h = varma->theta[h];
            for (int k = 1; k <= m; k++)
                for (int l = 1; l <= m; l++)
                    Psi[h][k][l] -= theta_h[k][l];   /* CORRECCIÓN: resta */
        }
    }

    /* Respuesta ortogonalizada: OIRF[h] = Psi[h] * L */
    for (int h = 0; h <= horizon; h++) {
        matrix_multiply(Psi[h], L, OIRF[h], m, m, m);
    }

    /* Impresión de resultados */
    fprintf(outputv, "\n\n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "           ORTHOGONALIZED IMPULSE RESPONSE FUNCTIONS        \n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "Shocks are orthogonalized via Cholesky decomposition of Sigma Matrix.\n");
    fprintf(outputv, "Order of variables: as in the model.\n\n");



    for (int shock = 1; shock <= m; shock++) {
        char nb[32];
        fprintf(outputv, "Shock to %s:\n", col_label(shock, "var", nb));
        fprintf(outputv, "Horizon");
        for (int i = 1; i <= m; i++)
            fprintf(outputv, "%9s", col_label(i, "var", nb));
        fprintf(outputv, "\n");
        for (int h = 0; h <= horizon; h++) {
            fprintf(outputv, "%5d  ", h);
            for (int i = 1; i <= m; i++)
                fprintf(outputv, "%9.4f", OIRF[h][i][shock]);
            fprintf(outputv, "\n");
        }
        fprintf(outputv, "\n");
    }

    /* ---------- Accumulated (cumulative) impulse responses ---------- */
    fprintf(outputv, "\n=============================================================\n");
    fprintf(outputv, "        ACCUMULATED IMPULSE RESPONSE FUNCTIONS               \n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "Cumulative response of each variable to a one-s.d. shock\n");
    fprintf(outputv, "(sum of responses up to each horizon).\n\n");

    for (int shock = 1; shock <= m; shock++) {
        char nb[32];
        real *acc = vector(1, m);
        for (int i = 1; i <= m; i++) acc[i] = 0.0;
        fprintf(outputv, "Shock to %s:\n", col_label(shock, "var", nb));
        fprintf(outputv, "Horizon");
        for (int i = 1; i <= m; i++)
            fprintf(outputv, "%9s", col_label(i, "var", nb));
        fprintf(outputv, "\n");
        for (int h = 0; h <= horizon; h++) {
            fprintf(outputv, "%5d  ", h);
            for (int i = 1; i <= m; i++) {
                acc[i] += OIRF[h][i][shock];
                fprintf(outputv, "%9.4f", acc[i]);
            }
            fprintf(outputv, "\n");
        }
        fprintf(outputv, "\n");
        free_vector(acc, 1, m);
    }

    /* ---------- Cálculo e impresión de la ganancia, desviaciones típicas
            y retardo medio de respuesta ---------- */
    fprintf(outputv, "\n=============================================================\n");
    fprintf(outputv, "  LONG‑RUN GAIN, SHOCK STD. DEVIATIONS AND MEAN LAG        \n");
    fprintf(outputv, "=============================================================\n");

    /* Desviaciones típicas de los shocks estructurales */
    fprintf(outputv, "\nStructural shock standard deviations (one‑s.d. shock size):\n");
    for (int j = 1; j <= m; j++) {
        char nb[32];
        fprintf(outputv, "  %-8s : %.4f\n", col_label(j, "shock", nb), OIRF[0][j][j]);
    }

    /* Ganancia y retardo medio para cada shock */
    for (int shock = 1; shock <= m; shock++) {
        char nb[32], nb2[32];
        fprintf(outputv, "\nShock to %s:\n", col_label(shock, "var", nb));
        for (int var = 1; var <= m; var++) {
            double gain = 0.0;
            double weighted_sum = 0.0;
            double gross = 0.0;            /* total absolute response */
            for (int h = 0; h <= horizon; h++) {
                double resp = OIRF[h][var][shock];
                gain += resp;
                weighted_sum += h * resp;
                gross += fabs(resp);
            }
            /* The mean lag = (sum h*resp)/gain is only meaningful when the net
               gain is a non-negligible fraction of the gross movement; with
               near-zero gain (cancelling responses) it is unstable. */
            if (gross > 1e-12 && fabs(gain) > 0.1 * gross) {
                double mean_lag = weighted_sum / gain;
                fprintf(outputv, "  %-8s : gain = %8.4f  mean lag = %6.2f months\n",
                        col_label(var, "var", nb2), gain, mean_lag);
            } else {
                fprintf(outputv, "  %-8s : gain = %8.4f  mean lag = undefined\n",
                        col_label(var, "var", nb2), gain);
            }
        }
    }
    fprintf(outputv, "=============================================================\n\n");


    /* Liberar memoria */
    for (int h = 0; h <= horizon; h++) {
        free_matrix(Psi[h], 1, m, 1, m);
        free_matrix(OIRF[h], 1, m, 1, m);
    }
    free(Psi);
    free(OIRF);
    free_matrix(Sigma, 1, m, 1, m);
    free_matrix(L, 1, m, 1, m);
}

/*---------------------------------------------------------------------------*/
/**
 * @brief Calcula y escribe la descomposición de varianza del error de predicción (FEVD).
 *
 * Utiliza las mismas respuestas ortogonalizadas que impulse_response().
 * Para cada variable i y cada horizonte h, la contribución del shock k es
 *   \f$ \frac{ \sum_{j=0}^{h-1} [\Theta_j^{(o)}]_{i,k}^2 }
 *          { \sum_{j=0}^{h-1} \sum_{l=1}^m [\Theta_j^{(o)}]_{i,l}^2 } \f$,
 * donde \f$\Theta_j^{(o)} = \Psi_j \mathbf{L}\f$ son las IRF ortogonalizadas.
 *
 * Los porcentajes se imprimen en el archivo de salida global `outputv`.
 *
 * @param[in] varma   Puntero a la estructura Tvarma con los parámetros del modelo.
 * @param[in] horizon Número máximo de períodos de predicción.
 * @param[in] outputv Puntero al archivo de salida donde se escriben los resultados.
 */
void variance_decomposition(struct Tvarma *varma, int horizon, FILE *outputv) {
    int m = varma->m;
    int p = varma->p;
    int q = varma->q;
    real sigma2 = varma->sigma2;
    real **Q = varma->qq;

    /* Matriz de covarianza Sigma = sigma2 * Q */
    real **Sigma = matrix(1, m, 1, m);
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++)
            Sigma[i][j] = sigma2 * Q[i][j];

    /* Factor de Cholesky L de Sigma (triangular inferior) */
    real **L = matrix(1, m, 1, m);
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= m; j++)
            L[i][j] = Sigma[i][j];

    int ifault;
    real d1, d2;
    choldcp(L, m, &d1, &d2, &ifault);
    if (ifault) {
        fprintf(outputv,
                "Error: descomposición de Cholesky fallida en variance_decomposition.\n");
        free_matrix(Sigma, 1, m, 1, m);
        free_matrix(L, 1, m, 1, m);
        return;
    }

    /* Reserva para pesos Psi y respuestas ortogonalizadas */
    real ***Psi = (real ***)malloc((horizon + 1) * sizeof(real**));
    real ***OIRF = (real ***)malloc((horizon + 1) * sizeof(real**));
    for (int h = 0; h <= horizon; h++) {
        Psi[h] = matrix(1, m, 1, m);
        OIRF[h] = matrix(1, m, 1, m);
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= m; j++)
                Psi[h][i][j] = 0.0;
    }

    /* Psi[0] = I */
    for (int i = 1; i <= m; i++) Psi[0][i][i] = 1.0;

    /* Cálculo de Psi[h] (idéntico a impulse_response) */
    for (int h = 1; h <= horizon; h++) {
        /* AR */
        for (int i = 1; i <= p; i++) {
            if (i > h) break;
            real **phi_i = varma->phi[i];
            real **Psi_hi = Psi[h - i];
            real **tmp = matrix(1, m, 1, m);
            matrix_multiply(phi_i, Psi_hi, tmp, m, m, m);
            for (int k = 1; k <= m; k++)
                for (int l = 1; l <= m; l++)
                    Psi[h][k][l] += tmp[k][l];
            free_matrix(tmp, 1, m, 1, m);
        }
        /* MA (resta) */
        if (h <= q) {
            real **theta_h = varma->theta[h];
            for (int k = 1; k <= m; k++)
                for (int l = 1; l <= m; l++)
                    Psi[h][k][l] -= theta_h[k][l];   /* CORRECCIÓN: resta */
        }
    }

    /* OIRF[h] = Psi[h] * L */
    for (int h = 0; h <= horizon; h++)
        matrix_multiply(Psi[h], L, OIRF[h], m, m, m);

    /* Acumular cuadrados de OIRF: cum[h][var][shock] = suma_{j=0}^{h} OIRF[j][var][shock]^2 */
    real ***cum = (real ***)malloc((horizon + 1) * sizeof(real**));
    for (int h = 0; h <= horizon; h++) {
        cum[h] = matrix(1, m, 1, m);
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= m; j++)
                cum[h][i][j] = 0.0;
    }
    for (int h = 0; h <= horizon; h++) {
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= m; j++) {
                real val = OIRF[h][i][j];
                cum[h][i][j] = (h == 0) ? (val * val)
                                        : (cum[h - 1][i][j] + val * val);
            }
        }
    }

    /* Impresión de resultados */
    fprintf(outputv, "\n\n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "           FORECAST ERROR VARIANCE DECOMPOSITION             \n");
    fprintf(outputv, "=============================================================\n");
    fprintf(outputv, "Values are percentages of forecast error variance accounted for by each shock.\n");
    fprintf(outputv, "Shocks are orthogonalized via Cholesky decomposition.\n\n");


    for (int var = 1; var <= m; var++) {
        char nb[32];
        fprintf(outputv, "%s:\n", col_label(var, "var", nb));
        fprintf(outputv, "Horizon");
        for (int j = 1; j <= m; j++)
            fprintf(outputv, "%9s", col_label(j, "shock", nb));
        fprintf(outputv, "\n");

        for (int h = 1; h <= horizon; h++) {
            /* La varianza del error para el horizonte h usa los pesos hasta j = h-1 */
            real total = 0.0;
            for (int j = 1; j <= m; j++) total += cum[h - 1][var][j];
            if (total == 0.0) total = 1.0;   /* evitar división por cero */

            fprintf(outputv, "%5d  ", h);
            for (int j = 1; j <= m; j++) {
                real percent = 100.0 * cum[h - 1][var][j] / total;
                fprintf(outputv, "%9.2f", percent);
            }
            fprintf(outputv, "\n");
        }
        fprintf(outputv, "\n");
    }

    /* Liberar memoria */
    for (int h = 0; h <= horizon; h++) {
        free_matrix(Psi[h], 1, m, 1, m);
        free_matrix(OIRF[h], 1, m, 1, m);
        free_matrix(cum[h], 1, m, 1, m);
    }
    free(Psi);
    free(OIRF);
    free(cum);
    free_matrix(Sigma, 1, m, 1, m);
    free_matrix(L, 1, m, 1, m);
}

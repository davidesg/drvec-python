/*****************************************************************************/
/*  fue_pre_reader.c -- part of drtran (Box-Jenkins transfer function models).
 *
 *  Original to drtran.
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
/*  fue_pre_reader.c — Lector de archivos .pre (DRVUS/FUE)                   */
/*  Parsea el modelo completo: deterministas (Omega/Delta), factores ARMA,   */
/*  media (valor + flag de estimación), Box-Cox, diferencias y la serie.     */
/*****************************************************************************/

/*  PROCEDENCIA — leído antes de copiar, no supuesto.  2026-08-17.
 *
 *  COPIA de drtran/src/fue_pre_reader.c.  No es un lector nuevo: en `fue` el
 *  lector vive DENTRO de src/fue.c y no está factorizado, así que el único
 *  artefacto en C reutilizable es éste.  Para drvec, que es C puro, la
 *  referencia es la implementación en C y no la de Python (drtran y fue existen
 *  en las dos).
 *
 *  MODIFICACIONES respecto al original: comentado el #include "drtran.h"
 *  de abajo; y desde 2026-10-16 (BUG-40) build_det_component no lleva
 *  filtro propio: llama a lib/fuedet, como el original de lib/fuepre.
 *  Comprobado que este fichero sólo usa struct Tusmodel, struct
 *  Tseries y los allocators de nlatools, cuyos nombres coinciden en los dos
 *  proyectos; sus únicos símbolos externos son DateToObs, ObsToDate, Easter,
 *  vector, ivector, matrix y stderr, todos del motor de drvec.
 *
 *  RIESGO: es una copia y puede derivar del original.  Si el formato .pre
 *  cambia, hay que revisar los dos.  La gramática autoritativa del formato está
 *  en atws/fue/fue/docs/FILE_CONTRACT.md, y ante una discrepancia manda el
 *  parser (fue/src/fue/inp.py), no la documentación.
 *
 *  Ver docs/PLAN_BETA.md F2 (F2.0 fuentes, F2.1 procedencia).
 */

#include "main.h"
/* #include "drtran.h" -- no se necesita: el lector solo usa Tusmodel,
   Tseries y los allocators de nlatools.  Ver PLAN_BETA.md F2
   (procedencia).                                                      */
#include "fue_pre_reader.h"
#include <string.h>
#include <math.h>

#include "fue_bridge.h"     /* free_fue_pre, for the error paths (BUG-39) */
#include "fuedet.h"         /* the deterministic filter of fue (BUG-40)   */

/*  BUG-39.  This used PRE_LINE, and main.h defines PRE_LINE as 80: the `#ifndef
 *  PRE_LINE / 512' that stood here never took effect.  A line longer than 80
 *  characters -- a long series name on the date line -- was split by fgets
 *  into two reads, and everything after it shifted by one: a lambda = 1 file
 *  was read as lambda = 0.  The reader has its own, generous line length.   */
#define PRE_LINE 4096

/* ─── CalcNonsOp (from fue.c) ─────────────────────────────────────────── */
static void CalcNonsOp( int sp, int d, int ds, int *ifds, int ord, real *op )
{
    real *pol1, *pol2, *pol3, *pol4;
    int  i, j, k, pp, pp1, pp2;

    pp1 = d + ds * sp;
    pol1 = vector( 0, pp1 );
    pol2 = vector( 0, pp1 );
    for ( i = 1; i <= pp1; i++ ) pol1[i] = 0.0;
    pol1[0] = -1.0;
    pp      = 0;

    if ( ds > 0 )
        for ( k = 1; k <= ds; k++ )
        {
            for ( i = 0; i <= pp1; i++ ) pol2[i] = 0.0;
            for ( j = 0; j <= pp + sp; j++ )
            {
                if ( (j >= 0) && (j < sp) )
                    pol2[j] = pol1[j];
                else if ( (j >= sp) && (j <= pp) )
                    pol2[j] = pol1[j] - pol1[j-sp];
                else if ( (j > pp) && (j <= pp + sp) )
                    pol2[j] = -pol1[j-sp];
            }
            pp += sp;
            for ( i = 0; i <= pp; i++ ) pol1[i] = pol2[i];
        }

    if ( d > 0 )
        for ( k = 1; k <= d; k++ )
        {
            for ( i = 0; i <= pp1; i++ ) pol2[i] = 0.0;
            for ( j = 0; j <= pp + 1; j++ )
            {
                if ( (j >= 0) && (j < 1) )
                    pol2[j] = pol1[j];
                else if ( (j >= 1) && (j <= pp) )
                    pol2[j] = pol1[j] - pol1[j-1];
                else if ( (j > pp) && (j <= pp + 1) )
                    pol2[j] = -pol1[j-1];
            }
            pp += 1;
            for ( i = 0; i <= pp; i++ ) pol1[i] = pol2[i];
        }

    free_vector( pol2, 0, pp1 );

    pp2 = ord - pp1;
    pol2 = vector( 0, pp2 );
    pol3 = vector( 0, pp2 );
    pol4 = vector( 0, 2 );
    for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0;
    pol3[0] = -1.0;
    pp      = 0;

    if ( ((sp == 12) && (ifds[0] == 1)) || ((sp == 4) && (ifds[0] == 1)) )
    {
        pol4[0] = -1.0; pol4[1] = 1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 1; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 1; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( (sp == 12) && (ifds[1] == 1) )
    {
        pol4[0] = -1.0; pol4[1] = sqrt(3.0); pol4[2] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 2; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 2; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( (sp == 12) && (ifds[2] == 1) )
    {
        pol4[0] = -1.0; pol4[1] = 1.0; pol4[2] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 2; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 2; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( ((sp == 12) && (ifds[3] == 1)) || ((sp == 4) && (ifds[1] == 1)) )
    {
        pol4[0] = -1.0; pol4[1] = 0.0; pol4[2] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 2; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 2; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( (sp == 12) && (ifds[4] == 1) )
    {
        pol4[0] = -1.0; pol4[1] = -1.0; pol4[2] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 2; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 2; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( (sp == 12) && (ifds[5] == 1) )
    {
        pol4[0] = -1.0; pol4[1] = -sqrt(3.0); pol4[2] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 2; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 2; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }
    if ( ((sp == 12) && (ifds[6] == 1)) || ((sp == 4) && (ifds[2] == 1)) )
    {
        /* Nyquist (f=s/2): factor (1 + B), root at B=-1  =>  pol4 = -(1+B).
           Al portar de fue.c se copio aqui el (1 - B) de la frecuencia CERO
           (pol4[1] = +1.0), que sobre-diferencia en f=0 y NO diferencia en
           Nyquist: rompia toda serie con raiz estacional en pi (EA/EP/EC de m6,
           sigma ~8x, el MA se iba a la raiz unitaria). El legacy trae -1.0.     */
        pol4[0] = -1.0; pol4[1] = -1.0;
        for ( i = 1; i <= pp2; i++ ) pol2[i] = 0.0; pol2[0] = -1.0; pol3[0] = -1.0;
        for ( i = 0; i <= pp; i++ ) for ( j = 0; j <= 1; j++ ) pol2[j+i] -= pol4[j] * pol3[i];
        pp += 1; for ( i = 1; i <= pp; i++ ) pol3[i] = pol2[i];
    }

    /* El operador no estacionario es el PRODUCTO del polinomio de diferencias
       regulares/estacionales (pol1, orden pp1) por el de factores de frecuencia
       fija (pol3, orden pp). Ambos se almacenan con signo cambiado
       (pol1 = -A, pol3 = -B), luego op = -(A*B) = -(pol1 * pol3).          */
    for ( i = 0; i <= ord; i++ ) op[i] = 0.0;
    for ( i = 0; i <= pp1; i++ )
        for ( j = 0; j <= pp; j++ )
            if ( i + j <= ord ) op[i+j] -= pol1[i] * pol3[j];
    free_vector( pol4, 0, 2 );
    free_vector( pol3, 0, pp2 );
    free_vector( pol2, 0, pp2 );
    free_vector( pol1, 0, pp1 );
}

/* Nota: DateToObs, ObsToDate y Easter las aporta el motor (diagnose.c /
   nlatools.c) y estan declaradas en main.h: son las mismas rutinas que fue. */


/* ─── Genera una variable determinista a partir de su línea del .pre ─────
   Replica exactamente los tipos que fue admite (fue.c):
     impulse <per> <año>   compimp <per> <año>   step <per> <año>
     ramp <per> <año>      easter (freq=12)      trend
     cos <r>               sin <r>               alter
   En series anuales (freq=1) las intervenciones llevan solo el año.
   Devuelve 1 si reconoció el tipo, 0 si es no estándar.                  */
static int gen_detvar( const char *line, struct Tseries *Ts, real *v )
{
    const double PI = 3.14159265358979323846;
    int  j, i1, i2, obs;
    double r1;

    for (j = 1; j <= Ts->nobs; j++) v[j] = 0.0;

    /* --- Intervenciones fechadas --- */
    {
        char kind[32];
        int  nread;

        nread = sscanf(line, "%31s %d %d", kind, &i1, &i2);

        if (nread >= 2 &&
            (strcmp(kind, "impulse") == 0 || strcmp(kind, "compimp") == 0 ||
             strcmp(kind, "step")    == 0 || strcmp(kind, "ramp")    == 0)) {

            if (Ts->freq == 1) { i2 = i1; i1 = 1; }   /* anual: solo el año */
            else if (nread < 3) return 0;             /* falta el año       */

            DateToObs(Ts->begyear, Ts->begtime, i2, i1, Ts->freq, &obs);

            if (strcmp(kind, "impulse") == 0) {
                if (obs >= 1 && obs <= Ts->nobs) v[obs] = 1.0;
            } else if (strcmp(kind, "compimp") == 0) {
                if (obs >= 1 && obs <= Ts->nobs)     v[obs]   =  1.0;
                if (obs >= 0 && obs + 1 <= Ts->nobs) v[obs+1] = -1.0;
            } else if (strcmp(kind, "step") == 0) {
                if (obs >= 1 && obs <= Ts->nobs)
                    for (j = obs; j <= Ts->nobs; j++) v[j] = 1.0;
            } else {   /* ramp */
                if (obs >= 1 && obs <= Ts->nobs)
                    for (j = obs; j <= Ts->nobs; j++) v[j] = j - obs + 1;
            }
            return 1;
        }
    }

    /* --- Semana Santa (solo mensual) --- */
    if (strncmp(line, "easter", 6) == 0 && Ts->freq == 12) {
        int year, month, eday, emonth;
        for (j = 1; j <= Ts->nobs; j++) {
            ObsToDate(Ts->begyear, Ts->begtime, j, Ts->freq, &year, &month);
            Easter(&eday, &emonth, year);
            if (emonth == 4 && month == emonth && eday >= 4)
                v[j] = 1.0;
            else if (emonth == 4 && month == emonth && eday < 4) {
                v[j] = 0.5;
                if (j > 1) v[j-1] = 0.5;
            } else if (emonth == 3 && month == emonth)
                v[j] = 1.0;
        }
        return 1;
    }

    /* --- Armónicos y componentes simples --- */
    if (sscanf(line, "cos %lf", &r1) == 1) {
        for (j = 1; j <= Ts->nobs; j++)
            v[j] = cos(2.0 * PI * r1 / Ts->freq * j);
        return 1;
    }
    if (sscanf(line, "sin %lf", &r1) == 1) {
        for (j = 1; j <= Ts->nobs; j++)
            v[j] = sin(2.0 * PI * r1 / Ts->freq * j);
        return 1;
    }
    if (strncmp(line, "alter", 5) == 0) {
        for (j = 1; j <= Ts->nobs; j++) v[j] = pow(-1.0, (double)j);
        return 1;
    }
    if (strncmp(line, "trend", 5) == 0 || strncmp(line, "time", 4) == 0) {
        for (j = 1; j <= Ts->nobs; j++) v[j] = j;
        return 1;
    }

    return 0;   /* no estándar: se leerá como columna extra de datos */
}

/* ─── Lee una sección de factores ARMA del .pre ─────────────────────────
   Puerto FIEL del lector de fue (fue.c [3.3.1]-[3.3.2]). fue usa fscanf, que es
   agnóstico al espaciado: los coeficientes pueden venir uno por línea (como los
   escribe fue) o varios en la misma. Gramática, idéntica para AR regular, AR
   anual, MA regular y MA anual:

       ** Number and orders of ... :        <- cabecera (fgets)
       <Num> <ord_1> <ord_2> ...            <- fscanf
       **                                   <- separador (fgets), por factor
       <coef> <flag> ...                    <- fscanf, ord_i coeficientes

   El "**" final de cada sección se fusiona con la cabecera de la siguiente.  */
static void read_arma_section(FILE *f, char *line, int *Num, int **ord,
                              real ***coef, int ***flag)
{
    int i, j;

    fgets(line, PRE_LINE, f);                     /* cabecera */

    *Num = 0;
    if (fscanf(f, "%d", Num) != 1 || *Num <= 0) {
        *Num = 0;
        fscanf(f, "\n");
        return;
    }

    *ord  = ivector(1, *Num);
    *coef = (real **)malloc((size_t)(*Num) * sizeof(real *)) - 1;
    *flag = (int  **)malloc((size_t)(*Num) * sizeof(int  *)) - 1;

    for (i = 1; i <= *Num; i++)
        if (fscanf(f, "%d", &(*ord)[i]) != 1) (*ord)[i] = 0;
    fscanf(f, "\n");

    for (i = 1; i <= *Num; i++) {
        (*coef)[i] = vector(0, (*ord)[i]);
        (*flag)[i] = ivector(0, (*ord)[i]);

        fgets(line, PRE_LINE, f);                 /* "**" */
        for (j = 1; j <= (*ord)[i]; j++) {
            if (fscanf(f, "%lf", &(*coef)[i][j]) != 1) (*coef)[i][j] = 0.0;
            if (fscanf(f, "%d\n", &(*flag)[i][j]) != 1) (*flag)[i][j] = 0;
        }
    }
}

/* ─── Lee una sección de factores de FRECUENCIA FIJA ────────────────────
   Puerto fiel de fue.c [3.3.9]-[3.3.10]. Cada factor es de orden 2 con un único
   coeficiente libre (el término en B²), y la primera línea lleva FRECUENCIAS
   (que fue lee como reales: "f = 3.0").                                      */
static void read_fixfreq_section(FILE *f, char *line, int *Num, int **fre,
                                 real ***coef, int **flag)
{
    int i;

    fgets(line, PRE_LINE, f);                     /* cabecera */

    *Num = 0;
    if (fscanf(f, "%d", Num) != 1 || *Num <= 0) {
        *Num = 0;
        fscanf(f, "\n");
        return;
    }

    *fre  = ivector(1, *Num);
    *coef = (real **)malloc((size_t)(*Num) * sizeof(real *)) - 1;
    *flag = ivector(1, *Num);

    for (i = 1; i <= *Num; i++) {
        double fr = 0.0;
        if (fscanf(f, "%lf", &fr) != 1) fr = 0.0;
        (*fre)[i] = (int)fr;
    }
    fscanf(f, "\n");

    for (i = 1; i <= *Num; i++) {
        (*coef)[i] = vector(0, 2);
        fgets(line, PRE_LINE, f);                 /* "**" */
        if (fscanf(f, "%lf", &(*coef)[i][2]) != 1) (*coef)[i][2] = 0.0;
        if (fscanf(f, "%d\n", &(*flag)[i]) != 1) (*flag)[i] = 0;
    }
}

/* ─── Componente determinista sobre un horizonte EXTENDIDO ──────────────
   Regenera cada variable determinista en t = 1..nobs_ext (son funciones del
   tiempo: armonicos, tendencias, intervenciones fechadas...), le aplica su
   filtro racional Omega(B)/Delta(B) y devuelve la contribucion total.

   Es lo que permite PREVER el nivel: la parte determinista del futuro se
   conoce exactamente, no se prevé.

   BUG-40 (2026-10-16).  El filtro NO es de esta copia: es lib/fuedet, la
   regla de fue (fue.c, cast_us [5]) que comparte toda la suite -- nu(B)
   truncada en Nomega retardos, o en 40 si hay delta, y las variables
   sumadas; quien llama resta la suma UNA vez.  Aqui estaba la recursion
   exacta, y un .pre con un delta proximo a uno y mas de 40 periodos tras el
   suceso dejaba de ser el optimo que fue habia escrito (mmb40.mink.pre:
   +0,47 de logL y otro veredicto del rango).                              */
void build_det_component(struct Tusmodel *Tm, struct Tseries *Ts,
                         int nobs_ext, real *det_out)
{
    struct Tseries Text = *Ts;      /* misma fecha de inicio, mas observaciones */
    real **x;
    int i, t;

    for (t = 1; t <= nobs_ext; t++) det_out[t] = 0.0;
    if (Tm->NdetVar <= 0) return;

    Text.nobs = nobs_ext;

    x = matrix(1, Tm->NdetVar, 1, nobs_ext);
    for (i = 1; i <= Tm->NdetVar; i++)
        gen_detvar(Tm->detspec[i], &Text, x[i]);

    if (fuedet_sum(Tm->NdetVar, Tm->Omega, Tm->Nomega, Tm->Delta, Tm->Ndelta,
                   x, nobs_ext, det_out) != 0)
        fprintf(stderr, "build_det_component: out of memory in fuedet_sum\n");

    free_matrix(x, 1, Tm->NdetVar, 1, nobs_ext);
}

/* ─── read_fue_pre ────────────────────────────────────────────────────── */
int read_fue_pre(const char *filename,
                 struct Tusmodel *Tm, struct Tseries *Ts, real ***DataMat)
{
    FILE *f;
    char  line[PRE_LINE];
    int   i, j;

    /*  Everything the reader hands back starts at zero, so that an error at
     *  any point can release what was reserved without walking garbage
     *  (BUG-39: on an early return free_fue_pre read uninitialised detspec
     *  entries), and so that no field is left for the caller to trust unset. */
    memset(Tm, 0, sizeof *Tm);
    memset(Ts, 0, sizeof *Ts);
    *DataMat = NULL;
    Ts->refactor = 1.0;

    if (NULL == (f = fopen(filename, "r"))) {
        fprintf(stderr, "Error opening %s\n", filename);
        return 1;
    }

#define PRE_FAIL(code) do { fclose(f); free_fue_pre(Tm, Ts, *DataMat); \
                            *DataMat = NULL; return (code); } while (0)

    /*  LA CABECERA ES LIBRE, y por eso aqui NO se cuentan lineas.  El parser
     *  autoritativo (fue/src/fue/inp.py [3.0], FILE_CONTRACT.md 2.0) descarta
     *  lo que venga hasta el separador cuyo texto dice "frequency", y ese es
     *  el unico sitio de todo el formato donde mira lo que un comentario DICE.
     *
     *  Contar cinco lineas, que es lo que hacia esta copia, no lee el FORMATO:
     *  lee un fichero concreto.  El .pre lo escribe FUE (report.py write_pre;
     *  DRVUS no escribe .pre -- el banner que dice DRVUS es texto heredado que
     *  fue copia), y lo escribe con una linea en blanco tras el banner: cinco
     *  renglones.  El .inp que escribe drvec no la trae, y el .inp de DRVUS
     *  trae ademas una linea de especificacion menos.  O sea que la familia de
     *  ficheros que este lector tiene delante NO es de cabecera uniforme, y el
     *  mismo lector acertaba con uno y se desplazaba un renglon con otro.  Y un
     *  renglon de desplazamiento no da error: da un nobs leido de la linea
     *  equivocada, con el que se pide un vector de ese tamano.  Medido: el
     *  .inp de la bateria daba nobs = 1787128427 y el proceso moria por
     *  memoria.  Ver docs/PLAN_BETA.md F2.1.                                 */
    {
        int seen = 0;
        while (fgets(line, PRE_LINE, f))
            if (strstr(line, "requency")) { seen = 1; break; }
        if (!seen) {
            fprintf(stderr, "ERROR: %s no trae el separador de frecuencia;"
                            " no es un fichero del formato fue\n", filename);
            PRE_FAIL(1);
        }
    }

    Tm->residuals = (char *)malloc(PRE_LINE);

    /* ── Frequency ── */
    fgets(line, PRE_LINE, f);
    if (strstr(line, "number") || strstr(line, "Number"))
        { Ts->freq = 1; Ts->numbering = 1; }
    else
        { sscanf(line, "%u", &Ts->freq); Ts->numbering = 0; }

    /* ── nobs, dates, name ── */
    fgets(line, PRE_LINE, f);  /* comment */
    fgets(line, PRE_LINE, f);
    {
        /*  BUG-39: the name was read with an unbounded %s into char[80], and
         *  the count was not checked, so the DRVUS form `62 1850' (two tokens)
         *  left the start year and the name as stack garbage.  The three
         *  numbers are required; the name and the residuals flag are not.  */
        char namef[PRE_LINE]; int outyear = 0, nread;
        namef[0] = '\0'; Tm->residuals[0] = '\0';
        Ts->nobs = 0;             /* si la linea no trae numero, se ve abajo */
        if (Ts->freq > 1)
            nread = sscanf(line, "%d %d %d %4095s %511s",
                           &Ts->nobs, &Ts->begtime, &Ts->begyear,
                           namef, Tm->residuals);
        else {
            nread = sscanf(line, "%d %d %d %4095s %511s",
                           &Ts->nobs, &outyear, &Ts->begyear,
                           namef, Tm->residuals);
            Ts->begtime = 1;
        }
        Ts->name = strdup(namef);
        if (nread < 3) {
            line[strcspn(line, "\r\n")] = '\0';
            fprintf(stderr, "ERROR: %s: the sample line needs `nobs period "
                            "year [name]', and has %d number(s): \"%.60s\"\n",
                    filename, (nread < 0 ? 0 : nread), line);
            PRE_FAIL(1);
        }
    }

    /*  Y si aun asi el numero no tiene sentido, se para AQUI.  Un nobs
     *  disparatado es la firma de una lectura desalineada, y pedir el vector
     *  antes de mirarlo convierte un fichero mal formado en una muerte por
     *  memoria, que no dice nada de lo que pasa.                             */
    if (Ts->nobs <= 0) {
        fprintf(stderr, "ERROR: %s declara %d observaciones\n",
                filename, Ts->nobs);
        PRE_FAIL(1);
    }

    Ts->data = vector(1, Ts->nobs);

    /* ── NdetVar ──  (puerto fiel de fue.c [3.2]: fgets cabecera + fscanf) */
    fgets(line, PRE_LINE, f);                 /* cabecera */
    Tm->NdetVar = 0;
    fscanf(f, "%d\n", &Tm->NdetVar);

    *DataMat = matrix(0, Tm->NdetVar, 1, Ts->nobs);

    if (Tm->NdetVar > 0) {

        /* ── [3.2.0] Nombres de las deterministas y generación de DataMat ──
           fue lee el tipo con fscanf("%s") y a continuación sus argumentos.
           Aquí se lee la línea completa y se delega en gen_detvar.          */
        fgets(line, PRE_LINE, f);             /* "**" */

        /* calloc: an error half-way must leave NULLs, not garbage (BUG-39) */
        Tm->detspec = (char **)calloc((size_t)Tm->NdetVar, sizeof(char *)) - 1;

        for (i = 1; i <= Tm->NdetVar; i++) {
            fgets(line, PRE_LINE, f);
            line[strcspn(line, "\r\n")] = '\0';

            /* Se guarda la especificacion: hace falta para regenerar la
               determinista en fechas FUTURAS al prever.                     */
            Tm->detspec[i] = strdup(line);

            if (!gen_detvar(line, Ts, (*DataMat)[i])) {
                fprintf(stderr,
                        "Error: variable determinista %d (\"%s\") no reconocida.\n"
                        "  Este lector admite: impulse, compimp, step, ramp, easter,\n"
                        "  trend, cos, sin, alter.\n"
                        "  Las variables NO ESTÁNDAR no se admiten por diseño: son\n"
                        "  una versión rudimentaria de un modelo de transferencia con\n"
                        "  input X. Especifica esa relación como transferencia (ω/δ, b),\n"
                        "  que es precisamente lo que drtran estima.\n",
                        i, line);
                PRE_FAIL(2);
            }
        }

        /* ── [3.2.1] Reserva de omegas y deltas ── */
        Tm->Nomega = ivector(1, Tm->NdetVar);
        Tm->Omega  = (real **)malloc((size_t)Tm->NdetVar * sizeof(real *)) - 1;
        Tm->Imega  = (int  **)malloc((size_t)Tm->NdetVar * sizeof(int  *)) - 1;
        Tm->Ndelta = ivector(1, Tm->NdetVar);
        Tm->Delta  = (real **)malloc((size_t)Tm->NdetVar * sizeof(real *)) - 1;
        Tm->Ielta  = (int  **)malloc((size_t)Tm->NdetVar * sizeof(int  *)) - 1;

        /* ── [3.2.2] Omegas: ω(B) de cada determinista ── */
        fgets(line, PRE_LINE, f);             /* cabecera "**" */
        for (i = 1; i <= Tm->NdetVar; i++)
            if (fscanf(f, "%d", &Tm->Nomega[i]) != 1) Tm->Nomega[i] = 0;
        fscanf(f, "\n");

        for (i = 1; i <= Tm->NdetVar; i++) {
            Tm->Omega[i] = vector(0, Tm->Nomega[i]);
            Tm->Imega[i] = ivector(0, Tm->Nomega[i]);
            fgets(line, PRE_LINE, f);         /* "**" */
            for (j = 0; j <= Tm->Nomega[i]; j++) {
                if (fscanf(f, "%lf", &Tm->Omega[i][j]) != 1) Tm->Omega[i][j] = 0.0;
                if (fscanf(f, "%d\n", &Tm->Imega[i][j]) != 1) Tm->Imega[i][j] = 0;
            }
        }

        /* ── [3.2.3] Deltas: δ(B) de cada determinista (solo si Ndelta > 0) ── */
        fgets(line, PRE_LINE, f);             /* cabecera "**" */
        for (i = 1; i <= Tm->NdetVar; i++)
            if (fscanf(f, "%d", &Tm->Ndelta[i]) != 1) Tm->Ndelta[i] = 0;
        fscanf(f, "\n");

        for (i = 1; i <= Tm->NdetVar; i++) {
            if (Tm->Ndelta[i] <= 0) continue;

            Tm->Delta[i] = vector(1, Tm->Ndelta[i]);
            Tm->Ielta[i] = ivector(1, Tm->Ndelta[i]);
            fgets(line, PRE_LINE, f);         /* "**" */
            for (j = 1; j <= Tm->Ndelta[i]; j++) {
                if (fscanf(f, "%lf", &Tm->Delta[i][j]) != 1) Tm->Delta[i][j] = 0.0;
                if (fscanf(f, "%d\n", &Tm->Ielta[i][j]) != 1) Tm->Ielta[i][j] = 0;
            }
        }

    } else {
        /* NdetVar == 0: dummy allocation */
        Tm->Nomega = ivector(1, 1); Tm->Nomega[1] = 0;
        Tm->Omega  = (real **)malloc(sizeof(real *)) - 1;
        Tm->Imega  = (int  **)malloc(sizeof(int  *)) - 1;
        Tm->Ndelta = ivector(1, 1); Tm->Ndelta[1] = 0;
        Tm->Delta  = (real **)malloc(sizeof(real *)) - 1;
        Tm->Ielta  = (int  **)malloc(sizeof(int  *)) - 1;
    }

    /* ── Factores ARMA: AR regular, AR anual, MA regular, MA anual ──
       Antes solo se parseaba el AR regular; los demás se declaraban pero no se
       leían (puntero reservado, sin órdenes ni coeficientes), lo que provocaba
       un segfault en cuanto el modelo tenía MA o estacionalidad.            */
    read_arma_section(f, line, &Tm->NumAr1, &Tm->p1, &Tm->Ar1, &Tm->Ia1);
    read_arma_section(f, line, &Tm->NumAr2, &Tm->p2, &Tm->Ar2, &Tm->Ia2);
    read_arma_section(f, line, &Tm->NumMa1, &Tm->q1, &Tm->Ma1, &Tm->Im1);
    read_arma_section(f, line, &Tm->NumMa2, &Tm->q2, &Tm->Ma2, &Tm->Im2);

    /* ── Factores de frecuencia fija (AR(2)/MA(2) irreducibles) ── */
    read_fixfreq_section(f, line, &Tm->NumAr1f, &Tm->pfre1, &Tm->Ar1f, &Tm->Ia1f);
    read_fixfreq_section(f, line, &Tm->NumMa1f, &Tm->qfre1, &Tm->Ma1f, &Tm->Im1f);

    /* ── mu ──
       FUE escribe "valor flag" si la media se estima (p.ej. "0.154472  1"),
       y un único "0" si la media NO forma parte del modelo. El flag es lo que
       decide si mu es un parámetro libre, no el que su valor sea o no cero.  */
    fgets(line, PRE_LINE, f); fgets(line, PRE_LINE, f);
    if (sscanf(line, "%lf %d", &Tm->mu, &Tm->Imu) < 2) {
        Tm->Imu = 0;   /* media fijada en el valor leído (típicamente 0) */
    }

    /* ── Box-Cox + diffs ── */
    fgets(line, PRE_LINE, f); fgets(line, PRE_LINE, f);
    sscanf(line, "%lf %d %d", &Tm->boxlam, &Tm->nrdiff, &Tm->nadiff);

    /* ── ifadf ──
       SEGUNDO DELTA respecto al original de drtran, y es un ARREGLO.  La
       seccion la escriben SIEMPRE los dos escritores de fue -- fue-1.13.1/
       src/fue.c:3485 y fue/src/fue/report.py:1203 --: con freq > 1 los
       freq/2+1 flags, y con datos ANUALES un " 0" literal.  El original leia
       las dos lineas solo dentro del if, asi que en un fichero anual todo lo
       que viene detras se desplazaba: cbands y refactor salian 0.0 y la SERIE
       NO SE LEIA, quedando a ceros, con read_fue_pre devolviendo exito.
       Comprobado sobre un .pre anual escrito por fue.  drtran no lo vio porque
       sus datos son mensuales; para drvec es el caso normal (el banco empieza
       en 1851).  Ver docs/PLAN_BETA.md F2.1.                                */
    fgets(line, PRE_LINE, f); fgets(line, PRE_LINE, f);
    if (Ts->freq > 1) {
        Tm->ifadf = ivector(0, Ts->freq / 2);
        Tm->sper  = Ts->freq;     /* free_fue_pre keys the release on sper */
        /*  BUG-39: an empty line left `off' uninitialised, and p += off
         *  walked off the buffer.  Every flag has to be there.              */
        { char *p = line; for (i = 0; i <= Ts->freq / 2; i++) {
            int off = 0;
            if (sscanf(p, "%d%n", &Tm->ifadf[i], &off) != 1) {
                fprintf(stderr, "ERROR: %s: the ifadf line needs %d flags "
                                "(frequency %d), and has %d\n",
                        filename, Ts->freq / 2 + 1, Ts->freq, i);
                PRE_FAIL(1);
            }
            p += off; } }
    } else {
        Tm->ifadf = NULL;   /* el original lo dejaba sin inicializar */
    }

    /* ── cbands + refactor ── */
    fgets(line, PRE_LINE, f); fgets(line, PRE_LINE, f);
    if (sscanf(line, "%lf %lf", &Tm->cbands, &Ts->refactor) < 2)
        Ts->refactor = 1.0;       /* a short line cannot leave it undefined */
    if (Ts->refactor == 0.0) Ts->refactor = 1.0;

    /* ── Data section ── */
/* lectura de la serie */
    fgets(line, PRE_LINE, f);  /* "** Time series..." */
    /*  BUG-39: a truncated file used to end in calloc zeros, an `NA' was read
     *  as 0 and a blank line shifted the series -- all with exit 0, and the
     *  zeros then estimated.  Every one of the nobs values has to be a number.
     *  fue.load rejects the same files.                                      */
    for (i = 1; i <= Ts->nobs; i++) {
        if (!fgets(line, PRE_LINE, f) || sscanf(line, "%lf", &Ts->data[i]) != 1) {
            fprintf(stderr, "ERROR: %s: observation %d of %d is %s\n",
                    filename, i, Ts->nobs,
                    feof(f) ? "missing (the file ends before the sample does)"
                            : "not a number");
            PRE_FAIL(1);
        }
    }

    fclose(f);
#undef PRE_FAIL

    /* ── Finalize model ── */
    Tm->sper = Ts->freq;
    for (i = 1; i <= Tm->NumAr1; i++) Tm->Ar1[i][0] = -1.0;
    for (i = 1; i <= Tm->NumAr2; i++) Tm->Ar2[i][0] = -1.0;
    for (i = 1; i <= Tm->NumMa1; i++) Tm->Ma1[i][0] = -1.0;
    for (i = 1; i <= Tm->NumMa2; i++) Tm->Ma2[i][0] = -1.0;
    for (i = 1; i <= Tm->NumAr1f; i++) Tm->Ar1f[i][0] = -1.0;
    for (i = 1; i <= Tm->NumMa1f; i++) Tm->Ma1f[i][0] = -1.0;

    Tm->ornsop = Tm->nrdiff + Tm->sper * Tm->nadiff;
    if (Tm->sper == 12) {
        if (Tm->ifadf[0] == 1) Tm->ornsop += 1;
        if (Tm->ifadf[1] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[2] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[3] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[4] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[5] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[6] == 1) Tm->ornsop += 1;
    } else if (Tm->sper == 4) {
        if (Tm->ifadf[0] == 1) Tm->ornsop += 1;
        if (Tm->ifadf[1] == 1) Tm->ornsop += 2;
        if (Tm->ifadf[2] == 1) Tm->ornsop += 1;
    }

    Tm->rnsop = vector(0, Tm->ornsop);
    CalcNonsOp(Tm->sper, Tm->nrdiff, Tm->nadiff, Tm->ifadf,
               Tm->ornsop, Tm->rnsop);

    return 0;
}

/*****************************************************************************/
/*  fue_bridge.c -- part of drvec (VEC estimation, Mauricio 2006).
 *
 *  Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.  Distributed WITHOUT ANY WARRANTY; see the
 *  GNU General Public License (file COPYING) for details.
 *****************************************************************************/

/*  Puente hacia la suite: lo que hace falta para consumir un .pre de fue y que
 *  el motor de drvec no aporta.  CUATRO funciones, las cuatro copiadas VERBATIM
 *  de drtran (2026-08-17), que es la implementacion en C del mismo puente:
 *  drtran usa el cast de fue y el motor de drvarma, que es exactamente lo que
 *  hace drvec.
 *
 *  PROCEDENCIA
 *
 *    ObsToDate          drtran/src/diagnose.c:128
 *    DateToObs          drtran/src/drtran.c:838
 *    expand_ar_factors  drtran/src/drtran.c:712
 *    expand_ma_factors  drtran/src/drtran.c:779
 *
 *  Unico cambio: DRTRAN_PI -> M_PI (que main.h ya define).
 *
 *  Las dos de fecha ya estaban DECLARADAS en include/main.h de drvec pero
 *  ningun .c las definia; son la unica dependencia de fue_pre_reader.c que el
 *  motor no cubria.  Solo las usa gen_detvar() para deterministas FECHADAS
 *  (impulse, step, ramp, easter): los .inp que drvec escribe llevan cero
 *  deterministas, asi que hoy la ruta no se ejercita.  Se traen completas y no
 *  como stub porque un stub daria fechas equivocadas EN SILENCIO si alguien
 *  alimenta un .pre que si las lleve.
 *
 *  Las dos de expansion son el espejo en C del _unscramble del cast de fue:
 *  convierten los operadores FACTORIZADOS del .pre en los polinomios planos
 *  Phi(B) = 1 - phi_1 B - ... y Theta(B) = 1 - theta_1 B - ..., que es la
 *  convencion con la que drvec imprime su modelo y la que espera elf().  El
 *  .pre guarda factores, no polinomios: "2 1 1" son DOS factores de primer
 *  orden y "1 2" es UNO de segundo, y la diferencia cambia el modelo
 *  (FILE_CONTRACT.md 2.3).  Expandir aqui, y no reimplementar la convolucion,
 *  es lo que garantiza que drvec lea el mismo modelo que fue estimo.
 *
 *  Ver docs/PLAN_BETA.md F2.1 (procedencia) y F2.3 (el cast).
 */

#include "main.h"

/*---------------------------------------------------------------------------*/
/*  ObsToDate lived here as a copy of drtran/src/diagnose.c:128, because that
 *  file was not vendored and only this one function was needed.  Since
 *  2026-08-24 the whole of diagnose.c IS vendored (src/diagnose.c, from
 *  drvarma v.04.1), so the copy is gone and the definition is the owner's.
 *  Two identical definitions in one binary is not a style question: the linker
 *  refuses it, which is the one kind of duplication that cannot rot quietly.
 */

/*---------------------------------------------------------------------------*/
/*  DateToObs: convierte (periodo, subperiodo) a numero de observacion       */
/*---------------------------------------------------------------------------*/
void DateToObs(int beg_per, int beg_sub, int per, int sub, int freq,
               int *obs_no)
{
    int srest, pcad, sad;

    srest = freq - beg_sub + 1;
    if (sub == freq) {
        pcad = per - beg_per;
        *obs_no = srest + freq * pcad;
    } else {
        pcad = per - beg_per - 1;
        sad  = sub;
        *obs_no = srest + freq * pcad + sad;
    }
}

/* Expande todos los factores AR en un único polinomio Φ(B)=1-φ₁B-φ₂B²-...
   phi_out[1..p] recibe los coeficientes (phi_out[0] = 1 implícito).
   Se asume que phi_out está pre-dimensionado con vector(1, p).            */
void expand_ar_factors(struct Tusmodel *Tm, real *phi_out, int p)
{
    int i, j, k;
    real *work;

    if (p == 0) return;

    work = vector(0, p);

    /* Inicializar: polinomio identidad 1 (coefs en work[1..p] = 0) */
    work[0] = 1.0;
    for (k = 1; k <= p; k++) work[k] = 0.0;

    /* Factores AR regulares: P_i(B) = -1 + a₁B + a₂B² + ...
       Convertido a VARMA:  -P_i(B) = 1 - a₁B - a₂B² - ...            */
    for (i = 1; i <= Tm->NumAr1; i++) {
        int ord_i = Tm->p1[i];
        for (k = p; k >= 0; k--) {
            real acc = work[k];
            for (j = 1; j <= ord_i && j <= k; j++)
                acc -= Tm->Ar1[i][j] * work[k - j];
            work[k] = acc;
        }
    }

    /* Factores AR ANUALES: 1 - a₁B^s - a₂B^2s - … (retardos múltiplos de sper) */
    for (i = 1; i <= Tm->NumAr2; i++) {
        int ord_i = Tm->p2[i];
        for (k = p; k >= 0; k--) {
            real acc = work[k];
            for (j = 1; j <= ord_i; j++) {
                int lag = j * Tm->sper;
                if (lag <= k) acc -= Tm->Ar2[i][j] * work[k - lag];
            }
            work[k] = acc;
        }
    }

    /* Factores AR de FRECUENCIA FIJA (irreducibles). fue los parametriza con un
       ÚNICO coeficiente libre c₂ (<0); el término en B se DERIVA de él y de la
       frecuencia (fue.c:4064):
           c₁ = 2·cos(2πf/s)·sqrt(−c₂)     →   1 − c₁B − c₂B²
       que es (1 − 2r·cos(ω)B + r²B²) con r = sqrt(−c₂).                     */
    for (i = 1; i <= Tm->NumAr1f; i++) {
        real c2 = Tm->Ar1f[i][2];
        real r  = (c2 < 0.0) ? sqrt(-c2) : 0.0;
        real c1 = 2.0 * cos(2.0 * M_PI * Tm->pfre1[i] / Tm->sper) * r;

        Tm->Ar1f[i][1] = c1;   /* fue guarda el término en B junto al factor */

        for (k = p; k >= 0; k--) {
            real acc = work[k];
            if (k >= 1) acc -= c1 * work[k - 1];
            if (k >= 2) acc -= c2 * work[k - 2];
            work[k] = acc;
        }
    }

    /* Copiar a phi_out[1..p] (negar porque work almacena P(B),
       pero VARMA usa Φ(B)=1-φ₁B-φ₂B²-... = 1 - work[1]B - work[2]B²-...) */
    for (k = 1; k <= p; k++) phi_out[k] = -work[k];

    free_vector(work, 0, p);
}

/* Expande todos los factores MA en un único polinomio Θ(B)=1-θ₁B-θ₂B²-...
   Misma lógica que expand_ar_factors.                                   */
void expand_ma_factors(struct Tusmodel *Tm, real *theta_out, int q)
{
    int i, j, k;
    real *work;

    if (q == 0) return;

    work = vector(0, q);
    work[0] = 1.0;
    for (k = 1; k <= q; k++) work[k] = 0.0;

    for (i = 1; i <= Tm->NumMa1; i++) {
        int ord_i = Tm->q1[i];
        for (k = q; k >= 0; k--) {
            real acc = work[k];
            for (j = 1; j <= ord_i && j <= k; j++)
                acc -= Tm->Ma1[i][j] * work[k - j];
            work[k] = acc;
        }
    }

    /* Factores MA ANUALES: 1 - θ₁B^s - θ₂B^2s - … (retardos múltiplos de sper) */
    for (i = 1; i <= Tm->NumMa2; i++) {
        int ord_i = Tm->q2[i];
        for (k = q; k >= 0; k--) {
            real acc = work[k];
            for (j = 1; j <= ord_i; j++) {
                int lag = j * Tm->sper;
                if (lag <= k) acc -= Tm->Ma2[i][j] * work[k - lag];
            }
            work[k] = acc;
        }
    }

    /* Factores MA de FRECUENCIA FIJA: igual que los AR (fue.c:4202) */
    for (i = 1; i <= Tm->NumMa1f; i++) {
        real c2 = Tm->Ma1f[i][2];
        real r  = (c2 < 0.0) ? sqrt(-c2) : 0.0;
        real c1 = 2.0 * cos(2.0 * M_PI * Tm->qfre1[i] / Tm->sper) * r;

        Tm->Ma1f[i][1] = c1;

        for (k = q; k >= 0; k--) {
            real acc = work[k];
            if (k >= 1) acc -= c1 * work[k - 1];
            if (k >= 2) acc -= c2 * work[k - 2];
            work[k] = acc;
        }
    }

    /* Copiar a theta_out[1..q] (misma convención de signos que AR) */
    for (k = 1; k <= q; k++) theta_out[k] = -work[k];

    free_vector(work, 0, q);
}

/*---------------------------------------------------------------------------*/
/*  free_fue_pre: suelta TODO lo que read_fue_pre reserva.                    */
/*---------------------------------------------------------------------------*/
/*  ESTO NO VIENE DE drtran: alli no existe.  read_fue_pre reserva y nadie
 *  libera -- ni en drtran ni en ningun otro consumidor --, que es una fuga por
 *  fichero leido.  Declarado como BUG-12 en drtran-python/docs/BUGS.md.
 *
 *  Va aqui y no dentro de fue_pre_reader.c a proposito, para que la copia del
 *  lector siga teniendo su delta contado respecto al original: lo que se anade
 *  es codigo nuevo, no una modificacion de codigo prestado.
 *
 *  DOS COSAS QUE HAY QUE RESPETAR, y equivocarse en ellas es peor que la fuga:
 *
 *   - los vectores 1-based del estilo Numerical Recipes se liberan con
 *     free_vector/free_ivector y SUS MISMOS LIMITES, no con free();
 *   - los arrays de punteros usan el idioma `malloc(n*sizeof(p)) - 1`, asi que
 *     el puntero que hay que devolver a free() es `p + 1`.
 *
 *  Verificado con valgrind: las rutas de siembra pasan a 0 bytes definitely
 *  lost y 0 errores, que es la unica forma de comprobar un desasignador escrito
 *  a mano contra un asignador ajeno.                                          */
/*  El lector guarda los arrays de punteros como `malloc(n*size) - 1`, asi que
 *  el bloque que hay que devolver a free() empieza una posicion mas alla.  Se
 *  pasa por esta funcion, y no se escribe `free(p + 1)` a la vista, porque gcc
 *  no ve la resta que hizo el lector y avisa de free-nonheap-object.  Aqui la
 *  aritmetica esta aislada y explicada en un solo sitio.                     */
static void free_base1( void *base1 )
{
    void **p = (void **) base1;
    if ( p != NULL ) free( (void *) &p[1] );
}

void free_fue_pre( struct Tusmodel *Tm, struct Tseries *Ts, real **DataMat )
{
    int i;

    if ( Ts != NULL )
        {
        if ( Ts->name ) free( Ts->name );
        if ( Ts->data ) free_vector( Ts->data, 1, Ts->nobs );
        Ts->name = NULL; Ts->data = NULL;
        }
    if ( Tm == NULL ) return;

    if ( DataMat && Ts ) free_matrix( DataMat, 0, Tm->NdetVar, 1, Ts->nobs );

    /* ── deterministas ── */
    if ( Tm->NdetVar > 0 )
        {
        for ( i = 1; i <= Tm->NdetVar; i++ )
            {
            if ( Tm->detspec && Tm->detspec[i] ) free( Tm->detspec[i] );
            if ( Tm->Omega && Tm->Omega[i] )
                free_vector( Tm->Omega[i], 0, Tm->Nomega[i] );
            if ( Tm->Imega && Tm->Imega[i] )
                free_ivector( Tm->Imega[i], 0, Tm->Nomega[i] );
            if ( Tm->Ndelta && Tm->Ndelta[i] > 0 )
                {
                if ( Tm->Delta && Tm->Delta[i] )
                    free_vector( Tm->Delta[i], 1, Tm->Ndelta[i] );
                if ( Tm->Ielta && Tm->Ielta[i] )
                    free_ivector( Tm->Ielta[i], 1, Tm->Ndelta[i] );
                }
            }
        if ( Tm->detspec ) free_base1( Tm->detspec );
        if ( Tm->Nomega )  free_ivector( Tm->Nomega, 1, Tm->NdetVar );
        if ( Tm->Ndelta )  free_ivector( Tm->Ndelta, 1, Tm->NdetVar );
        }
    else
        {
        /* la reserva ficticia del caso sin deterministas */
        if ( Tm->Nomega ) free_ivector( Tm->Nomega, 1, 1 );
        if ( Tm->Ndelta ) free_ivector( Tm->Ndelta, 1, 1 );
        }
    if ( Tm->Omega ) free_base1( Tm->Omega );
    if ( Tm->Imega ) free_base1( Tm->Imega );
    if ( Tm->Delta ) free_base1( Tm->Delta );
    if ( Tm->Ielta ) free_base1( Tm->Ielta );
    Tm->detspec = NULL; Tm->Nomega = NULL; Tm->Ndelta = NULL;
    Tm->Omega = NULL; Tm->Imega = NULL; Tm->Delta = NULL; Tm->Ielta = NULL;

    /* ── factores ARMA: los cuatro bloques tienen la misma forma ── */
    {
    int   nums[4];
    int  *ords[4];
    real **cfs[4];
    int  **fls[4];
    int b;

    nums[0] = Tm->NumAr1; ords[0] = Tm->p1; cfs[0] = Tm->Ar1; fls[0] = Tm->Ia1;
    nums[1] = Tm->NumAr2; ords[1] = Tm->p2; cfs[1] = Tm->Ar2; fls[1] = Tm->Ia2;
    nums[2] = Tm->NumMa1; ords[2] = Tm->q1; cfs[2] = Tm->Ma1; fls[2] = Tm->Im1;
    nums[3] = Tm->NumMa2; ords[3] = Tm->q2; cfs[3] = Tm->Ma2; fls[3] = Tm->Im2;

    for ( b = 0; b < 4; b++ )
        {
        if ( nums[b] <= 0 ) continue;
        for ( i = 1; i <= nums[b]; i++ )
            {
            if ( cfs[b] && cfs[b][i] ) free_vector( cfs[b][i], 0, ords[b][i] );
            if ( fls[b] && fls[b][i] ) free_ivector( fls[b][i], 0, ords[b][i] );
            }
        if ( ords[b] ) free_ivector( ords[b], 1, nums[b] );
        if ( cfs[b] )  free_base1( cfs[b] );
        if ( fls[b] )  free_base1( fls[b] );
        }
    }
    Tm->p1 = Tm->p2 = Tm->q1 = Tm->q2 = NULL;
    Tm->Ar1 = Tm->Ar2 = Tm->Ma1 = Tm->Ma2 = NULL;
    Tm->Ia1 = Tm->Ia2 = Tm->Im1 = Tm->Im2 = NULL;
    Tm->NumAr1 = Tm->NumAr2 = Tm->NumMa1 = Tm->NumMa2 = 0;

    /* ── factores de frecuencia fija: coef es (0..2) por factor ── */
    if ( Tm->NumAr1f > 0 )
        {
        for ( i = 1; i <= Tm->NumAr1f; i++ )
            if ( Tm->Ar1f && Tm->Ar1f[i] ) free_vector( Tm->Ar1f[i], 0, 2 );
        if ( Tm->pfre1 ) free_ivector( Tm->pfre1, 1, Tm->NumAr1f );
        if ( Tm->Ia1f )  free_ivector( Tm->Ia1f,  1, Tm->NumAr1f );
        if ( Tm->Ar1f )  free_base1( Tm->Ar1f );
        }
    if ( Tm->NumMa1f > 0 )
        {
        for ( i = 1; i <= Tm->NumMa1f; i++ )
            if ( Tm->Ma1f && Tm->Ma1f[i] ) free_vector( Tm->Ma1f[i], 0, 2 );
        if ( Tm->qfre1 ) free_ivector( Tm->qfre1, 1, Tm->NumMa1f );
        if ( Tm->Im1f )  free_ivector( Tm->Im1f,  1, Tm->NumMa1f );
        if ( Tm->Ma1f )  free_base1( Tm->Ma1f );
        }
    Tm->Ar1f = Tm->Ma1f = NULL; Tm->pfre1 = Tm->qfre1 = NULL;
    Tm->Ia1f = Tm->Im1f = NULL;
    Tm->NumAr1f = Tm->NumMa1f = 0;

    /* ── resto ── */
    if ( Tm->ifadf && Tm->sper > 1 ) free_ivector( Tm->ifadf, 0, Tm->sper / 2 );
    if ( Tm->rnsop ) free_vector( Tm->rnsop, 0, Tm->ornsop );
    if ( Tm->residuals ) free( Tm->residuals );
    Tm->ifadf = NULL; Tm->rnsop = NULL; Tm->residuals = NULL;
}

/*
 * fuedet.h -- la componente determinista de un modelo de fue, con la regla
 * de fue.c [5] (cast_us, paso 5), la de Mauricio.
 *
 *     ruido_t = dato_t - SUM_j nu_j(B) x_{j,t}
 *
 *   - nu_j(B) = omega_j(B) / delta_j(B), con los signos de Box-Jenkins
 *     (omega_j(B) = w0 - w1 B - ..., delta_j(B) = 1 - d1 B - ...), sacada por
 *     calcnu;
 *   - nu_j se TRUNCA en Nomega_j retardos si la variable no tiene delta, y en
 *     FUEDET_NULAG_DELTA (40) si lo tiene;
 *   - las variables se SUMAN (tmp2 por variable, tmp1 += tmp2) y la suma se
 *     resta UNA vez.
 *
 * Es la unica implementacion del filtro determinista de la suite. La usan
 * fue, fuf, lib/fuepre (build_det_component), lib/prewhiten
 * (apply_univariate_model) y drvec; y el paquete de Python de fue lleva una
 * copia byte a byte de estos dos ficheros (csrc/fuedet.[ch]), vigilada por
 * su sha256. Ver lib/fuedet/README.md.
 *
 * C puro: no depende del anfitrion (ni de su `real`, ni de sus asignadores).
 * Los indices son los de fue (Numerical Recipes): variables j = 1..nvar,
 * tiempo t = 1..n.
 *
 * Copyright (C) 1995-2026 J.A. Mauricio, A.B. Treadway & D.E. Guerrero.
 * License: GPL v2 or later.
 */

#ifndef FUEDET_H
#define FUEDET_H

#ifdef __cplusplus
extern "C" {
#endif

/* Retardos de nu cuando la variable tiene delta (fue.c [5]). */
#define FUEDET_NULAG_DELTA 40

/* El retardo maximo de nu_j: nomega si ndelta == 0, si no 40. */
int fuedet_nulag( int nomega, int ndelta );

/* calcnu de fue.c, tal cual: nu[0..lags] de omega(B)/delta(B).
     omega[0..s], delta[1..r] (delta[0] no se lee).                         */
void fuedet_nu( double *omega, int s, double *delta, int r,
                double *nu, int lags );

/* La componente determinista: det[t] = SUM_j nu_j(B) x[j][t], t = 1..n.
     omega[j][0..nomega[j]], delta[j][1..ndelta[j]], x[j][1..n], j = 1..nvar.
   Con nvar <= 0, det = 0. Devuelve 0, o -1 si falta memoria.               */
int fuedet_sum( int nvar, double **omega, int *nomega,
                double **delta, int *ndelta,
                double **x, int n, double *det );

/* El ruido: out[t] = data[t] - SUM_j nu_j(B) x[j][t], t = 1..n, con la
   suma restada una vez (fue.c [5]: vtmp1 = DataMat[0] - tmp1). Con
   nvar <= 0, out = data. Devuelve 0, o -1 si falta memoria.                */
int fuedet_noise( int nvar, double **omega, int *nomega,
                  double **delta, int *ndelta,
                  double **x, const double *data, int n, double *out );

#ifdef __cplusplus
}
#endif

#endif /* FUEDET_H */

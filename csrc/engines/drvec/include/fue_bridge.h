/*****************************************************************************/
/*  fue_bridge.h -- part of drvec (VEC estimation, Mauricio 2006).
 *
 *  Copyright (C) 1995-2026 A.B. Treadway, J.A. Mauricio & D.E. Guerrero.
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation; either version 2 of the License, or (at your
 *  option) any later version.  Distributed WITHOUT ANY WARRANTY; see the
 *  GNU General Public License (file COPYING) for details.
 *****************************************************************************/

#ifndef FUE_BRIDGE_H
#define FUE_BRIDGE_H

#include "main.h"

/* Expanden los operadores FACTORIZADOS de un .pre a los polinomios planos
   Phi(B) = 1 - phi_1 B - ... y Theta(B) = 1 - theta_1 B - ...
   phi_out/theta_out se esperan dimensionados con vector(1, p) / vector(1, q).
   Copiadas de drtran; ver la cabecera de src/fue_bridge.c.                  */
void expand_ar_factors( struct Tusmodel *Tm, real *phi_out,   int p );
void expand_ma_factors( struct Tusmodel *Tm, real *theta_out, int q );

/* Suelta todo lo que read_fue_pre reserva.  No existe en drtran -- alli nadie
   libera, que es su BUG-12 --, asi que es codigo nuevo y por eso vive aqui y no
   dentro de la copia del lector.                                            */
void free_fue_pre( struct Tusmodel *Tm, struct Tseries *Ts, real **DataMat );

#endif

"""The bootstrap's generator (drvec.c:2203-2223, `boot_normal`).

A 64-bit LCG with the PCG constants and Box-Muller with a cached spare:

    state <- state * 6364136223846793005 + 1442695040888963407   (mod 2^64)
    u      = ((state >> 33) & 0x7FFFFFFF) / 2^31
    u1 drawn again while u1 <= 1e-12; u2 = the next u;
    returns sqrt(-2 log u1) cos(2 pi u2), and caches sqrt(-2 log u1) sin(2 pi u2)

seeded with 987654321. In the C the state and the spare are STATIC: they are
never reset, so they run on across replications and across rank pairs; here one
`Rng` object is threaded through a whole rank test for the same effect.

This is the stream of the oracle (Linux, `unsigned long` of 64 bits). On
Windows `unsigned long` has 32 bits and the C's own stream is a different one
(TODO.md: a C defect candidate); the port computes the 64-bit stream everywhere.
"""
from __future__ import annotations

import math

_MASK = (1 << 64) - 1
_A = 6364136223846793005
_C = 1442695040888963407
SEED = 987654321


class Rng:
    def __init__(self, seed=SEED):
        self.state = int(seed) & _MASK
        self.have = False
        self.spare = 0.0

    def uniform(self):
        self.state = (self.state * _A + _C) & _MASK
        return float((self.state >> 33) & 0x7FFFFFFF) / 2147483648.0

    def normal(self):
        if self.have:
            self.have = False
            return self.spare
        while True:
            u1 = self.uniform()
            if not (u1 <= 1.0e-12):
                break
        u2 = self.uniform()
        rad = math.sqrt(-2.0 * math.log(u1))
        th = 2.0 * math.pi * u2
        self.spare = rad * math.sin(th)
        self.have = True
        return rad * math.cos(th)

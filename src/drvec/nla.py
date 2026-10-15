"""The small linear algebra of drvec's C, transcribed operation for operation.

Why not numpy: the oracle is reproduced BIT FOR BIT (docs/PORT.md §1), and the
optimiser's trajectory amplifies a last-bit difference into a different point
on a flat likelihood (on case 1 the four starts end between -1.50 and 6.48).
So the regressions are solved by the C's normal equations with its LU
(`nlatools.c:52 ludcp`, `:91 lusol`), the products follow `matrix_multiply`'s
i-j-k loops (`nlatools.c:1068`), and the Cholesky is the MODIFIED one
(`nlatools.c:124 choldcp`, Gill-Murray-Wright), not a strict one.

Every matrix here is a list of lists, 0-based; the loops run in the C's order.
"""
from __future__ import annotations

import math

MACHEPS = 2.0 ** -52          # cmacheps() (nlatools.c:493) on IEEE doubles


def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def eye(n):
    a = zeros(n, n)
    for i in range(n):
        a[i][i] = 1.0
    return a


def mm(A, B):
    """matrix_multiply: C = A B, sum over k in increasing order."""
    n, m, p = len(A), len(B), len(B[0]) if B else 0
    C = zeros(n, p)
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += Ai[k] * B[k][j]
            Ci[j] = s
    return C


def ludcp(a, n):
    """LU with partial pivoting, in place on `a` (n x n, 0-based). Returns ip
    (0-based pivot rows; ip[n-1] carries the sign, 0 if singular, as the C)."""
    ip = [0] * n
    ip[n - 1] = 1
    for k in range(n):
        if k != n - 1:
            kp1 = k + 1
            m = k
            for i in range(kp1, n):
                if abs(a[i][k]) > abs(a[m][k]):
                    m = i
            ip[k] = m
            if m != k:
                ip[n - 1] = -ip[n - 1]
            tmp = a[m][k]
            a[m][k] = a[k][k]
            a[k][k] = tmp
            if tmp != 0.0:
                for i in range(kp1, n):
                    a[i][k] /= -tmp
                for j in range(kp1, n):
                    tmp = a[m][j]
                    a[m][j] = a[k][j]
                    a[k][j] = tmp
                    if tmp != 0.0:
                        for i in range(kp1, n):
                            a[i][j] += a[i][k] * tmp
        if a[k][k] == 0.0:
            ip[n - 1] = 0
    return ip


def lusol(a, b, n, ip):
    """Solve with ludcp's factors, in place on b (0-based)."""
    if n != 1:
        for k in range(n - 1):
            m = ip[k]
            tmp = b[m]
            b[m] = b[k]
            b[k] = tmp
            for i in range(k + 1, n):
                b[i] += a[i][k] * tmp
        for k1 in range(1, n):
            k = n - k1                     # 0-based index of the C's km1 + 1
            b[k] /= a[k][k]
            tmp = -b[k]
            for i in range(k):
                b[i] += a[i][k] * tmp
    b[0] /= a[0][0]


def choldcp(S):
    """nlatools.c:choldcp. Returns (L, ifault): L the lower factor (upper
    triangle zeroed), ifault 1 when a negative pivot exceeds sqrt(eps)*maxoffl.
    It reads the UPPER triangle for the off-diagonal entries, as the C."""
    n = len(S)
    mat = [row[:] for row in S]
    sqrteps = math.sqrt(MACHEPS)
    maxoffl = 0.0
    for j in range(n):
        v = math.sqrt(abs(mat[j][j]))
        if v > maxoffl:
            maxoffl = v
    if maxoffl * maxoffl <= sqrteps:
        return zeros(n, n), 0
    minl = 0.0
    minl2 = sqrteps * maxoffl
    for j in range(n):
        s1 = mat[j][j]
        for i in range(j):
            s1 -= mat[j][i] * mat[j][i]
        if s1 != abs(s1) and abs(s1) > minl2:
            return None, 1
        mat[j][j] = s1
        minljj = 0.0
        for i in range(j + 1, n):
            s1 = mat[j][i]
            for k in range(j):
                s1 -= mat[i][k] * mat[j][k]
            mat[i][j] = s1
            if abs(s1) > minljj:
                minljj = abs(s1)
        minljj = minljj / maxoffl if (minljj / maxoffl) > minl else minl
        if mat[j][j] > minljj * minljj:
            mat[j][j] = math.sqrt(mat[j][j])
        else:
            if minljj < minl2:
                minljj = minl2
            mat[j][j] = minljj
        for i in range(j + 1, n):
            mat[i][j] /= mat[j][j]
    for j in range(1, n):
        for i in range(j):
            mat[i][j] = 0.0
    return mat, 0


def chol_ok(S):
    """Only choldcp's verdict: True when S passes."""
    return choldcp(S)[1] == 0


def choldcp_det(S):
    """choldcp with its determinant (nlatools.c:124-210): -> (L, d1, d2, ifault),
    det = d1 * 2^d2, d1 kept in [1/16, 1) as the C keeps it."""
    n = len(S)
    mat = [row[:] for row in S]
    d1, d2 = 1.0, 0.0
    sqrteps = math.sqrt(MACHEPS)
    maxoffl = 0.0
    for j in range(n):
        v = math.sqrt(abs(mat[j][j]))
        if v > maxoffl:
            maxoffl = v
    if maxoffl * maxoffl <= sqrteps:
        return zeros(n, n), d1, d2, 0
    minl = 0.0
    minl2 = sqrteps * maxoffl
    for j in range(n):
        s1 = mat[j][j]
        for i in range(j):
            s1 -= mat[j][i] * mat[j][i]
        if s1 != abs(s1) and abs(s1) > minl2:
            return None, d1, d2, 1
        mat[j][j] = s1
        minljj = 0.0
        for i in range(j + 1, n):
            s1 = mat[j][i]
            for k in range(j):
                s1 -= mat[i][k] * mat[j][k]
            mat[i][j] = s1
            if abs(s1) > minljj:
                minljj = abs(s1)
        minljj = minljj / maxoffl if (minljj / maxoffl) > minl else minl
        if mat[j][j] > minljj * minljj:
            mat[j][j] = math.sqrt(mat[j][j])
        else:
            if minljj < minl2:
                minljj = minl2
            mat[j][j] = minljj
        d1 *= mat[j][j] * mat[j][j]
        while abs(d1) >= 1.0:
            d1 *= 0.0625
            d2 += 4.0
        while abs(d1) < 0.0625:
            d1 *= 16.0
            d2 -= 4.0
        for i in range(j + 1, n):
            mat[i][j] /= mat[j][j]
    for j in range(1, n):
        for i in range(j):
            mat[i][j] = 0.0
    return mat, d1, d2, 0

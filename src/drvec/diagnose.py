"""The per-series residual diagnosis of the .out: diagnose.c (drvec-v0.12,
engines/drvec/src/diagnose.c, vendored from drvarma), transcribed line for line
-- the statistics, the ASCII plot, the histogram, the ACF and PACF with their
Ljung-Box, and the cross-correlation functions. Everything is printed, so the
arithmetic keeps the C's order (sums in increasing t, pow() for the moments)."""
from __future__ import annotations

import math

GUIONS = "-------------+-------------------------+-------------------------+--------------"


def round_local(num):
    """nlatools.c:round_local: half away from zero."""
    t1 = abs(num)
    t2 = math.floor(t1)
    if t1 - t2 >= 0.5:
        t2 = math.ceil(t1)
    i = int(t2)
    return i if num >= 0.0 else -i


def obs_to_date(beg_per, beg_sub, obs_no, freq):
    """diagnose.c:ObsToDate -> (per, sub)."""
    if obs_no + beg_sub - 1 <= freq:
        return beg_per, beg_sub + obs_no - 1
    q, rem = divmod(obs_no - (freq - beg_sub + 1), freq)
    if rem > 0:
        return beg_per + q + 1, rem
    return beg_per + q, freq


def mean(d):
    s = 0.0
    for v in d:
        s += v
    return s / len(d)


def stdev(d):
    ave = mean(d)
    s = 0.0
    for v in d:
        s += (v - ave) * (v - ave)
    return math.sqrt(s / len(d))


def skew(d):
    ave, sd = mean(d), stdev(d)
    if sd < 1e-20:
        return 0.0
    s = 0.0
    for v in d:
        s += math.pow((v - ave) / sd, 3.0)
    return s / len(d)


def kurt(d):
    ave, sd = mean(d), stdev(d)
    if sd < 1e-20:
        return 0.0
    s = 0.0
    for v in d:
        s += math.pow((v - ave) / sd, 4.0)
    return s / len(d) - 3.0


def maxval(d):
    im, mx = 0, d[0]
    for i in range(1, len(d)):
        if d[i] > mx:
            mx, im = d[i], i
    return im


def minval(d):
    im, mn = 0, d[0]
    for i in range(1, len(d)):
        if d[i] < mn:
            mn, im = d[i], i
    return im


def acf(d, lags, mu, var):
    n = len(d)
    out = [0.0] * (lags + 1)
    for j in range(1, lags + 1):
        c = 0.0
        for i in range(n - j):
            c += (d[i] - mu) * (d[i + j] - mu)
        out[j] = c / (n * var)
    return out


def pacf(corr, lags):
    Mt = [[0.0] * (lags + 1) for _ in range(lags + 1)]
    Mt[1][1] = corr[1]
    for i in range(2, lags + 1):
        s1 = s2 = 0.0
        for j in range(1, i):
            s1 += Mt[i - 1][j] * corr[i - j]
            s2 += Mt[i - 1][j] * corr[j]
        Mt[i][i] = (corr[i] - s1) / (1.0 - s2)
        for j in range(1, i):
            Mt[i][j] = Mt[i - 1][j] - Mt[i][i] * Mt[i - 1][i - j]
    return [0.0] + [Mt[i][i] for i in range(1, lags + 1)]


def chitest(corr, lags, n):
    c = 0.0
    for i in range(1, lags + 1):
        c += corr[i] * corr[i] / (n - i)
    return c * (n * (n + 2))


def chitest_c(corr, lags, n):
    c = 0.0
    for i in range(1, lags + 1):
        c += corr[i] * corr[i] / (n - i + 1)
    return c * (n * (n + 2))


def plot_cor(corr, lags, isacf, n, freq, npar):
    marcas = ("            -1                         0                         1  L-B Q  DF"
              if isacf else "            -1                         0                         1")
    o = "Autocorrelation function (acf " if isacf else "Partial autocorrelation function (pacf "
    o += "bands =  %5.3f):\n\n%s\n%s\n" % (2.0 / math.sqrt(n), marcas, GUIONS)
    for i in range(1, lags + 1):
        t = [" "] * 53
        if freq != 1 and i % freq == 0:
            o += "%4d %7.3f +" % (i, corr[i])
            sym = "+"
            t[51] = "+"
        else:
            o += "%4d %7.3f |" % (i, corr[i])
            sym = "*"
            t[51] = "|"
        pos = corr[i] * 25.0
        posi = abs(round_local(pos))
        if pos <= 0.0:
            for j in range(25 - posi, 26):
                t[j] = sym
        else:
            for j in range(25, 26 + posi):
                t[j] = sym
        t[25] = "|"
        posi = round_local(2.0 / math.sqrt(n) * 25.0)
        if t[25 + posi] == " ":
            t[25 + posi] = ":"
        if t[25 - posi] == " ":
            t[25 - posi] = ":"
        o += "".join(t)
        lb = "%6.2f %3d" % (chitest(corr, i, n), i - npar)
        if freq != 1 and i % freq == 0 and isacf and i - npar >= 1:
            o += lb
        if i % freq != 0 and isacf and i == lags and i - npar >= 1:
            o += lb
        if freq == 1 and isacf and i == lags and i - npar >= 1:
            o += lb
        o += "\n"
    return o + "%s\n%s\n\n" % (GUIONS, marcas)


def plot_ccf(corr, lags, n):
    marcas = "            -1                         0                         1"
    o = "CCF BANDS  2.0/SQRT(N) =  %2.5f:\n" % (2.0 / math.sqrt(n))
    o += "\n%s\n%s\n" % (marcas, GUIONS)
    for i in range(1, 2 * lags + 2):
        t = [" "] * 80
        o += "%4d %7.3f |" % (i - lags - 1, corr[i])
        t[51] = "|"
        pos = corr[i] * 25.0
        posi = int(pos + (0.5 if pos >= 0 else -0.5))
        if pos <= 0.0:
            for j in range(25 - abs(posi), 26):
                t[j] = "*"
        else:
            for j in range(25, 26 + posi):
                t[j] = "*"
        t[25] = "|"
        posi = int(2.0 / math.sqrt(n) * 25.0 + 0.5)
        if t[25 + posi] == " ":
            t[25 + posi] = ":"
        if t[25 - posi] == " ":
            t[25 - posi] = ":"
        o += "".join(t) + "\n"
    return o + "%s\n%s\n" % (GUIONS, marcas)


def ccf(d1, d2, lags, m1, m2, sd1, sd2):
    n = len(d1)
    out = [0.0] * (lags + 2)
    for j in range(1, lags + 2):
        c = 0.0
        for i in range(n - j + 1):
            c += (d1[i] - m1) * (d2[i + j - 1] - m2)
        out[j] = c / (n * sd1 * sd2)
    return out


def stat_ser(d, freq, by, bt):
    n = len(d)
    ey, et = obs_to_date(by, bt, n, freq)
    mu = mean(d)
    tmp = stdev(d)
    sk, ku = skew(d), kurt(d)
    imax, imin = maxval(d), minval(d)
    o = "Unconditional residuals (seasonal period: %d)\n" % freq
    o += "%d observations: " % n
    o += ("from %d/%d to %d/%d\n" % (bt, by, et, ey)) if freq > 1 else ("from %d to %d\n" % (by, ey))
    o += "\n"
    o += "                  Mean: %18.6f\n" % mu
    o += "Standard error of mean: %18.6f\n" % (tmp / math.sqrt(n))
    o += "              Variance: %18.6f\n" % (tmp * tmp)
    o += "    Standard deviation: %18.6f\n" % tmp
    o += "              Skewness: %18.6f\n" % sk
    o += "              Kurtosis: %18.6f\n" % ku
    My, Mt = obs_to_date(by, bt, imax + 1, freq)
    my, mt = obs_to_date(by, bt, imin + 1, freq)
    if freq > 1:
        o += "               Minimum: %18.6f at %2d/%d (observation %3d)\n" % (d[imin], mt, my, imin + 1)
        o += "               Maximum: %18.6f at %2d/%d (observation %3d)\n" % (d[imax], Mt, My, imax + 1)
    else:
        o += "               Minimum: %18.6f at %d (observation %3d)\n" % (d[imin], my, imin + 1)
        o += "               Maximum: %18.6f at %d (observation %3d)\n" % (d[imax], My, imax + 1)
    return o + "\n", mu, tmp * tmp, imax, imin


def plot_ser(d, freq, by, bt, mu, var, imax, imin):
    guions = list(GUIONS)
    marcas = list("                                       0                          ")
    r1, r2, r3, r4 = d[imax], d[imin], mu, math.sqrt(var)
    if not (r4 > 0.0):
        return "(residual series is constant; plot skipped)\n\n"
    absmax = abs((r1 - r3) / r4)
    if abs((r2 - r3) / r4) > absmax:
        absmax = abs((r2 - r3) / r4)
    if absmax <= 2.0:
        absmax = 3.0
    if not (absmax <= 8.0):
        return "Warning: at least one observation above 8 sigmas\n"
    hor = 25.0 / absmax
    b1, b2 = hor, 2.0 * hor
    for i in range(1, 9):
        if absmax >= i:
            c = str(i)[0]
            k = round_local(i * hor)
            guions[39 - k] = "+"
            marcas[39 - k] = c
            marcas[39 - k - 1] = "-"
            guions[39 + k] = "+"
            marcas[39 + k] = c
            marcas[39 + k - 1] = "+"
    G, Mk = "".join(guions), "".join(marcas)
    o = "Standardized time series plot (original values on right-side column):\n\n"
    o += "%s\n%s\n" % (Mk, G)
    for i in range(1, len(d) + 1):
        o += "%4d" % i
        Ap, As = obs_to_date(by, bt, i, freq)
        o += ("%7d " % Ap) if freq == 1 else ("%3d/%4d" % (As, Ap))
        t = [" "] * 55
        if freq != 1 and As == freq:
            t[1] = t[53] = "+"
        else:
            t[1] = t[53] = "|"
        z = (d[i - 1] - r3) / r4
        if abs(z) >= 2.0:
            t[0] = t[54] = ">"
        pos = z * hor
        t[27 + round_local(pos)] = "*"
        if t[27] == " ":
            t[27] = "|"
        for k in (round_local(b1), -round_local(b1), round_local(b2), -round_local(b2)):
            if t[27 + k] == " ":
                t[27 + k] = ":"
        o += "".join(t) + "%13.10f\n" % d[i - 1]
    o += "%s\n%s\n\n" % (G, Mk)
    o += ("                 +------------------------------------------+\n"
          "                 |       Table of standardized values       |\n"
          "                 |       greater than or equal to 2.0       |\n"
          "                 +------------------------------------------+\n"
          "                 |                                          |\n"
          "                 | Observation    Date   Standardized value |\n"
          "                 |                                          |\n")
    for i in range(1, len(d) + 1):
        z = (d[i - 1] - r3) / r4
        if abs(z) >= 2.0:
            Ap, As = obs_to_date(by, bt, i, freq)
            o += "                 |" + "%7d" % i
            o += ("%13d " % Ap) if freq == 1 else ("%9d/%4d" % (As, Ap))
            o += "%13.2f" % z + "        |\n"
    o += "                 +------------------------------------------+\n\n"
    return o


def hist_ser(d, mu, var, imax, imin):
    NumFil, NumCol = 17, 64
    r1, r2 = mu, math.sqrt(var)
    if not (r2 > 0.0):
        return "(residual series is constant; histogram skipped)\n\n"
    xmax = abs((d[imax] - r1) / r2)
    if abs((d[imin] - r1) / r2) > xmax:
        xmax = abs((d[imin] - r1) / r2)
    if not (xmax <= 8.0):
        return "Warning: at least one observation above 8 sigmas\n"
    xmax = 4.0 if xmax <= 4.0 else 8.0
    if xmax == 4.0:
        nphor, no, yes = 4, "    ", "...."
        base1 = "        +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+"
        base2 = "       -4      -3      -2      -1       0      +1      +2      +3      +4"
    else:
        nphor, no, yes = 2, "  ", ".."
        base1 = "        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
        base2 = "       -8  -7  -6  -5  -4  -3  -2  -1   0  +1  +2  +3  +4  +5  +6  +7  +8"
    brk = [0.0, -xmax + 0.5]
    while brk[-1] < xmax:
        brk.append(brk[-1] + 0.5)
    NumCat = len(brk) - 1
    freqs = [0] * (NumCat + 1)
    a1 = a2 = 0
    for v in d:
        num = (v - r1) / r2
        if num <= brk[1]:
            freqs[1] += 1
        else:
            for j in range(2, NumCat + 1):
                if brk[j - 1] < num <= brk[j]:
                    freqs[j] += 1
                    break
        if abs(num) >= 2.0:
            a2 += 1
        if abs(num) >= 1.0:
            a1 += 1
    fmax = max(freqs[1:NumCat + 1])
    opf = fmax / 16.0
    shist = [""] * NumFil
    aux = [""] * NumFil
    for j in range(2, NumFil + 1):
        row = ""
        for i in range(1, NumCat + 1):
            row += yes if freqs[i] > opf * (NumFil - j) else no
        shist[j - 1] += row + "|"
    chk = [0] * (NumCat + 1)
    for j in range(2, NumFil + 1):
        for i in range(1, NumCat + 1):
            if freqs[i] > opf * (NumFil - j) and chk[i] == 0:
                s1 = "%d" % freqs[i]
                if nphor == 2:
                    if len(s1) == 1:
                        s1 += " "
                else:
                    if len(s1) == 2:
                        s1 = " " + s1
                    elif len(s1) == 1:
                        s1 = "  " + s1
                    elif len(s1) == 3:
                        s1 += " "
                if len(s1) > nphor:
                    s1 = "*" * nphor
                aux[j - 2] += s1
                chk[i] = 1
            else:
                aux[j - 2] += no
    # strcpy(shist[0], aux[0]); shist[0][NumCol-1] = '|'; shist[0][NumCol] = 0:
    # a 3-character label in a 4-character cell leaves aux[0] short, and then
    # the '|' lands past the string's terminator and is not printed
    a0 = aux[0]
    if len(a0) >= NumCol - 1:
        shist[0] = a0[:NumCol - 1] + "|"
    else:
        shist[0] = a0
    for j in range(2, NumFil):
        row = list(shist[j - 1])
        for k, ch in enumerate(aux[j - 1]):
            if ch != " ":
                row[k] = ch
        shist[j - 1] = "".join(row)
    o = "Standardized time series histogram:\n\n%s\n%s\n" % (base2, base1)
    for i in range(NumFil):
        o += "        |%s\n" % shist[i]
    o += "%s\n%s\n\n" % (base1, base2)
    n = len(d)
    o += "%16d values outside (-1,+1): %5.2f %% (31.74 %% expected)\n" % (a1, (a1 * 100.0) / n)
    o += "%16d values outside (-2,+2): %5.2f %% ( 4.56 %% expected)\n" % (a2, (a2 * 100.0) / n)
    return o + "\n"


def diagnose(a, names, freq, start, offset, npar_ser):
    """diagnose (diagnose.c:372). a: n x m residuals (lists); names: the m
    labels; start: (year, sub) of the RAW sample; offset: where the residuals
    start in it (drvec passes nobs_raw - nobs)."""
    n, m = len(a), len(a[0])
    o = "\n\n" + "=" * 61 + "\n                RESIDUAL DIAGNOSTICS                         \n" + "=" * 61 + "\n"
    by, bt = obs_to_date(start[0], start[1], offset + 1, freq)
    for i in range(m):
        d = [a[t][i] for t in range(n)]
        o += "\n--- Residual series a[%d] (%s) ---\n" % (i + 1, names[i])
        t, mu, var, imax, imin = stat_ser(d, freq, by, bt)
        o += t
        o += plot_ser(d, freq, by, bt, mu, var, imax, imin)
        o += hist_ser(d, mu, var, imax, imin)
        lags = (n - freq // 2) if n < 3 * (freq + 1) else 3 * (freq + 2)
        if lags > n - 2:
            lags = n - 2
        corr = acf(d, lags, mu, var)
        o += plot_cor(corr, lags, True, n, freq, npar_ser[i])
        corr = pacf(corr, lags)
        o += plot_cor(corr, lags, False, n, freq, npar_ser[i])
    if m > 1:
        o += "\n--- Cross-correlation functions between residuals ---\n"
        lags = n if n < 6 else 9
        if lags > n - 2:
            lags = n - 2
        for i in range(m - 1):
            r1 = [a[t][i] for t in range(n)]
            m1 = mean(r1)
            sd1 = stdev(r1)
            for j in range(i + 1, m):
                r2 = [a[t][j] for t in range(n)]
                m2 = mean(r2)
                sd2 = stdev(r2)
                o += "\nCROSS CORRELATION a[%d] (%s) - a[%d] (%s)\n" % (j + 1, names[j], i + 1, names[i])
                o += "      %s --> %s IF k > 0\n" % (names[j], names[i])
                o += "      %s --> %s IF k < 0\n" % (names[i], names[j])
                c1 = ccf(r1, r2, lags, m1, m2, sd1, sd2)
                c2 = ccf(r2, r1, lags, m2, m1, sd2, sd1)
                tot = [0.0] * (2 * lags + 2)
                for h in range(1, lags + 2):
                    tot[h] = c1[lags + 2 - h]
                for h in range(2, lags + 2):
                    tot[lags + h] = c2[h]
                o += plot_ccf(tot, lags, n)
                q2 = chitest_c(c2, lags + 1, n)
                q1 = chitest_c(c1, lags + 1, n)
                o += "Q(%2d) = %6.3f     k >= 0      Q(%2d) = %6.3f      k > 0\n" % (
                    lags + 1, q2, lags, q2 - (c2[1] * c2[1] * (n + 2)))
                o += "Q(%2d) = %6.3f     k <= 0      Q(%2d) = %6.3f      k < 0\n" % (
                    lags + 1, q1, lags, q1 - (c1[1] * c1[1] * (n + 2)))
    return o + "\n" + "=" * 61 + "\n"

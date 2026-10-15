#include <stdio.h>
#include <math.h>
typedef double real;
static unsigned long boot_rng = 987654321UL;

static real boot_normal(void)
{
    /* Box-Muller over an LCG of our own; deterministic and independent of libc. */
    static int have = 0;
    static real spare = 0.0;
    real u1, u2, r, th;
    if (have) { have = 0; return spare; }
    do {
        boot_rng = boot_rng * 6364136223846793005UL + 1442695040888963407UL;
        u1 = ((real)((boot_rng >> 33) & 0x7FFFFFFF)) / 2147483648.0;
    } while (u1 <= 1.0e-12);
    boot_rng = boot_rng * 6364136223846793005UL + 1442695040888963407UL;
    u2 = ((real)((boot_rng >> 33) & 0x7FFFFFFF)) / 2147483648.0;
    r  = sqrt(-2.0 * log(u1));
    th = 2.0 * M_PI * u2;
    spare = r * sin(th); have = 1;
    return r * cos(th);
}

int main(void){int i; for(i=0;i<12;i++) printf("%.17g\n", boot_normal()); printf("state %lu\n", boot_rng); return 0;}

"""-rungs: the ladder below the rank, emitted (run_rungs, drvec.c:9047-9178).

Rungs 0 (F, Theta and Sigma diagonal), 1 (Sigma free) and 2 (F and Theta
free), all at r = 0, each from the optimum of the one below and fitted with
the search: exactly the C's r = 0 ladder (fit.fit_r0_ladder), with rung 0's
gate, and the ordinary nested chi2 LRs between them. The three rungs are
always fitted, whatever -diagar/-diagma/-diagcov say; rung 2 keeps the MA
class only when it is -marow or -matri.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

NAMES = ("0  F, Theta, Sigma diagonal", "1  Sigma free", "2  F and Theta free")


@dataclass
class Rungs:
    spec: object            # the command line's spec (its r is the header's)
    data: object
    ladder: object          # fit.Ladder (rungs, gate)
    npar: list
    nobs: int

    @property
    def ll(self):
        return [r[3] for r in self.ladder.rungs] + [None] * (3 - len(self.ladder.rungs))

    def lr(self, k):
        a, b = self.ll[k], self.ll[k + 1]
        return None if a is None or b is None else 2.0 * (b - a)

    def out_text(self, base):
        from .report import _chisq_q, gate_block, header
        o = header(self.data, self.spec, base + ".out")
        o += "\n=== The ladder: rungs at r = 0 ===\n"
        if self.ladder.gate is not None:
            o += gate_block(self.ladder.gate)
        o += ("\n  rung                          npar         logL         AIC         BIC\n"
              "  ---------------------------------------------------------------------\n")
        n = self.nobs
        for k in range(3):
            ll = self.ll[k]
            if ll is None:
                o += "  %-28s  --   estimation failed\n" % NAMES[k]
                continue
            o += "  %-28s %4d %12.4f %11.4f %11.4f\n" % (
                NAMES[k], self.npar[k], ll, (-2.0 * ll + 2.0 * self.npar[k]) / n,
                (-2.0 * ll + self.npar[k] * math.log(n)) / n)
        o += ("\n  step        LR = 2*[L(k+1) - L(k)]    df    p-value\n"
              "  -----------------------------------------------------\n")
        for k in range(2):
            lr = self.lr(k)
            if lr is None:
                o += "  %d -> %d      not available\n" % (k, k + 1)
                continue
            df = self.npar[k + 1] - self.npar[k]
            o += "  %d -> %d   %14.4f       %3d   %9.4f%s\n" % (
                k, k + 1, lr, df, _chisq_q(lr, df) if (df > 0 and lr > 0.0) else 1.0,
                "   *** NEGATIVE: the wider fit is worse, so it did not converge"
                if lr < -1.0e-6 else "")
        o += ("\n  These are ordinary nested comparisons: the restricted model is\n"
              "  an INTERIOR point of the wider one, so the log-likelihood cannot\n"
              "  fall and the statistic is chi2 on the stated degrees of freedom.\n"
              "  The next rung -- adding the VEC matrix, r = 1 -- is not ordinary\n"
              "  in either respect: the null sits ON the boundary of the alternative\n"
              "  and B2 is unidentified under it.  It is reported by -lrtest, with\n"
              "  a parametric bootstrap available for its distribution.\n")
        return o


def run_rungs(data, spec):
    """spec: the command line's (p, q, case, MA class, r for the header)."""
    from .fit import fit_r0_ladder
    ma2 = spec.ma if spec.ma in ("marow", "matri") else "free"
    top = spec.but(r=0, diag_ar=False, diag_cov=False, ma=ma2)
    lad = fit_r0_ladder(top, data)
    s0 = top
    npar = [s0.but(diag_ar=True, ma="diag", diag_cov=True).npar(),
            s0.but(diag_ar=True, ma="diag", diag_cov=False).npar(), s0.npar()]
    return Rungs(spec, data, lad, npar, data.at(0).n)

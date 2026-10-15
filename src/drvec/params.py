"""The specification, the parameter vector and its two row orders.

THE TWO ROW ORDERS (the BUG-17 family of the C, REVIEW_2026-09-23 §5.1). Every
matrix of the VEC coordinates -- the rows of Lambda, the rows AND columns of
F_k, Theta_k and Sigma (Q) -- is in the INTERNAL order [Y1 ; Y2], so Q[0][0] = 1
is the variance of the first Y1 series. The data, the names, the mean, B2, w and
the residuals are in the .inp order [Y2 ; Y1] = [nabla Y2 ; W]. `RowOrder` is the
one place that maps between them (drvec.c:2841 inp2lam, 2850 lam2inp); it
depends on r.

THE VECTOR (par_blocks 881, calc_nparametrs 923; walked by vec_shootx 4439-4583):

    mean      case 1: none; case 2: E[W] (r); case 3: E[nabla Y2] (s) then E[W] (r)
    Lambda    M x r, ROW-major, internal rows
    F_k       k = 1..p-1, M x M row-major, internal (diagonal under diag_ar)
    Theta_k   k = 1..q, in the class's pattern, internal
    Q         Q[i][i] for i = 2..M, then Q[i][j], i = 2..M, j < i  (Q[1][1] = 1)
    B2        s x r, COLUMN-major, .inp order of the Y2 block

Lambda and B2 run in opposite orders, as in the C. One walker, here, is used by
every other module: the C has seven hand walks of this vector, and two of them
disagreeing is how BUG-30 and BUG-32 happened.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# the moving-average classes, as zero patterns on Theta (internal order)
MA_CLASSES = ("free", "diag", "marow", "matri")


@dataclass(frozen=True)
class Spec:
    M: int
    p: int
    q: int
    r: int
    case: int = 2
    ma: str = "free"          # free (the default since 2026-09-23), diag, marow, matri
    diag_ar: bool = False
    diag_cov: bool = False
    drift: tuple = None       # case 3: drift[i] False holds E[nabla Y2_i] = 0 (-nodrift,
                              # -case pre); by series, .inp order; None = all free
    trend: bool = False       # -trend (case 4): a trend in W, restricted to beta
    b2_hold: tuple = ()       # -fixb2row: ((row, value), ...), rows of B2 held, 0-based

    def __post_init__(self):
        if self.ma not in MA_CLASSES:
            raise ValueError(f"unknown MA class {self.ma!r}")
        if self.case not in (1, 2, 3):
            raise ValueError("case must be 1, 2 or 3")
        if self.p < 1 or self.q < 0 or not (0 <= self.r < self.M):
            raise ValueError("p >= 1, q >= 0 and 0 <= r < M")

    def but(self, **kw):
        return replace(self, **kw)

    def mu_free(self, i):
        """y2mu_free (drvec.c:796): series i (0-based) carries a free drift."""
        if self.case != 3:
            return False
        return self.drift is None or i >= len(self.drift) or bool(self.drift[i])

    def b2_held(self, i):
        """The value row i of B2 is held at (-fixb2row), or None."""
        for row, v in self.b2_hold:
            if row == i:
                return v
        return None

    def b2_nheld(self):
        return sum(1 for i in range(self.s) if self.b2_held(i) is not None)

    @property
    def s(self):
        return self.M - self.r

    @property
    def nf(self):
        return self.p - 1 if self.p > 1 else 0

    def theta_class(self):
        """ma_class_now (drvec.c:4818). The structured classes are written on the
        r/s partition and collapse at r = 0 (marow_on needs r > 0)."""
        if self.q == 0:
            return "q0"
        if self.ma == "marow":
            return "marow" if self.r > 0 else "free"
        return self.ma

    def f_free(self, i, j):
        return i == j if self.diag_ar else True

    def f_nfree(self):
        M = self.M
        return sum(self.f_free(i, j) for i in range(M) for j in range(M))

    def nsig(self):
        M = self.M
        return (M if self.diag_cov else M * (M + 1) // 2) - 1

    def blocks(self):
        """(nmean, nlam, nmid, ntail): par_blocks."""
        M, r, s = self.M, self.r, self.s
        nmean = (r if self.case == 2 else
                 (sum(self.mu_free(i) for i in range(s)) + r if self.case == 3 else 0))
        nmean += r if self.trend else 0
        nlam = M * r
        nmid = self.nf * self.f_nfree() + theta_len(self.theta_class(), self.q, M, r) + self.nsig()
        return nmean, nlam, nmid, (s - self.b2_nheld()) * r

    def npar(self):
        return sum(self.blocks())

    def head_tail(self):
        """search_blocks (4838): head = mean, Lambda, F; tail = Q, B2."""
        nmean, nlam, nmid, ntail = self.blocks()
        return nmean + nlam + self.nf * self.f_nfree(), self.nsig() + ntail


def theta_len(cls, q, M, r):
    s = M - r
    return {"q0": 0, "diag": q * M, "marow": q * r * M,
            "matri": q * (M * M - s * r), "free": q * M * M}[cls]


def theta_unpack(x, off, cls, q, M, r):
    """theta_unpack (4862): q M x M matrices, zeros where the class has none."""
    Th = [[[0.0] * M for _ in range(M)] for _ in range(q)]
    idx = off
    for k in range(q):
        T = Th[k]
        if cls == "diag":
            for i in range(M):
                T[i][i] = x[idx]; idx += 1
        elif cls in ("marow", "matri"):
            for i in range(r):
                for j in range(M):
                    T[i][j] = x[idx]; idx += 1
            if cls == "matri":
                for i in range(r, M):
                    for j in range(r, M):
                        T[i][j] = x[idx]; idx += 1
        elif cls == "free":
            for i in range(M):
                for j in range(M):
                    T[i][j] = x[idx]; idx += 1
    return Th, idx


def theta_pack(Th, cls, q, M, r):
    out = []
    for k in range(q):
        T = Th[k]
        if cls == "diag":
            out += [T[i][i] for i in range(M)]
        elif cls in ("marow", "matri"):
            out += [T[i][j] for i in range(r) for j in range(M)]
            if cls == "matri":
                out += [T[i][j] for i in range(r, M) for j in range(r, M)]
        elif cls == "free":
            out += [T[i][j] for i in range(M) for j in range(M)]
    return out


def embed_ma(xs, spec_src, spec_dst):
    """embed_ma (4900): the optimum of a poorer MA class as a point of the
    richer one. Same head and tail, Theta completed with zeros."""
    M, r = spec_dst.M, spec_dst.r
    head, tail = spec_dst.head_tail()
    cs, qs = spec_src.theta_class(), spec_src.q
    cd, qd = spec_dst.theta_class(), spec_dst.q
    ls, ld = theta_len(cs, qs, M, r), theta_len(cd, qd, M, r)
    qm = min(qs, qd)
    Th = [[[0.0] * M for _ in range(M)] for _ in range(qd)]
    if qs > 0:
        Tu, _ = theta_unpack(xs, head, cs, qm, M, r)
        for k in range(qm):
            Th[k] = Tu[k]
    xd = list(xs[:head]) + theta_pack(Th, cd, qd, M, r) + list(xs[head + ls: head + ls + tail])
    assert len(xd) == head + ld + tail
    return xd


class RowOrder:
    """inp2lam / lam2inp for (M, r), 0-based."""

    def __init__(self, M, r):
        self.M, self.r, self.s = M, r, M - r

    def inp2lam(self, i):
        return self.r + i if i < self.s else i - self.s

    def lam2inp(self, a):
        return self.s + a if a < self.r else a - self.r


def slot_labels(spec, names):
    """One label per slot of x, in the vector's order, with the C's wording
    (report_fit 8340-8550): what the reader sees, in the .inp's names.

    Returns a list of (label, kind, sign): `sign = -1` for Lambda, which is
    reported as alpha = -Lambda."""
    M, r, s, q = spec.M, spec.r, spec.s, spec.q
    ro = RowOrder(M, r)
    nm = lambda a: names[ro.lam2inp(a)]          # noqa: E731  (internal row -> name)
    out = []
    if spec.case == 2:
        out += [(f"E[ec{j + 1}]", "mean", 1) for j in range(r)]
    elif spec.case == 3:
        out += [(f"E[D.{names[i]}]", "mean", 1) for i in range(s) if spec.mu_free(i)]
        out += [(f"E[ec{j + 1}]", "mean", 1) for j in range(r)]
    if spec.trend:
        out += [(f"trend[ec{j + 1}]", "trend", 1) for j in range(r)]
    for a in range(M):
        for j in range(r):
            out.append((f"D.{nm(a)} <- ec{j + 1}", "alpha", -1))
    for k in range(spec.nf):
        for a in range(M):
            for b in range(M):
                if spec.f_free(a, b):
                    out.append((f"D.{nm(a)} <- D.{nm(b)}(-{k + 1})", "gamma", 1))
    cls = spec.theta_class()
    pattern = []
    for k in range(q):
        if cls == "diag":
            pattern += [(k, a, a) for a in range(M)]
        elif cls in ("marow", "matri"):
            pattern += [(k, a, b) for a in range(r) for b in range(M)]
            if cls == "matri":
                pattern += [(k, a, b) for a in range(r, M) for b in range(r, M)]
        elif cls == "free":
            pattern += [(k, a, b) for a in range(M) for b in range(M)]
    for k, a, b in pattern:
        out.append((f"D.{nm(a)} <- A.{nm(b)}(-{k + 1})", "theta", 1))
    out += [(f"var {nm(a)}", "q", 1) for a in range(1, M)]
    if not spec.diag_cov:
        out += [(f"cov {nm(a)}, {nm(b)}", "q", 1) for a in range(1, M) for b in range(a)]
    for j in range(r):
        for i in range(s):
            if spec.b2_held(i) is None:
                out.append((f"{names[i]} in ec{j + 1}", "beta", 1))
    assert len(out) == spec.npar()
    return out

"""drvec-py: the command line of the C's .pre route.

    drvec-py s1.pre s2.pre [... sM.pre] p q r [options]

C: `parse_cli` (drvec.c:10318), `validate_cli` (10214) and `main` (10878). The
syntax, the option table, the checks on p, q and r, the refusals and the
products are the C's: <base>.out always, <base>.boot with -lrtest -bootstrap,
<base>.irfboot with -irfboot, <base>.forecast with -f and <base>.recursive (or
-C FILE) with -estwin; <base> is -name's, or the files' stems joined by `_`.

What the port does not have is refused with exit 2 and says so: the .inp
route, and the options of docs/PORT.md §10 that are not ported. The terminal
output is a summary, not the C's progress log; the files are the C's.

Exit codes, as the C's: 0 done (and -h, --version), 1 a malformed command
line or an unreadable input, 2 a combination refused or an estimation failed.
"""
from __future__ import annotations

import os
import sys

PROG = "drvec-py"

# (name, argument kind, what the value is called, ported)
OPT_TABLE = [
    ("-mean", "none", None, True),
    ("-case", "case", "1|2|3|pre", True),
    ("-nodrift", "int", "i", True),
    ("-m", "method", "1|2", True),
    ("-diagar", "none", None, True),
    ("-trend", "none", None, True),
    ("-diagma", "none", None, True),
    ("-diagcov", "none", None, True),
    ("-levels", "none", None, True),
    ("-differenced", "none", None, True),
    ("-fixb2", "real_opt", "[v]", True),
    ("-fixb2row", "int_real", "i v", True),
    ("-xpre", "str", "FILE", True),
    ("-xlink", "str", "FILE", True),
    ("-xsys", "int", "i", True),
    ("-lrtest", "none", None, True),
    ("-bootstrap", "int", "N", True),
    ("-irfboot", "int", "N", True),
    ("-rungs", "none", None, True),
    ("-specs", "none", None, False),
    ("-warma", "none", None, False),
    ("-mawarma", "none", None, False),
    ("-marow", "none", None, True),
    ("-mafree", "none", None, True),
    ("-f", "int", "H", True),
    ("-estwin", "int", "E", True),
    ("-C", "str", "FILE", True),
    ("-matri", "none", None, True),
    ("-rankadm", "tol_opt", "[tol]", False),
    ("-matest", "int", "N", False),
    ("-artest", "int", "N", False),
    ("-alpha", "str", "FILE", True),
    ("-weakex", "int", "i", True),
    ("-multistart", "int", "n", True),
    ("-eval", "none", None, False),
    ("-fdhess", "none", None, True),
    ("-seed", "str", "pfx", False),
    ("-seedybar", "str", "pfx", False),
    ("-seedgate", "none", None, False),
    ("-noladder", "none", None, False),
    ("-seedjoh", "none", None, False),
    ("-seedb2", "real", "v", False),
    ("-interv", "str", "pfx", True),
    ("-writeres", "str", "pfx", False),
    ("-writeinp", "str", "pfx", False),
    ("-name", "str", "NAME", True),
]
_OPTS = {o[0]: o for o in OPT_TABLE}
B2ROW_MAX = 64


# the C's words for the combinations it refuses (parse_cli, drvec.c:10634-10741)
MSG_XLINK = ("-xlink needs the .pre route in the levels layout, and a\n"
             "       single fit: it cannot be combined with -differenced, -warma,\n"
             "       -lrtest, -specs, -rungs, -matest, -artest, -eval, -f, -estwin,\n"
             "       -writeinp or -writeres.")
MSG_IRFBOOT = ("-irfboot bootstraps the responses of a single fit in the levels\n"
               "       layout: it cannot be combined with -differenced, -xlink, -trend,\n"
               "       -warma, -estwin, -writeinp, -writeres or the modes -lrtest,\n"
               "       -specs, -rungs, -matest, -artest and -eval (not yet).")
MSG_MIXED = ("WARNING: with some drifts held (-nodrift / -case pre) the rank test's\n"
             "         asymptotic tables are case 3's, which do not apply to the\n"
             "         mixed case: read the bootstrap (-bootstrap N), not the table.\n")
MSG_TREND = ("-trend (case 4) goes on case 3's drifts (-case 3 or -case pre)\n"
             "       for a single fit: it cannot be combined with -warma, -f,\n"
             "       -estwin, -writeinp, -writeres or the modes -lrtest, -specs,\n"
             "       -rungs, -matest, -artest and -eval (not yet).")
MSG_XSYS = ("-xsys embeds an input in the system for a single fit with\n"
            "       q = 0.  It cannot be combined with -alpha/-weakex (it sets\n"
            "       alpha itself), -xlink, -warma, a moving average (q > 0, not\n"
            "       yet), or the modes -lrtest, -specs, -rungs, -matest, -artest\n"
            "       and -eval.")
MSG_FIXB2ROW = ("-fixb2row holds rows of B2 at a single fit.  It cannot be\n"
                "       combined with -fixb2 (which holds every row), -warma, or the\n"
                "       modes -lrtest, -specs, -rungs, -matest, -artest and -eval.")


class CliError(Exception):
    def __init__(self, msg, code=1, usage=False):
        super().__init__(msg)
        self.code = code
        self.usage = usage


def _option_list():
    out = ["", "Every option the C accepts (* = not ported yet; refused):", " "]
    col = 0
    for name, _kind, val, ported in OPT_TABLE:
        item = (name + (" " + val if val else "")) + ("" if ported else "*")
        if col and col + len(item) + 2 > 72:
            out.append(" ")
            col = 0
        out[-1] += " %-*s" % (len(item) + 1, item)
        col += len(item) + 2
    return "\n".join(out) + "\n"


def usage():
    from . import __version__
    return f"""
{PROG} {__version__} -- VEC model EML estimation (Mauricio 2006), the Python port of drvec

  {PROG} s1.pre s2.pre [... sM.pre] p q r [options]
        ONE UNIVARIATE MODEL PER SERIES, as in drtran: each .pre
        brings its series, its Box-Cox and rescaling.  The files
        are lined up by DATE and the common sample is used.  The
        column order is the .inp's: the first M-r files are the
        nabla Y2 block and the last r the Y1 block.  Products are
        named <stem1>_<stem2>...; -name NAME overrides that.
        (The C's .inp route, `drvec file p q r', is not ported.)

  p     : AR order of stationary VARMA on Ybar_t
  q     : MA order
  r     : cointegration rank (0 <= r < M; 0 is the diagonal rung;
          ignored with -lrtest)

Deterministic cases (Mauricio 2006, Remark 6):
  -case 1 : E[nabla Y2]=0, E[W]=0     (default)
  -case 2 : E[nabla Y2]=0, E[W]!=0    (-mean alone also selects it)
  -case 3 : E[nabla Y2]!=0, E[W]!=0
  -case pre : case 3 with each drift as its .pre says
  -nodrift i : hold E[nabla Y2_i] = 0 inside case 3 (repeatable)
  -trend  : case 4, a linear trend in W (restricted to beta)

Restrictions (each but -fixb2 with its LR against the free model):
  -fixb2 [v]     hold B2 fixed: at v (an a-priori value, a valid LR) or,
                 without a value, at its static-OLS estimate
  -fixb2row i v  hold ROW i of B2 at v in every relation (repeatable)
  -alpha FILE    alpha = A*psi, A read from FILE (`M sa', then M rows)
  -weakex i      series i weakly exogenous (A = I without column i)
  -xsys i        series i an input embedded in the system (q = 0)
  -xpre FILE -xlink FILE   exogenous transfer-function inputs

Modes and products:
  -lrtest [-bootstrap N]   the sequential LR test for the rank (<base>.boot)
  -rungs         the ladder below the rank: rungs 0, 1 and 2 at r = 0,
                 with the gate and their chi2 LRs
  -irfboot N     bands for the level responses and the FEVD (<base>.irfboot)
  -f H           forecast H steps in LEVELS, with bands (<base>.forecast)
  -estwin E      estimate on observations 1..E and roll the origin (needs
                 -f H; per-origin errors to <base>.recursive, or -C FILE)
  -fdhess        standard errors from the finite-difference Hessian at
                 the optimum: use it before quoting a standard error
  -multistart n  n - 1 jittered starts added to the search (the C's generator)
  -diagar, -diagcov, -diagma, -marow, -matri, -mafree: the structure
""" + _option_list()


def _int(s):
    try:
        if s is None or s.strip() != s or s == "":
            return None
        return int(s, 10)
    except ValueError:
        return None


def _real(s):
    try:
        if s is None or s == "" or s.strip() != s:
            return None
        return float(s)
    except ValueError:
        return None


def _nearest(name):
    best, hit = 0, None
    for o in _OPTS:
        k = 0
        while k < len(name) and k < len(o) and name[k] == o[k]:
            k += 1
        if k > best:
            best, hit = k, o
    return hit if best >= 4 else None


def validate(argv, first):
    """validate_cli: the preliminary pass, which stops at the first problem."""
    i = first
    while i < len(argv):
        a = argv[i]
        if not a.startswith("-"):
            raise CliError(f"unexpected argument `{a}' -- options start with `-'")
        o = _OPTS.get(a)
        if o is None:
            near = _nearest(a)
            msg = f"unknown option `{a}'" + (f"\n       did you mean `{near}'?" if near else "")
            raise CliError(msg, usage=True)
        kind, val = o[1], o[2]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if kind in ("str", "int", "case", "method", "real") and nxt is None:
            raise CliError(f"{a} needs {val}")
        if kind == "str":
            i += 1
        elif kind == "int":
            v = _int(nxt)
            if v is None or v < 1:
                raise CliError(f"{a} needs an integer >= 1, got `{nxt}'")
            i += 1
        elif kind == "case":
            v = _int(nxt)
            if nxt != "pre" and (v is None or not 1 <= v <= 3):
                raise CliError(f"{a} must be 1, 2, 3 (Mauricio 2006, Remark 6) or pre "
                               f"(case 4 is -trend), got `{nxt}'")
            i += 1
        elif kind == "method":
            v = _int(nxt)
            if v is None or not 1 <= v <= 2:
                raise CliError(f"{a} must be 1 (exact ML) or 2 (approximate), got `{nxt}'")
            i += 1
        elif kind == "real":
            if _real(nxt) is None:
                raise CliError(f"{a} needs a number, got `{nxt}'")
            i += 1
        elif kind == "int_real":
            if i + 2 >= len(argv):
                raise CliError(f"{a} needs {val}")
            v = _int(argv[i + 1])
            if v is None or not 1 <= v <= B2ROW_MAX:
                raise CliError(f"{a} needs a row number >= 1, got `{argv[i + 1]}'")
            if _real(argv[i + 2]) is None:
                raise CliError(f"{a} needs a value for the row, got `{argv[i + 2]}'")
            i += 2
        elif kind == "real_opt":
            if nxt is not None and _real(nxt) is not None:
                i += 1
        elif kind == "tol_opt":
            if nxt is not None and not nxt.startswith("-"):
                v = _real(nxt)
                if v is None or v <= 0.0:
                    raise CliError(f"{a} needs a tolerance > 0, got `{nxt}'")
                i += 1
            elif nxt is not None and _real(nxt) is not None:
                raise CliError(f"{a} needs a tolerance > 0, got `{nxt}'")
        i += 1


def parse(argv):
    """argv without the program name -> a dict of the run, or raise CliError."""
    if len(argv) < 4:
        raise CliError("", usage=True)
    k = 0
    while k < len(argv) and argv[k].endswith(".pre"):
        k += 1
    if k == 0:
        raise CliError("the .inp route (`drvec file p q r') is not ported: give one .pre "
                       "per series,\n       `drvec-py s1.pre s2.pre ... p q r [options]'",
                       code=2)
    if len(argv) < k + 3:
        raise CliError("ERROR: after the .pre files come p, q and r", usage=True)
    files = argv[:k]
    first = k + 3
    p, q, r = (_int(v) for v in argv[k:k + 3])
    if p is None or p < 1:
        raise CliError(f"p (the AR order) must be an integer >= 1, got `{argv[k]}'")
    if q is None or q < 0:
        raise CliError(f"q (the MA order) must be an integer >= 0, got `{argv[k + 1]}'")
    if r is None or r < 0:
        raise CliError(f"r (the cointegration rank) must be an integer >= 0, got "
                       f"`{argv[k + 2]}'")
    validate(argv, first)
    o = dict(files=files, p=p, q=q, r=r, case=1, case_given=False, mean=False,
             nodrift=[], diag_ar=False, diag_cov=False, ma=None, fixb2=None, fixb2row={},
             xpre=[], xlink=None, xsys=[], lrtest=False, bootstrap=0, irfboot=0,
             forecast=0, estwin=0, csv=None, alpha=None, weakex=0, fdhess=False,
             name=None, method=1, multistart=0, rungs=False, differenced=False, interv=False, unported=[])
    i = first
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        kind = _OPTS[a][1]
        if not _OPTS[a][3]:
            o["unported"].append(a)
        if a == "-mean":
            o["mean"] = True
        elif a == "-case":
            if nxt == "pre":
                o["case"] = "pre"
            else:
                o["case"] = int(nxt)
            o["case_given"] = True
        elif a == "-nodrift":
            o["nodrift"].append(int(nxt))
        elif a == "-m":
            o["method"] = int(nxt)
        elif a == "-diagar":
            o["diag_ar"] = True
        elif a == "-trend":
            o["trend"] = True
        elif a == "-diagma":
            o["ma"] = "diag"
        elif a == "-diagcov":
            o["diag_cov"] = True
        elif a == "-marow":
            o["ma"] = "marow"
        elif a == "-matri":
            o["ma"] = "matri"
        elif a == "-mafree":
            o["ma"] = o["ma"] or "free"
        elif a == "-differenced":
            o["differenced"] = True
        elif a == "-levels":
            o["differenced"] = False
        elif a == "-fixb2":
            v = _real(nxt) if nxt is not None else None
            if v is not None:
                o["fixb2"] = v
                i += 1
            else:
                o["fixb2"] = True
        elif a == "-fixb2row":
            o["fixb2row"][int(argv[i + 1])] = float(argv[i + 2])
            i += 2
        elif a == "-xpre":
            o["xpre"].append(nxt)
        elif a == "-xlink":
            o["xlink"] = nxt
        elif a == "-xsys":
            o["xsys"].append(int(nxt))
        elif a == "-lrtest":
            o["lrtest"] = True
        elif a == "-rungs":
            o["rungs"] = True
        elif a == "-bootstrap":
            o["bootstrap"] = int(nxt)
        elif a == "-irfboot":
            o["irfboot"] = int(nxt)
        elif a == "-f":
            o["forecast"] = int(nxt)
        elif a == "-estwin":
            o["estwin"] = int(nxt)
        elif a == "-C":
            o["csv"] = nxt
        elif a == "-alpha":
            o["alpha"] = nxt
        elif a == "-weakex":
            o["weakex"] = int(nxt)
        elif a == "-fdhess":
            o["fdhess"] = True
        elif a == "-multistart":
            o["multistart"] = int(nxt)
        elif a == "-interv":
            o["interv"] = True
        elif a == "-name":
            o["name"] = nxt
        if kind in ("str", "int", "case", "method", "real"):
            i += 1
        elif kind == "tol_opt" and nxt is not None and not nxt.startswith("-"):
            i += 1
        i += 1
    o.setdefault("trend", False)
    # -mean is case 2 unless a case was given (BUG-45: -case 1 -mean refused)
    if o["mean"] and o["case"] == 1:
        if o["case_given"]:
            raise CliError("ERROR: -case 1 has no deterministic term and -mean asks for "
                           "one.  Drop one of them\n       (-mean alone means case 2).",
                           code=2)
        o["case"] = 2
    if o["name"]:
        o["base"] = o["name"]
    else:
        stems = []
        for f in files:
            b = os.path.basename(f)
            stems.append(b[:b.rfind(".")] if "." in b else b)
        o["base"] = "_".join(stems)
    return o


def _refusals(o):
    """The C's checks of combinations (parse_cli's tail), in its order."""
    if o["interv"]:
        raise CliError("ERROR: -interv does nothing on the .pre route -- the deterministic\n"
                       "       terms are already subtracted, from those same models.\n"
                       "       Accepting it would subtract them twice.")
    if o["differenced"]:
        raise CliError("ERROR: -differenced does not apply to the .pre route.  A .pre\n"
                       "       carries the series in levels and says how it is differenced;\n"
                       "       drvec forms nabla Y2 itself.")
    if o["unported"]:
        raise CliError("ERROR: %s %s not ported to drvec-py yet (docs/PORT.md §10);\n"
                       "       the C program drvec has %s." % (
                           ", ".join(o["unported"]),
                           "is" if len(o["unported"]) == 1 else "are",
                           "it" if len(o["unported"]) == 1 else "them"), code=2)
    if o["method"] == 2:
        raise CliError("ERROR: -m 2 (the xi sequence not truncated) is not ported to "
                       "drvec-py yet.", code=2)
    if o["lrtest"] and o["rungs"]:
        raise CliError("ERROR: one mode per run: -lrtest, -specs, -rungs, -matest/-artest "
                       "and -eval\n       cannot be combined.", code=2)
    if o["rungs"] and (o["forecast"] or o["estwin"]):
        raise CliError("ERROR: -f and -estwin act on a single fit; the mode asked for "
                       "does not\n       produce one.  Run them separately.", code=2)
    if o["rungs"]:
        for on, msg in ((o["xlink"] or o["xpre"], MSG_XLINK), (o["irfboot"], MSG_IRFBOOT),
                        (o["trend"], MSG_TREND), (o["xsys"], MSG_XSYS),
                        (o["fixb2row"], MSG_FIXB2ROW)):
            if on:
                raise CliError("ERROR: " + msg, code=2)
        late = [n for n, on in (("-alpha", o["alpha"]), ("-weakex", o["weakex"]),
                                ("-fixb2", o["fixb2"] is not None),
                                ("-nodrift", o["nodrift"]), ("-case pre", o["case"] == "pre"))
                if on]
        if late:
            raise CliError("ERROR: -rungs with %s is not ported to drvec-py yet."
                           % ", ".join(late), code=2)
    if o["lrtest"]:
        if o["forecast"] or o["estwin"]:
            raise CliError("ERROR: -f and -estwin act on a single fit; the mode asked for "
                           "does not\n       produce one.  Run them separately.", code=2)
        if o["irfboot"]:
            raise CliError("ERROR: " + MSG_IRFBOOT, code=2)
        if o["fixb2row"]:
            raise CliError("ERROR: " + MSG_FIXB2ROW, code=2)
        late = [n for n, on in (("-alpha", o["alpha"]), ("-weakex", o["weakex"]),
                                ("-fixb2", o["fixb2"] is not None),
                                ("-diagar", o["diag_ar"]), ("-diagcov", o["diag_cov"]))
                if on]
        if late:
            raise CliError("ERROR: -lrtest with %s is not ported to drvec-py yet (the C "
                           "program drvec has it)." % ", ".join(late), code=2)
    if o["bootstrap"] > 0 and not o["lrtest"]:
        raise CliError("ERROR: -bootstrap calibrates -lrtest and does nothing without it.",
                       code=2)


def run(argv, out=sys.stdout, err=sys.stderr):
    from . import __version__
    if argv and argv[0] in ("-h", "--help"):
        out.write(usage())
        return 0
    if argv and argv[0] in ("--version", "-version"):
        out.write(f"{PROG} {__version__}\nVEC model EML estimation (Mauricio 2006), "
                  "the Python port of drvec, on the drvarma engine\nGPL v2 or later\n")
        return 0
    try:
        o = parse(argv)
        _refusals(o)
    except CliError as e:
        if str(e):
            msg = str(e)
            err.write((msg if msg.startswith("ERROR") else f"drvec-py: {msg}") + "\n")
            if not e.usage and not msg.startswith("ERROR"):
                err.write(f"       `{PROG} -h' lists every option.\n")
        if e.usage:
            err.write(usage())
        return e.code
    return _execute(o, out, err)


def _execute(o, out, err):
    import drvec
    from . import __version__
    base = o["base"]
    for f in o["files"]:
        if not os.path.exists(f):
            err.write(f"ERROR: cannot open {f}\n")
            return 1
    out.write(f"\nDRVEC {__version__} (python) -- VEC model EML estimation (Mauricio 2006)\n"
              f"Input  : the .pre files\nOutput : {base}.out\n"
              f"Model  : VEC({o['r']}) with stationary VARMA({o['p']},{o['q']}) on Ybar_t\n"
              f"Case   : {o['case']}\n")
    try:
        if o["rungs"]:
            t = drvec.rungs(o["files"], o["p"], o["q"], o["r"], case=o["case"],
                            ma=o["ma"] or "free")
            _write(base + ".out", t.out_text(base))
            for k in range(3):
                out.write("  rung %d : %s\n" % (k, "ok" if t.ll[k] is not None
                                                 else "estimation failed"))
            out.write(f"Done. Output written to {base}.out\n")
            return 0
        if o["lrtest"]:
            if o["nodrift"] or o["case"] == "pre":
                err.write(MSG_MIXED)
            boot = base + ".boot" if o["bootstrap"] > 0 else None
            t = drvec.rank_test(o["files"], o["p"], o["q"], case=o["case"],
                                ma=o["ma"] or "free", bootstrap=o["bootstrap"],
                                boot_path=boot, trend=o["trend"], xsys=o["xsys"] or None,
                                xpre=o["xpre"] or None, xlink=o["xlink"],
                                multistart=o["multistart"], nodrift=o["nodrift"] or None)
            text = t.out_text(base)
            _write(base + ".out", text)
            for rr in range(t.spec.M - 1):
                out.write("  r = %d: LR = %.6f\n" % (rr, t.lr(rr)))
            if boot:
                out.write(f"Bootstrap replications written to {boot}\n")
            out.write(f"Done. Output written to {base}.out\n")
            return 0
        res = drvec.fit(o["files"], o["p"], o["q"], o["r"], case=o["case"],
                        ma=o["ma"] or "free", diag_ar=o["diag_ar"], diag_cov=o["diag_cov"],
                        fixb2row=o["fixb2row"] or None, nodrift=o["nodrift"] or None,
                        trend=o["trend"], irfboot=o["irfboot"], xsys=o["xsys"] or None,
                        xpre=o["xpre"] or None, xlink=o["xlink"],
                        forecast=o["forecast"], fdhess=o["fdhess"], alpha=o["alpha"],
                        weakex=o["weakex"] or None, fixb2=o["fixb2"], estwin=o["estwin"],
                        multistart=o["multistart"])
    except (ValueError, NotImplementedError) as e:
        msg = str(e)
        err.write((msg if msg.startswith("ERROR") else "ERROR: " + msg) + "\n")
        return 2
    except RuntimeError as e:
        err.write(f"ERROR: {e}\n")
        return 2
    except OSError as e:
        err.write(f"ERROR: {e}\n")
        return 1
    if res.fdhess is not None and not res.fdhess.ok:
        err.write(res.fdhess.warning(res.npar))
    _write(base + ".out", res.out_text(base))
    out.write("  logL = %.10f   npar = %d   %s\n" % (res.loglik, res.npar,
                                                     res.search.best.label))
    ib = res.irfboot_text(base) if res.irfboot is not None else None
    if ib is not None:
        # too few replications for bands: the C writes no .irfboot, and its
        # .out says so (bugs/py-cli-irfboot-few)
        _write(base + ".irfboot", ib)
        out.write(f"Bands written to {base}.irfboot\n")
    if res.forecast_h:
        _write(base + ".forecast", res.forecast_text(base))
        out.write(f"Forecasts written to {base}.forecast\n")
    rec = res.recursive_text()
    if rec is not None:
        path = o["csv"] or base + ".recursive"
        _write(path, rec)
        out.write(f"Per-origin errors: {path}\n")
    out.write(f"Done. Output written to {base}.out\n")
    return 0


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()

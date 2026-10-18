# drvec

**VARMA-VEC cointegration models by exact maximum likelihood** (Mauricio
2006): a vector error-correction model whose short-run part is a full VARMA,
not only a VAR, estimated by the exact Gaussian likelihood. drvec fits the
model at a given cointegrating rank, tests the rank (the sequential LR test,
with a parametric bootstrap), and writes the C program's reports: estimates
and standard errors, the long run (alpha, beta, Pi), restrictions on alpha and
beta with their LR tests, diagnostics, forecasts and impulse responses.

It is the Python port of the C program **drvec** of the ATSW suite
([atsw-gui](https://github.com/davidesg/atsw-gui), `engines/drvec`), and its
fourth rung: [fue](https://pypi.org/project/fue/) and
[art](https://pypi.org/project/art-tseries/) build each series' univariate
model, drtran the transfer functions, drvarma and sima the stationary VARMA,
and drvec the cointegrated system. The assistant that drives it is **sima**
([sima-tseries](https://pypi.org/project/sima-tseries/)).

## Installation

```sh
pip install drvec
```

drvec's code is Python. What must be fast comes with binary wheels:
[drvarma](https://pypi.org/project/drvarma/) (>= 0.3.0) brings the compiled
exact VARMA likelihood (AS 311), its optimiser and GSL's eigensolvers,
[fue](https://pypi.org/project/fue/) (>= 0.1.18) the reader and cast of the
`.pre` files, and drvec's own wheels carry **drvec C 0.13**, the program the
port mirrors, for the long bootstraps (below): Linux x86_64 and aarch64 and
macOS arm64, with GSL linked in. Nothing to compile, and no system GSL
needed. Python 3.10 or later. Elsewhere, Windows included, pip takes the
pure-Python wheel and the bootstraps run in Python (on Windows drvec C 0.13's
bootstrap does not end: a C defect, bugs/drvec-c-rng-long32).

Check that the compiled engine is in use:

```sh
python -c "import drvarma._engine as E, drvec._gsl as G; print(E.engine_backend(), G.backend())"
# c gsl
python -c "from drvec import c_engine; print(c_engine.find())"
# CBinary(path='.../site-packages/drvec/_bin/drvec', version='0.13', source='bundled')
```

## A minimal example

drvec reads one fue `.pre` file per series: each series' univariate model
(its Box-Cox, deterministic terms and ARMA), from which it takes the series
in levels. Jenkins and Alavi's muskrat and mink (62 annual observations from
1850), with the files of the test suite:
[mmpre.muskrat.pre](https://raw.githubusercontent.com/davidesg/drvec-python/main/tests/data/fixtures/mmpre.muskrat.pre)
and
[mmpre.mink.pre](https://raw.githubusercontent.com/davidesg/drvec-python/main/tests/data/fixtures/mmpre.mink.pre).

```python
import drvec

pre = ["mmpre.muskrat.pre", "mmpre.mink.pre"]

# the rank: sequential LR tests (add bootstrap=200 for the bootstrap p-values)
rt = drvec.rank_test(pre, p=1, q=0, case=2)
open("mmrank.out", "w").write(rt.out_text("mmrank"))   # the C program's .out

# a VAR(1)-VEC with one cointegrating relation, Johansen's case 2
r = drvec.fit(pre, p=1, q=0, r=1, case=2)
print(r.loglik)                       # -18.980152265473677
for p in r.params:
    print(f"{p.label:24s} {p.value:9.4f} {p.se:8.4f}")
open("mm.out", "w").write(r.out_text("mm"))
```

The same from the command line, with the C program's syntax and files:

```sh
drvec-py mmpre.muskrat.pre mmpre.mink.pre 1 0 1 -case 2 -name mm
drvec-py mmpre.muskrat.pre mmpre.mink.pre 2 0 1 -case 2 -lrtest -bootstrap 200
drvec-py -h
```

It is `drvec-py` and not `drvec` because `drvec` is the C program's name on a
system with the ATSW suite.

## The results as numbers

Everything the `.out` prints is also there as arrays, computed by the same
routine that prints it (the suite checks that each equals the printed number):

```python
r.alpha, r.beta, r.Pi, r.gamma, r.theta, r.Sigma   # in the files' order
r.residuals, r.stationary                          # Ybar's innovations and VARMA
r.level_irf(), r.fevd(), r.long_run()              # the responses in levels
r.diagnostics().hosking_p, r.diagnostics().jb_p    # the residual tests
r.forecasts(8).level                               # (H, M), in each series' units
r.restriction_lr                                   # an LR with its df, when restricted
```

The list, with the conventions, is in
[docs/API.md](https://github.com/davidesg/drvec-python/blob/main/docs/API.md).

## Long bootstraps: workers, and drvec C as the engine

A rank test with B = 999 on three series is hours of Python on one core.
Two things make it affordable, and neither changes a number:

- **`workers=`**: the replications are drawn in sequence, in the C's order,
  and refitted in a process pool. The default is `DRVEC_WORKERS`, else the
  cores less one; `workers=1` runs in-process. The output is byte-identical
  whatever the number.
- **`engine=`**: `"auto"` (the default) runs the bootstrap on **drvec C 0.13**
  when it finds the binary -- `DRVEC_BIN`, then the one bundled in the
  wheel (`drvec/_bin/drvec`), then `drvec` on the `PATH`, then the ATSW
  install directory -- and on the parallel Python otherwise. The C's replications are read back into
  the same result objects; on Linux x86_64 the files are the same bytes.
  `"python"` and `"c"` force one or the other.

```python
rt = drvec.rank_test(pre, p=2, q=0, case=2, bootstrap=999, boot_path="mm.boot",
                     progress=lambda b, rep: print(rep.h0, b, rep.status))
print(rt.provenance.engine, rt.provenance.version)   # c 0.13, or python 0.1.0
```

`seed=` gives another stream for a Monte Carlo check (always in Python: the
C's generator is fixed at its seed); the default is the C's.

## The C oracle and the platform convention

The port is homologated, case by case, against its oracle: the C program
**drvec 0.13** ([atsw-gui](https://github.com/davidesg/atsw-gui) commit
`ae95446`; its tag `drvec-v0.13` comes with the suite's release). On these
cases it is not "within tolerance" of the C: it is the same computation.

- **Exact on Linux x86_64**, the oracle's platform: every number and every
  line of the `.out`, `.boot`, `.forecast` and `.irfboot` files is byte for
  byte the C's.
- **A tolerance tier elsewhere** (Linux aarch64, macOS, Windows): the
  compiled likelihood and libm round the last bits differently, and the
  optimiser can amplify them on a flat likelihood. There the log-likelihood
  agrees to 1e-5 and the estimates to 1e-4, and every *decision* -- the
  winning start, the rank test's verdicts, the bootstrap's statuses and
  p-values -- must still be the C's.
- **Ill-defined fits** (a search that stops on a flat ridge, a gate verdict
  that rounding can flip) are compared off that platform on what they
  declare, not on the point.

The details, decision by decision, are in
[docs/PORT.md](https://github.com/davidesg/drvec-python/blob/main/docs/PORT.md);
the rules of the convention in
[tests/conftest.py](https://github.com/davidesg/drvec-python/blob/main/tests/conftest.py).

Not ported in this version, and refused by name when asked for (exit 2, with
the C's words): the options starred in `drvec-py -h` -- `-specs`, `-warma`,
`-mawarma`, `-matest`, `-artest`, `-rankadm`, `-eval`, the seed options
(`-seed`, `-seedybar`, `-seedgate`, `-noladder`, `-seedjoh`, `-seedb2`) and
the writers (`-writeres`, `-writeinp`) --, `-m 2`, the `.inp` route, and the
rank test's and `-rungs`' combinations that the C has and the port does not
yet (docs/PORT.md §10).

## Documentation and changes

- [CHANGELOG.md](https://github.com/davidesg/drvec-python/blob/main/CHANGELOG.md)
- [docs/API.md](https://github.com/davidesg/drvec-python/blob/main/docs/API.md): the API, the accessors, the engines
- [docs/PORT.md](https://github.com/davidesg/drvec-python/blob/main/docs/PORT.md): how the C was ported
- [TODO.md](https://github.com/davidesg/drvec-python/blob/main/TODO.md): the milestones and the open risks
- [bugs/](https://github.com/davidesg/drvec-python/tree/main/bugs): the C defects found by the port, each with its repro

## License

GPL-2.0-or-later (see
[COPYING](https://github.com/davidesg/drvec-python/blob/main/COPYING)).

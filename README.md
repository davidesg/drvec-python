# drvec

VARMA-VEC cointegration models by **exact maximum likelihood** (Mauricio 2006):
the Python port of the C program drvec of the ATSW suite.

**Status: in development.** The port is homologated, case by case, against
its oracle, the C program at the tag `drvec-v0.12` of
[atsw-gui](https://github.com/davidesg/atsw-gui): the log-likelihood to 1e-6
and the estimates to 1e-4.

## Design

Like drtran's port, drvec reuses what its siblings already have:

- the univariate models come as one fue `.pre` file per series, read with
  `fue.load()`;
- the exact VARMA likelihood (AS 311) and the optimiser are drvarma's;
- the only new piece is the **VEC cast**: from the error-correction
  parameters (α, β, Γ, Θ, Σ and the deterministic terms) to the stationary
  VARMA the likelihood evaluates.

drvec is pure Python; the compiled engine arrives with drvarma's binary
wheels. A compiled cast of its own is added only if the parametric bootstraps
need it.

The assistant that drives drvec is **sima** (sima-tseries), from its 0.2.0.

## License

GPL-2.0-or-later.

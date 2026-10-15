# drvec-python — TODO

## Milestone 1 — the cast, one fit, one bootstrap timed

- [ ] Evaluate the exact log-likelihood at the oracle's estimates (`.pre`
      route, muskrat and mink, p = 2, q = 1, r = 1, `-mean -case 2`).
- [ ] The full fit of that case, homologated against `tests/data/oracle/proute.out`.
- [ ] The rank test with the parametric bootstrap (`-lrtest -bootstrap 20`),
      replication by replication against `rankboot.boot`, and its time against
      the C's. This decides whether the cast needs C (option b).

## The port's first version (scope decided 2026-10-11)

Cases 1-4, the `.pre` route with `-case pre`/`-nodrift`, the rank test,
the impulse responses and variance decomposition with `-irfboot`, `-f`,
`-fixb2row`, `-xpre`/`-xlink` and `-xsys`, `-alpha`, `-weakex`, `-fixb2`,
`-fdhess`. The gaps the C declares are ported as refusals, with the C's
message.

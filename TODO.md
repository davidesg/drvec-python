# drvec-python — TODO

## Milestone 1 — the cast, one fit, one bootstrap timed

- [x] Evaluate the exact log-likelihood at the oracle's estimates (`.pre`
      route, muskrat and mink, p = 2, q = 1, r = 1, `-mean -case 2`): equal to the
      C's at every est() optimum (tests/test_loglik.py).
- [x] The full fit of that case, homologated against `tests/data/oracle/proute.out`:
      the 24 est() calls in the C's order, logL, every estimate and s.e. to the
      printed digits, the ladder, the gate, the profile, the search table
      (tests/test_fit_case1.py, tests/test_trajectory.py).
- [x] The rank test with the parametric bootstrap (`-lrtest -bootstrap 20`),
      replication by replication against `rankboot.boot` (tests/test_rank.py), and
      its time against the C's (`tools/time_bootstrap.py`): 5.7 s against 0.48 s,
      ~12x; at that ratio the article's bootstraps (13, 89, 23 min in C) would take
      ~2.6, ~17.5 and ~4.5 h. Decision input for option (b), a cast in C.

## Open risks (docs/PORT.md)

- Johansen's seed uses scipy's `eigh` (decision D2): a few ulps off GSL's
  gensymmv. Measured harmless on both cases (no winner changes; 2 of 20 LR* off by
  ~1e-13), but on another data set a near-tie between `johansen` and another start
  could flip the winner, and then the estimates move by up to the flatness of the
  likelihood (1e-4 here). If that is ever seen: call libgsl through ctypes (checked
  bit-exact on 2026-10-15).
- The case-1 optimum is on the MA invertibility wall (BUG-49) and depends on the
  bit-exact trajectory of one start: any change to the cast's arithmetic (numpy
  products, a different Cholesky) must re-run tests/test_fit_case1.py.
- `nrerror` (the line search's ROUNDOFF) is an exception, not an exit (D3).
- The `.pre` reader: deterministic variables and D > 0 are refused; alignment is
  the C's (date intersection), not drvarma's ladder rule.
- C defect candidate, not filed: the bootstrap generator's `unsigned long`
  (drvec.c:2203) is 32-bit on Windows, so the C's bootstrap stream (and its
  `.boot` goldens) are platform-dependent. The port computes the 64-bit stream.
- The suite was run against drvarma 0.2.0 and fue 0.1.17.dev0 installed as
  EDITABLE dev trees; re-run against the released wheels.

## The port's first version (scope decided 2026-10-11)

Cases 1-4, the `.pre` route with `-case pre`/`-nodrift`, the rank test,
the impulse responses and variance decomposition with `-irfboot`, `-f`,
`-fixb2row`, `-xpre`/`-xlink` and `-xsys`, `-alpha`, `-weakex`, `-fixb2`,
`-fdhess`. The gaps the C declares are ported as refusals, with the C's
message.

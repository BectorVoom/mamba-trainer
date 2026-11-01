# Mamba-CLM contracts

The exact rules shared by the Rust code (`finance/crates/*`), the Lean 4 model
(`finance/lean/ClmVerify`) and the tests that compare them. If code and this file
disagree, one of them is a bug; fix the code or change this file on purpose.

All indices are **trading-day indices** `t = 0 .. N-1` on one common calendar (the
trading days of the market index, SPY). Every security uses the same calendar; a
day without a bar for a security is *missing* for that security.

## C1. Bars and information time

For security `s` and day `t` a bar is `(open, high, low, close, adj_close, volume)`.
Prices used by features and labels are **adjusted**: `f = adj_close / close`, and
`O = open*f, H = high*f, L = low*f, C = adj_close`. Only scale-invariant quantities
(ratios, log ratios, z-scores) of adjusted prices are features; no price level is
ever a feature (the adjustment factor of a past bar depends on later dividends and
splits, but ratios inside one adjustment regime do not).

* The bar of day `t` is known **after the close of day `t`**.
* A decision for day `t` is made after the close of `t`, using bars `<= t` only.
* It is executed at the **open of day `t+1`** (`O[t+1]`).

## C2. Features

`features(s, t)` may read bars of days `t - K .. t` for a fixed maximum lookback
`K` (`K = 252` for the feature set below), of security `s`, of the market index and
of the security's sector ETF, and the same-day (`t`) features of other securities
(cross-sectional z-scores). It never reads a day `> t`.

A feature row is *valid* iff every bar it reads exists. The model input of a sample
is the window of feature rows `t - W + 1 .. t` (`W` = sequence length); the sample's
input is valid iff all `W` rows are valid.

Feature set v1 (`F = 20`), all `f32`:

| # | name | definition (log = natural log) |
|---|------|-----------|
| 0 | r1 | `log(C[t]/C[t-1])` |
| 1 | r5 | `log(C[t]/C[t-5])` |
| 2 | r20 | `log(C[t]/C[t-20])` |
| 3 | r60 | `log(C[t]/C[t-60])` |
| 4 | gap | `log(O[t]/C[t-1])` |
| 5 | intraday | `log(C[t]/O[t])` |
| 6 | range | `log(H[t]/L[t])` |
| 7 | rv20 | population std of `r1` over `t-19..t`, times `sqrt(252)` |
| 8 | rv60 | same over `t-59..t` |
| 9 | relvol | `log(V[t] / mean(V[t-19..t]))`, `V = volume` (0 volume: row invalid) |
| 10 | mom | `log(C[t-20]/C[t-120])` |
| 11 | hi52 | `log(C[t] / max(H[t-251..t]))` |
| 12 | amihud | `log(1e-12 + mean_{k=t-19..t} |r1[k]| / (close[k]*V[k]))` — **reported** (unadjusted) close: the traded value. The adjusted level `C[k]` depends on later dividends (C1), so it is not used in a level-dependent denominator. |
| 13 | mkt_r1 | `r1` of the market index |
| 14 | mkt_r20 | `r20` of the market index |
| 15 | mkt_rv20 | `rv20` of the market index |
| 16 | sec_r1 | `r1 - r1` of the sector ETF |
| 17 | sec_r20 | `r20 - r20` of the sector ETF |
| 18 | xs_r20 | cross-sectional z-score of `r20` over securities valid on day `t` (0 if fewer than 3) |
| 19 | xs_relvol | cross-sectional z-score of `relvol` likewise |

## C3. Labels (horizon `h >= 1`)

For a decision at day `t`:

* `label_start(t) = t + 1`, `label_end(t) = t + 1 + h`.
* forward return `r_h(t) = log(O[t+1+h] / O[t+1])` (enter at the next open, exit at
  the open `h` days later).
* path low `mae_h(t) = log(min(L[t+1 .. t+h]) / O[t+1])` (max adverse excursion).
* The label exists iff every bar of `t+1 .. t+1+h` exists.

Classes (position-independent target exposures, long-only):

```
theta(t) = 2*kappa + m * sigma_h(t)        sigma_h(t) = rv20(t) * sqrt(h / 252)
BUY  (0)   if r_h(t) >  theta(t)           be long over the horizon
SELL (1)   if r_h(t) < -theta(t)           be flat over the horizon
HOLD (2)   otherwise                       the edge does not pay for a trade
```

`kappa` is the one-way cost (fraction), `m >= 0` a margin multiplier (default 0).
Auxiliary targets: `ret = r_h / max(sigma_h, 1e-4)` (vol-scaled return), and
`stop = 1[mae_h(t) < -stop_frac]`.

Leakage facts the Lean model proves: every index a feature reads is `<= t`; every
index a label reads is `>= t + 1`; `label_end(t) = t + 1 + h`.

## C4. Walk-forward folds

Inputs: the calendar's year of each day `year[t]` (non-decreasing), `N`, and

```
FoldConfig { scheme: Expanding | Rolling { train_days },
             first_test_year, last_test_year,
             val_days, cal_days, horizon h, embargo e >= 0, min_train_days }
```

For each test year `Y` in `first_test_year ..= last_test_year` (in order):

```
test  = [ts, te)   ts = first t with year[t] == Y,  te = first t with year[t] > Y (or N)
cal   = [ts - cal_days, ts)
val   = [cal.start - val_days, cal.start)
train = [0, val.start)                                  Expanding
        [max(0, val.start - train_days), val.start)     Rolling
```

A fold is *emitted* only if year `Y` occurs in the calendar, `ts >= cal_days + val_days`
and `train.len() >= min_train_days`; otherwise it is skipped (not an error).

### Purging and embargo

Segments are ordered `train < val < cal < test`. A sample at decision day `t` in a
segment with a **next** segment starting at `s_next` is kept iff

```
label_end(t) + e < s_next          i.e.   t + 1 + h + e < s_next
```

(strict: no price a kept label reads lies inside a later segment, and `e` more days
of gap). Test samples (no next segment) are kept iff `label_end(t) <= N - 1`. A
sample is also dropped when its input window or its label is invalid (C2, C3) —
that is a data rule, independent of the fold.

Properties (proved in Lean for the model, tested in Rust):

1. `train`, `val`, `cal`, `test` are pairwise disjoint, contiguous, in that order, and
   inside `[0, N)`.
2. For a kept sample `t` of a segment and **any** later segment `S` of the same
   fold, every label index of `t` is `< S.start - e <= ` every decision day of `S`.
3. Test segments of different folds are pairwise disjoint and increasing.
4. Everything fitted on a fold (normalisation, weights, early stopping, temperature,
   thresholds) uses only samples kept in `train`, `val` or `cal` of that fold.
5. Expanding: `train(k) ⊆ train(k+1)`.

### Normalisation

Per fold, per feature: `mu, sd` over the feature rows of days in `train`
(days, not samples: row `t` for `t` in `train`, valid rows of every security),
`sd` floored at `1e-6`; every input row is mapped to `clamp((x - mu)/sd, -8, 8)`.
Rows of later days are transformed with the same `mu, sd`.

## C5. Policy and backtest

Per security, the deterministic policy turns the action of decision day `t` into the
position held from the open of `t+1` to the open of `t+2`:

```
p_before(first decision day of the segment) = 0
p(t) = 1 if a(t) = BUY,  0 if a(t) = SELL,  p(t-1) if a(t) = HOLD
```

(`p(t)` is the position chosen at decision `t`; `p(t-1)` the one before it.) A
security whose **input** window is invalid on a decision day (C2) gets HOLD. Whether
the day's *label* exists (C3) must not influence trading: it depends on bars after
`t` (a halt, a delisting), so the model is run on every input-valid test day, labelled
or not; labels only gate the prediction-quality metrics. If `O[t+1]` or `O[t+2]` is
missing, `gross(t) = 0` (the position is still recorded and costs are still charged).
Decision days run over the test segment `[ts, te)`; the P&L of decision `t` may read
`O[t+2]` past `te` (it is reported on day `t+1`). If `t + 2 > N - 1` the decision is
not traded (`gross = cost = 0`, position unchanged).

Decision rule (identical for every model): from the calibrated probabilities,
`edge = p_buy - p_sell`; BUY if `edge > tau`, SELL if `edge < -tau`, else HOLD.
The **primary** policy uses one fixed `tau` (`policy_tau`, 0 in the study) for
every model and fold. A **secondary** policy, reported separately, chooses `tau`
per fold and model from a fixed grid by the Sharpe of this policy on the kept part
of the calibration segment (whose prices all precede the test segment). (A first version vetoed trades whenever HOLD was the most likely class;
since HOLD is the most frequent class, every model then bought once and held —
found on a two-fold smoke run before the main study, and replaced.)

Accounting, per security, all as simple (not log) returns, cost `kappa` one way:

```
trade(t)   = |p(t) - p(t-1)|                     executed at O[t+1]
gross(t)   = p(t) * (O[t+2] / O[t+1] - 1)        accrues on day t+1
cost(t)    = kappa * trade(t)
net(t)     = gross(t) - cost(t)
```

Portfolio: `N_s` securities, one sleeve each of weight `1/N_s` (positions are therefore
capped at `1/N_s` of capital per name, gross exposure `<= 1`). Portfolio day return
`R(d) = (1/N_s) * sum_s net_s(t)` for the decision day `t = d - 1`.
Turnover per day `= (1/N_s) * sum_s trade_s(t)`.

Facts the Lean model proves: execution day `t+1 > t`; every price a decision's
P&L reads has index `>= t+1`; `sum net = sum gross - kappa * sum trade`;
`p(t) ∈ {0,1}`; HOLD never trades.

Reported metrics on the concatenated daily portfolio returns `R`:
annualised return `252 * mean(R)`, Sharpe `sqrt(252) * mean/std` (population std),
Sortino `sqrt(252) * mean / sqrt(mean(min(R,0)^2))`, max drawdown of the compounded
equity `prod(1+R)`, profit factor `sum(R>0) / |sum(R<0)|`, mean daily turnover,
total net return.

## C6. Statistical comparison

* Stationary bootstrap (Politis-Romano) of daily returns, mean block length `20`,
  `B = 2000`, fixed seed; percentile 95% intervals. Paired: the same resampled
  indices for both strategies, giving intervals for `Sharpe(A) - Sharpe(B)` and
  `mean(A - B)`.
* Paired fold comparison: per test year, Sharpe and log loss of A and B; count of
  years A wins; mean and sd of the per-year difference.
* Cost sensitivity: kappa in {0, 2.5, 5, 10, 20} bps one way.
* Regimes: days split by the market index's `rv20` at the decision day above or below
  its median over the *training* days of the fold (so the split is point-in-time).

## C7. Selective scan (Mamba-1)

Shapes: `x, dt : [B, T, D]`, `A : [D, N]`, `Bm, Cm : [B, T, N]`, `Dskip : [D]`.

```
h[b,-1,d,n] = 0
h[b,t,d,n]  = exp(dt[b,t,d] * A[d,n]) * h[b,t-1,d,n] + dt[b,t,d] * Bm[b,t,n] * x[b,t,d]
y[b,t,d]    = sum_n Cm[b,t,n] * h[b,t,d,n] + Dskip[d] * x[b,t,d]
```

`dt > 0` (a softplus output), `A < 0` (`A = -exp(A_log)`), so `0 < exp(dt*A) < 1`.

Element-wise this is the affine recurrence `h_t = a_t h_{t-1} + u_t`. Pairs
`(a, u)` compose as `(a1,u1) ⊕ (a2,u2) = (a2*a1, a2*u1 + u2)` (apply 1 then 2),
which is associative; any chunking of time therefore gives the sequential result
(the chunked kernel: per chunk, a local scan from zero and the product of `a`; a
sequential pass over chunk summaries for the carried state; a local re-scan from
the carried state). Adjoint: with `gy = dL/dy`,

```
g[t] = Cm[t] * gy[t] + a[t+1] * g[t+1]        g[T] = 0          (per b,d,n)
dL/du[t]   = g[t]
dL/da[t]   = g[t] * h[t-1]
dx  += sum_n g * dt * Bm   + Dskip * gy
ddt += sum_n g * (A * a * h[t-1] + Bm * x)
dA  += sum_{b,t} g * dt * a * h[t-1]
dBm += sum_d g * dt * x
dCm += sum_d gy * h
dDskip += sum_{b,t} gy * x
```

The chunked forward's chunk summary is the running product of the per-step
factors `exp(dt*A)` (not `exp(A*sum dt)`), so it rounds like the sequential pass.
The backward kernel walks checkpoint segments last to first, recomputing each
segment's states from its checkpoint, then runs the reverse recurrence through it.

The Lean model proves associativity, chunked = sequential for every chunk size,
causality (`y[t]` is unchanged by any change of inputs after `t`), and the adjoint
identity `sum_t gy[t]*Cm[t]*h[t] = sum_t g[t]*u[t] + g[0]*a[0]*h[-1]` (so `g` is the
exact gradient of the loss with respect to `u`).

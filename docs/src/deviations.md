# Deliberate deviations

SAShE implements the algorithms in the [References](@ref) as closely as it can. Where it
departs from one of them on purpose, it is listed here, so a difference from a paper is not
mistaken for a bug. A deviation need not be a departure from *every* reference — the papers
sometimes disagree with each other, in which case this page says which one SAShE follows and
which it departs from. It states *what* differs and *why*; it makes no claim about which
choice is better.

## Effects are returned unnormalized

The references differ here. [1]'s Eq. (14) divides by `Var(Y)`, so its Shapley effects sum
to 1, while [2] and [4] do not: [2]'s Algorithm 1 ends at `Shᵢ = Shᵢ/m` and its Eq. (10)
states `Σᵢ Shᵢ = Var[Y]` as a property of the definition, and [4] likewise ends at
`Σⱼ νⱼ = σ²` (§3.2). SAShE returns effects unnormalized, summing to `Var(Y)` — matching [2]
and [4] directly, and departing from [1] only.

Reason: the normalized form is one division away, and `sum(Φ)` vs. `var(Y)` is meaningful to
look at either way — though what it actually tells you depends on the estimator:

- For [`CallableModel`](@ref) + pick-and-freeze, it's a genuine (if noisy) check on the
  estimator: the walk telescopes through real `func` evaluations end to end, so a mismatch
  reflects the estimator not having converged yet — more base samples narrows it.
- For [`DataModel`](@ref), [`MixModel`](@ref), and [`CallableModel`](@ref) + double Monte
  Carlo, the walk's terminal step is a `var(Y)` computed directly from data rather than from
  the estimator being tested, so the telescoping sum equals that `var(Y)` **exactly** by
  construction — regardless of whether the intermediate steps are correct. It confirms the
  aggregation is wired correctly, not that the estimator itself is.

```julia
Φ_normalized = Φ ./ var(Yₙ)
```

## `DataModel` requires its estimator explicitly

There is no default estimator for `analyze(::DataModel, n, estimator)`. The papers
do not prescribe one; requiring it keeps the choice visible at the call site rather than
implied.

## Quasi-Monte Carlo sampling is an extension, not from the papers

`QuasiMonteCarloSampling` lets factor values be drawn from a stratified point set
rather than by i.i.d. Monte Carlo. None of [1]–[5] use this, and [4]'s Remark 1 notes that
quasi-Monte Carlo is hard to apply to Shapley-effect estimation, because generating the random
permutation breaks the smoothness that quasi-Monte Carlo theory relies on.

It is available for experimentation. The documented workflows all use
`MonteCarloSampling`, and no accuracy advantage over it has been measured here. Only
algorithms that randomize each dimension independently are accepted — see the type's docstring
for the measured reason.

## Confidence intervals assume independent permutations, which the nearest-neighbour estimators don't satisfy

`SAShE.margin_of_error` (and so `SAShE.confint`) treats each permutation's contribution —
each column of `Φₙ` — as an independent draw. For [`CallableModel`](@ref) that holds: every
base sample is drawn fresh, and [4]'s Theorem 1 gives the matching unbiased variance
estimator.

It does **not** hold for [`DataModel`](@ref) or [`MixModel`](@ref), and [1] says so directly
(§6.2): *"Although Assumption 1 does not hold with the estimators `Ê_{u,MC}` and `V̂_{u,PF}`
(the summands of these estimators are not independent), we keep choosing `N_u = N_O = 1`"*.
Every permutation's `Ŵ` is computed from the same fixed dataset, so the columns are positively
dependent. SAShE follows [1] in using the procedure anyway, but reports the interval with the
same formula, which makes it **anti-conservative** there: true coverage is below the nominal
95%.

The gap widens as `n_permutations` grows relative to the dataset size. The reported interval
shrinks like `1/√n_permutations`, while the nearest-neighbour approximation's own error floor
(`O(N^{-1/(2|u|)})`, [1]'s Corollary 2) depends only on the dataset and does not shrink at
all — a formula that only sees scatter *across* permutations cannot see that floor. Treat the
interval as a lower bound on uncertainty, and add permutations to stabilize the estimate
rather than to earn a tighter interval.

## The nearest-neighbour search excludes the query point

[1] §6 defines `k^v_N(l, n)` over the full sample *including* `l`, so its first "nearest
neighbour" of `l` is `l` itself. `SAShE`'s internal search excludes `l` and its callers add it
back where the estimator needs it (see [1]'s Remark 20). The estimated quantity is identical;
only the indexing convention differs.

## Aggregation over coalitions

Only the random-permutation W-aggregation procedure is implemented — originally [2]'s, in the
form [1] states it (§4.2). [1]'s own subset W-aggregation procedure (§4.1) is not.

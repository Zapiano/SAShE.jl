# Deliberate deviations

SAShE implements the algorithms in the [References](@ref) as closely as it can. Where it
departs from them on purpose, it is listed here, so a difference from a paper is not mistaken
for a bug. This page states *what* differs and *why*; it makes no claim about which choice is
better.

## Effects are returned unnormalized

[1]'s Eq. (14) and [2]'s Algorithm 1 (step 5) both divide by `Var(Y)`, so their Shapley
effects sum to 1. SAShE returns them unnormalized, summing to `Var(Y)` instead.

Reason: it keeps `sum(Φ) ≈ var(Y)` usable as a sanity check on a run, and the normalized form
is one division away.

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

## The nearest-neighbour search excludes the query point

[1] §6 defines `k^v_N(l, n)` over the full sample *including* `l`, so its first "nearest
neighbour" of `l` is `l` itself. `SAShE`'s internal search excludes `l` and its callers add it
back where the estimator needs it (see [1]'s Remark 20). The estimated quantity is identical;
only the indexing convention differs.

## Aggregation over coalitions

Only [1]'s random-permutation W-aggregation procedure is implemented. Its subset
W-aggregation procedure is not.

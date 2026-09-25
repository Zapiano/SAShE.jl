# Next steps

This documentation is an initial pass: the home page, [Getting started](@ref), and
[How it works](@ref). Everything below is planned but not yet written — notes for the next
session.

## Pages to add

### Workflows (how-to)

Promote the "model and sample, always together" section of [How it works](@ref) to its own
page. `analyze` always takes a model and a sample built independently — `CallableModel`
never holds sample data, and the sample never holds the model:

| You build | `analyze` runs | Use when |
| :-- | :--------- | :------- |
| `CallablePickAndFreezeSample(X1, X2)` | pick-and-freeze — `analyze(model, S)` | independent factors, known distribution |
| `CallablePickAndFreezeSample(names, n, dists, MonteCarloSampling())` | pick-and-freeze, with `A`/`B` drawn for you | you have per-factor distributions rather than pre-built `X1`/`X2` |
| `CallablePickAndFreezeSample(X1, X2; conditional_sampler=...)` | pick-and-freeze, dependent factors | some factors depend on others |
| `CallablePickAndFreezeSample(X, perms)` | pick-and-freeze over an already-built sample table | any other custom sampling scheme, e.g. QMC-derived permutations |
| `CallableDoubleMonteCarloSample(names, dists, m, strategy)` | double Monte Carlo — `analyze(model, S)` | independent factors, want lower variance per evaluation |

### Dependent factors (how-to) — the important one

- **Why the estimator assumes independence.** It relies on `(x_u, y₋ᵤ) ∼ ρ`, which only
  holds when the factors are independent. With a dependency (say `x3 = x1 + x2`) the
  swapped rows of `Z` land off the constraint and you silently estimate a different
  problem.
- **The `conditional_sampler` contract**, argument by argument:
  `conditional_sampler(X1_param_idx, X2_param_idx, base_row, noise_row)` returns the values
  for the factors in `X2_param_idx`, drawn conditional on the `X1_param_idx` factors being
  fixed at their `base_row` values.
- **How to use `noise_row`** (the part that trips people up):
  - `base_row` is the frozen `X1` values; `noise_row` is an independent `X2` draw.
  - a factor being resampled with no active constraint → just take its `noise_row` value,
    it already has the right marginal.
  - a constrained factor → compute it from the others; if it still has a free direction,
    remap a `noise_row` value into the conditional support (`Uniform(a, b) → Uniform(0, 1)
    → Uniform(lb, ub)` is two affine rescalings).
  - Why not `rand()`? Then the result is no longer reproducible from `(samples,
    permutations)` alone, and every `pmap` worker needs its own seed.
- **Full worked example**: `x3 = x1 + x2`, all six non-trivial frozen/resampled cases,
  ending in the `conditional_sampling` function from `test/conditional.jl`.
- **The two equivalent workflows** (build `Z` yourself vs. pass `conditional_sampler`),
  with the exact-equivalence test as evidence.
- **Caveat** from [4] §5.2 (see [References](@ref)): independent conditional draws at each
  permutation step can inflate the confidence intervals.
- **Pointer**: for correlated-but-not-deterministic inputs, the general recipe is marginal
  distributions + a copula. Not built in.

### Reproducibility (explanation)

Seeding, `Distributed` / `pmap` and per-worker RNG, why deterministic conditional samplers
matter, and custom sampling schemes via `CallablePickAndFreezeSample(samples, perms)` — build the
table yourself, then wrap it directly.

## Infrastructure

- **GitHub Pages deploy**: add `.github/workflows/documentation.yml` (Documenter's standard
  job), create an empty `gh-pages` branch, enable Pages in the repo settings. `deploydocs`
  is already wired up in `docs/make.jl`; it needs either a `DOCUMENTER_KEY` secret or the
  default `GITHUB_TOKEN` with Pages write permission.
- `docs/build/` is git-ignored; `docs/Manifest.toml` is tracked (pinned docs-environment
  versions, including `LiveServer` for local live-reload previews via `servedocs()`).
- Consider **Literate.jl** for the dependent-factors tutorial, so the example is a runnable
  script that doubles as a test.

## Housekeeping noticed along the way

- `README.md` is stale: it still refers to `SAShE.Problem` and `SAShE.solve`, which are now
  `CallableModel` and `analyze`. Either trim it to point at these docs or update it.
- Consider exporting `shapley_effects`, `confint`, and `margin_of_error` — they are part of
  the normal workflow but currently need the `SAShE.` prefix.

## Building the docs locally

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The rendered site lands in `docs/build/`.

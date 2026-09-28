# Estimators

SAShE.jl implements two *estimation methods* for the Shapley effects — `PickAndFreeze` and `DoubleMonteCarlo` — across the three *data settings* (`CallableModel`, `MixModel` and `DataModel`). Each *estimation method* follows the same logic in every *data setting*, but the details differ, because what each container has available differs: a callable model and a known input distribution, a callable model and only a dataset, or a dataset alone. Not every combination is implemented yet — see Table 1 on the [home page](index.md).

<!--
TODO
## Pick-And-Freeze

## Double Monte Carlo
-->

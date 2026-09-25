# Naming conventions

Conventions for identifiers, spelling, and citations in this package's code.

- **Never abbreviate "nearest neighbour" to "NN".** "NN" is heavily
  overloaded with "neural network" in this field. Spell it out (`_nearest_neighbour_indices`, `_nearest_neighbour_pick_freeze`, etc).
- **UK spelling** ("neighbour," not "neighbor") is used throughout this package's
   code, for consistency.
- **Cite papers by number** (`[1]`, `[2]`, ...) in docstrings and comments, never by
  embedding the full citation — the [References](@ref) page is the single source of truth
  for what a number means.
- For the mathematical object a given identifier represents (notation, paper source, plain
  description), see that identifier's own docstring rather than looking for a separate
  index — e.g. `_nearest_neighbour_pick_freeze`'s docstring gives its notation (`V̂^knn_{u,s,PF}`),
  citation (`[1]`, §6.1.2, Eq. 23), and a plain-English description together.
- **Sample types are named `<Container><Method>Sample`.** The prefix names the model
  container the sample pairs with (`CallableModel` → `Callable...`, `MixModel` → `Mix...`),
  and the middle name identifies the estimation method (`PickAndFreeze`, `DoubleMonteCarlo`).
  No container gets to omit its prefix, even where it reads redundantly (e.g.
  `CallablePickAndFreezeSample`, not `PickAndFreezeSample`) — the point is that every sample
  type name answers "which container is this for" on sight, without having to remember which
  one is the unprefixed exception.

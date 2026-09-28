# Concepts

Here we assume the existence of a model $f(x_1, ..., x_N): \mathbb{R}^N \rightarrow \mathbb{R}$ of a random variable $X = (X_1, ..., X_N)$ and we are interested in determining how much each variable $X_j$ contributes to the variance in $f$.

## tl dr;
- Historically, the most widespread method for performing variance-based sensitivity analysis uses Sobol' *main effect* and *total effect* indices.
- Sobol' indices are derived from ANOVA decomposition and require the inputs to be independent.
- Another property of Sobol' indices is that the sum of all *main effects* is not guaranteed to be equal to the sum of all *total effects* and neither of these sums is guaranteed to be equal to the total model variance.
- With the Game Theory concept of *Shapley values*, we can define a new family of indices (called *Shapley effects*) that have the following desired properties:
  - They sum up to the model's total variance
  - They can be compound (you can sum the *Shapley effects* for two variables to get their share in the model's total variance)

<!-- TODO: complete this page-->

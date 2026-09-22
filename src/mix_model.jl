"""
    MixModel(func::Function, X::DataFrame, Y::Vector)

Container for the "callable model, plus an existing dataset" case: a nearest-neighbour
lookup into `(X, Y)` picks *where* to evaluate `func`, rather than reusing a stored `Y`
value. Estimation logic not yet implemented — this type exists so callers can construct it,
but no `analyze` method is defined for it yet.

# Arguments
- `func` : A function that accepts a vector of inputs as argument.
- `X` : Existing sample of inputs, one row per observation, one column per factor.
- `Y` : Corresponding outputs, `Y[n]` is the already-known output for row `n` of `X`.
"""
struct MixModel
    func::Function
    X::DataFrame
    Y::Vector
end

"""
    AbstractMargin

Abstract supertype for all feature-margin models.
Each concrete subtype (e.g. [`GaussianMargin`](@ref), [`PoissonMargin`](@ref),
[`GammaMargin`](@ref), [`MultinomialMargin`](@ref)) stores the prior hyperparameters,
the variational posterior parameters (one set per cluster), and the fixed background
parameters estimated from the full dataset.
"""
abstract type AbstractMargin end

"""
    ModelSetting

Abstract supertype for the two feature-relevance models.
Use [`SFRM`](@ref) or [`LFRM`](@ref) as the `model_setting` argument to [`mixClust`](@ref).
"""
abstract type ModelSetting end

"""
    SFRM()

**Shared Feature Relevance Model.** Each feature has a single relevance
probability shared across all clusters. The relevance indicator `sⱼ` follows
a Bernoulli distribution with a Beta-distributed probability common to every cluster.

Use this model as a fast baseline or when the expectation is that a feature is
either universally informative or universally uninformative.

See also: [`LFRM`](@ref), [`mixClust`](@ref).
"""
struct SFRM <: ModelSetting end

"""
    LFRM()

**Local Feature Relevance Model.** Each feature–cluster pair `(j, k)` has its own
relevance probability, so a feature can be relevant in some clusters and irrelevant
in others. The relevance indicator `sᵢⱼ` depends on the cluster assignment of
individual `i`.

Use this model when you expect subgroup-specific signal: for example, a biomarker
that separates two out of three subtypes but is uninformative for the third.

See also: [`SFRM`](@ref), [`mixClust`](@ref).
"""
struct LFRM <: ModelSetting end

"""
    MixClustResult

Result returned by [`mixClust`](@ref) (and [`prune_and_merge_clusters`](@ref)).

# Fields

| Field | Type | Description |
|:--- |:--- |:--- |
| `w` | `Matrix{Float64}` (n × K̂) | Soft cluster assignments; each row is a probability vector summing to 1 |
| `labels` | `Vector{Int}` (length n) | Hard assignment: `labels[i] = argmax(w[i, :])` |
| `pip` | `Matrix{Float64}` (n × p) | Posterior Inclusion Probabilities `γᵢⱼ ∈ (0,1)` |
| `alpha_star` | `Vector{Float64}` (length K̂) | Variational Dirichlet parameters `α★ₖ = α₀ + Σᵢ wᵢₖ` |
| `delta_star` | `Array{Float64}` | Variational Beta parameters for relevance: shape `(p,2)` under SFRM, `(K̂,p,2)` under LFRM |
| `margins` | `Vector{AbstractMargin}` (length p) | Fitted margin objects, one per feature |
| `elbo_history` | `Vector{Float64}` | ELBO value at each CAVI iteration (useful for convergence diagnostics) |

# Accessing results

```julia
results = mixClust(dataset, 10)

K_hat  = size(results.w, 2)           # estimated number of clusters
labels = results.labels                # hard assignments
eig    = compute_eig(results.margins, results.w, results.pip)
```
"""
struct MixClustResult
    w::Matrix{Float64}              # n × K soft cluster assignment probabilities
    labels::Vector{Int}             # MAP cluster assignment: argmax of each row of w
    pip::Matrix{Float64}            # n × p Posterior Inclusion Probabilities γᵢⱼ
    alpha_star::Vector{Float64}     # K variational Dirichlet parameters α★
    delta_star::Array{Float64}      # Beta parameters for relevance: (p,2) SFRM or (K,p,2) LFRM
    margins::Vector{AbstractMargin} # fitted margin objects, one per feature
    elbo_history::Vector{Float64}   # ELBO history across CAVI iterations
end

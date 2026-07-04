using SpecialFunctions
using Statistics

# Interface functions that must be implemented by each concrete margin subtype.

"""
    update_margin!(margin::AbstractMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)

Updates the variational parameters of the feature `margin` based on the observed data vector `y_j` (length n),
the cluster assignment weights matrix `w` (n x K), and the feature inclusion probabilities `gamma_j` (length n).
"""
function update_margin!(margin::AbstractMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    error("update_margin! not implemented for type $(typeof(margin))")
end

"""
    expected_log_density(margin::AbstractMargin, y_j::AbstractVector) -> Matrix{Float64}

Computes the expected log-density of the data vector `y_j` (length n) under the variational posterior of the cluster-specific parameters.
Returns an n x K matrix where the (i, k)-th entry is E_q[ ln f(y_{i,j} | alpha_{k,j}) ].
"""
function expected_log_density(margin::AbstractMargin, y_j::AbstractVector)
    error("expected_log_density not implemented for type $(typeof(margin))")
end

"""
    background_log_density(margin::AbstractMargin, y_j::AbstractVector) -> Vector{Float64}

Computes the log-density of the data vector `y_j` (length n) under the fixed background parameters beta_j.
Returns a vector of length n where the i-th entry is ln f(y_{i,j} | beta_j).
"""
function background_log_density(margin::AbstractMargin, y_j::AbstractVector)
    error("background_log_density not implemented for type $(typeof(margin))")
end

"""
    expected_kl_divergence(margin::AbstractMargin) -> Vector{Float64}

Computes the expected KL divergence between the cluster-specific densities and the global background density.
Returns a vector of length K where the k-th entry is E_q[ KL( f(· | alpha_{k,j}) || f(· | beta_j) ) ].
"""
function expected_kl_divergence(margin::AbstractMargin)
    error("expected_kl_divergence not implemented for type $(typeof(margin))")
end

"""
    predictive_density(margin::AbstractMargin, y_new::AbstractVector) -> Matrix{Float64}

Computes the marginal predictive density of new observations under the variational parameter posterior.
Returns an n_new x K matrix where the (i, k)-th entry is E_{q(alpha_j)}[ f(y_new_i | alpha_{k,j}) ].
"""
function predictive_density(margin::AbstractMargin, y_new::AbstractVector)
    error("predictive_density not implemented for type $(typeof(margin))")
end

"""
    update_background!(margin::AbstractMargin, y_j::AbstractVector, gamma_j::AbstractVector)

Updates the background parameter beta_j based on the observed data vector `y_j` (length n)
and the feature inclusion probabilities `gamma_j` (length n) using a weighted M-step.
"""
function update_background!(margin::AbstractMargin, y_j::AbstractVector, gamma_j::AbstractVector)
    error("update_background! not implemented for type $(typeof(margin))")
end



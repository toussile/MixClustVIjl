module MixClustVIjl

# Import dependencies
using DelimitedFiles
using SpecialFunctions
using Statistics
using LinearAlgebra
using Random
using Plots

# Include files
include("types.jl")
include("margins.jl")
include("margins/gaussian.jl")
include("margins/multinomial.jl")
include("margins/poisson.jl")
include("margins/gamma.jl")
include("cavi.jl")
include("post_hoc.jl")
include("visualization.jl")
include("datasets.jl")

# Export types
export AbstractMargin, ModelSetting, SFRM, LFRM, MixClustResult
export GaussianMargin, MultinomialMargin, PoissonMargin, GammaMargin

# Export functions
export mixClust, compute_eig, filter_features
export predict_proba, predictive_log_likelihood, prune_and_merge_clusters
export plot_elbo, plot_pips, plot_eig, plot_assignments, plot_profiles
export plot_cluster_sizes, plot_assignment_confidence, plot_local_pips

# Export bundled datasets
export load_heart_disease, simulate_synthetic_cohort

end # module MixClustVIjl

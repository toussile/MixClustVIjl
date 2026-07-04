# MixClustVIjl

**Bayesian mixture model clustering for heterogeneous data via Variational Inference.**

`MixClustVIjl` fits finite mixture models to datasets containing any combination of
continuous, count, and categorical variables, using Coordinate Ascent Variational
Inference (CAVI). It simultaneously selects the number of clusters, identifies which
features actually drive the cluster structure, and produces diagnostic plots — all from
a single model fit.

---

## Features

- **Mixed data types** — Gaussian (continuous), Poisson (count), Gamma (positive
  continuous), and Multinomial (categorical) variables can be freely combined in the
  same model.
- **Automatic cluster order selection** — a sparse symmetric Dirichlet prior on an
  overfitted mixture automatically prunes redundant components; no need to specify
  *K* in advance.
- **Two feature-relevance models**
  - `SFRM()` — each feature has one shared relevance probability across all clusters.
  - `LFRM()` — each feature can be relevant in some clusters and irrelevant in others,
    capturing finer subgroup-specific signal.
- **Post-hoc EIG filter** — an Expected Information Gain criterion controls the False
  Discovery Rate for feature selection.
- **Multi-start screening** — automatic warm-start strategy selects the best
  initialization before the full CAVI run.
- **Out-of-sample prediction** — assign new observations to the learned cluster
  structure via `predict_subtypes`.
- **8 built-in visualizations** — covering the full analysis workflow from convergence
  diagnostics to cluster-specific feature profiles.

---

## Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/toussile/MixClustVIjl.git")
```

### Local development install

If you have cloned the repository locally:

```julia
using Pkg
Pkg.develop(path="/path/to/MixClustVIjl")
```

### From the Julia General Registry (planned)

Once registered:

```julia
Pkg.add("MixClustVIjl")
```

---

## Example: clustering a synthetic patient cohort

The following self-contained example generates a dataset of 150 patients described by
five heterogeneous features (two continuous, one count, one positive-continuous, one
categorical) plus two uninformative noise features. Three subtypes are present.

```julia
using MixClustVIjl, Random, Statistics

Random.seed!(2026)

# ── 1. Simulate a mixed-type patient dataset ───────────────────────────────
n       = 150          # 150 patients
K_true  = 3            # 3 true subtypes, 50 patients each
labels  = vcat(fill(1, 50), fill(2, 50), fill(3, 50))

# Helper: multinomial draw
function rand_cat(probs)
    u = rand(); cum = 0.0
    for (i, p) in enumerate(probs)
        cum += p; u <= cum && return i
    end
    return length(probs)
end

# Feature 1 – Age (Gaussian, standardized). Mean differs across subtypes.
age_raw = [labels[i] == 1 ? 45.0 + 8randn() :
           labels[i] == 2 ? 62.0 + 8randn() : 55.0 + 8randn() for i in 1:n]
age = (age_raw .- mean(age_raw)) ./ std(age_raw)

# Feature 2 – BMI (Gaussian, standardized). Subtype 3 has elevated BMI.
bmi_raw = [labels[i] == 3 ? 31.0 + 4randn() : 24.0 + 4randn() for i in 1:n]
bmi = (bmi_raw .- mean(bmi_raw)) ./ std(bmi_raw)

# Feature 3 – Mutation count (Poisson). Subtype 2 is hypermutated.
function rpois(λ)
    L = exp(-λ); k = 0; p = 1.0
    while p > L; k += 1; p *= rand(); end
    return k - 1
end
mutation_count = Float64[labels[i] == 1 ? rpois(2.0) :
                         labels[i] == 2 ? rpois(18.0) : rpois(6.0) for i in 1:n]

# Feature 4 – Tumour size in mm (Gamma, positive continuous).
rand_gamma(shape, rate) = -sum(log(rand()) for _ in 1:round(Int, shape)) / rate
tumour_size = [labels[i] == 1 ? rand_gamma(3, 0.3) :
               labels[i] == 2 ? rand_gamma(3, 0.1) : rand_gamma(3, 0.2) for i in 1:n]

# Feature 5 – Histological subtype (Multinomial, 3 categories).
# P(category | subtype): subtypes have distinct profiles.
cat_probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
histology  = [[rand_cat(cat_probs[labels[i]]) == c ? 1 : 0 for c in 1:3] for i in 1:n]

# Noise features (uninformative)
noise_gaussian = randn(n)
noise_count    = Float64[rpois(4.0) for _ in 1:n]

# Assemble the dataset as Vector{Any} (one element per feature)
feature_names = ["age", "bmi", "mutations", "tumour_size", "histology",
                 "noise_cont", "noise_count"]
dataset = Any[age, bmi, mutation_count, tumour_size, histology,
              noise_gaussian, noise_count]

# ── 2. Fit the model ───────────────────────────────────────────────────────
# K_fit = 10 (overfitted mixture); the model will prune down to the true K.
results = mixClust(dataset, 10;
                   model_setting = LFRM(),
                   alpha_0       = 0.01,
                   max_iter      = 500,
                   tol           = 1e-4,
                   prune         = true,
                   n_init        = 10,
                   max_iter_init = 10)

K_hat = size(results.w, 2)
println("Estimated number of clusters: K̂ = ", K_hat)   # → 3

# ── 3. Feature selection via EIG ───────────────────────────────────────────
τ = 0.10                                      # EIG detection threshold
eig     = compute_eig(results.margins, results.w, results.pip)
active  = filter_features(eig, τ)
println("Active features (EIG ≥ $τ): ", feature_names[active])
# → ["age", "mutations", "tumour_size", "histology"]

# ── 4. Hard cluster assignments ────────────────────────────────────────────
k_hat = results.labels                        # Vector{Int} of length n
println("Cluster sizes: ", [count(==(k), k_hat) for k in 1:K_hat])

# ── 5. Visualizations ─────────────────────────────────────────────────────
plot_elbo(results)                            # convergence check
plot_cluster_sizes(results)                   # individuals per cluster
plot_assignment_confidence(results)           # crisp assignments?
plot_pips(results; feature_names)             # global PIPs
plot_eig(eig, τ; feature_names)              # EIG bar chart
plot_local_pips(results; feature_names)       # per-cluster PIPs (LFRM)
plot_profiles(results, dataset;               # feature z-scores per cluster
              feature_names, threshold=τ)
plot_assignments(results, dataset;            # 2-D scatter coloured by cluster
                 x_feature=1, y_feature=3,
                 feature_names)

# ── 6. Out-of-sample prediction ────────────────────────────────────────────
w_new  = predict_subtypes(results, dataset)   # reuse training data as demo
ll_new = predictive_log_likelihood(results, dataset)
println("Log-predictive density: ", round(ll_new, digits=1))
```

The full real-data analyses from the paper (UCI Heart Disease, synthetic simulations)
are available in [`experiments/scripts/`](../experiments/scripts/).

---

## Data format

The `dataset` argument to `mixClust` is a `Vector{Any}` with one element per feature:

| Feature type | Julia type | Notes |
| :--- | :--- | :--- |
| Continuous (Gaussian) | `Vector{Float64}` | Standardize to mean 0, variance 1 |
| Count (Poisson) | `Vector{Float64}` | Non-negative integers as `Float64` |
| Positive continuous (Gamma) | `Vector{Float64}` | Strictly positive values |
| Categorical (Multinomial) | `Vector{Vector{Int}}` | One-hot vectors of length `n_levels` |

The type of each margin is inferred automatically: `Vector{Vector{Int}}` → Multinomial;
non-negative integer-valued `Vector{Float64}` → Poisson; strictly positive
`Vector{Float64}` → Gamma; otherwise Gaussian. Features of different types can appear
in any order.

To override the auto-detection for specific features, pass a `feature_types` vector:

```julia
# Force feature 3 to Gaussian even though all values are positive
results = mixClust(dataset, 10;
                   feature_types = [nothing, nothing, :gaussian, nothing, nothing])
```

Each entry is `:gaussian`, `:poisson`, `:gamma`, `:multinomial`, or `nothing` (auto-detect).
`mixClust` will print the resolved type of every feature at `@info` level so you can
verify the detection before the CAVI run starts.

---

## API Reference

### Model fitting

```julia
results = mixClust(dataset, K_fit;
                   model_setting  = SFRM(),   # or LFRM()
                   alpha_0        = 0.01,     # Dirichlet sparsity (< 1; smaller → more pruning)
                   delta_prior    = (1.0, 1.0), # Beta prior for relevance indicators
                   max_iter       = 500,      # CAVI iterations for the final run
                   tol            = 1e-4,     # convergence tolerance
                   prune          = true,     # post-hoc pruning & merging
                   size_threshold = 0.02,     # min relative cluster size to retain
                   merge_threshold= 0.85,     # cosine similarity above which clusters are merged
                   beta_estimation= :two_stage, # or :iterative
                   n_init         = 10,       # number of screening restarts
                   max_iter_init  = 10,       # iterations per screening run
                   feature_types  = nothing)  # optional override per feature (see below)
```

`results` is a `MixClustResult` struct:

| Field | Type | Description |
| :--- | :--- | :--- |
| `results.w` | `n × K̂ Matrix{Float64}` | Soft cluster assignments (rows sum to 1) |
| `results.labels` | `Vector{Int}` | Hard assignment: `argmax` of each row of `w` |
| `results.pip` | `n × p Matrix{Float64}` | Posterior Inclusion Probabilities |
| `results.margins` | `Vector{AbstractMargin}` | Fitted margin objects (one per feature) |
| `results.elbo_history` | `Vector{Float64}` | ELBO values over iterations |
| `results.alpha_star` | `Vector{Float64}` | Variational Dirichlet parameters (length K̂) |
| `results.delta_star` | `Array{Float64}` | Variational Beta parameters for relevance |

### Model settings

| Setting | Description |
| :--- | :--- |
| `SFRM()` | Shared Feature Relevance: one relevance probability per feature |
| `LFRM()` | Local Feature Relevance: one relevance probability per (feature, cluster) pair |

Use `LFRM()` when you expect some features to separate only a subset of clusters.
Use `SFRM()` as a faster baseline or when relevance is expected to be uniform.

### Feature relevance

```julia
eig     = compute_eig(results.margins, results.w, results.pip)
# → Vector{Float64} of length p

active  = filter_features(eig, 0.10)
# → Vector{Int} of indices where EIG ≥ threshold
```

### Out-of-sample prediction

```julia
w_new  = predict_subtypes(results, new_dataset)
# → (n_new × K̂) matrix; rows sum to 1

ll_new = predictive_log_likelihood(results, new_dataset)
# → scalar total log-predictive density
```

### Visualization

All plot functions accept optional keyword arguments forwarded to `Plots.jl`.

| Function | What it shows |
| :--- | :--- |
| `plot_elbo(results)` | ELBO convergence curve |
| `plot_cluster_sizes(results)` | Number of individuals per cluster |
| `plot_assignment_confidence(results; threshold=0.9)` | Histogram of max posterior responsibility |
| `plot_pips(results; feature_names)` | Bar chart of mean PIP per feature |
| `plot_eig(eig, τ; feature_names)` | Bar chart of EIG with threshold line |
| `plot_local_pips(results; feature_names)` | K × p heatmap of per-cluster PIPs (LFRM) |
| `plot_profiles(results, dataset; feature_names)` | Feature z-scores per cluster vs. background |
| `plot_assignments(results, dataset; x_feature, y_feature, feature_names)` | 2-D scatter coloured by cluster |

**Recommended workflow:**

```
plot_elbo              → convergence ok?
plot_cluster_sizes     → how many individuals per group?
plot_assignment_confidence → are assignments crisp?
plot_pips / plot_eig   → which features matter globally?
plot_local_pips        → which features matter per cluster? (LFRM)
plot_profiles          → what does each subtype look like?
plot_assignments       → where do individuals fall in feature space?
```

---

## Citation

If you use `MixClustVIjl` in your research, please cite:

```bibtex
@article{toussile2026mixclustvi,
  author  = {Toussile, Wilson and Fotso, Simeon and Takam Soh, Patrice},
  title   = {Local Feature Relevance and Cluster Order Selection in Model-Based
             Clustering for Mixed Data via Variational Inference},
  journal = {Computational Statistics \& Data Analysis},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Roadmap

- [ ] Negative Binomial margin for overdispersed count data
- [ ] Block-diagonal covariance for continuous features (relax within-cluster independence)
- [ ] Native missing-data handling within CAVI

---

## License

[MIT License](LICENSE)

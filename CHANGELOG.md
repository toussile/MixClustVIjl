# Changelog

All notable changes to MixClustVIjl are documented here.
Versioning follows [Semantic Versioning](https://semver.org):
`MAJOR.MINOR.PATCH` — breaking changes bump MAJOR (or MINOR while pre-1.0),
new features bump MINOR, bug fixes bump PATCH.

---

## [0.2.0] — 2026-07-05

### Breaking changes

- **`results.alpha_star` renamed to `results.u_star`** — aligns with the paper
  notation where $\bm{u} = (u_1, \dots, u_K)^T$ denotes the variational Dirichlet
  parameter for mixing proportions. Any code accessing `results.alpha_star` must be
  updated to `results.u_star`.
- **`alpha_0` keyword renamed to `u0`** in `mixClust(...)` — aligns with the paper
  notation $u^{(0)}$ for the symmetric Dirichlet hyperparameter. Any call using
  `mixClust(...; alpha_0=...)` must be updated to `mixClust(...; u0=...)`.

### Bug fixes

- **ELBO monotonicity** — the ELBO sequence is now guaranteed non-decreasing at
  every CAVI iteration. Two root causes were fixed:
  - The KL divergence between variational and prior distributions for margin
    parameters (`kl_from_prior`) was missing from the ELBO computation. Implemented
    for all four margin types (Gaussian/NIG, Poisson/Gamma, Gamma, Multinomial/Dirichlet).
  - The ELBO is now computed *before* `update_margin!`, using refreshed expectations
    from the just-updated $\bm{u}$ and $\bm{\delta}$ parameters, ensuring the logged
    sequence corresponds to a consistent set of variational parameters.

### Improvements

- README: Iris dataset example now loads data via `RDatasets.jl` instead of
  hardcoded matrix.
- README: `K_fit` renamed to `K_max` throughout examples for clarity.
- README: Features section now explicitly lists post-hoc cluster refinement
  (pruning & merging) as a distinct step.

---

## [0.1.0] — 2026-07-04

Initial release. Core features:

- CAVI inference for finite overfitted mixtures on heterogeneous data
  (Gaussian, Poisson, Gamma, Multinomial margins).
- Shared Feature Relevance Model (`SFRM`) and Local Feature Relevance Model (`LFRM`).
- Sparse Dirichlet prior for automatic cluster order selection.
- Post-hoc cluster refinement: size-based pruning and cosine-similarity merging.
- Post-hoc feature selection via Expected Information Gain (EIG).
- Multi-start screening strategy.
- Out-of-sample prediction (`predict_proba`, `predictive_log_likelihood`).
- 8 built-in diagnostic visualizations.
- Bundled datasets: UCI Heart Disease and synthetic clinicogenomic cohort.

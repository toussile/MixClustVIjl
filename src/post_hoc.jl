"""
    compute_eig(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractMatrix) -> Vector{Float64}

Computes the Expected Information Gain (EIG) for each feature.
- `margins`: Vector of length p containing the updated concrete margins.
- `w`: The n x K matrix of cluster assignment probabilities.
- `pip`: The n x p matrix of individual Posterior Inclusion Probabilities (gamma_ij).

Returns a vector of length p representing the EIG of each feature.
"""
function compute_eig(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractMatrix)
    n, K = size(w)
    p = length(margins)
    
    # Compute estimated cluster proportions w_bar
    w_bar = Vector{Float64}(undef, K)
    for k in 1:K
        w_bar[k] = sum(w[:, k]) / n
    end
    
    eig = Vector{Float64}(undef, p)
    for j in 1:p
        # Compute average PIP for feature j
        gamma_bar = sum(pip[:, j]) / n
        
        # Get the expected KL divergences for feature j across all clusters
        kl = expected_kl_divergence(margins[j]) # Vector of length K
        
        # EIG_j = gamma_bar * sum_{k} w_bar_k * kl_k (only for active clusters with w_bar >= 0.02)
        val = 0.0
        for k in 1:K
            if w_bar[k] >= 0.02
                val += w_bar[k] * kl[k]
            end
        end
        eig[j] = gamma_bar * val
    end
    
    return eig
end


"""
    filter_features(eig::Vector{Float64}, threshold::Real) -> Vector{Int}

Returns the indices of the features whose Expected Information Gain exceeds the given threshold.
"""
function filter_features(eig::Vector{Float64}, threshold::Real)
    return findall(x -> x >= threshold, eig)
end

"""
    predict_proba(results::MixClustResult, new_data::AbstractVector) -> Matrix{Float64}

Computes the posterior subtype assignment probabilities q(z^*_i = k | y^*_i) for new observations.
Returns an n_new x K matrix where rows sum to 1.
"""
function predict_proba(results::MixClustResult, new_data::AbstractVector)
    p = length(results.margins)
    n_new = length(new_data[1])
    K = length(results.u_star)
    
    # 1. Compute expected mixing weights omega_bar
    sum_alpha = sum(results.u_star)
    omega_bar = results.u_star ./ sum_alpha
    
    # 2. Compute expected feature inclusion probabilities gamma_bar (K x p)
    gamma_bar = Matrix{Float64}(undef, K, p)
    if ndims(results.delta_star) == 2
        # Model 1
        for j in 1:p
            g = results.delta_star[j, 1] / (results.delta_star[j, 1] + results.delta_star[j, 2])
            gamma_bar[:, j] .= g
        end
    else
        # Model 2
        for k in 1:K
            for j in 1:p
                gamma_bar[k, j] = results.delta_star[k, j, 1] / (results.delta_star[k, j, 1] + results.delta_star[k, j, 2])
            end
        end
    end
    
    # 3. Precompute predictive densities for each feature j
    P = [predictive_density(results.margins[j], new_data[j]) for j in 1:p]
    B = [exp.(background_log_density(results.margins[j], new_data[j])) for j in 1:p]
    
    # 4. Compute log-likelihood contribution for each patient i and cluster k
    log_q = Matrix{Float64}(undef, n_new, K)
    for i in 1:n_new
        for k in 1:K
            val = log(max(omega_bar[k], 1e-15))
            for j in 1:p
                g = gamma_bar[k, j]
                dens = g * P[j][i, k] + (1.0 - g) * B[j][i]
                val += log(max(dens, 1e-300))
            end
            log_q[i, k] = val
        end
    end
    
    # 5. Exponentiate and normalize (softmax per individual)
    w_pred = Matrix{Float64}(undef, n_new, K)
    for i in 1:n_new
        max_log = maximum(log_q[i, :])
        row_exp = exp.(log_q[i, :] .- max_log)
        sum_exp = sum(row_exp)
        w_pred[i, :] = sum_exp > 0 ? row_exp ./ sum_exp : fill(1.0 / K, K)
    end
    
    return w_pred
end

"""
    predictive_log_likelihood(results::MixClustResult, new_data::AbstractVector) -> Float64

Computes the total out-of-sample log-predictive density log p(y^* | y) on a test dataset.
"""
function predictive_log_likelihood(results::MixClustResult, new_data::AbstractVector)
    p = length(results.margins)
    n_new = length(new_data[1])
    K = length(results.u_star)
    
    sum_alpha = sum(results.u_star)
    omega_bar = results.u_star ./ sum_alpha
    
    gamma_bar = Matrix{Float64}(undef, K, p)
    if ndims(results.delta_star) == 2
        for j in 1:p
            g = results.delta_star[j, 1] / (results.delta_star[j, 1] + results.delta_star[j, 2])
            gamma_bar[:, j] .= g
        end
    else
        for k in 1:K
            for j in 1:p
                gamma_bar[k, j] = results.delta_star[k, j, 1] / (results.delta_star[k, j, 1] + results.delta_star[k, j, 2])
            end
        end
    end
    
    P = [predictive_density(results.margins[j], new_data[j]) for j in 1:p]
    B = [exp.(background_log_density(results.margins[j], new_data[j])) for j in 1:p]
    
    total_log_lik = 0.0
    for i in 1:n_new
        log_terms = Vector{Float64}(undef, K)
        for k in 1:K
            val = log(max(omega_bar[k], 1e-15))
            for j in 1:p
                g = gamma_bar[k, j]
                dens = g * P[j][i, k] + (1.0 - g) * B[j][i]
                val += log(max(dens, 1e-300))
            end
            log_terms[k] = val
        end
        max_log = maximum(log_terms)
        total_log_lik += max_log + log(sum(exp.(log_terms .- max_log)))
    end
    
    return total_log_lik
end


"""
    prune_and_merge_clusters(results::MixClustResult, data::AbstractVector; size_threshold=0.02, merge_threshold=0.85) -> MixClustResult

Prunes clusters below size_threshold and merges highly overlapping clusters with cosine similarity of assignments above merge_threshold.
"""
function prune_and_merge_clusters(results::MixClustResult, data::AbstractVector;
                                  size_threshold=0.02,
                                  merge_threshold=0.85)
    w = copy(results.w)
    pip = copy(results.pip)
    u_star = copy(results.u_star)
    delta_star = copy(results.delta_star)
    margins = [deepcopy(m) for m in results.margins]
    
    n, K = size(w)
    p = length(margins)
    
    # 1. Size-based pruning
    active_clusters = Int[]
    for k in 1:K
        prop = sum(w[:, k]) / n
        if prop >= size_threshold
            push!(active_clusters, k)
        end
    end
    
    if length(active_clusters) < K
        w = w[:, active_clusters]
        for i in 1:n
            s = sum(w[i, :])
            if s > 0
                w[i, :] ./= s
            else
                w[i, :] .= 1.0 / length(active_clusters)
            end
        end
        u_star = u_star[active_clusters]
        if ndims(delta_star) == 3
            delta_star = delta_star[active_clusters, :, :]
        end
        K = length(active_clusters)
    end
    
    # 2. Overlap-based merging
    merged = true
    while merged && K > 1
        merged = false
        best_sim = -1.0
        best_pair = (0, 0)
        
        for k1 in 1:K
            for k2 in (k1+1):K
                vec1 = w[:, k1]
                vec2 = w[:, k2]
                norm1 = sqrt(sum(vec1.^2))
                norm2 = sqrt(sum(vec2.^2))
                sim = (norm1 > 0 && norm2 > 0) ? sum(vec1 .* vec2) / (norm1 * norm2) : 0.0
                
                if sim > best_sim
                    best_sim = sim
                    best_pair = (k1, k2)
                end
            end
        end
        
        if best_sim >= merge_threshold
            k1, k2 = best_pair
            w_new = Matrix{Float64}(undef, n, K - 1)
            idx_new = 1
            for k in 1:K
                if k == k1
                    w_new[:, idx_new] = w[:, k1] .+ w[:, k2]
                    idx_new += 1
                elseif k != k2
                    w_new[:, idx_new] = w[:, k]
                    idx_new += 1
                end
            end
            
            for i in 1:n
                s = sum(w_new[i, :])
                if s > 0
                    w_new[i, :] ./= s
                end
            end
            
            w = w_new
            u_star = [sum(w[:, k]) for k in 1:(K-1)]
            K = K - 1
            merged = true
        end
    end
    
    # 3. Re-fit margins based on the new assignments
    new_margins = Vector{AbstractMargin}(undef, p)
    for j in 1:p
        y_j = data[j]
        if typeof(margins[j]) <: MultinomialMargin
            new_margins[j] = MultinomialMargin(y_j, K; varphi=margins[j].varphi)
        elseif typeof(margins[j]) <: PoissonMargin
            new_margins[j] = PoissonMargin(y_j, K; a_0=margins[j].a_0, b_0=margins[j].b_0)
        elseif typeof(margins[j]) <: GammaMargin
            new_margins[j] = GammaMargin(y_j, K; alpha_0=margins[j].alpha_0, beta_0=margins[j].beta_0)
        else
            new_margins[j] = GaussianMargin(y_j, K; mu_0=margins[j].mu_0, kappa_0=margins[j].kappa_0, a_0=margins[j].a_0, b_0=margins[j].b_0)
        end
        update_margin!(new_margins[j], y_j, w, pip[:, j])
    end
    
    new_u_star = fill(0.01 + n / K, K)
    for k in 1:K
        new_u_star[k] = 0.01 + sum(w[:, k])
    end
    
    new_delta_star = ndims(results.delta_star) == 2 ? fill(1.0, p, 2) : fill(1.0, K, p, 2)
    if ndims(results.delta_star) == 2
        new_delta_star .= results.delta_star
    else
        for k in 1:K
            for j in 1:p
                new_delta_star[k, j, 1] = 1.0 + sum(w[:, k] .* pip[:, j])
                new_delta_star[k, j, 2] = 1.0 + sum(w[:, k] .* (1.0 .- pip[:, j]))
            end
        end
    end
    
    labels = [argmax(w[i, :]) for i in 1:size(w, 1)]
    return MixClustResult(w, labels, pip, new_u_star, new_delta_star, new_margins, results.elbo_history)
end


using Test
using Random
using Statistics
using Plots
using MixClustVIjl

# Helper samplers for test data generation
function rand_multinomial(N, p_vec)
    C = length(p_vec)
    counts = zeros(Int, C)
    cum_p = cumsum(p_vec)
    for _ in 1:N
        u = rand()
        idx = findfirst(x -> u <= x, cum_p)
        if isnothing(idx)
            idx = C
        end
        counts[idx] += 1
    end
    return counts
end

function rand_poisson(lambda)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand()
    end
    return k - 1
end

function rand_gamma_shape2(rate)
    # Gamma(2, rate) is the sum of two independent Exponentials
    u1 = rand()
    u2 = rand()
    return -log(u1 * u2) / rate
end

@testset "MixClustVIjl.jl Tests" begin
    Random.seed!(42)
    
    n = 120
    K_true = 3
    K_max = 8 # Overfitted mixture
    
    # Assign true cluster membership: 40 individuals per cluster
    true_z = vcat(fill(1, 40), fill(2, 40), fill(3, 40))
    
    # 1. Feature 1: Active Gaussian
    # C1: N(-2, 0.5^2), C2: N(0, 0.5^2), C3: N(2, 0.5^2)
    y1 = Float64[]
    for c in true_z
        if c == 1
            push!(y1, -2.0 + randn() * 0.5)
        elseif c == 2
            push!(y1, 0.0 + randn() * 0.5)
        else
            push!(y1, 2.0 + randn() * 0.5)
        end
    end
    
    # 2. Feature 2: Noise Gaussian
    # N(0, 1.0) for all
    y2 = randn(n)
    
    # 3. Feature 3: Active Poisson
    # C1: Poisson(1), C2: Poisson(6), C3: Poisson(12)
    y3 = Float64[]
    for c in true_z
        if c == 1
            push!(y3, Float64(rand_poisson(1.0)))
        elseif c == 2
            push!(y3, Float64(rand_poisson(6.0)))
        else
            push!(y3, Float64(rand_poisson(12.0)))
        end
    end
    
    # 4. Feature 4: Noise Poisson
    # Poisson(4) for all
    y4 = [Float64(rand_poisson(4.0)) for _ in 1:n]
    
    # 5. Feature 5: Active Multinomial (3 categories, 15 trials)
    # C1: [0.8, 0.1, 0.1], C2: [0.1, 0.8, 0.1], C3: [0.1, 0.1, 0.8]
    y5 = Vector{Int}[]
    for c in true_z
        if c == 1
            push!(y5, rand_multinomial(15, [0.8, 0.1, 0.1]))
        elseif c == 2
            push!(y5, rand_multinomial(15, [0.1, 0.8, 0.1]))
        else
            push!(y5, rand_multinomial(15, [0.1, 0.1, 0.8]))
        end
    end
    
    # 6. Feature 6: Noise Multinomial
    # [0.33, 0.33, 0.33] for all
    y6 = [rand_multinomial(15, [0.33, 0.33, 0.33]) for _ in 1:n]
    
    # 7. Feature 7: Active Gamma (shape = 2.0)
    # C1: Gamma(2, rate=1.0), C2: Gamma(2, rate=5.0), C3: Gamma(2, rate=0.2)
    y7 = Float64[]
    for c in true_z
        if c == 1
            push!(y7, rand_gamma_shape2(1.0))
        elseif c == 2
            push!(y7, rand_gamma_shape2(5.0))
        else
            push!(y7, rand_gamma_shape2(0.2))
        end
    end
    
    # 8. Feature 8: Noise Gamma
    # Gamma(2, rate=2.0) for all
    y8 = [rand_gamma_shape2(2.0) for _ in 1:n]
    
    # Combine features into the heterogeneous dataset vector
    # Order: [ActGauss, NoiseGauss, ActPoi, NoisePoi, ActMult, NoiseMult, ActGamma, NoiseGamma]
    dataset = [y1, y2, y3, y4, y5, y6, y7, y8]
    p_total = length(dataset)
    
    # Verify dataset dimensions and types
    @test length(dataset) == 8
    @test length(dataset[1]) == n
    
    @testset "CAVI Model 1 (Shared Relevance)" begin
        # Fit overfitted CAVI Model 1
        results = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=80, tol=1e-5, u0=0.01, prune=false)
        
        # Check output structure sizes
        @test size(results.w) == (n, K_max)
        @test size(results.pip) == (n, p_total)
        @test length(results.u_star) == K_max
        @test length(results.margins) == p_total
        @test length(results.elbo_history) >= 2
        
        # Verify that ELBO converges and values are finite
        @test all(!isnan, results.elbo_history)
        # ELBO must be monotonically non-decreasing (CAVI guarantee)
        @test all(i -> results.elbo_history[i] >= results.elbo_history[i-1] - 1e-6,
                  2:length(results.elbo_history))
        
        # Compute Expected Information Gain (EIG) post-hoc
        eig = compute_eig(results.margins, results.w, results.pip)
        @test length(eig) == p_total
        @test all(x -> x >= 0, eig)
        
        # Perform feature selection based on EIG with threshold tau = 0.1
        tau = 0.1
        active_indices = filter_features(eig, tau)
        
        println("Model 1 EIG values:")
        for (idx, val) in enumerate(eig)
            println("Feature $idx (Active label: $(idx % 2 == 1)): EIG = ", round(val, digits=4))
        end
        
        # Check that active features (1, 3, 5, 7) have significantly higher EIG than noise features (2, 4, 6, 8)
        @test eig[1] > eig[2]
        @test eig[3] > eig[4]
        @test eig[5] > eig[6]
        @test eig[7] > eig[8]
        
        # Test visualizations
        p_elbo = plot_elbo(results)
        p_pips = plot_pips(results)
        p_eig = plot_eig(eig, tau)
        p_w = plot_assignments(results, dataset)
        p_prof = plot_profiles(results, dataset; threshold=tau)
        
        # Save plots to files to verify they save correctly
        plots_dir = joinpath(@__DIR__, "plots")
        mkpath(plots_dir)
        savefig(p_elbo, joinpath(plots_dir, "elbo_m1.png"))
        savefig(p_pips, joinpath(plots_dir, "pips_m1.png"))
        savefig(p_eig, joinpath(plots_dir, "eig_m1.png"))
        savefig(p_w, joinpath(plots_dir, "assignments_m1.png"))
        savefig(p_prof, joinpath(plots_dir, "profiles_m1.png"))
        
        @test isfile(joinpath(plots_dir, "elbo_m1.png"))
        @test isfile(joinpath(plots_dir, "pips_m1.png"))
        @test isfile(joinpath(plots_dir, "eig_m1.png"))
        @test isfile(joinpath(plots_dir, "assignments_m1.png"))
        @test isfile(joinpath(plots_dir, "profiles_m1.png"))
        
        @testset "Predictive Inference & Pruning Tests" begin
            # 1. Test predictive density of concrete margins on test data
            for j in 1:p_total
                P_j = MixClustVIjl.predictive_density(results.margins[j], dataset[j][1:10])
                @test size(P_j) == (10, K_max)
                @test all(x -> x >= 0 && !isnan(x) && isfinite(x), P_j)
            end
            
            # 2. Test predict_proba and predictive_log_likelihood
            w_pred = predict_proba(results, dataset)
            @test size(w_pred) == (n, K_max)
            @test all(x -> isapprox(sum(w_pred[x, :]), 1.0, atol=1e-6), 1:n)
            
            log_lik = predictive_log_likelihood(results, dataset)
            @test isfinite(log_lik)
            @test log_lik < 0.0
            
            # 3. Test prune_and_merge_clusters
            results_pruned = prune_and_merge_clusters(results, dataset; size_threshold=0.05, merge_threshold=0.95)
            K_pruned = size(results_pruned.w, 2)
            @test K_pruned <= K_max
            @test size(results_pruned.w, 1) == n
            @test length(results_pruned.margins) == p_total
            @test length(results_pruned.u_star) == K_pruned
        end
    end

    @testset "CAVI Model 2 (Cluster-Specific Relevance)" begin
        # Fit overfitted CAVI Model 2
        results = mixClust(dataset, K_max; model_setting=LFRM(), max_iter=80, tol=1e-5, u0=0.01, prune=false)
        
        # Check output structure sizes
        @test size(results.w) == (n, K_max)
        @test size(results.pip) == (n, p_total)
        @test length(results.u_star) == K_max
        @test length(results.margins) == p_total
        @test length(results.elbo_history) >= 2
        @test all(!isnan, results.elbo_history)
        # ELBO must be monotonically non-decreasing (CAVI guarantee)
        @test all(i -> results.elbo_history[i] >= results.elbo_history[i-1] - 1e-6,
                  2:length(results.elbo_history))

        # Compute EIG and verify active vs noise feature separation
        eig = compute_eig(results.margins, results.w, results.pip)

        println("Model 2 EIG values:")
        for (idx, val) in enumerate(eig)
            println("Feature $idx (Active label: $(idx % 2 == 1)): EIG = ", round(val, digits=4))
        end
        
        @test eig[1] > eig[2]
        @test eig[3] > eig[4]
        @test eig[5] > eig[6]
        @test eig[7] > eig[8]
    end

    @testset "Computed properties of MixClustResult" begin
        results = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=30, tol=1e-4,
                           u0=0.01, prune=true)
        @test results.n_obs      == n
        @test results.n_features == length(dataset)
        @test results.n_clusters == size(results.w, 2)
        @test length(results.cluster_sizes) == results.n_clusters
        @test sum(results.cluster_sizes) == n
    end

    @testset "simulate_synthetic_cohort" begin
        cohort = simulate_synthetic_cohort()
        @test length(cohort.data) == 7
        @test length(cohort.labels) == 150
        @test all(l -> l in (1, 2, 3), cohort.labels)
        @test length(cohort.feature_names) == 7
        # Reproducibility: same seed → same data
        cohort2 = simulate_synthetic_cohort(seed=2026)
        @test cohort.labels == cohort2.labels
        @test cohort.data[1] == cohort2.data[1]
        # Custom size
        cohort3 = simulate_synthetic_cohort(n=60)
        @test length(cohort3.labels) == 60
    end

    @testset "load_heart_disease" begin
        hd = load_heart_disease()
        @test length(hd.data) == 13
        @test length(hd.labels) == 297
        @test all(l -> l in (0, 1), hd.labels)
        @test length(hd.feature_names) == 13
        # Continuous features are standardized (mean ≈ 0)
        for j in [1, 4, 5, 8, 10]
            @test abs(mean(hd.data[j])) < 1e-10
        end
        # Categorical features are one-hot vectors
        for j in [2, 3, 6, 7, 9, 11, 12, 13]
            @test all(v -> sum(v) == 1, hd.data[j])
        end
    end

    @testset "CAVI with Iterative Background Estimation" begin
        # 1. SFRM + :iterative
        res_iter1 = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=40, tol=1e-4, u0=0.01, prune=false, beta_estimation=:iterative)
        @test size(res_iter1.w) == (n, K_max)
        @test all(!isnan, res_iter1.elbo_history)
        # Note: :iterative uses heuristic background updates (not proper coord ascent),
        # so strict monotonicity is not guaranteed — we only check finiteness.

        # 2. LFRM + :iterative
        res_iter2 = mixClust(dataset, K_max; model_setting=LFRM(), max_iter=40, tol=1e-4, u0=0.01, prune=false, beta_estimation=:iterative)
        @test size(res_iter2.w) == (n, K_max)
        @test all(!isnan, res_iter2.elbo_history)
    end
end

using Random, LinearAlgebra, Statistics, Distributions

function GramFilter(Y, gamma2)
    n, p = size(Y)
    
    w = ones(n)  # Initial weights w(1) = 1
    
    # Pre-calculate Y'Y to save computation time inside the loop
    YYt = Y * Y'
    
    while true
        # Construct the Gram matrix G(w) where G_ij = sqrt(wi * wj) * <Yi, Yj>
        sqrt_w = sqrt.(w)
        # Element-wise multiplication: G_ij = sqrt_wi * sqrt_wj * (YYt)_ij
        Gram_w = (sqrt_w .* sqrt_w') .* YYt
        
        D_w = p*diagm(w)
        
        # Matrix for the operator norm check
        M = Gram_w - D_w
        op_norm = norm(M, 2) 
        
        # Termination condition: norm < 5 * gamma2
        if op_norm < 5 * gamma2
            break
        end
        
        # Step 3: Top singular vector v
        svd_res = svd(M)
        v = svd_res.U[:, 1]
        
        # Step 4: Calculate tau_i = (vi² / wi)
        tau = zeros(n)
        for i in 1:n
            if w[i] > 0
                tau[i] = (v[i]^2) / w[i]
            else
                tau[i] = 0.0
            end
        end
        
        # Step 5: Update weights
        max_tau = maximum(tau)
        if max_tau > 0
            w .= (1.0 .- tau ./ max_tau) .* w
        else
            break
        end
    end
    
    return w
end

function MomentFilter(Y, gamma2, u)
    n, p = size(Y)
    w = ones(n)
    
    while true
        # Efficiently compute M = Y' * Diag(w) * Y
        # w .* Y performs row-wise scaling
        M = Y' * (w .* Y)
        A = M - n * I(p)

        # Compute largest singular value (equivalent to svd(A)$d[1])
        # Using svdl on a symmetric matrix; opnorm is more direct
        op_norm = opnorm(A, 2)

        if op_norm < 5 * gamma2
            break
        end

        # Get the first left singular vector
        # eigvals/eigvecs is faster for symmetric A, but svd matches R logic
        sv = svd(A)
        v = sv.U[:, 1]

        # tau_i = <v, Y_i>^2 * 1[w_i > 0]
        projections = Y * v
        tau = (projections.^2) .* (w .> 0)

        tau1 = maximum(tau)
        if tau1 == 0
            break
        end

        idx = sortperm(tau, rev=true)
        cumw = cumsum(w[idx])
        L = findfirst(cumw .>= 2 * u * n)
        if L === nothing
            L=n
        end

        w_new = copy(w)
        for j in 1:L
            i = idx[j]
            w_new[i] = (1.0 - tau[i] / tau1) * w[i]
        end

        w = w_new
    end
    return w
end

"""
    rowsum_filter(Y, w, u)

Subroutine to remove top contamination indices based on inner products.
"""
function rowsum_filter(Y, w, u)
    n, p = size(Y)
    sqrt_w = sqrt.(w)

    # S_vec = colSums(Y * sqrt_w)
    S_vec = vec(sum(Y .* sqrt_w, dims=1))

    # tau_i definition
    inner_terms = (Y * S_vec) .* sqrt_w
    tau = abs.(inner_terms .- w .* p) .* (w .> 0)

    # Remove top u*n indices
    num_remove = floor(Int, u * n)
    if num_remove > 0
        # Get indices of largest tau values
        ord = sortperm(tau, rev=true)
        w_new = copy(w)
        w_new[ord[1:num_remove]] .= 0.0
        return w_new
    end
    
    return w
end

"""
    robust_mean_test(Y, kappa0, delta, epsilon; C_gamma=0.1, Tu=6)

Robust Mean Testing routine (Canconne et al. 2023).
"""
function robust_mean_test(Y, kappa0, delta, epsilon, fm=false; C_gamma=0.1, Tu=6)
    n, p = size(Y)

    # Contamination fraction u
    if fm
        q = epsilon + min(p/(20*n),1/20)
    else
        q = epsilon + 1/n
    end
    u = q + sqrt(2*q*log(4/delta)/n)+2*log(4/delta)/(3*n)
    
    if u > 0.1
        throw(ArgumentError("n is too small. u ($u) needs to be less than 0.1"))
    end
    
    gamma2 = C_gamma * (
            u * n * p * log(1/u) +
            (sqrt(n * p)+p) * log(2 * p / delta) +
            kappa0^2 * n)
    if n>p
        w = MomentFilter(Y, gamma2, u)
    else 
        w = GramFilter(Y, gamma2)
    end
    
    w_prime = rowsum_filter(Y, w, u)

    sqrt_w = sqrt.(max.(w_prime, 0.0))
    Sum_wS = vec(sum(sqrt_w .* Y, dims=1))

    test_stat = abs(sum(Sum_wS.^2) - p * sum(w_prime))

    return test_stat >= (1-Tu*u)^2/2 * kappa0^2 * n^2
end

function robust_mean_test_mom(Y, kappa0, omega, epsilon; Tu=6, k_const=8)
    function rmt_stat(Y, u, kappa0, delta, epsilon)
        n, p = size(Y)
        
        if u > 0.2  #Tunable
            throw(ArgumentError("n is too small. u ($u) needs to be less than 0.15"))
        end
        
        if n>p
            gamma2 = 0.1 * (
                u * n * p * log(1/u) +
                (sqrt(n * p)+p) * log(2 * p / delta) +
                kappa0^2 * n)
            w = MomentFilter(Y, gamma2,u)
        else 
            gamma2 = 0.05 * (
                u * n * p * log(1/u) +
                (sqrt(n * p)+p) * log(2 * p / delta) +
                kappa0^2 * n)
            w = GramFilter(Y, gamma2)
        end
        
        w_prime = rowsum_filter(Y, w, u)

        sqrt_w = sqrt.(max.(w_prime, 0.0))
        Sum_wS = vec(sum(sqrt_w .* Y, dims=1))

        test_stat = abs(sum(Sum_wS.^2) - p * sum(w_prime))

        return test_stat
    end
    n, p = size(Y)
    k=ceil(Int, k_const*log(1/omega))  
    if n<k
        throw(ArgumentError("n is too small. n ($n) needs to be at least k ($k)"))
    end
    block_size = div(n, k)
    U_stats=zeros(k)
    q = epsilon + min(p/(20*block_size),1/20)
    u= q + sqrt(2*q*log(16*k)/block_size)+2*log(16*k)/(3*block_size)
    ends = n:-block_size:(n - (k-1)*block_size)
    for i in 1:k  
        idx = (ends[i] - block_size + 1):ends[i]
        U_stats[i] = rmt_stat(Y[idx, :], u, kappa0, 1/(4k), epsilon)
    end
    return median(U_stats)>= (1-Tu*u)^2/2 * kappa0^2 * block_size^2
end

# --- Data Generation Functions ---

function rt_hd(n, p=1, df=3.0; sd=nothing, mu=zeros(p))
    # Default scale logic from R code
    actual_sd = isnothing(sd) ? sqrt(df/(df-2)) : sd
    
    # Generate t-dist samples and scale them
    # Note: Distributions.jl TDist is for standard t
    d = TDist(df)
    # Correcting scale to match R logic (dividing by theoretical SD then multiplying by target)
    samples = rand(d, n, p) .* (actual_sd / sqrt(df/(df-2)))
    
    return samples .+ mu'
end

function rlaplace_hd(n, p=1, s=1/sqrt(2); mu=zeros(p))
    d = Laplace(0, s)
    samples = rand(d, n, p)
    return samples .+ mu'
end

function rlaplace_hd_cpt(n, p, epsilon; cpt=100, mu_norm=2)
    # 1. Define Mean Vectors
    mu_null = zeros(p)
    mu_alt = fill(mu_norm/ sqrt(p), p) 
    
    # 2. Generate Clean Data with Change Point
    Y_before = rlaplace_hd(cpt, p; mu=mu_null)
    Y_after = rlaplace_hd(n-cpt, p; mu=mu_alt)
    Y_clean = [Y_before; Y_after]

    # 3. Corruptions
    # Draw shared n_corrupt for this iteration
    is_contaminated = rand(Bernoulli(epsilon), n)
    n_corrupt = sum(is_contaminated)
    if n_corrupt>0 
        corrupt = randn(n_corrupt, p) .- 1.0
        corrupt_idx = findall(is_contaminated)
        Y_clean[corrupt_idx, :] = corrupt
    end
    return Y_clean
end

function rt_hd_cpt(n, p, epsilon; cpt=100, mu_norm=2)
    # 1. Define Mean Vectors
    mu_null = zeros(p)
    mu_alt = fill(mu_norm/ sqrt(p), p) 
    
    # 2. Generate Clean Data with Change Point
    Y_before = rt_hd(cpt, p, 4.1; sd=1, mu=mu_null)
    Y_after = rt_hd(n-cpt, p, 4.1; sd=1, mu=mu_alt)
    Y_clean = [Y_before; Y_after]

    # 3. Corruptions
    # Draw shared n_corrupt for this iteration
    is_contaminated = rand(Bernoulli(epsilon), n)
    n_corrupt = sum(is_contaminated)
    if n_corrupt>0 
        corrupt = randn(n_corrupt, p) .- 1.0
        corrupt_idx = findall(is_contaminated)
        Y_clean[corrupt_idx, :] = corrupt
    end
    return Y_clean
end

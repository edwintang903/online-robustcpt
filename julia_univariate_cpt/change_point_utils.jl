using Random, Distributions, Statistics

# --- Robust Univariate Mean Estimator ---------------------------------------
function rume(X::Vector{Float64}; epsilon=0.0, delta=0.05)
    n_total = length(X)
    n = n_total ÷ 2
    X1 = sort(X[1:n])
    X2 = X[end-n+1:end]

    vareps = max(epsilon, log(1/delta)/n)
    prop=2*vareps + 2*sqrt(vareps*log(1/delta)/n) + log(1/delta)/n
    if  prop>= 0.5
        error("Not enough sample size to achieve required confidence level.")
    end
    datapts = floor(Int, n*(1 - prop))

    candidates = [X1[i+datapts-1] - X1[i] for i in 1:(n-datapts+1)]
    min_index = argmin(candidates)
    interval = (X1[min_index], X1[min_index+datapts-1])

    filtered_X2 = filter(x -> interval[1] ≤ x ≤ interval[2], X2)
    return mean(filtered_X2)
end

# --- Contaminated t-distribution sampler -----------------------------------
function contaminated_sample_t(n, mu=0.0; df=3.0, epsilon=0.0)
    t_samples = rand(TDist(df), n) .+ mu
    contam_mask = rand(Binomial(1, epsilon), n) .== 1
    contam_idx = findall(contam_mask)
    if !isempty(contam_idx)
        t_samples[contam_idx] = rand(Normal(0, 100), length(contam_idx))
    end
    return t_samples
end

# --- Change-point data generator -------------------------------------------
function change_point_model(n; mechanism=contaminated_sample_t, cpt=nothing, kappa=1.0)
    if isnothing(cpt)
        return mechanism(n)
    else
        sample = mechanism(cpt)
        append!(sample, mechanism(n-cpt, kappa))
        return sample
    end
end

# --- RUMEDIAN change-point detection ---------------------------------------
function rumedian_v(online_data::Vector{Float64}, sigma; v=2, epsilon=0.0, alpha=0.1, C1=1.59, C2=2.25)
    n = length(online_data)
    if epsilon > 0.1
        error("epsilon > 0.1 is not supported.")
    end

    for t in 2:n
        delta_t = (8*alpha)/(3*t^3 - 3*t)
        h_t = ceil(Int, 20*log(1/delta_t))
        for s in 1:floor(Int, t/2)
            if s >= h_t
                vareps = max(epsilon, 2*log(1/delta_t)/s)
                diff_rume = abs(rume(online_data[(t-s+1):t]; epsilon=epsilon) -
                                rume(online_data[1:s]; epsilon=epsilon))
                zeta = 2*sigma*C1*max(vareps^(1-1/v), sqrt(2/s*log(1/delta_t)))
                if diff_rume > zeta
                    return Dict("method" => "RUME", "subsample" => s, "location" => t)
                end
            else
                conf=0.5*exp(-1)*(delta_t/2)^(2/s) - epsilon
                if conf>0
                    diff_median = abs(median(online_data[(t-s+1):t]) - median(online_data[1:s]))
                    chi = 2*sigma*C2*(conf^(-1/v))
                    if diff_median > chi
                        return Dict("method" => "median", "subsample" => s, "location" => t)
                    end
                end
            end
        end
    end
    return Dict("method" => "no changepoint", "subsample" => -1, "location" => -1)
end

# --- Contaminated laplace sampler -----------------------------------
function contaminated_laplace(n, mu=0.0; epsilon=0.0)
    samples = rand(Laplace(mu, 3.24037), n)
    contam_mask = rand(Bernoulli(epsilon), n)
    
    # Replace only those indices with high-variance noise
    for i in 1:n
        if contam_mask[i]
            samples[i] = rand(Normal(0, 100)) # The "outlier" distribution
        end
    end
    
    return samples
end

function rumedian_theta(online_data::Vector{Float64}, sigma; theta=1, epsilon=0.0, alpha=0.1, C1=0.53, C2=0.075)
    n = length(online_data)
    if epsilon > 0.1
        error("epsilon > 0.1 is not supported.")
    end

    for t in 2:n
        delta_t = (8*alpha)/(3*t^3 - 3*t)
        h_t = ceil(Int, 20*log(1/delta_t))
        for s in 1:floor(Int, t/2)
            if s >= h_t
                vareps = max(epsilon, 2*log(1/delta_t)/s)
                diff_rume = abs(rume(online_data[(t-s+1):t]; epsilon=epsilon) -
                                rume(online_data[1:s]; epsilon=epsilon))
                zeta = 2*sigma*C1*max(vareps*(log(1/vareps))^(1+1/theta), sqrt(2/s*log(1/delta_t)))
                if diff_rume > zeta
                    return Dict("method" => "RUME", "subsample" => s, "location" => t)
                end
            else
                conf=0.5*exp(-1)*(delta_t/2)^(2/s) - epsilon
                if conf>0
                    diff_median = abs(median(online_data[(t-s+1):t]) - median(online_data[1:s]))
                    chi = 2*sigma*C2*(log(2/conf))^(1/theta)
                    if diff_median > chi
                        return Dict("method" => "median", "subsample" => s, "location" => t)
                    end
                end
            end
        end
    end
    return Dict("method" => "no changepoint", "subsample" => -1, "location" => -1)
end


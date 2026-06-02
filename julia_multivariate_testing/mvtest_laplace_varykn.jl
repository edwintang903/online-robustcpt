using Distributed, DataFrames, CSV

@everywhere begin
    using Random, Statistics, LinearAlgebra, Distributions
    include("../julia_utils/hd_utils.jl")

    # Define fixed constants 
    const p_ = 100
    const s_=1/sqrt(2)
    const epsilon_ = 0.01
    const delta_ = 0.1   
    const C_gamma_ = 0.1
end

# Define grid of (n, kappa)
n_seq = 400:100:3000
kappa_seq = 0.1:0.01:0.45
reps = 2000

# 1. Create a single master DataFrame
results_df = DataFrame(Iterators.product(n_seq, kappa_seq))
rename!(results_df, [:n, :kappa0])
results_df.type1_error = zeros(nrow(results_df))
results_df.type2_error = zeros(nrow(results_df))

for idx in 1:nrow(results_df)
    curr_n = results_df.n[idx]
    curr_k = results_df.kappa0[idx]
    
    # pmap returns a vector of tuples [(t1, pwr), (t1, pwr), ...]
    sim_results = pmap(1:reps) do _
        # --- Type I Error ---
        n_corrupt = rand(Binomial(curr_n, epsilon_))
        n_clean = curr_n - n_corrupt
        corrupt = randn(n_corrupt, p_) .- 1/sqrt(p_)
        
        Y_t1 = [rlaplace_hd(n_clean, p_, s_, mu=zeros(p_)); corrupt]
        t1_rej = robust_mean_test(Y_t1, curr_k, delta_, epsilon_, C_gamma=C_gamma_)

        # --- Power ---
        n_corrupt = rand(Binomial(curr_n, epsilon_))
        n_clean = curr_n - n_corrupt
        corrupt = randn(n_corrupt, p_) .- 1/sqrt(p_)

        mu_vec = fill(curr_k / sqrt(p_), p_)
        Y_p = [rlaplace_hd(n_clean, p_, s_, mu=mu_vec); corrupt]
        pwr_rej = robust_mean_test(Y_p, curr_k, delta_, epsilon_, C_gamma=C_gamma_)

        return (t1_rej, pwr_rej)
    end

    # 2. Extract and assign directly to the DataFrame row
    results_df.type1_error[idx] = sum(r[1] for r in sim_results) / reps
    results_df.type2_error[idx] = 1 - (sum(r[2] for r in sim_results) / reps)
end

# 3. Save as a single combined file
CSV.write("samp_comp/th1e4p100e1.csv", results_df)
println("All results saved to th1e4p100.csv")


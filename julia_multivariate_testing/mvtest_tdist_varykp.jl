using Distributed, DataFrames, CSV

# Define grid (p,kappa)
k_seq = 0.2:0.02:0.6
p_seq = 10:10:500
reps = 2000

@everywhere begin
    using Random, Statistics, LinearAlgebra, Distributions
    include("../julia_utils/hd_utils.jl") 

    # Define constants 
    const n_ = 500
    const epsilon_ = 0.01
    const delta_ = 0.1
end

# 1. Create a single master DataFrame
results_df = DataFrame(Iterators.product(k_seq, p_seq))
rename!(results_df, [:kappa, :p])
results_df.type1_error = fill(NaN, nrow(results_df))
results_df.type2_error = fill(NaN, nrow(results_df))

for idx in 1:nrow(results_df)
    curr_k = results_df.kappa[idx]
    curr_p = results_df.p[idx]
    q = epsilon_ + min(curr_p/(20*n_),1/20)
    if q + sqrt(2*q*log(4/delta_)/n_)+2*log(4/delta_)/(3*n_)>0.1
        results_df.type1_error[idx] = NaN
        results_df.type2_error[idx] = NaN
    else
    # pmap returns a vector of tuples [(t1, pwr), (t1, pwr), ...]
    sim_results = pmap(1:reps) do _
        for C_gamma_ in [0.1]
            # --- Type I Error ---
            n_corrupt = rand(Binomial(n_, epsilon_))
            n_clean = n_ - n_corrupt
            corrupt = randn(n_corrupt, curr_p) .- 1/sqrt(curr_p)
            
            Y_t1 = [rt_hd(n_clean, curr_p, 4.1, sd=1, mu=zeros(curr_p)); corrupt]
            t1_rej = robust_mean_test(Y_t1, curr_k, delta_, epsilon_, C_gamma=C_gamma_)
 
            # --- Power ---
            n_corrupt = rand(Binomial(n_, epsilon_))
            n_clean = n_ - n_corrupt
            corrupt = randn(n_corrupt, curr_p) .- 1/sqrt(curr_p)

            mu_vec = fill(curr_k / sqrt(curr_p), curr_p)
            Y_p = [rt_hd(n_clean, curr_p, 4.1, sd=1, mu=mu_vec); corrupt]
            pwr_rej = robust_mean_test(Y_p, curr_k, delta_, epsilon_, C_gamma=C_gamma_)
            return (t1_rej, pwr_rej)
#            if (t1_rej == 0 && pwr_rej == 1) #|| C_gamma_ == 0.06
#                return (t1_rej, pwr_rej)
#            end
         end
    end
    # 2. Extract and assign directly to the DataFrame row
    results_df.type1_error[idx] = sum(r[1] for r in sim_results) / reps
    results_df.type2_error[idx] = 1 - (sum(r[2] for r in sim_results) / reps)
    end
end

# 3. Save as a single combined file
CSV.write("samp_comp/v4varyp_k.csv", results_df)
println("All results saved to v4varyp_k.csv")


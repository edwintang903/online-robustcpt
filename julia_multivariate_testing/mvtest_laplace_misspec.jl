using Distributed, DataFrames, CSV

@everywhere begin
    using Random, Statistics, LinearAlgebra, Distributions
    include("../julia_utils/hd_utils.jl")

    # Define constants for the workers
    const n_seq = 500
    const p_ = 600
    const s_=1/sqrt(2)
    const epsilon_ = 0.01
    const delta_ = 0.1
end

# Define your grid
C_gamma_seq = [0.01, 0.03, 0.05]
threshold_seq= 0.1:0.1:2.5
kappa_seq = 1
reps = 1000

heatmap_data = DataFrame(
    (n=n, t=t, C=C) 
    for n in n_seq, t in threshold_seq, C in C_gamma_seq
)

rename!(heatmap_data, [:n, :kappa0, :C_gamma])
heatmap_data.error_rate = zeros(nrow(heatmap_data))
heatmap_data.power = zeros(nrow(heatmap_data))

for idx in 1:nrow(heatmap_data)
    curr_n = heatmap_data.n[idx]
    curr_k0 = heatmap_data.kappa0[idx]
    curr_Cgam = heatmap_data.C_gamma[idx]
    println(curr_k0, curr_Cgam)
    Random.seed!(12)
    # pmap returns a vector of tuples [(t1, pwr), (t1, pwr), ...]
    sim_results = pmap(1:reps) do _
        # --- Type I Error ---
        n_corrupt = rand(Binomial(curr_n, epsilon_))
        if n_corrupt >0
            n_clean = curr_n - n_corrupt
            corrupt = randn(n_corrupt, p_) .- 1
            Y_t1 = [rlaplace_hd(n_clean, p_, s_, mu=zeros(p_)); corrupt]
        else
            Y_t1 = rlaplace_hd(curr_n, p_, s_, mu=zeros(p_))
        end
        t1_rej = robust_mean_test(Y_t1, curr_k0, delta_, epsilon_, C_gamma=curr_Cgam)

        # --- Power ---
        mu_vec = fill(kappa_seq / sqrt(p_), p_)
        n_corrupt = rand(Binomial(curr_n, epsilon_))
        if n_corrupt>0
            n_clean = curr_n - n_corrupt
            corrupt = randn(n_corrupt, p_) .- 1
            Y_p = [rlaplace_hd(n_clean, p_, s_, mu=mu_vec); corrupt]
        else
            Y_p = rlaplace_hd(curr_n, p_, s_, mu=mu_vec)
        end
        pwr_rej = robust_mean_test(Y_p, curr_k0, delta_, epsilon_, C_gamma=curr_Cgam)

        return (t1_rej, pwr_rej)
    end

    # 2. Extract and assign directly to the DataFrame row
    heatmap_data.error_rate[idx] = sum(r[1] for r in sim_results) / reps
    heatmap_data.power[idx] = sum(r[2] for r in sim_results) / reps
end

# 3. Save as a single combined file
CSV.write("sensitive/th1sensitive_p600smol.csv", heatmap_data)
println("All results saved to th1sensitive_p600.csv")


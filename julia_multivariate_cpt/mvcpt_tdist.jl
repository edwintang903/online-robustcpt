using Distributed, DataFrames, CSV

# Define your grid
kappa_seq = 0.15:0.1:0.35
reps = 1000

@everywhere begin
    using Random, Statistics, LinearAlgebra, Distributions
    include("../julia_utils/hd_utils.jl")

    # Define constants for the workers
    const p_ = 10
    const kappa_guess_ = 0.5
    const Tu_= 0.2
    const epsilon_ = 0.01
    const alpha_ = 0.1   
    const n_=7500
    const cpt_=3000
    const k_const_=1
    
    function run_one_sim(idx, curr_k, i)
        Y = rt_hd_cpt(n_, p_, epsilon_; cpt=cpt_, mu_norm=curr_k)
        r = 2
        low_limit = 1400
        for t in 2800:n_
            tested = false
            delta_t = 4*alpha_/(t*r*(r+1))
            limit = fld(t, 2)
            k = ceil(Int, k_const_*log(1/delta_t)) #Group numbers
            for s in low_limit:limit
                samplesize=fld(s,2)
                block_size = div(samplesize, k)
                q = 2*epsilon_ + min(p_/(20*block_size),1/20)
                u= q + sqrt(2*q*log(16k)/block_size)+2*log(16k)/(3*block_size)
                if u <= 0.15
                    current_Y = Y[1:t, :]
                    X_t = zeros(samplesize, p_)

                    for c in 1:samplesize
                        @inbounds X_t[c, :] .= (current_Y[t-c+1, :] .- current_Y[c, :]) ./ sqrt(2)
                    end

                    if robust_mean_test_mom(X_t, kappa_guess_/sqrt(2),
                                            delta_t, 2*epsilon_, Tu=Tu_, k_const=k_const_)
                        return (idx, i, t)
                    end
                    tested = true
                else 
                   low_limit=s+1
                end
            end
            if tested
                r += 1
            end
        end
        return (idx, i, n_+1)  # no detection case
    end
end

# 1. Create a single master DataFrame
df = DataFrame(kappa = kappa_seq)
for i in 1:reps
    df[!, Symbol("sim_$i")] = zeros(length(kappa_seq))
end

for idx in 1:nrow(df)
    curr_k = df.kappa[idx]

    sim_results = pmap(i -> run_one_sim(idx, curr_k, i), 1:reps)

    for (idx_res, i_res, t_res) in sim_results
        df[idx_res, Symbol("sim_$i_res")] = t_res
    end
end

# 3. Save as a single combined file
CSV.write("cpt/hdcpt_t_p10_small2.csv", df)
println("All results saved to cpt/hdcpt_t_p10.csv")


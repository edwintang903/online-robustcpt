using Distributed, DataFrames, CSV

# Define your grid
kappa_seq = 0.025:0.05:0.425
reps = 1000

@everywhere begin
    using Random, Statistics, LinearAlgebra, Distributions
    include("../julia_utils/hd_utils.jl")

    # Define constants for the workers
    const p_ = 10
    const kappa_guess_ = 0.5
    const Tu_= 2.8
    const epsilon_ = 0.01
    const alpha_ = 0.1  
    const n_=2000
    const cpt_=500
    
    function run_one_sim(idx, curr_k, i)
        Y = rlaplace_hd_cpt(n_, p_, epsilon_; cpt=cpt_, mu_norm=curr_k)
        r = 2
        for t in 2:n_
            current_Y = Y[1:t, :]
            delta_t = 4*alpha_/(t*r*(r+1))
            limit = fld(t, 2)
            L = log(1/delta_t)
            A = 1 + 2*L/3
            D = 0.09 - 2*epsilon_

            threshold_s = ceil(Int,(A^2-2*L)/(A*D+L*2*epsilon_ -
                sqrt(L*(4*L*epsilon_^2+4*A*epsilon_*D+2*D^2))))

            if threshold_s <= limit
                X_t = zeros(limit, p_)

                for s in 1:limit
                    @inbounds X_t[s, :] .= (current_Y[t-s+1, :] .- current_Y[s, :]) ./ sqrt(2)
                end

                for s in threshold_s:limit
                    if robust_mean_test(X_t[1:s,:], kappa_guess_/sqrt(2),
                                        delta_t, 2*epsilon_, Tu=Tu_)
                        return (idx, i, t)
                    end
                end
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
CSV.write("cpt/hdcpt_laplace_p10smaller.csv", df)
println("All results saved to hd_laplace_p10smaller.csv")


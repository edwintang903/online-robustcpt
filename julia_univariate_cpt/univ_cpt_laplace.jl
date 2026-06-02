using Distributed

@everywhere begin
    using Random, Statistics, StatsBase, DelimitedFiles
    include("change_point_utils.jl")
end

# ------------------------------------------------------------------
# Parameters (defined once, broadcast to workers)
# ------------------------------------------------------------------

epsilon = 0.1
theta = 1
sigma = sqrt(21) # Same as v=2
alpha = 0.2
n = 2400
C1=0.395 # RUME constant 0.35 0.37 (works) 0.375 0.39 0.38 
C2=0.0479 # median constant 0.054 0.05 (works) 0.047 0.048 0.483 
reps = 2000

#kappa_sizes = sigma *  # R2R3 (0.24:0.02:1) (0.25:0.02:0.53) # R1R2: (0.0:0.01:0.23)
powers_of_two = 2.0 .^ (0:0.25:20)
kappa_sizes = sigma .* powers_of_two

@everywhere begin
    const epsilon_ = $epsilon
    const theta_ = $theta
    const sigma_ = $sigma
    const alpha_ = $alpha
    const n_ = $n
    const C1_ = $C1
    const C2_ = $C2

    mechanism_df_ = (n, mu=0.0) ->
        contaminated_laplace(n, mu; epsilon=epsilon_)
end

# ------------------------------------------------------------------
# Parallel computation
# ------------------------------------------------------------------

locations = zeros(length(kappa_sizes), reps)

for (k_idx, kappa) in enumerate(kappa_sizes)

    println("Running kappa index $k_idx on $(nworkers()) workers")

    results = pmap(1:reps) do _
        online_data = change_point_model(
            n_;
            mechanism=mechanism_df_,
            cpt=600,
            kappa=kappa
        )

        result = rumedian_theta(online_data, sigma_; theta=theta_, epsilon=epsilon_, alpha=alpha_, C1=C1_, C2=C2_) 

        result["location"]
    end

    locations[k_idx, :] .= results
end

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

writedlm("locations_th1e10_R3R4.csv", locations, ',')
println("Finished successfully.")

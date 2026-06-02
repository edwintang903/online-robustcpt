using Distributed

@everywhere begin
    using Random, Statistics, StatsBase, DelimitedFiles
    include("change_point_utils.jl")

    function sigma_t(df, v)
        if v == 2
            return sqrt(df / (df - 2))
        elseif v == 4
            return (3 * df^2 / ((df - 2) * (df - 4)))^(1/4)
        elseif v == 6
            return (15 * df^3 / ((df - 2) * (df - 4) * (df - 6)))^(1/6)
        else
            error("Unsupported v")
        end
    end
end

# ------------------------------------------------------------------
# Parameters (defined once, broadcast to workers)
# ------------------------------------------------------------------

epsilon = 0.1
df = 2.1
v = 2
sigma = sigma_t(df, v)
alpha = 0.2
n = 2400
C1 = 0.21
C2 = 0.013
reps = 2000

kappa_sizes = sigma * (0.09:0.01:0.7) #(0:0.005:0.08) # (0.085:0.01:0.495)
#powers_of_two = 2.0 .^ (-1:0.25:20)
#kappa_sizes = sigma .* powers_of_two

@everywhere begin
    const epsilon_ = $epsilon
    const df_ = $df
    const v_ = $v
    const sigma_ = $sigma
    const alpha_ = $alpha
    const n_ = $n
    const C1_ = $C1
    const C2_ = $C2

    mechanism_df_ = (n, mu=0.0) ->
        contaminated_sample_t(n, mu; df=df_, epsilon=epsilon_)
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

        result = rumedian_v(
            online_data,
            sigma_;
            v=v_,
            epsilon=epsilon_,
            alpha=alpha_,
            C1=C1_,
            C2=C2_
        )

        result["location"]
    end

    locations[k_idx, :] .= results
end

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

writedlm("locations_v2e10_R2R3more.csv", locations, ',')
println("Finished successfully.")


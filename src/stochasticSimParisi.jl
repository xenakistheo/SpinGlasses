"""
This file contains a "naive" calculation of the solution 
to the Parisi PDE using a stochastic simulation. 
The algorithm is O(N^m) where N is the number of simulations 
and m is the number of steps in the discretization of the time interval [0,1]. 
The code is not optimized and serves as a proof of concept for the algorithm.

Run at your own risk :) 
[It might take a while to run]
"""

### Equation 7.3 in Optimizing Mean Field Spin Glasses
function phi(j, x, gammas, rootR, G, Nsim, m)
    if j == m
        return abs(x)
    else
        γ = gammas[j+1]
        vals = [
            exp(γ * phi(j+1, x + rootR[j+1] * G[j+1, k], gammas, rootR, G, Nsim, m))
            for k in 1:Nsim
        ]
        return (1/γ) * log(sum(vals) / Nsim)
    end
end



m = 10
Nsim_vals = 1:8
time_vals = zeros(length(Nsim_vals))

for (i, Nsim) in enumerate(Nsim_vals)
    # Nsim = 4

    G = randn((m+1, Nsim))
    tsteps = collect(range(0,1,m+1))
    dt = 1/(m)
    ξ_c = 1 # Constant of mixture function


    rootR = sqrt.(3*ξ_c * dt .* (2*tsteps[2:end] .- dt^2))

    gammas = ones(m)


    time_vals[i] = @elapsed phi(0, 0, ones(m), rootR, G, Nsim, m)
end

using Plots
plot(Nsim_vals, time_vals, xlabel="N",
    ylabel="time [s]", title="time complexity, m=10")



using FastGaussQuadrature
using Interpolations
using Statistics


"""
    solve_parisi_p3(tgrid, g; L=8.0, Nx=1001, Q=40)

Solves the Parisi PDE for pure p=3 with piecewise constant gamma.

gamma(t) = g[i] for t in [tgrid[i], tgrid[i+1])

Inputs:
- tgrid: vector of length K+1, e.g. range(0, 1, length=K+1)
- g: vector of length K, nonnegative, nondecreasing values of gamma

Returns:
- phi00: approximation of Phi_gamma(0,0)
- xgrid: spatial grid
- Phi: approximation of Phi_gamma(0,x) on xgrid
"""
function solve_parisi_p3(tgrid, g; L=8.0, Nx=1001, Q=40)
    K = length(g)

    @assert length(tgrid) == K + 1
    @assert tgrid[1] ≈ 0.0
    @assert tgrid[end] ≈ 1.0
    @assert all(g .>= 0.0)

    # spatial grid
    xgrid = collect(range(-L, L, length=Nx))

    # terminal condition Phi(1,x) = |x|
    Phi = abs.(xgrid)

    # Gauss-Hermite nodes and weights
    nodes, weights = gausshermite(Q)

    # Convert Gauss-Hermite rule to standard normal expectation:
    # E[f(Z)] ≈ 1/sqrt(pi) * sum_q w_q f(sqrt(2)*x_q)
    weights = weights ./ sqrt(pi)
    z_nodes = sqrt(2.0) .* nodes

    # Work backwards over intervals
    for i in K:-1:1
        a = tgrid[i]
        b = tgrid[i+1]
        m = g[i]

        # For p=3: xi'(t) = 3t^2
        # A_i = xi'(b) - xi'(a)
        A = 3.0 * (b^2 - a^2)
        sqrtA = sqrt(A)
        shifts = sqrtA .* z_nodes  # Q-vector: shift per quadrature node

        interp = LinearInterpolation(xgrid, Phi, extrapolation_bc=Line())

        # Build Nx×Q matrix: interp_vals[j,q] = Phi(xgrid[j] + shifts[q])
        interp_vals = similar(Phi, Nx, Q)
        for q in eachindex(z_nodes)
            interp_vals[:, q] .= interp.(xgrid .+ shifts[q])
        end

        if abs(m) < 1e-10
            # m = 0 limit: Gaussian average — matrix-vector multiply
            Phi = interp_vals * weights
        else
            # stable log-sum-exp, vectorized over all x at once
            scaled = m .* interp_vals                        # Nx × Q
            maxvals = maximum(scaled, dims=2)                # Nx × 1
            expectation = exp.(scaled .- maxvals) * weights  # Nx-vector
            Phi = (vec(maxvals) .+ log.(expectation)) ./ m
        end
    end

    # interpolate Phi(0,0)
    interp0 = LinearInterpolation(xgrid, Phi, extrapolation_bc=Line())
    phi00 = interp0(0.0)

    return phi00, xgrid, Phi
end


# K = 20
# tgrid = collect(range(0.0, 1.0, length=K+1))

# # Example monotone gamma values
# g = collect(range(0.1, 2.0, length=K))

# phi00, xgrid, Phi0 = solve_parisi_p3(tgrid, g)

# println("Phi_gamma(0,0) = ", phi00)



function parisi_functional_p3(tgrid, g; L=8.0, Nx=1001, Q=40)
    phi00, _, _ = solve_parisi_p3(tgrid, g; L=L, Nx=Nx, Q=Q)

    penalty = 0.0
    for i in eachindex(g)
        penalty += g[i] * (tgrid[i+1]^3 - tgrid[i]^3)
    end

    return phi00 - penalty
end

# Pval = parisi_functional_p3(tgrid, g)
# println("P(gamma) = ", Pval)



function theta_to_g(theta)
    increments = log1p.(exp.(theta))  # softplus
    return cumsum(increments)
end

function make_objective(tgrid; L=8.0, Nx=1001, Q=40)
    function objective(theta)
        g = theta_to_g(theta)
        return parisi_functional_p3(tgrid, g; L=L, Nx=Nx, Q=Q)
    end
    return objective
end

function make_objective_g(tgrid; L=8.0, Nx=1001, Q=40)
    function objective(g)
        return parisi_functional_p3(tgrid, g; L=L, Nx=Nx, Q=Q)
    end
    return objective
end

K = 30
tgrid = collect(range(0.0, 1.0, length=K+1))

objective = make_objective(tgrid; L=8.0, Nx=1001, Q=40)
objective_g = make_objective_g(tgrid; L=8.0, Nx=1001, Q=40)

theta0 = log.(fill(0.05, K))  # gives small positive increments


g0 = theta_to_g(theta0)
val = objective(theta0) #1.2512224485245205

println(val)

using ADTypes: AutoForwardDiff

# lower = zeros(K)  # theta can be any real number, but we want g to be nonnegative
# upper = ones(K)


# @time res = optimize(objective, theta0, BFGS()) #14s, 16GB
# @time res = optimize(objective, theta0, LBFGS(; m=10)) #29s, 33GB
@time res = optimize(objective, theta0, LBFGS(); autodiff=AutoForwardDiff()) #6s, 17GB

# @time res = optimize(objective_g, g0, LBFGS(); autodiff=AutoForwardDiff()) #
# @time res = optimize(objective_g, lower, upper, g0, Fminbox(LBFGS()); autodiff=AutoForwardDiff()) #


# g_star = Optim.minimizer(res)

theta_star = Optim.minimizer(res)
g_star = theta_to_g(theta_star)

println("Optimized P(gamma) = ", objective(theta_star))# 1.1505790804832279
println(g_star) #0.9783148176005934

# println("Optimized P(gamma) = ", objective_g(g_star))
# println(g_star) #0.9783148176005934

plot(
    tgrid[1:end-1],
    g_star,
    seriestype=:steppost,
    xlabel="t",
    ylabel="γ*(t)",
    linewidth=2,
    label="optimized γ"
)

diff(g_star) 


best_res = nothing
best_val = Inf

for seed in 1:10
    @show seed
    theta0 = randn(K)

    res = optimize(objective, theta0, LBFGS(); autodiff=AutoForwardDiff())
    val = Optim.minimum(res)

    if val < best_val
        @show val
        best_val = val
        best_res = res
    end
end

theta_star = Optim.minimizer(best_res)
g_star = theta_to_g(theta_star)

plot(
    tgrid[1:end-1],
    g_star,
    seriestype=:steppost,
    xlabel="t",
    ylabel="γ*(t)",
    linewidth=2,
    label="optimized γ"
)


# We expect flat region for p=3
# for p=2 we should not see flat region, but rather a smooth increase in gamma.
# for p\geq 4 we should see flatness

# For both K=10 and K=30 we see flatness in gamma all the way until t = 1. 
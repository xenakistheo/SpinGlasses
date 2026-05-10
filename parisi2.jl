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

        # interpolation of Phi at next time level
        interp = LinearInterpolation(xgrid, Phi, extrapolation_bc=Line())

        Phi_new = similar(Phi)

        for (j, x) in enumerate(xgrid)
            vals = similar(z_nodes)

            for q in eachindex(z_nodes)
                y = x + sqrtA * z_nodes[q]
                vals[q] = interp(y)
            end

            if abs(m) < 1e-10
                # m = 0 limit: Gaussian average
                Phi_new[j] = sum(weights .* vals)
            else
                # stable log-sum-exp version
                maxval = maximum(m .* vals)
                expectation = sum(weights .* exp.(m .* vals .- maxval))
                Phi_new[j] = (maxval + log(expectation)) / m
            end
        end

        Phi = Phi_new
    end

    # interpolate Phi(0,0)
    interp0 = LinearInterpolation(xgrid, Phi, extrapolation_bc=Line())
    phi00 = interp0(0.0)

    return phi00, xgrid, Phi
end


K = 20
tgrid = collect(range(0.0, 1.0, length=K+1))

# Example monotone gamma values
g = collect(range(0.1, 2.0, length=K))

phi00, xgrid, Phi0 = solve_parisi_p3(tgrid, g)

println("Phi_gamma(0,0) = ", phi00)



function parisi_functional_p3(tgrid, g; L=8.0, Nx=1001, Q=40)
    phi00, _, _ = solve_parisi_p3(tgrid, g; L=L, Nx=Nx, Q=Q)

    penalty = 0.0
    for i in eachindex(g)
        penalty += g[i] * (tgrid[i+1]^3 - tgrid[i]^3)
    end

    return phi00 - penalty
end

Pval = parisi_functional_p3(tgrid, g)
println("P(gamma) = ", Pval)



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

K = 10
tgrid = collect(range(0.0, 1.0, length=K+1))

objective = make_objective(tgrid; L=8.0, Nx=1001, Q=40)

theta0 = log.(fill(0.05, K))  # gives small positive increments

g0 = theta_to_g(theta0)
val = objective(theta0) #1.2512224485245205

println(val)

using Optim

@time res = optimize(objective, theta0, BFGS()) #24s, 20.057

theta_star = Optim.minimizer(res)
g_star = theta_to_g(theta_star)

println("Optimized P(gamma) = ", objective(theta_star))
println(g_star) #0.9783148176005934

using Plots

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

for seed in 1:20
    theta0 = randn(K)

    res = optimize(objective, theta0, BFGS())
    val = Optim.minimum(res)

    if val < best_val
        best_val = val
        best_res = res
    end
end

theta_star = Optim.minimizer(best_res)
g_star = theta_to_g(theta_star)
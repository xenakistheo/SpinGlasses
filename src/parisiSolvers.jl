using FastGaussQuadrature
using Interpolations
using Statistics
using Plots




"""
    solve_parisi(tgrid, g; p=3, L=8.0, Nx=1001, Q=40)

Solves the Parisi PDE for pure p with piecewise constant gamma.

gamma(t) = g[i] for t in [tgrid[i], tgrid[i+1])

Inputs:
- tgrid: vector of length K+1, e.g. range(0, 1, length=K+1)
- g: vector of length K, nonnegative, nondecreasing values of gamma

Returns:
- phi00: approximation of Phi_gamma(0,0)
- xgrid: spatial grid
- Phi: approximation of Phi_gamma(0,x) on xgrid
"""

function solve_parisi(tgrid, g; p=3, L=8.0, Nx=1001, Q=40)
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

        # For p=p: xi'(t) = pt^(p-1)
        # A_i = xi'(b) - xi'(a)
        A = p * (b^(p-1) - a^(p-1))
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



function parisi_functional(tgrid, g; p=3, L=8.0, Nx=1001, Q=40)
    phi00, _, _ = solve_parisi(tgrid, g; p=p, L=L, Nx=Nx, Q=Q)

    penalty = 0.0
    for i in eachindex(g)
        penalty += g[i] * (tgrid[i+1]^p - tgrid[i]^p)
    end

    return phi00 - penalty
end


function theta_to_g(theta)
    increments = log1p.(exp.(theta))  # softplus
    return cumsum(increments)
end

function make_objective(tgrid; p=3, L=8.0, Nx=1001, Q=40)
    function objective(theta)
        g = theta_to_g(theta)
        return parisi_functional(tgrid, g; p=p, L=L, Nx=Nx, Q=Q)
    end
    return objective
end


function make_objective_g(tgrid; p=3, L=8.0, Nx=1001, Q=40)
    function objective(g)
        return parisi_functional(tgrid, g; p=p, L=L, Nx=Nx, Q=Q)
    end
    return objective
end

# K = 20
# tgrid = collect(range(0.0, 1.0, length=K+1))

# # Example monotone gamma values
# g = collect(range(0.1, 2.0, length=K))

# phi00, xgrid, Phi0 = solve_parisi(tgrid, g)

# println("Phi_gamma(0,0) = ", phi00)

# Pval = parisi_functional_p3(tgrid, g)
# println("P(gamma) = ", Pval)



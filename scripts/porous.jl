using Revise
using StabilityNonlinearDiffusion, RadiiPolynomial
using GLMakie, CairoMakie, Polynomials





#################
# Porous medium #
#################

model = Porous(; d₁₁ = I"2.4259", d₁₂ = I"0.6938", d₂₁ = I"9.2038", d₂₂ = I"2.8059",
                 r₁  = I"9.9224", a₁  = I"9.0512", b₁  = I"6.6474",
                 r₂  = I"6.0865", b₂  = I"2.3864", a₂  = I"5.7359")

# domain [0, 1]

ω = interval(π) # frequency

# Newton's method

mid_model = Porous(; d₁₁ = 2.4259, d₁₂ = 0.6938, d₂₁ = 9.2038, d₂₂ = 2.8059,
                     r₁  = 9.9224, a₁  = 9.0512, b₁  = 6.6474,
                     r₂  = 6.0865, b₂  = 2.3864, a₂  = 5.7359)

K = 40

# some completely random data -- failure...

#u_guess = Sequence(CosFourier(K, mid(ω))^2, [[0.1 * (-1)^rand(Bool) * rand()/2.0^k for k = 0:K] ; [0.01 * (-1)^rand(Bool) * rand()/2.0^k for k = 0:K]])

#u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)

# using Antoine's data

function extract_uv_series(filepath::String)
    # Read the file content
    lines = readlines(filepath)

    # Initialize containers
    u_values = Float64[]
    v_values = Float64[]
    current_var = ""

    # Loop over the lines and sort values into the correct container
    for line in lines
        line = strip(line)
        if isempty(line)
            continue
        elseif line == "u"
            current_var = "u"
        elseif line == "v"
            current_var = "v"
        else
            value = parse(Float64, line)
            if current_var == "u"
                push!(u_values, value)
            elseif current_var == "v"
                push!(v_values, value)
            end
        end
    end

    return u_values, v_values
end

function interpolate(u_grid, t)
    time_grid = LinRange(0, 1, length(u_grid))
    ts = [[time_grid[i], time_grid[i+1]] for i ∈ 1:length(u_grid)-1]
    t = mod(t, 2)
    if t ≤ 1
        (idx = findfirst(v -> v[1] ≤ t ≤ v[2], ts))
        tmin = time_grid[idx]
        tmax = time_grid[idx+1]
        ymin = u_grid[idx]
        ymax = u_grid[idx+1]
        return ( ymax + (ymax - ymin)/(tmax - tmin) * (t - tmax) )
    else
        t ≥ 1
        return interpolate(u_grid, -t)
    end
end

u₁_grid, u₂_grid = extract_uv_series("scripts/profils.txt") #Needed to add scripts/...

import ApproxFun, ApproxFunFourier
# u1_approx = ApproxFun.Fun(t -> interpolate(u₁_grid, t), ApproxFun.Laurent(ApproxFun.PeriodicSegment(-1, 1)), 200)
# u2_approx = ApproxFun.Fun(t -> interpolate(u₂_grid, t), ApproxFun.Laurent(ApproxFun.PeriodicSegment(-1, 1)), 200)
u1_approx_cos = ApproxFun.Fun(t -> interpolate(u₁_grid, t), ApproxFun.CosSpace(ApproxFun.PeriodicSegment(-1, 1)), 4000)
u2_approx_cos = ApproxFun.Fun(t -> interpolate(u₂_grid, t), ApproxFun.CosSpace(ApproxFun.PeriodicSegment(-1, 1)), 4000)

## visual check

fig = Figure()

ax1 = Axis(fig[1,1])
# lines!(ax1, LinRange(0, 1, length(u₁_grid)), t -> real(u1_approx(t)); linewidth = 4, color = :red)
lines!(ax1, LinRange(0, 1, length(u₁_grid)), t -> u1_approx_cos(t); linewidth = 4, color = :green)
scatter!(ax1, Point2f.(LinRange(0, 1, length(u₁_grid)), u₁_grid))

ax2 = Axis(fig[1,2])
# lines!(ax2, LinRange(0, 1, length(u₂_grid)), t -> real(u2_approx(t)); linewidth = 4, color = :red)
lines!(ax2, LinRange(0, 1, length(u₁_grid)), t -> u2_approx_cos(t); linewidth = 4, color = :green)
scatter!(ax2, Point2f.(LinRange(0, 1, length(u₂_grid)), u₂_grid))

##

N = 100
# u1 = Sequence(CosFourier(K, mid(ω)), [real(u1_approx.coefficients[i]) for i = 1:2:2K+1])
# u2 = Sequence(CosFourier(K, mid(ω)), [real(u2_approx.coefficients[i]) for i = 1:2:2K+1])

# u1_cos = Sequence(CosFourier(K, mid(ω)), u1_approx_cos.coefficients[1:K+1])
# u2_cos = Sequence(CosFourier(K, mid(ω)), u2_approx_cos.coefficients[1:K+1])
u1_cos = Sequence(CosFourier(N, mid(ω)), u1_approx_cos.coefficients[1:N+1] .* [1 ; fill(0.5, N)])
u2_cos = Sequence(CosFourier(N, mid(ω)), u2_approx_cos.coefficients[1:N+1] .* [1 ; fill(0.5, N)])

# u_guess = Sequence(CosFourier(K, mid(ω))^2, [u1.coefficients ; u2.coefficients])
u_guess_cos = Sequence(CosFourier(N, mid(ω))^2, [u1_cos.coefficients ; u2_cos.coefficients])

# u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)
u_approx_cos, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess_cos)



# fig = Figure()

# ax1 = Axis(fig[1,1])
# lines!(ax1, LinRange(0, 1, length(u₁_grid)), t -> component(u_approx_cos, 1)(t); linewidth = 4, color = :green)
# scatter!(ax1, Point2f.(LinRange(0, 1, length(u₁_grid)), u₁_grid))

# ax2 = Axis(fig[1,2])
# lines!(ax2, LinRange(0, 1, length(u₁_grid)), t -> component(u_approx_cos, 2)(t); linewidth = 4, color = :green)
# scatter!(ax2, Point2f.(LinRange(0, 1, length(u₂_grid)), u₂_grid))

#-------#
# Proof #
#-------#

K = 30#3600

#component(u_approx_cos, 1)[1:2:end] .= 0
#component(u_approx_cos, 2)[1:2:end] .= 0
CFK = CosFourier(K, ω)^2
CFK2 = CosFourier(2K, ω)^2
CFK3 = CosFourier(3K, ω)^2
CFK4 = CosFourier(4K, ω)^2
coeff_u_bar = coefficients(project(u_approx_cos, CosFourier(K, mid(ω))^2))
u_bar = Sequence(CFK, interval(
        coeff_u_bar)
        )

# construct approx inverse

A_bar = StabilityNonlinearDiffusion.A(model, [component(u_bar, 1), component(u_bar, 2)])
L_bar = DF(model, u_bar, CFK3, CFK4)
L_bar_small = project(L_bar, CFK2, CFK3)


θ = approx_inv(A_bar)
approx_finit_inv = inv(project(mid.(L_bar), CFK3, CFK3))
approx_DF⁻¹ = ApproxInverse(project(approx_finit_inv, CFK2, CFK2), θ)

# θΔ⁻¹_N∞ = project(approx_DF⁻¹, CFK3, CFK2) - approx_DF⁻¹.finite_matrix
# approx_DF⁻¹_bis = ApproxInverse( (I - θΔ⁻¹_N∞*L_bar_small)*approx_DF⁻¹.finite_matrix, θ)
#

F_bar = F(model, u_bar, CFK2)

ν = 1.001
X = Ell1(GeometricWeight(ν))
X² = NormedCartesianSpace(X, Ell1())

Y = norm(project(approx_DF⁻¹, space(F_bar), CFK3) * F_bar, X²)

C_bar = StabilityNonlinearDiffusion.C(model, [component(u_bar, 1),component(u_bar, 2)])
B_bar = StabilityNonlinearDiffusion.B(model, [component(u_bar, 1),component(u_bar, 2)])
normθ = opnorm(norm.(θ, [X]), 1)

Z₁_a = opnorm(I - project(approx_DF⁻¹, CFK3, CFK4) * L_bar_small, X²)
# Z₁_a_bis = opnorm(I - project(approx_DF⁻¹_bis, CFK3, CFK4) * L_bar_small, X²)
Z₁_b1 = opnorm(norm.([1. 0. ; 0. 1.] - θ * A_bar, [X]), 1)
Z₁_b2 = inv((2*K + 1) * ω) * normθ * opnorm(norm.(B_bar[1], [X]), 1)
Z₁_b3 = inv((2*K + 1) * ω)^2 * normθ * opnorm(norm.(C_bar, [X]), 1)

Z₁ = max(Z₁_a, Z₁_b1 + Z₁_b2 + Z₁_b3)

#
# spy(abs.(mid.(coefficients(project(approx_DF⁻¹, domain(L_bar_small), CFK3)))))
# spy(abs.(mid.(coefficients(I - project(approx_DF⁻¹, domain(L_bar_small), CFK3) * L_bar))))
# spy(abs.(mid.(coefficients(L_bar))))

opnorm_approxDF⁻¹Δ = max(opnorm(approx_DF⁻¹.finite_matrix * project(Laplacian(),CFK2, CFK2), X²), normθ)

normA′ = 2 * maximum(abs.([model.d₁₁, model.d₁₂, model.d₂₁, model.d₂₂]))
norm∇⁻¹B′ = abs(model.d₁₂) + abs(model.d₂₁)
normΔ⁻¹C′ = maximum([2*abs(model.a₁), 2*abs(model.a₂), abs(model.b₁)+ abs(model.b₂)])
Z₂ = opnorm_approxDF⁻¹Δ * (normA′ + norm∇⁻¹B′ + normΔ⁻¹C′)
ϵ_u = interval(inf(interval_of_existence(Y, Z₁, Z₂, Inf)))

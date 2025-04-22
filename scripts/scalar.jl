# using Revise
using StabilityNonlinearDiffusion, RadiiPolynomial, LinearAlgebra
using GLMakie





##################
# Scalar example #
##################



model = ScalarExample(; α = I"1", β = I"1")

# domain [0, 1]

ω = interval(π) # frequency

# Newton's method

mid_model = ScalarExample(; α = 1.0, β = 1.0)

K = 20

u_guess = Sequence(CosFourier(K, mid(ω))^1, [1.362741344081890 ; 0.052107816731015 ; 0.008200891820525 ; -0.002635040629728 ; 0.007018830076427 ; zeros(K-4)])

u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)

# lines(LinRange(0, 1, 201), x -> u_approx(x)[1])



#-------#
# Proof #
#-------#

u_bar = Sequence(CosFourier(K, ω)^1, interval(coefficients(u_approx)))

# construct approx inverse

A_bar = StabilityNonlinearDiffusion.A(model, [component(u_bar, 1)])
L_bar = DF(model, u_bar, CosFourier(2K, ω)^1, CosFourier(3K, ω)^1)

approx_DF⁻¹ = ApproxInverse(project(inv(project(mid.(L_bar), CosFourier(2K, ω)^1, CosFourier(2K, ω)^1)), space(u_bar), space(u_bar)), approx_inv(A_bar))

#

F_bar = F(model, u_bar, CosFourier(2K, ω)^1)

Y = norm(project(approx_DF⁻¹, space(F_bar), CosFourier(3K, ω)^1) * F_bar, 1)

#

C_bar = StabilityNonlinearDiffusion.C(model, [component(u_bar, 1)])

Z₁ = max(opnorm(I - project(approx_DF⁻¹, codomain(L_bar), CosFourier(4K, ω)^1) * L_bar, 1),
         opnorm(norm.([1.] - approx_DF⁻¹.sequence_tail * A_bar, 1), 1) + inv((K + 1) * ω)^2 * opnorm(norm.(approx_DF⁻¹.sequence_tail, 1), 1) * opnorm(norm.(C_bar, 1), 1))

#

opnorm_approxDF⁻¹Δ = max(opnorm(approx_DF⁻¹.finite_matrix * project(Laplacian(), space(u_bar), space(u_bar)), 1),
                        opnorm(norm.(approx_DF⁻¹.sequence_tail, 1), 1))

Z₂ = opnorm_approxDF⁻¹Δ * 2 * (1 + model.β)

#

ϵ_u = interval(inf(interval_of_existence(Y, Z₁, Z₂, Inf)))



#--------------------#
# Stability analysis #
#--------------------#

N = 10

# construct P

Γ = LinearOperator(CosFourier(2N, ω)^1, CosFourier(2N, ω)^1, Diagonal(repeat([1 ; fill(1/2, 2N)], 1)))
Γ⁻¹ = inv(Γ)
L = DF(model, u_bar, CosFourier(2N, ω)^1, CosFourier(2N, ω)^1)

P_finite_2N = Γ * LinearOperator(CosFourier(2N, ω)^1, CosFourier(2N, ω)^1, Symmetric(lyap(adjoint(mid.(coefficients(L))), coefficients(Γ⁻¹))))
P_finite = project(P_finite_2N, CosFourier(N, ω)^1, CosFourier(N, ω)^1)

# Xcos = Γ⁻¹ * P_finite_2N; Xcos * L + adjoint(L) * Xcos
# P_finite_2N * L + Γ * adjoint(L) * Γ⁻¹ * P_finite_2N

P_bar = solve_lyap(A_bar)

P = OperatorP(P_finite, P_bar)

# construct Q

M = N+K # this is assumed everywhere
P_ext = project(P, CosFourier(M+2K, ω) ^ 1, CosFourier(M+2K, ω) ^ 1)
L_ext = DF(model, u_bar, CosFourier(M+2K, ω)^1, CosFourier(M+2K, ω)^1)

Γ = LinearOperator(CosFourier(M+2K, ω)^1, CosFourier(M+2K, ω)^1, Diagonal(repeat([1 ; fill(1/2, M+2K)], 1)))
Γ⁻¹ = inv(Γ)

Q = - (project(P_ext, CosFourier(M+K, ω)^1, CosFourier(M+2K, ω)^1) * project(L_ext, CosFourier(M, ω)^1, CosFourier(M+K, ω)^1) +
       project(Γ * adjoint(L_ext) * Γ⁻¹, CosFourier(M+K, ω)^1, CosFourier(M+2K, ω)^1) * project(P_ext, CosFourier(M, ω)^1, CosFourier(M+K, ω)^1))

#

μ = 1 - (C₀(P, opnorm_approxDF⁻¹Δ, Z₂, ϵ_u; N, K) +
            max(opnorm(Q - I, 1),
                opnorm(norm.(P.W_bar * A_bar + A_bar * P.W_bar - [1.], 1), 1) + C₁(P.W_bar, A_bar; N, K) + C₃(P.W_bar, C_bar; N, K)))

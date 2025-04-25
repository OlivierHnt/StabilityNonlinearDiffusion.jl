using Revise
using StabilityNonlinearDiffusion, RadiiPolynomial, LinearAlgebra
using GLMakie, MAT



#######
# SKT #
#######

file = matopen("solutions SKT 1D/datanontri1.mat")
vars = read(file)
u₁ = read(file,"u")
u₂ = read(file,"v")
d = read(file, "d")
close(file)


model = SKT(;d₁ = interval(d), d₂ = interval(d), d₁₁ = I"0", d₁₂ = I"3", d₂₁ = I"0.055", d₂₂ = I"0.",
                r₁  = I"5", a₁  = I"3", b₁  = I"1",
                r₂  = I"2", b₂  = I"1", a₂  = I"3")

# domain [0, 1]

ω = interval(π) # frequency

# Newton's method

mid_model = SKT(;d₁ = d, d₂ = d, d₁₁ = 0., d₁₂ = 3., d₂₁ = 0.055, d₂₂ = 0.,
r₁  = 5., a₁  = 3., b₁  = 1.,
r₂  = 2., b₂  = 1., a₂  = 3.)

K = 100

u_guess = Sequence(CosFourier(K, mid(ω))^2, [u₁[1:K+1] ; u₂[1:K+1]])

u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)


##
t = time()


#-------#
# Proof #
#-------#

u_bar = Sequence(CosFourier(K, ω)^2, interval(coefficients(u_approx)))

# construct approx inverse

A_bar = StabilityNonlinearDiffusion.A(model, [component(u_bar, 1), component(u_bar, 2)])
L_bar = DF(model, u_bar, CosFourier(3K, ω)^2, CosFourier(4K, ω)^2)
L_bar_small = project(L_bar, CosFourier(2K, ω)^2, CosFourier(3K, ω)^2)

approx_DF⁻¹ = ApproxInverse(
    project(inv(project(mid.(L_bar), CosFourier(3K, ω)^2, CosFourier(3K, ω)^2)), CosFourier(2K, ω)^2, CosFourier(2K, ω)^2),
    approx_inv(A_bar))

#

F_bar = F(model, u_bar, CosFourier(2K, ω)^2)

ν = 1.001
X = Ell1(GeometricWeight(ν))
X² = NormedCartesianSpace(X, Ell1())

Y = norm(project(approx_DF⁻¹, space(F_bar), CosFourier(3K, ω)^2) * F_bar, X²)

C_bar = StabilityNonlinearDiffusion.C(model, [component(u_bar, 1),component(u_bar, 2)])
#B_bar = StabilityNonlinearDiffusion.B(model, [component(u_bar, 1),component(u_bar, 2)])
normθ = opnorm(norm.(approx_DF⁻¹.sequence_tail, [X]), 1)

Z₁_a = opnorm(I - project(approx_DF⁻¹, CosFourier(3K, ω)^2, CosFourier(4K, ω)^2) * L_bar_small, X²)
Z₁_b1 = opnorm(norm.([1. 0. ; 0. 1.] - approx_DF⁻¹.sequence_tail * A_bar, [X]), 1)

Z₁_b3 = inv((2*K + 1) * ω)^2 * normθ * opnorm(norm.(C_bar, [X]), 1)

Z₁ = max(Z₁_a, Z₁_b1 + Z₁_b3)

#
# spy(abs.(mid.(coefficients(project(approx_DF⁻¹, domain(L_bar_small), CosFourier(3K, ω)^2)))))
# spy(abs.(mid.(coefficients(I - project(approx_DF⁻¹, domain(L_bar_small), CosFourier(3K, ω)^2) * L_bar))))
# spy(abs.(mid.(coefficients(L_bar))))

opnorm_approxDF⁻¹Δ = max(opnorm(approx_DF⁻¹.finite_matrix * project(Laplacian(), CosFourier(2K,ω)^2, CosFourier(2K,ω)^2), X²), normθ)

normA′ = 2 * maximum(abs.([model.d₁₁, model.d₁₂, model.d₂₁, model.d₂₂]))

normΔ⁻¹C′ = maximum([2*abs(model.a₁), 2*abs(model.a₂), abs(model.b₁)+ abs(model.b₂)])
Z₂ = opnorm_approxDF⁻¹Δ * (normA′ + normΔ⁻¹C′)
ϵ_u = interval(inf(interval_of_existence(Y, Z₁, Z₂, Inf)))
time() - t

#-----------#
# Stability #
#-----------#

N = 3K #need to be > 2K

# construct P
D = Diagonal(repeat([1 ; fill(1/2, 2N)], 1))
Γ = LinearOperator(CosFourier(2N, ω)^2, CosFourier(2N, ω)^2, [D zero(D); zero(D) D])
Γ⁻¹ = inv(Γ)
L = DF(model, u_bar, CosFourier(2N, ω)^2, CosFourier(2N, ω)^2)

P_finite_2N = Γ * LinearOperator(CosFourier(2N, ω)^2, CosFourier(2N, ω)^2, Symmetric(lyap(adjoint(mid.(coefficients(L))), coefficients(Γ⁻¹))))
P_finite = project(P_finite_2N, CosFourier(N, ω)^2, CosFourier(N, ω)^2)

# Xcos = Γ⁻¹ * P_finite_2N; Xcos * L + adjoint(L) * Xcos
# P_finite_2N * L + Γ * adjoint(L) * Γ⁻¹ * P_finite_2N

P_bar = solve_lyap(A_bar)

P = OperatorP(P_finite, P_bar)

# construct Q

M = N+K # this is assumed everywhere
P_ext = project(P, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
L_ext = DF(model, u_bar, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
D = Diagonal(repeat([1 ; fill(1/2, M+2K)], 1))
Γ = LinearOperator(CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2, [D zero(D); zero(D) D])
Γ⁻¹ = inv(Γ)

Q = - (project(P_ext, CosFourier(M+K, ω)^2, CosFourier(M+2K, ω)^2) * project(L_ext, CosFourier(M, ω)^2, CosFourier(M+K, ω)^2) +
       project(Γ * adjoint(L_ext) * Γ⁻¹, CosFourier(M+K, ω)^2, CosFourier(M+2K, ω)^2) * project(P_ext, CosFourier(M, ω)^2, CosFourier(M+K, ω)^2))

adjA_bar = copy(A_bar)
adjA_bar[1,2] = A_bar[2,1]
adjA_bar[2,1] = A_bar[1,2]

μ = 1 - (C₀(P, opnorm_approxDF⁻¹Δ, Z₂, ϵ_u; N, K) +
            max(opnorm(Q - I, X²),
                opnorm(norm.(P.W_bar * A_bar + adjA_bar * P.W_bar - [1. 0; 0 1], [X]), 1) + C₁(P.W_bar, A_bar; N, K) + C₃(P.W_bar, C_bar; N, K)))

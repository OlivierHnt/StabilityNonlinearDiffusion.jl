using Revise
using RadiiPolynomial, LinearAlgebra, StabilityNonlinearDiffusion
using GLMakie, CairoMakie, MAT



#######
# SKT #
#######

# 1d

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

K = 60
if length(u₁) > K
    u₁ = u₁[1:K+1]
    u₂ = u₂[1:K+1]
else
    u₁ = [u₁; fill(0, K-length(u₁)+1)]
    u₂ = [u₂; fill(0, K-length(u₂)+1)]
end

u_guess = Sequence(CosFourier(K, mid(ω))^2, [u₁[1:K+1] ; u₂[1:K+1]])

u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)



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
B_bar = StabilityNonlinearDiffusion.B(model, [component(u_bar, 1),component(u_bar, 2)])
norm_w = opnorm(norm.(approx_DF⁻¹.sequence_tail, [X]), 1)

Z₁_a = opnorm(I - project(approx_DF⁻¹, CosFourier(3K, ω)^2, CosFourier(4K, ω)^2) * L_bar_small, X²)
Z₁_b1 = opnorm(norm.([1. 0. ; 0. 1.] - approx_DF⁻¹.sequence_tail * A_bar, [X]), 1)

Z₁_b3 = inv((2*K + 1) * ω)^2 * norm_w * opnorm(norm.(C_bar, [X]), 1)

Z₁ = max(Z₁_a, Z₁_b1 + Z₁_b3)

#
# spy(abs.(mid.(coefficients(project(approx_DF⁻¹, domain(L_bar_small), CosFourier(3K, ω)^2)))))
# spy(abs.(mid.(coefficients(I - project(approx_DF⁻¹, domain(L_bar_small), CosFourier(3K, ω)^2) * L_bar))))
# spy(abs.(mid.(coefficients(L_bar))))

opnorm_approxDF⁻¹Δ = max(opnorm(approx_DF⁻¹.finite_matrix * project(Laplacian(), CosFourier(2K,ω)^2, CosFourier(2K,ω)^2), X²), norm_w)

normA′ = 2 * maximum(abs.([model.d₁₁, model.d₁₂, model.d₂₁, model.d₂₂]))

normΔ⁻¹C′ = maximum([2*abs(model.a₁), 2*abs(model.a₂), abs(model.b₁)+ abs(model.b₂)])
Z₂ = opnorm_approxDF⁻¹Δ * (normA′ + normΔ⁻¹C′)
ϵ_u = interval(inf(interval_of_existence(Y, Z₁, Z₂, Inf)))


#-----------#
# Stability #
#-----------#


N = 120

# construct P with Lyapunov equations
L = DF(model, u_bar, CosFourier(2N, ω)^2, CosFourier(2N, ω)^2)
Δ = project(Laplacian(), CosFourier(2N, ω)^2, CosFourier(2N, ω)^2)
component(Δ,1,1)[0,0] += -1
component(Δ,2,2)[0,0] += -1

g = project(I, CosFourier(0, ω)^2, CosFourier(0, ω)^2)
g_bar = 0.5*I([Sequence(CosFourier(0,ω), [1]);;].*I(2))
Γ_op = NewOperatorP(g, g_bar)
invΓ_op = NewOperatorP(g, 4*g_bar)
null_op = LinearOperator( EmptySpace()^2, EmptySpace()^2, [;;])


Γ = project(Γ_op, CosFourier(2N, ω)^2, CosFourier(2N, ω)^2)
Γ⁻¹ = project(invΓ_op, CosFourier(2N, ω)^2, CosFourier(2N, ω)^2)
#Δ_max = (ω*2N)^2


P_finite_2N = Γ * LinearOperator(CosFourier(2N, ω)^2, CosFourier(2N, ω)^2, Symmetric(lyap(adjoint(mid.(coefficients(L))), -mid.(coefficients(Γ⁻¹*Δ)))))
P_finite = project(P_finite_2N, CosFourier(N, ω)^2, CosFourier(N, ω)^2)

# Xcos = Γ⁻¹ * P_finite_2N; Xcos * L + adjoint(L) * Xcos
# P_finite_2N * L + Γ * adjoint(L) * Γ⁻¹ * P_finite_2N

P_bar = solve_lyap(A_bar, true)
isinv(det_Seq(P_bar))
#lines(LinRange(0, 1, 201),  x -> mid(P_bar[1](x)), color = :red)
M = N+K # this is assumed everywhere
L_ext = DF(model, u_bar, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
P_ext = project(NewOperatorP(P_finite, P_bar), CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)


Γ = project(Γ_op, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
Γ⁻¹ = project(invΓ_op, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)


E,V = eigen(mid.(coefficients(P_ext)))
μ₀ = E[1]*0.5

## Proof P-μ₀ still positive
Pμ = NewOperatorP(mid.(P_finite) - μ₀*I, map(v -> mid.(v), P_bar) - μ₀*I(2))

Dom = CosFourier(N, ω)^2
Band = CosFourier(K, ω)^2

## Approx a square root of Pμ.W_bar
Pμ_bar_mid = map(v -> mid.(v), Pμ.W_bar)
tr_bar = sum(l -> Pμ_bar_mid[l,l], 1:2)
det_bar = project(det_Seq(Pμ_bar_mid), Band.space ⊕ Band.space)
s_bar = sqrt(det_bar)
norm(s_bar^2 - det_bar)
if sup(s_bar(0)) < 0
    s_bar = -s_bar
end
t_bar = sqrt(tr_bar + 2*s_bar)
if sup(t_bar(0)) < 0
    t_bar = -t_bar
end
norm(tr_bar + 2*s_bar - t_bar^2)
sqrt_P_bar = (Pμ_bar_mid + [s_bar 0; 0 s_bar]).*[1/t_bar]
norm(norm.(Pμ_bar_mid - sqrt_P_bar*sqrt_P_bar),1)

# finer method to get square root
# N_root = 2N
# D_square = x -> project(Multiplication(2x), CosFourier(N_root,ω), CosFourier(N_root,ω))
# s_bar2, _ = newton(x -> (x^̄2 - det_bar, D_square(x)), project(s_bar, CosFourier(N_root, ω)); tol=1e-14, maxiter=20)
# norm(s_bar2^2 - det_bar)
# t_bar2, _ = newton(x -> (x^̄2 - tr_bar - 2*s_bar2, D_square(x)), project(t_bar, CosFourier(N_root, ω)); tol=1e-14, maxiter=20)
# norm(tr_bar + 2*s_bar - t_bar2^2)
# sqrt_P_bar2 = (Pμ_bar_mid + [s_bar2 0; 0 s_bar2]).*[1/t_bar2]
# norm(norm.(Pμ_bar_mid - sqrt_P_bar2*sqrt_P_bar2),1)
# sqrt_P_bar2 = map(v -> interval.(v), sqrt_P_bar2) 

Bigdom = Dom ⊕ Band
C_bar = map(v -> mid.(v), sqrt_P_bar)
invC_bar = approx_inv(C_bar) #map(v -> interval.(inv(mid.(v))), sqrt_P_bar)
norm(norm.(approx_inv(invC_bar) - sqrt_P_bar),1)
#MC_bar = NewOperatorP(null_op, C_bar)
MinvC_bar = NewOperatorP(null_op, map(v -> interval.(v), invC_bar))
MP_bar = NewOperatorP(null_op, map(v -> interval.(v), Pμ.W_bar))

valid_Pμ_W_bar = map(v -> ValidatedSequence(v, Ell1(GeometricWeight(interval(1)))), MP_bar.W_bar)
valid_invC_bar = map(v -> ValidatedSequence(v, Ell1(GeometricWeight(interval(1)))), MinvC_bar.W_bar)

#norm(norm.(valid_Pμ_W_bar - approx_inv(valid_invC_bar)^2),1) # should be small

P̃ = project(MinvC_bar, Dom, Bigdom) * (Pμ.P_finite - project(MP_bar, Dom, Dom)) * project(MinvC_bar, Bigdom, Dom) + I
opnorm(P̃*Γ- Γ*P̃', 1) #is sym ?
sym_P̃ = Symmetric(coefficients(P̃*project(Γ, Bigdom, Bigdom)))#is sym ?
Ẽ = eigen(mid.(sym_P̃))
Ṽ, invṼ = Ẽ.vectors, Ẽ.vectors'
D̃ = invṼ*sym_P̃*Ṽ
disk_P̃ = gershgorin(D̃)
# fig_gersh = plot_gershgorin(mid.(disk_P̃))
# save("fig/skt_gershP.eps", fig_gersh)

if prod(inf.(disk_P̃[:,1]) - sup.(disk_P̃[:,2]) .> 0) > 0
       "P̃ is positive"
else
       "P̃ is non postive"
end

μ = μ₀ - norm(norm.(valid_Pμ_W_bar - approx_inv(valid_invC_bar)^2),1)
if inf(μ) > 0
       display("P is positive definite")
else
       display("We cannot conclude")
end

# Gershgorin analysis of Q
P = NewOperatorP(P_finite, P_bar)
L_ext = DF(model, u_bar, CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
Q_ext = - (project(P, CosFourier(M+K, ω)^2, CosFourier(M+2K, ω)^2) * project(L_ext, CosFourier(M, ω)^2, CosFourier(M+K, ω)^2) + project(Γ * adjoint(L_ext) * Γ⁻¹, CosFourier(M+K, ω)^2, CosFourier(M+2K, ω)^2) * project(P_ext, CosFourier(M, ω)^2, CosFourier(M+K, ω)^2))

## Build of Q
# order 0 terms
# ∇ = StabilityNonlinearDiffusion.Gradient{1}()

#!# Incorrect Non Adapté pour 2D pour le moment !!
q₀_bar = q₀(A_bar, B_bar, C_bar, P_bar)
n_q₀ = 2*sum(abs.(q₀_bar[1][1:end]))
# order 1 terms
q₁_bar = q₁(A_bar, B_bar, P_bar)
n_q₁ = sum(l-> 2*sum(abs.(q₁_bar[l][1][2:end])), 1:1)
# order 2 terms
q₂_bar = q₂(A_bar, P_bar)
n_q₂ = 2*sum(abs.(q₂_bar[1][2:end]))

Q0 = zeros(Interval{Float64},CosFourier(M+2K, ω)^2, CosFourier(M+2K, ω)^2)
Q1 = zero(Q0)
Q2 = zero(Q0)
for j ∈ 1:2, i ∈ 1:2
    project!(component(Q0,i,j), Multiplication(q₀_bar[i,j]))
    mul!(component(Q1,i,j), Multiplication(sum(l -> q₁_bar[l][i,j], 1)), Derivative(1)) 
    mul!(component(Q2,1,1), Multiplication(q₂_bar[i,j]), Laplacian())
end
Q = Q0 + Q1 + Q2
for j ∈ 1:2, i ∈ 1:2
    _tmp_ = component(Q_ext, i, j)
    component(Q, i, j)[indices(codomain(_tmp_)),indices(domain(_tmp_))] .= _tmp_
end
Q

## Find index of truncation


## Diagonalization if needed
#E_Q, V_Q = eigen(mid.(coefficients(Q_ext)[1:M+1, 1:M+1]))

# Q̃ = zero(Q)
# project!(component(Q̃,1,1), component(Q,1,1))
# coefficients(component(Q̃,1,1))[1:M0+1,1:M0+1] .= V_Q[1:M0+1,1:M0+1]*coefficients(Q)[1:M0+1,1:M0+1]*V_Q[1:M0+1,1:M0+1]^(-1)


## Compute μ_∞
normp =  norm(norm.(interval.(P.W_bar)),1)
opnormP = max( opnorm(interval.(project(P, domain(P.P_finite), domain(P.P_finite) ⊕ space(P.W_bar[1,1])^2)), 1), normp)

normp1 = norm( norm.(differentiate.(interval.(P.W_bar)), 1) ,1)
normp2 = norm( norm.(differentiate.(interval.(P.W_bar), 2), 1), 1) 
Δ = project(Laplacian(), domain(P.P_finite), domain(P.P_finite))
Δ[1,1] -= 1
_r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
invΔ = LinearOperator(domain(P.P_finite), domain(P.P_finite), RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))
opnormDPD = max( opnorm(invΔ*P.P_finite*Δ, 1), normp + 2*normp1/(ω*N) + normp2/(ω*N)^2)
C_L = Z₂*ϵ_u/opnorm_approxDF⁻¹Δ
C_Q = (opnormP + opnormDPD)*C_L
C_V = C_Q #norminvV*normV*C_Q

eta = 0.125
epsilon = 0.5

small = (1-(C_V + n_q₂ + eta)/(-q₂_bar[1][0]*(1-epsilon))) ## must be smaller than 1, if not adjust eta and epsilon
μ_inf = small*Q̃[M0+K,M0+K]



## Determine Q positive definite
disk_Q̃ = gershgorin((coefficients(Q_ext))[1:M+1, 1:M+2K+1])
# gersh_Q̃ = plot_gershgorin(mid.(disk_Q̃))
# save("fig/skt_gershQ.eps", gersh_Q̃)

MU = inf.(disk_Q̃[:,1]) - sup.(disk_Q̃[:,2])
@show μ = min(minimum(MU), inf(μ_inf))

if μ > 0
       display("Q is positive definite")
else
       display("Q is not positive")
end

## A bound on the spectral gap of L
# @show λ = μ/(2*opnormP)
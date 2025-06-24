using Revise
using RadiiPolynomial, LinearAlgebra, StabilityNonlinearDiffusion
# using CairoMakie#GLMakie#, CairoMakie





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
B_bar = StabilityNonlinearDiffusion.B(model, [component(u_bar, 1)])
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

# fig = Figure()
# ax = Axis(fig[1,1], aspect = DataAspect())
# lines!(ax, LinRange(0, 1, 100), x -> component(u_approx,1)(x); linewidth = 2, color = :green, linestyle = :solid)
# xlims!(0,1)
# save("fig/scalar_solution.eps",fig)

#--------------------#
# Stability analysis #
#--------------------#

N = 40

g = project(I, CosFourier(0, ω)^1, CosFourier(0, ω)^1)
g_bar = 0.5*I([Sequence(CosFourier(0, ω), [1]);;])
Γ_op = NewOperatorP(g, g_bar)
invΓ_op = NewOperatorP(g, 4*g_bar)


## P_bar
Band = CosFourier(K, ω)^1
P_bar = map(v -> Sequence(Band.space, mid.(coefficients(v))), solve_lyap(A_bar))
# isinv(P_bar[1])


## Build Q̂
# order 0 terms
q₀_bar = q₀(A_bar, B_bar, C_bar, P_bar)
n_q₀ = 2*sum(abs.(q₀_bar[1][1:end]))
# order 1 terms
q₁_bar = q₁(A_bar, B_bar, P_bar)
n_q₁ = sum(l-> 2*sum(abs.(q₁_bar[l][1][2:end])), 1:1)
# order 2 terms
q₂_bar = q₂(A_bar, P_bar)
n_q₂ = 2*sum(abs.(q₂_bar[1][2:end]))


## Find index od truncation
d1 = (q₁_bar[1][1][1]^2 + 4q₀_bar[1][0]*q₂_bar[1][0])
if inf(d1) > 0
       k1  =max((q₁_bar[1][1][1] + sqrt(d1))/(2*q₂_bar[1][0]*ω),(q₁_bar[1][1][1] - sqrt(d1))/(2*q₂_bar[1][0]*ω))
else
       k1 = 0
end
epsilon = 0.5 #...
d2 = (q₁_bar[1][1][1]^2 + 4*epsilon*q₀_bar[1][0]*q₂_bar[1][0])
if inf(d2) > 0
       k2 = max(-(q₁_bar[1][1][1] + sqrt(d2))/(q₂_bar[1][0]*ω),-(q₁_bar[1][1][1] - sqrt(d2))/(q₀_bar[1][0]*ω))
else
       k2 = 0
end
eta = 0.125#-q₂_bar[1][0]/2 - n_q₂  #a refaire 
d3 = (n_q₁^2 + 4*eta*(n_q₀+ω^2*K^2*n_q₂))
k3 = (n_q₁ + sqrt(d3))/(2*eta*ω)
M0 = Int(sup(max(N, K, k1, k2, k3)))

Littledom = CosFourier(M0, ω)^1
Dom = Littledom ⊕ Band
Bigdom = Dom ⊕ Band


Q0 = zeros(Interval{Float64},Bigdom, Bigdom)
project!(component(Q0,1,1), Multiplication(q₀_bar[1,1]))
Q1 = zero(Q0)
mul!(component(Q1,1,1), Multiplication(q₁_bar[1][1,1]), Derivative(1))
Q2 = zero(Q0)
mul!(component(Q2,1,1), Multiplication(q₂_bar[1,1]), Laplacian())
Q̂ = Q0 + Q1 + Q2


## P_finite 

Γ = project(Γ_op, Dom, Dom)
Γ⁻¹ = project(invΓ_op, Dom, Dom)
L = DF(model, u_bar, Dom, Dom)
Δ = project(Laplacian(), Dom, Dom)
Δ[1,1] -= 1


P_finite_2N = Γ * LinearOperator(Dom, Dom, Symmetric(lyap(adjoint(mid.(coefficients(L))), -mid.(coefficients(Γ⁻¹*Δ)))))
P_finite = project(P_finite_2N, Littledom, Littledom)
# where 
# Xcos = Γ⁻¹ * P_finite_2N; Xcos * L + adjoint(L) * Xcos
# P_finite_2N * L + Γ * adjoint(L) * Γ⁻¹ * P_finite_2N

P_ext = project(NewOperatorP(P_finite, P_bar), Bigdom, Bigdom)
# E,V = eigen(mid.(coefficients(P_ext)))
μ₀ = 1e-3#E[1]

Pμ  = NewOperatorP(P_finite - μ₀*I, P_bar - I(1)*μ₀)
sqrt_P_bar = sqrt.(Pμ.W_bar)

null_op = LinearOperator(EmptySpace()^1, EmptySpace()^1, [;;])

## P positive definite

invC_bar = map(v -> interval(inv(mid.(v))), sqrt_P_bar)
# C_bar = map(v -> mid.(v), sqrt_P_bar)
# MC_bar = NewOperatorP(null_op, C_bar)
MinvC_bar = NewOperatorP(null_op, invC_bar)
MP_bar = NewOperatorP(null_op, map(v -> interval.(v), Pμ.W_bar))

valid_Pμ_W_bar = ValidatedSequence.(MP_bar.W_bar, [Ell1(GeometricWeight(interval(1)))])
valid_invC_bar = ValidatedSequence.(invC_bar, [Ell1(GeometricWeight(interval(1)))])

#norm.(valid_Pμ_W_bar - inv.(valid_invC_bar)^2)[1] # should be small

P̃ = project(MinvC_bar, Littledom, Dom) * (interval.(Pμ.P_finite) - project(MP_bar, Littledom, Littledom)) * project(MinvC_bar, Dom, Littledom) + I
opnorm(P̃*Γ- Γ*P̃', 1) #is sym ?
sym_P̃ = Symmetric(coefficients(P̃*project(Γ, Dom, Dom)))
Ẽ = eigen(mid.(sym_P̃))
Ṽ, invṼ = Ẽ.vectors, Ẽ.vectors'
D̃ = invṼ*sym_P̃*Ṽ
disk_P̃ = gershgorin(D̃)

# fig_gersh = plot_gershgorin(mid.(disk_P̃))
# save("fig/scalar_gershP.eps", fig_gersh)

if prod(inf.(disk_P̃[:,1]) - sup.(disk_P̃[:,2]) .> 0) > 0
       "P̃ is positive"
else
       "P̃ is non postive"
end

μ = μ₀ - norm.(valid_Pμ_W_bar - inv.(valid_invC_bar)^2)[1]

if inf(μ) > 0
       display("P is positive-definite")
else
       display("We cannot conclude")
end

## Gershgorin analysis of Q
P = NewOperatorP(P_finite, P_bar)
L_ext = DF(model, u_bar, Bigdom, Bigdom)
Q_ext = - (project(P, Dom, Bigdom) * project(L_ext, Littledom, Dom) +
       project(Γ * adjoint(L_ext) * Γ⁻¹, Dom, Bigdom) * project(P_ext, Littledom, Dom))

Q = copy(Q̂)
_tmp_ = component(Q_ext, 1, 1)
component(Q, 1, 1)[indices(codomain(_tmp_)),indices(domain(_tmp_))] .= _tmp_
Q

Q̃ = Q

## Change of Basis (if necessary)
# E_Q, V_Q = eigen(mid.(coefficients(Q)))
# V = NewOperatorP(LinearOperator(CosFourier(M0,ω)^1, CosFourier(M0,ω)^1, V_Q[1:M0+1,1:M0+1]), 2*g_bar)
# Vt = NewOperatorP(LinearOperator(CosFourier(M0,ω)^1, CosFourier(M0,ω)^1, inv(V_Q)[1:M0+1,1:M0+1]), 2*g_bar)
# Q̃ = project(V, domain(Q), domain(Q))*Q*project(Vt, domain(Q), domain(Q))

# normV = max( opnorm(interval.(V.P_finite),1), 1 )
# Δ = project(Laplacian(), domain(V.P_finite), domain(V.P_finite))
# Δ[1,1] -= 1
# _r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
# invΔ = LinearOperator(domain(V.P_finite), domain(V.P_finite), RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))
# norminvV = max( opnorm(invΔ*Vt.P_finite*Δ, 1), 1)

## Compute μ_∞
normp =  norm(interval.(P.W_bar[1]),1)
opnormP = max( opnorm(interval.(project(P, domain(P.P_finite), domain(P.P_finite) ⊕ space(P.W_bar[1])^1)), 1), normp)

normp1 = norm(differentiate(interval.(P.W_bar[1]),1),1)
normp2 = norm(differentiate(interval(P.W_bar[1]),2), 1) 
Δ = project(Laplacian(), domain(P.P_finite), domain(P.P_finite))
Δ[1,1] -= 1
_r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
invΔ = LinearOperator(domain(P.P_finite), domain(P.P_finite), RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))
opnormDPD = max( opnorm(invΔ*P.P_finite*Δ, 1), normp + 2*normp1/(ω*M0) + normp2/(ω*M0)^2)
C_L = 2 * (1 + model.β)*ϵ_u
C_Q = (opnormP + opnormDPD)*C_L
C_V = C_Q #norminvV*normV*C_Q if diagonalization

small = (1-(C_V + n_q₂ + eta)/(-q₂_bar[1][0]*(1-epsilon))) ## must be smaller than 1, if not adjust eta and epsilon
μ_inf = small*Q̃[M0+K,M0+K]

## Determine Q postive definite
disk_Q̃ = gershgorin((coefficients(Q̃))[1:M0+K, 1:M0+2K+1])
disk_Q̃[1,2] += C_V
disk_Q̃[2:end,2] += [C_V*(ω*k)^2 for k=1:size(disk_Q̃,1)-1]

# gersh_Q̃ = plot_gershgorin(mid.(disk_Q̃))
# save("fig/scalar_gershQ.eps", gersh_Q̃)


MU = inf.(disk_Q̃[:,1]) - sup.(disk_Q̃[:,2])
@show μ = min(minimum(MU), inf(μ_inf))
if μ > 0
       display("Q is positive definite")
else
       display("Q is not positive")
end

## A bound on the spectral gap of L
# @show λ = μ/(2*opnormP)
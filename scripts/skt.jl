using Revise
using StabilityNonlinearDiffusion, RadiiPolynomial, LinearAlgebra
using GLMakie, MAT



#######
# SKT #
#######

# attempt at 2d solutions // cf. the article FINITE-VOLUME SCHEME FOR THE SKT SYSTEM

ω = interval(π) # frequency

# Newton's method

mid_model = SKT(; d₁ = 0.05, d₂ = 0.05, d₁₁ = 2.5e-5, d₂₂ = 2.5e-5, d₁₂ = 1.025, d₂₁ = 0.075,
            r₁ = 59.7, r₂ = 49.75, a₁ = 24.875, b₁ = 19.9, a₂ =  19.9, b₂ = 19.9)

K = 10

u_guess = zeros((CosFourier(K, mid(ω)) ⊗ CosFourier(K, mid(ω)))^2)
component(u_guess, 1)[(0,0)] = 1.
component(u_guess, 2)[(0,0)] = 1.

##
# component(u_guess, 1)[(4,0)] = 0.1
# component(u_guess, 1)[(0,4)] = 0.1

# component(u_guess, 2)[(2,2)] = 0.1
##

## to produce some 2d solutions
for k1 = 2:2:K, k2 = 2:2:K
    component(u_guess, 1)[(k1,k2)] = rand()/2.0^(abs(k1)+abs(k2))
    # component(u_guess, 1)[(0,k2)] = rand()/2.0^(abs(k1)+abs(k2))
end
for k1 = 1:2:K, k2 = 1:2:K
    component(u_guess, 2)[(k1,k2)] = rand()/2.0^(abs(k1)+abs(k2))
end
##

##
# reshape(view(component(u_guess, 1), (0:10,0:10)), 1:11, 1:11) .=
# [  1.70288       0.0          -0.11505       0.0           0.0540458     0.0          -0.011848      0.0           0.00337627    0.0          -0.00095635
# 0.0          -1.32118e-26   0.0           5.05502e-26   0.0          -1.45466e-26   0.0           3.31953e-27   0.0           1.32489e-27   0.0
# -2.98505e-26   0.0          -1.24085e-25   0.0           2.87078e-26   0.0          -1.23189e-26   0.0           2.67739e-27   0.0          -9.6241e-29
# 0.0          -0.135478      0.0           0.0520527     0.0          -0.018525      0.0           0.00499812    0.0          -0.00135507    0.0
# 4.26364e-26   0.0          -6.73687e-28   0.0          -2.89275e-27   0.0           1.90904e-28   0.0           3.80034e-28   0.0          -2.25516e-28
# 0.0           6.48949e-26   0.0           2.02453e-26   0.0          -8.1939e-27    0.0           2.01633e-27   0.0           2.1457e-28    0.0
# 0.0367017     0.0           0.0103023     0.0          -0.00815193    0.0           0.00304667    0.0          -0.000715515   0.0           0.000101363
# 0.0          -2.05388e-26   0.0          -3.11206e-28   0.0          -1.36157e-27   0.0           1.8696e-28    0.0           2.78012e-28   0.0
# -2.93243e-26   0.0          -1.36915e-26   0.0           1.81517e-27   0.0           5.87307e-28   0.0           2.51844e-28   0.0          -5.02283e-29
# 0.0          -0.00441349    0.0          -0.000337298   0.0           0.00112088    0.0          -0.00047912    0.0           8.02874e-5    0.0
# 8.93858e-27   0.0           5.694e-27     0.0           5.52104e-28   0.0           4.68386e-29   0.0           1.77087e-28   0.0          -2.34779e-28]
# reshape(view(component(u_guess, 2), (0:10,0:10)), 1:11, 1:11) .=
# [  0.828312      0.0           0.0876932     0.0          -0.0317349     0.0           0.00471024    0.0          -0.000849789   0.0           0.000170747
# 0.0           1.48748e-26   0.0          -2.53319e-26   0.0           2.53855e-27   0.0           1.17383e-27   0.0          -2.50661e-28   0.0
# 3.41167e-26   0.0           7.85958e-26   0.0          -1.11951e-26   0.0           1.41206e-27   0.0           6.22608e-28   0.0          -7.23878e-28
# 0.0           0.0912182     0.0          -0.0232152     0.0           0.00291717    0.0           0.00038764    0.0          -0.000299717   0.0
# -2.42527e-26   0.0          -4.14941e-27   0.0           9.5255e-29    0.0          -3.7106e-28    0.0           2.07668e-28   0.0          -4.02812e-28
# 0.0          -2.64844e-26   0.0          -5.66639e-27   0.0           1.21879e-28   0.0           8.19035e-28   0.0          -2.70481e-28   0.0
# -0.00744062    0.0          -0.00457353    0.0           0.00112449    0.0           0.00053481    0.0          -0.000383755   0.0           0.00014532
# 0.0           3.43588e-27   0.0           1.52684e-27   0.0          -2.42525e-28   0.0           4.73317e-28   0.0          -2.0678e-28    0.0
# 5.94407e-28   0.0           9.36181e-28   0.0           5.61866e-28   0.0          -1.01664e-28   0.0           9.00288e-29   0.0          -1.1754e-28
# 0.0          -0.000536213   0.0           0.000483636   0.0          -4.09955e-5    0.0          -0.000120147   0.0           6.80527e-5    0.0
# 1.4515e-27    0.0           3.74216e-28   0.0          -2.24677e-28   0.0           5.99288e-29   0.0          -7.39904e-29   0.0           7.22917e-30]
##

## volcano / radially symmetric
# reshape(view(component(u_guess, 1), (0:10,0:10)), 1:11, 1:11) .=
# [ 1.79098       0.0           0.132064      0.0           0.0304946     0.0           0.00193064    0.0           0.000417192   0.0           6.47389e-5
# 0.0          -2.24841e-14   0.0           1.22879e-14   0.0           2.0645e-15    0.0          -3.61905e-16   0.0          -5.86348e-17   0.0
# 0.132064      0.0           0.0429094     0.0          -0.0115226     0.0          -0.00360012    0.0          -0.000642628   0.0          -5.18846e-5
# 0.0           1.14754e-14   0.0           9.19842e-15   0.0           3.95715e-16   0.0          -8.19634e-16   0.0          -1.31802e-16   0.0
# 0.0304946     0.0          -0.0115226     0.0          -0.00933324    0.0          -0.001904      0.0           0.000120327   0.0           0.000105903
# 0.0           1.94313e-15   0.0           5.22327e-16   0.0          -1.60361e-15   0.0          -6.66792e-16   0.0          -1.63581e-17   0.0
# 0.00193064    0.0          -0.00360012    0.0          -0.001904      0.0           0.000420632   0.0           0.000368373   0.0           8.18535e-5
# 0.0          -2.97881e-16   0.0          -7.51968e-16   0.0          -6.6935e-16    0.0          -5.31936e-17   0.0           1.14148e-16   0.0
# 0.000417192   0.0          -0.000642628   0.0           0.000120327   0.0           0.000368373   0.0           0.000113329   0.0          -1.07306e-5
# 0.0          -4.99556e-17   0.0          -1.34505e-16   0.0          -3.25619e-17   0.0           1.09493e-16   0.0           6.89693e-17   0.0
# 6.47389e-5    0.0          -5.18846e-5    0.0           0.000105903   0.0           8.18535e-5    0.0          -1.07306e-5    0.0          -2.18409e-5]
# reshape(view(component(u_guess, 2), (0:10,0:10)), 1:11, 1:11) .=
# [  0.733551      0.0          -0.0904099    0.0          -0.012206      0.0           0.000282503   0.0           6.17457e-5    0.0           2.78033e-5
# 0.0           1.7753e-14    0.0         -7.93226e-15   0.0           4.68119e-17   0.0           1.80638e-16   0.0          -3.17996e-17   0.0
# -0.0904099     0.0          -0.0181951    0.0           0.00943807    0.0           0.00100718    0.0          -4.85799e-5    0.0          -2.044e-5
# 0.0          -7.51418e-15   0.0         -2.74733e-15   0.0           1.00677e-15   0.0           2.27559e-16   0.0          -3.39084e-17   0.0
# -0.012206      0.0           0.00943807   0.0           0.0027234     0.0          -0.000406068   0.0          -0.000198182   0.0          -1.52138e-5
# 0.0           6.50986e-18   0.0          9.53967e-16   0.0           6.17445e-16   0.0          -6.0073e-17    0.0          -5.83031e-17   0.0
# 0.000282503   0.0           0.00100718   0.0          -0.000406068   0.0          -0.000358089   0.0          -1.98332e-5    0.0           2.02646e-5
# 0.0           1.56759e-16   0.0          2.32982e-16   0.0          -5.18252e-17   0.0          -1.05063e-16   0.0          -1.54681e-17   0.0
# 6.17457e-5    0.0          -4.85799e-5   0.0          -0.000198182   0.0          -1.98332e-5    0.0           3.56643e-5    0.0           1.06722e-5
# 0.0          -2.79913e-17   0.0         -2.55079e-17   0.0          -5.65957e-17   0.0          -1.65715e-17   0.0           1.05532e-17   0.0
# 2.78033e-5    0.0          -2.044e-5     0.0          -1.52138e-5    0.0           2.02646e-5    0.0           1.06722e-5    0.0          -6.61534e-7]
##

##
# reshape(view(component(u_guess, 1), (0:10,0:10)), 1:11, 1:11) .=
# [  1.78225       0.0          0.171925      0.0           0.0848931     0.0           0.0115792     0.0           0.00144018   0.0          -0.000325919
#   0.0          -0.0207678    0.0           0.00581356    0.0           0.00510314    0.0           0.00228501    0.0          0.000514026   0.0
#   0.0529659     0.0          0.0220271     0.0          -0.00892128    0.0          -0.00544614    0.0          -0.00242031   0.0          -0.000575163
#   0.0           0.0211798    0.0           0.000236801   0.0          -0.000926985   0.0          -0.0014319     0.0         -0.000634647   0.0
#   0.000760157   0.0         -0.000782376   0.0          -0.00114943    0.0          -0.000617596   0.0          -3.05316e-5   0.0           6.41866e-5
#   0.0           0.00129236   0.0          -0.00149547    0.0          -0.00124718    0.0          -0.000348439   0.0         -1.67085e-5    0.0
#   0.000426817   0.0         -3.28533e-5    0.0          -0.000286626   0.0          -0.000117671   0.0          -2.30579e-5   0.0           1.40218e-5
#   0.0          -5.48483e-5   0.0          -0.000144068   0.0          -4.9375e-5     0.0           4.77392e-5    0.0          4.86747e-5    0.0
#   7.67412e-6    0.0         -4.7519e-5     0.0          -6.37951e-5    0.0          -3.86587e-6    0.0           2.09676e-5   0.0           1.4362e-5
#   0.0          -2.31657e-6   0.0          -8.18752e-6    0.0           2.54291e-6    0.0           9.32033e-6    0.0          5.50113e-6    0.0
#  -2.22272e-6    0.0         -3.06458e-6    0.0           1.01027e-6    0.0           7.05873e-6    0.0           5.96948e-6   0.0           1.55097e-6]
# reshape(view(component(u_guess, 2), (0:10,0:10)), 1:11, 1:11) .=
# [ 0.742758      0.0         -0.118602      0.0          -0.0366376     0.0           0.00164065   0.0           0.00125111   0.0          0.000303197
#   0.0           0.0167724    0.0          -0.00583007    0.0          -0.00250289    0.0         -0.000221304   0.0          0.00013826   0.0
#   -0.0365451     0.0         -0.009447      0.0           0.00825309    0.0           0.00212      0.0           8.53806e-5   0.0         -0.00014555
#    0.0          -0.0117625    0.0           0.00257023    0.0           0.00142824    0.0          0.000496442   0.0          3.50716e-5   0.0
#   -0.000114068   0.0          0.000639972   0.0           0.000340506   0.0          -3.90477e-5   0.0          -9.0238e-5    0.0         -2.76042e-5
#    0.0           6.4303e-5    0.0           0.000754951   0.0           0.000164842   0.0         -0.000114513   0.0         -6.88037e-5   0.0
#   -2.21321e-5    0.0          9.22358e-5    0.0           7.07463e-5    0.0          -1.4241e-5    0.0          -1.89159e-5   0.0         -5.09802e-6
#    0.0           3.21778e-5   0.0          -5.97763e-6    0.0          -3.97604e-5    0.0         -2.20922e-5    0.0         -1.17799e-6   0.0
#    1.84946e-5    0.0          8.84382e-6    0.0          -1.09896e-5    0.0          -1.40575e-5   0.0          -4.47891e-6   0.0          8.73496e-7
#    0.0           1.89373e-6   0.0          -2.40832e-6    0.0          -3.1575e-6     0.0         -1.83322e-7    0.0          1.08199e-6   0.0
#    8.97742e-7    0.0         -9.47066e-7    0.0          -2.30759e-6    0.0          -4.83405e-7   0.0           1.02106e-6   0.0          7.45263e-7]
##


##


fig = Figure()
ax = Axis3(fig[1,1])
surface!(ax, [x for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[y for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[component(u_guess, 1)(x, y) for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)])
ax2 = Axis3(fig[1,2])
surface!(ax2, [x for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[y for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[component(u_guess, 2)(x, y) for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)])




u_approx, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_guess)

u_approx2, _ = newton(u -> (F(mid_model, u, space(u)), DF(mid_model, u, space(u), space(u))), u_approx)


fig = Figure()
ax = Axis3(fig[1,1])
surface!(ax, [x for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[y for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[component(u_approx2, 1)(x, y) for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)])
ax2 = Axis3(fig[1,2])
surface!(ax2, [x for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[y for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)],
[component(u_approx2, 2)(x, y) for x = LinRange(0, 1, 101), y = LinRange(0, 1, 101)])







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

K = 100
u₁ = [u₁; fill(0, K-length(u₁)+1)]
u₂ = [u₂; fill(0, K-length(u₂)+1)]

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


#-----------#
# Stability #
#-----------#

N = 4K #need to be > 2K

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

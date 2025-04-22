module StabilityNonlinearDiffusion

using RadiiPolynomial, GLMakie

# models

@kwdef struct ScalarExample{T}
    # parameters of the reaction
    α :: T
    β :: T
end

R(model::ScalarExample, u) = [model.α * u[1] - model.β * u[1]^2 + Sequence(CosFourier(4, frequency(u[1])), [0.5, 1.5, 1, -0.5, 3])]

A(::ScalarExample, u) = [2u[1];;]

B(::ScalarExample, u) = [[zero(differentiate(u[1]));;]]

C(model::ScalarExample, u) = [model.α - 2model.β * u[1];;]

    export ScalarExample

#-

# @kwdef struct SKT{T} # Shigesada-Kawasaki-Teramoto system
#     # parameters of the diffusion
#     d₁  :: T
#     d₂  :: T
#     d₁₁ :: T
#     d₁₂ :: T
#     d₂₁ :: T
#     d₂₂ :: T
#     # parameters of the reaction
#     r₁  :: T
#     r₂  :: T
#     a₁  :: T
#     b₁  :: T
#     b₂  :: T
#     a₂  :: T
# end

# function Φ(model::SKT, u)
#     u₁, u₂ = eachcomponent(u)
#     Φ₁ = (model.d₁ + model.d₁₁ * u₁ + model.d₁₂ * u₂) * u₁
#     Φ₂ = (model.d₂ + model.d₂₁ * u₁ + model.d₂₂ * u₂) * u₂
#     return [Φ₁, Φ₂] # Sequence(space(Φ₁) × space(Φ₂), [coefficients(Φ₁) ; coefficients(Φ₂)])
# end

# function Φ′(model::SKT, u)
#     u₁, u₂ = eachcomponent(u)
#     Φ′₁₁ = model.d₁ + 2model.d₁₁ * u₁ +  model.d₁₂ * u₂
#     Φ′₁₂ =                               model.d₁₂ * u₁
#     Φ′₂₁ =             model.d₂₁ * u₂
#     Φ′₂₂ = model.d₂ +  model.d₂₁ * u₁ + 2model.d₂₂ * u₂
#     return [Φ′₁₁ Φ′₁₂ ; Φ′₂₁ Φ′₂₂]
# end

# function R(model::SKT, u)
#     u₁, u₂ = eachcomponent(u)
#     R₁ = (model.r₁ - model.a₁ * u₁ - model.b₁ * u₂) * u₁
#     R₂ = (model.r₂ - model.b₂ * u₁ - model.a₂ * u₂) * u₂
#     return [R₁, R₂]
# end

# function R′(model::SKT, u)
#     u₁, u₂ = eachcomponent(u)
#     R′₁₁ = model.r₁ - 2model.a₁ * u₁ -  model.b₁ * u₂
#     R′₁₂ =                           -  model.b₁ * u₁
#     R′₂₁ =          -  model.b₂ * u₂
#     R′₂₂ = model.r₂ -  model.b₂ * u₁ - 2model.a₂ * u₂
#     return [R′₁₁ R′₁₂ ; R′₂₁ R′₂₂]
# end

#     export SKT

#-

@kwdef struct Porous{T}
    # parameters of the diffusion
    d₁₁ :: T
    d₁₂ :: T
    d₂₁ :: T
    d₂₂ :: T
    # parameters of the reaction
    r₁  :: T
    a₁  :: T
    b₁  :: T
    r₂  :: T
    b₂  :: T
    a₂  :: T
end

A(model::Porous, u) = [model.d₁₁ * u[1]     model.d₁₂ * u[1]
                       model.d₂₁ * u[2]     model.d₂₂ * u[2]]

R(model::Porous, u) = [(model.r₁ + model.a₁ * u[1] + model.b₁ * u[2]) * u[1]
                       (model.r₂ + model.b₂ * u[1] + model.a₂ * u[2]) * u[2]]

B(model::Porous, u) = [[ model.d₁₂ * differentiate(u[2])     -model.d₁₂ * differentiate(u[1])
                        -model.d₂₁ * differentiate(u[2])      model.d₂₁ * differentiate(u[1])]]

C(model::Porous, u) = [model.r₁ + 2model.a₁ * u[1] + model.b₁ * u[2]                model.b₁ * u[1]
                                                     model.b₂ * u[2]     model.r₂ + model.b₂ * u[1] + 2model.a₂ * u[2]]

    export Porous



# zero-finding problem for the steady-state

struct Gradient{N} end

Base.getindex(::Gradient{1}, i)           = Derivative(1)
Base.getindex(::Gradient{N}, i) where {N} = Derivative(ntuple(j -> ifelse(j == i, 1, 0), Val(N)))

_nspaces(::BaseSpace) = 1
_nspaces(s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = N

function F(model, u, s)
    _u_ = [component(u, i) for i ∈ 1:nspaces(s)]
    _A_ = A(model, _u_)
    _R_ = R(model, _u_)

    d = _nspaces(s[1])
    ∇ = Gradient{d}()

    _F_ = sum(l -> [∇[l]] .* (_A_ * ([∇[l]] .* _u_)), 1:d) + _R_

    seq = zeros(eltype(u), s)
    for i ∈ 1:nspaces(s)
        project!(component(seq, i), _F_[i])
    end

    return seq
end

function DF(model, u, dom, codom)
    _u_ = [component(u, i) for i ∈ 1:nspaces(dom)]
    _A_ = A(model, _u_); _multA_ = project.(Multiplication.(_A_), spaces(dom), spaces(codom))
    _B_ = B(model, _u_)
    _C_ = C(model, _u_); _multC_ = project.(Multiplication.(_C_), spaces(dom), spaces(codom))

    d = _nspaces(dom[1])
    ∇ = Gradient{d}()
    _div_multB_ = sum(l -> [∇[l]] .* project.(Multiplication.(_B_[l]), spaces(dom), RadiiPolynomial.image.([∇[l]], spaces(codom))), 1:d)
    _DF_ = [Laplacian()] .* _multA_ + _div_multB_ + _multC_

    linop = zeros(eltype(u), dom, codom)
    for j ∈ 1:nspaces(dom), i ∈ 1:nspaces(codom)
        project!(component(linop, i, j), _DF_[i,j])
    end

    return linop
end

    export F, DF

#

struct ApproxInverse{T<:LinearOperator,S<:Sequence}
    finite_matrix :: T
    sequence_tail :: Matrix{S}
end

function RadiiPolynomial.project(A::ApproxInverse, dom, codom)
    #
    Δ = project(Laplacian(), dom[1], dom[1])
    _r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
    Δ⁻¹ = LinearOperator(dom[1], dom[1], RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))
    #
    V = zeros(Interval{Float64}, dom, codom)
    for j ∈ 1:nspaces(dom), i ∈ 1:nspaces(codom)
        mul!(component(V, i, j), Multiplication(A.sequence_tail[i,j]), Δ⁻¹)
        _tmp_ = component(A.finite_matrix, i, j)
        component(V, i, j)[indices(codomain(_tmp_)),indices(domain(_tmp_))] .= _tmp_
    end
    return V
end

function approx_inv(A)
    if size(A) == (1, 1)
        # inv 1-by-1, i.e., scalar case
        return inv.(A)
    elseif size(A) == (2, 2)
        # inv 2-by-2
        detA = A[1,1] * A[2,2] - A[1,2] * A[2,1]
        if detA == zero(A[1,1])
            error("Matrix is singular")
        end
        return project.([inv(detA)] .* [A[2,2] -A[1,2] ; -A[2,1] A[1,1]], space.(A))
    else
        error()
    end
end

    export ApproxInverse, approx_inv



# stability

struct OperatorP{T<:LinearOperator,S<:Sequence}
    P_finite :: T
    W_bar    :: Matrix{S}
end

function RadiiPolynomial.project(A::OperatorP, dom, codom)
    #
    Δ = project(Laplacian(), dom[1], dom[1])
    _r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
    Δ⁻¹ = LinearOperator(dom[1], dom[1], RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))
    #
    V = zeros(Interval{Float64}, dom, codom)
    for j ∈ 1:nspaces(dom), i ∈ 1:nspaces(codom)
        mul!(component(V, i, j), Multiplication(A.W_bar[i,j]), Δ⁻¹)
        mul!(component(V, i, j), Δ⁻¹, Multiplication(A.W_bar[i,j]), -0.5, -0.5)
        _tmp_ = component(A.P_finite, i, j)
        component(V, i, j)[indices(codomain(_tmp_)),indices(domain(_tmp_))] .= _tmp_
    end
    return V
end

function solve_lyap(A)
    if size(A) == (1, 1)
        # lyapunov 1-by-1, i.e., scalar case
        return inv.(2 .* A)
    elseif size(A) == (2, 2)
        # lyapunov 2-by-2
        detA = RadiiPolynomial.LinearAlgebra.det(A)
        trA = RadiiPolynomial.LinearAlgebra.tr(A)
        den = inv(2 * detA * trA)
        P₁_bar =  (detA + A[2,1]^2 + A[2,2]^2)        * den
        P₂_bar = -(A[1,1] * A[2,1] + A[1,2] * A[2,2]) * den
        P₃_bar =  (detA + A[1,1]^2 + A[1,2]^2)        * den
        return [P₁_bar P₂_bar ; P₂_bar P₃_bar]
    else # In literature we can find algorithm for lyapunov n-by-n
        error()
    end
end

    export OperatorP, solve_lyap

#

# function adj_DF(model, u, dom, codom)
#     _adj_Φ′_ = adjoint(Φ′(model, u))
#     _adj_R′_ = adjoint(R′(model, u))
#     mult_adj_Φ′ = zeros(eltype(eltype(_adj_Φ′_)), dom, codom)
#     mult_adj_R′ = zeros(eltype(eltype(_adj_R′_)), dom, codom)
#     for j = 1:nspaces(dom), i = 1:nspaces(codom)
#         project!(component(mult_adj_Φ′, i, j), Multiplication(_adj_Φ′_[i,j]))
#         project!(component(mult_adj_R′, i, j), Multiplication(_adj_R′_[i,j]))
#     end
#     return mult_adj_Φ′ * Laplacian() + mult_adj_R′
# end

function C₀(P, opnorm_approxinvΔ, Z₂, ϵ_u; N, K)
    n = size(P.W_bar, 1)
    ω = frequency(space(P.W_bar[1]))
    M = N+K
    Δ = Laplacian()
    d = _nspaces(space(P.W_bar[1]))
    ∇ = Gradient{d}()
    PΔ = project(P, CosFourier.(M, ω) ^ n, CosFourier.(M+K, ω) ^ n) * project(Δ, CosFourier.(M, ω) ^ n, CosFourier.(M, ω) ^ n)
    #
    x = opnorm(norm.(P.W_bar, 1), 1)
    y = inv(ω*(M-K+1))^2 * opnorm(norm.([Δ] .* P.W_bar, 1), 1)
    z = inv(ω*(M-K+1)) * sum(l -> opnorm(norm.([∇[l]] .* P.W_bar, 1), 1), 1:d)
	return 2max(opnorm(PΔ, 1), x + y / 2 + z) * Z₂ / opnorm_approxinvΔ * ϵ_u
end

function C₁(W_bar, A_bar; N, K)
    M = N+K
    Δ = Laplacian()
    d = _nspaces(space(W_bar[1]))
    ω = frequency(space(W_bar[1]))
    ∇ = Gradient{d}()
    #
    x = inv(ω*(M-2K+1))^2 * opnorm(norm.(([Δ] .* W_bar) * A_bar, 1), 1) +
        inv(ω*(M+1))^2    * opnorm(norm.(([Δ] .* W_bar) * A_bar, 1), Inf)
    y = inv(ω*(M-2K+1)) * sum(l -> opnorm(norm.(([∇[l]] .* W_bar) * A_bar, 1), 1), 1:d)
    z = inv(ω*(M+1))    * sum(l -> opnorm(norm.(([∇[l]] .* W_bar) * A_bar, 1), Inf), 1:d)
	return x / 2 + y + z
end

function C₂(W_bar, B_bar; N, K)
    M = N+K
    d = _nspaces(space(W_bar[1]))
    ω = frequency(space(W_bar[1]))
    ∇ = Gradient{d}()
    #
    x = sum(l -> inv(ω*(M-K+1)) * (opnorm(norm.(B_bar[l], 1), 1) + opnorm(norm.(B_bar[l], 1), Inf)), 1:d) * opnorm(norm.(W_bar, 1), 1)
    y = sum(l -> inv(ω*(M-2K+1)) * opnorm(norm.(W_bar * B_bar[l], 1), 1) + inv(ω*(M+1)) * opnorm(norm.(W_bar * B_bar[l], 1), Inf), 1:d)
    z = sum(l -> inv(ω*(M-2K+1))^2 * opnorm(norm.(([∇[l]] .* W_bar) * B_bar[l], 1), 1) + inv(ω*(M+1))^2 * opnorm(norm.(([∇[l]] .* W_bar) * B_bar[l], 1), Inf), 1:d)
    return (x + y + z) / 2
end

function C₃(W_bar, C_bar; N, K)
    M = N+K
    ω = frequency(space(W_bar[1]))
    #
    x = inv(ω*(M-K+1))^2  * (opnorm(norm.(C_bar, 1), 1) + opnorm(norm.(C_bar, 1), Inf)) * opnorm(norm.(W_bar, 1), 1)
    y = inv(ω*(M-2K+1))^2 * opnorm(norm.(W_bar * C_bar, 1), 1) +
        inv(ω*(M+1))^2    * opnorm(norm.(W_bar * C_bar, 1), Inf)
    return (x + y) / 2
end

    export C₀, C₁, C₂, C₃



end

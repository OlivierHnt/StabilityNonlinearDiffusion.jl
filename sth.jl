

function _test_(k,ρ)
    return  k*ρ^k
end

function Bound_on_∇(d,ν,ν′,N_max)
    if ν <= 1
        error("ν must be greater than 1")
    end
    if 1< ν′ && ν′ > ν  
        error("ν′ must belongs to (1,ν)")
    end
    ρ = ν′/ν
    N_min = ceil(-1/ log(ρ)) #from N_min, test(k,ρ) is decreasing in k
    if N_min > N_max
        error("N_min > N_max")
    end
    N = N_min
    while N <= N_max && _test_(N,ρ) > 1
        N += 1
    end
    return d*N, _test_(N,ρ) < 1
end
export Bound_on_∇

function gershgorin(A)
    # Compute the Gershgorin disks for a given matrix A
    n = size(A, 1)
    disks = []
    for i in 1:n
        radius = sum(abs.(A[i, :])) - abs(A[i, i])
        push!(disks, (A[i, i], radius))
    end
    return disks
end
export gershgorin

function plot_gershgorin(A)
    # Plot the Gershgorin disks for a given matrix A
    disks = gershgorin(A)
    f = Figure()
    ax = Axis(f[1,1], aspect = DataAspect())
    ax2 = Axis(f[2,1], aspect = DataAspect())
    t = LinRange(0, 2π, 100)
    eit = [exp(im * tt) for tt in t]
    for disk in disks
        center = disk[1]
        radius = disk[2]
        scatter!(ax, [center 0], color = :blue, markersize = 4, marker = :circle)
        truc = radius .* eit .+ complex(center)
        lines!(ax, real(truc),imag(truc), color = :blue, linewidth = 2)
    end
    for disk in disks[end-3:end]
        center = disk[1]
        radius = disk[2]
        scatter!(ax2, [center 0], color = :blue, markersize = 4, marker = :circle)
        truc = radius .* eit .+ complex(center)
        lines!(ax2, real(truc),imag(truc), color = :blue, linewidth = 2)
    end
    display(f)
    return f
end

# function isinv(s)
#     # Check if a Sequence is invertible
#     n_ech = 2*length(s) + 1
#     ω = frequency(space(s))
#     tt = LinRange(0, ω/π, n_ech)
#     ss_ = s.(tt)
#     return prod(@. ss_ > 0) || prod(@. ss_ < 0)
# end

export plot_gershgorin

K = 100
ω = 1
spaceF = CosFourier(K, ω)
p = Sequence(spaceF, rand(Float64, K+1).*[ (1/5)^k for k in 0:K])
xx = LinRange(0, 1, 2*K+1)
lines(xx, x -> p(x))
Matp = [p zero(p); zero(p) p]
if !isinv(p)
       error("not invertible")
end
spaceBig = CosFourier(K, ω)
Δ = project(Laplacian(), spaceBig, spaceBig)
_r, _c, _v = RadiiPolynomial.SparseArrays.findnz(coefficients(Δ))
Δ⁻¹ = LinearOperator(spaceBig, spaceBig, RadiiPolynomial.SparseArrays.sparse(_r, _c, inv.(_v)))

Op = zero(Δ)
mul!(Op, Multiplication(p), Δ⁻¹)
mul!(Op, Δ⁻¹, Multiplication(p), -0.5, -0.5)
plot_gershgorin(coefficients(Op))
eig_Op = eigen(coefficients(Op))
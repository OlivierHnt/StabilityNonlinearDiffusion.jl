function test(k,ρ)
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
    while N <= N_max && test(N,ρ) > 1
        N += 1
    end
    return d*N, test(N,ρ) < 1
end
export Bound_on_∇

N_max = 100000
Bound_on_∇(1, 1.001, 1, N_max)
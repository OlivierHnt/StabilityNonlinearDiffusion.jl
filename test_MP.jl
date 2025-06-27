
function plot_gershgorin(disks)
    # Plot the Gershgorin disks for a given matrix A
    function my_log10(value)
        if value > 0
            format = rich("10", superscript("$(round(Int,log10(value)))"))
        else
            format = rich("$value")
        end
        return format 
    end
    f = Figure()
    # ax = Axis(f[1,1], 
    # # xticks = [0, 1e-3, 1e-2, 1e-1],
    # # aspect = DataAspect(),
    # aspect = AxisAspect(3),
    # # xtickformat = values -> [my_log10(value) for value in values]
    # )
    # # ax2 = Axis(f[2,1])#, aspect = DataAspect())
    ax3 = Axis(f[1,1], 
    aspect = AxisAspect(3),
    xscale = Makie.pseudolog10,
    # xticks = [0, 1, 2],
    xticks = [0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],#[0, 1, 10],#
    xtickformat = values -> [my_log10(value) for value in values]
    # aspect = DataAspect()
    )
    t = LinRange(0, 2π, 100)
    eit = [exp(im * tt) for tt in t]

    # for i in 1:2
    #     center = disks[i,1]
    #     radius = disks[i,2]
    #     scatter!(ax, [center 0], color = :black, markersize = 4, marker = :circle)
    #     truc = radius .* eit .+ complex(center)
    #     lines!(ax, real(truc), imag(truc), color = :green, linewidth = 2)
    # end
    # for i in 2:4
    #     center = disks[i,1]
    #     radius = disks[i,2]
    #     scatter!(ax2, [center 0], color = :black, markersize = 4, marker = :circle)
    #     truc = radius .* eit .+ complex(center)
    #     lines!(ax2, real(truc), imag(truc), color = :green, linewidth = 2)
    # end
    for i in 1:size(disks,1)
        center = disks[i,1]
        radius = disks[i,2]
        scatter!(ax3, [center 0], color = :black, markersize = 4, marker = :circle)
        truc = radius .* eit .+ complex(center)
        lines!(ax3, real(truc), imag(truc), color = :green, linewidth = 2)
    end
    # my = 1.1*maximum(disks[:,2])
    # mdown = min(0,1.1*maximum(disks[:,1])-my)
    # mup = 1.1*maximum(disks[:,1])+my
    # xlims!(ax, 0, 3e-2)
    xlims!(ax3, 0, 1e6)
    # ylims!(ax3, -my, my)
    display(f)
    return f
end

export plot_gershgorin


function RadiiPolynomial.project(a::ValidatedSequence, space_dest::VectorSpace, ::Type{T}=eltype(a)) where {T}
    a_seq = a.sequence
    c_seq = project(a_seq,space_dest)
    c_err = norm( a_seq - c_seq, a.banachspace)
    return ValidatedSequence(c_seq, a.sequence_error + c_err, a.banachspace)
end




fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect())
# truc = project(P_ext - μ₀*I, CosFourier(M, ω)^1,CosFourier(M+2K, ω)^1) - project(C, CosFourier(M+K, ω)^1, CosFourier(M+2K, ω)^1)*project(Γ*C, CosFourier(M, ω)^1, CosFourier(M+K, ω)^1)
# findmax(mid.(coefficients(truc)))
sp = spy!(ax, mid.(coefficients(component(Q̃,2,1))[1:M0+K, 1:M0+K+1]))
Colorbar(fig[:, end+1], sp)
display(fig)

fig = Figure()
ax = Axis(fig[1,1], aspect = DataAspect(), yreversed = true, xaxisposition = :top)
sp = spy!(ax, mid.(coefficients(Q)'))
Colorbar(fig[:, end+1], sp)
fig

fig = Figure()
ax11 = Axis(fig[1,1], 
# yscale = log10,
aspect = DataAspect(), yreversed = true, xaxisposition = :top
)
ax12 = Axis(fig[1,2], 
# yscale = log10,
aspect = DataAspect(), yreversed = true, xaxisposition = :top
)
ax21 = Axis(fig[2,1], 
# yscale = log10,
aspect = DataAspect(), yreversed = true, xaxisposition = :top
)
ax22 = Axis(fig[2,2], 
# yscale = log10,
aspect = DataAspect(), yreversed = true, xaxisposition = :top
)
sp = spy!(ax11, (mid.(coefficients(component(Q0,1,1)))))
spy!(ax12, (mid.(coefficients(component(Q0,1,2)))))
spy!(ax21, (mid.(coefficients(component(Q0,2,1)))))
spy!(ax22, (mid.(coefficients(component(Q0,2,2)))))
Colorbar(fig[:, end+1], sp)
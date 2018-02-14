using StaticArrays

abstract type MeshElement end
abstract type Tri <: MeshElement end
abstract type Tet <: MeshElement end

struct RefBasisFunction
    ϕ::Vector{Float64}
    ∇ϕ::Matrix{Float64}
end

function get_basis_funcs(::Type{Tri})
    ϕs = (
        x -> 1.0 - x[1] - x[2],
        x -> x[1],
        x -> x[2]
    )

    ∇ϕs = (
        x -> (-1.0, -1.0),
        x -> ( 1.0,  0.0),
        x -> ( 0.0,  1.0)
    )

    return ϕs, ∇ϕs
end

function get_basis_funcs(::Type{Tet})
    ϕs = (
        x -> 1.0 - x[1] - x[2] - x[3],
        x -> x[1],
        x -> x[2],
        x -> x[3],
    )

    ∇ϕs = (
        x -> (-1.0, -1.0, -1.0),
        x -> ( 1.0,  0.0,  0.0),
        x -> ( 0.0,  1.0,  0.0),
        x -> ( 0.0,  0.0,  1.0)
    )

    return ϕs, ∇ϕs
end

"""
Evaluate ϕ and ∇ϕ in the provided quadrature points
"""
function element_basis(::Type{T}, quad_points::NTuple{n,Coord{d}}) where {T<:MeshElement,n,d}
    ϕs_sym, ∇ϕs_sym = get_basis_funcs(T)
   
    bases = Vector{RefBasisFunction}(length(ϕs_sym))

    for (i, (ϕ_sym, ∇ϕ_sym)) in enumerate(zip(ϕs_sym, ∇ϕs_sym))
        ϕ = zeros(n)
        ∇ϕ = zeros(d,n)

        for (idx, point) in enumerate(quad_points)
            ϕ[idx] = ϕ_sym(point)
            ∇ϕ[:, idx] .= ∇ϕ_sym(point)
        end

        bases[i] = RefBasisFunction(ϕ, ∇ϕ)
    end

    return bases
end
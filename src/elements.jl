using StaticArrays

abstract type MeshElement end
abstract type Tri <: MeshElement end
abstract type Tet <: MeshElement end

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

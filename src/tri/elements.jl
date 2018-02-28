abstract type Tri <: MeshElement end

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
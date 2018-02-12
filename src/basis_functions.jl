struct EvaluatedBasisFunctions{points, dim}
    ϕ::SVector{points, Float64}
    ∇ϕ::SVector{points, SVector{dim, Float64}}
end

struct EvaluatedBasisFunction{dim}
    ϕ::Float64
    ∇ϕ::SVector{dim, Float64}
end

function calculate_interpolation_polynomial_derivatives(ϕ; vars = (:x, :y))
    return [differentiate(ϕ, var) for var in vars]
end

"""
    create_basis((0.5, 0.0), (0.0, 0.5), (0.5, 0.5), :(1 - x - y))

For a given basis function ϕ and quadrature points xs, we return
a BasisFunction with ϕ and ∇ϕ evaluated in all xs.
"""
function create_basis(xs::NTuple{points,Coord}, ϕ::Union{Expr,Symbol}) where {points}
    ∇ϕ = calculate_interpolation_polynomial_derivatives(ϕ)
    create_basis(xs, ϕ, ∇ϕ)
end

function create_basis(xs::NTuple{n,Coord}, ϕ, ∇ϕ) where {n}
    ϕ_values = Vector{Float64}(n)
    ∇ϕ_values = Matrix{Float64}(2, n)
    
    for (idx, point) in enumerate(xs)
        Q = Expr(:block)
        push!(Q.args, :((x, y) = $point), :($ϕ))
        ϕ_values[idx] = eval(Q)
        
        R = Expr(:block)
        push!(R.args, :((x, y) = $point), :($∇ϕ))
        ∇ϕ_values[:, idx] .= eval(R)
    end

    ϕ_svec = SVector{n, Float64}(ϕ_values)
    ∇ϕ_svec = SVector{n, SVector{2, Float64}}([SVector{2, Float64}(∇ϕ_values[:, i]) for i = 1 : n]...)
    
    return EvaluatedBasisFunctions(ϕ_svec, ∇ϕ_svec)
end

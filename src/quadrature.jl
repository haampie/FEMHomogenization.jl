abstract type QuadRule end

"""
Get the default quadrature rule for an element type
"""
default_quadrature(::Type{<:MeshElement}) = throw("Not implemented")

"""
Basisfunction is a pair (ϕ, grad ϕ) evaluated in a quadrature point in the
reference basis element. It also has space allocated for the gradient ∇ϕ when
a change of coordinates is applied.
"""
struct BasisFunction{d,T}
    ϕ::T
    grad::SVector{d,T}
    ∇ϕ::MVector{d,T}

    BasisFunction{d,T}(ϕ, grad, ∇ϕ) where {d,T} = new(ϕ, grad, ∇ϕ)
end

BasisFunction(ϕ::T, grad::SVector{d,T}) where {d,T} = 
  BasisFunction{d,T}(ϕ, grad, zeros(MVector{d,T}))

"""
Evaluate ϕs and ∇ϕs in all quadrature points xs.
"""
function evaluate_basis_funcs(ϕs, ∇ϕs, xs)
    d = length(xs[1])
    n = length(xs)
    basis = Vector{Vector{BasisFunction{d,Float64}}}(n)

    # Go over each quad point x
    for (i, x) in enumerate(xs)
        inner = Vector{BasisFunction{d,Float64}}(n)

        # Evaluate ϕ and ∇ϕ in x
        for (j, (ϕ, ∇ϕ)) in enumerate(zip(ϕs, ∇ϕs))
            inner[j] = BasisFunction(ϕ(x), SVector(∇ϕ(x)))
        end

        basis[i] = inner
    end

    return basis
end
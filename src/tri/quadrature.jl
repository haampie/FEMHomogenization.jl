abstract type Tri1 <: QuadRule end
abstract type Tri3 <: QuadRule end
abstract type Tri4 <: QuadRule end

"""
Maps an element type to a quadrature type
"""
default_quadrature(::Type{Tri}) = Tri3

function quadrature_rule(::Type{Tri1})
    weights = (1,)
    points = (Coord{2}(0.25,0.25),)
    return weights, points
end

function quadrature_rule(::Type{Tri3})
    weights = (1/6, 1/6, 1/6)
    points = (Coord{2}(0.0, 0.5), Coord{2}(0.5, 0.0), Coord{2}(0.5, 0.5))
    return weights, points
end

function quadrature_rule(::Type{Tri4})
    weights = (-27/96, 25/96, 25/96, 25/96)
    points = (Coord{2}(1/3, 1/3), Coord{2}(1/5, 1/5), Coord{2}(1/5, 3/5), Coord{2}(3/5, 1/5))
    return weights, points
end

"""
Returns the affine map from the blueprint element to the given element.
"""
function affine_map(m::Mesh{Tri,Tv,Ti}, el::SVector{3,Ti}) where {Tv,Ti}
    @inbounds begin
        p1 = m.nodes[el[1]]
        p2 = m.nodes[el[2]]
        p3 = m.nodes[el[3]]
        return [p2 - p1 p3 - p1], p1
    end
end

abstract type QuadRule end
abstract type Tri3 <: QuadRule end
abstract type Tri4 <: QuadRule end

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
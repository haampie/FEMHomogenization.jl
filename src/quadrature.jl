abstract type QuadRule end
abstract type Tri3 <: QuadRule end
abstract type Tri4 <: QuadRule end
abstract type Tet4 <: QuadRule end
abstract type Tet5 <: QuadRule end

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

function quadrature_rule(::Type{Tet4})
    weights = (1/24, 1/24, 1/24, 1/24)
    a, b = 0.5854101966249685, 0.1381966011250105
    points = (Coord{3}(a, b, b), Coord{3}(b,a,b), Coord{3}(b,b,a), Coord{3}(b,b,b))

    return weights, points
end

function quadrature_rule(::Type{Tet5})
    weights = (-2/15, 3/40, 3/40, 3/40, 3/40)
    points = (Coord{3}(1/4,1/4,1/4), Coord{3}(1/2,1/6,1/6), Coord{3}(1/6,1/6,1/6), Coord{3}(1/6,1/6,1/2), Coord{3}(1/6,1/2,1/6))
    
    return weights, points
end

"""
Maps an element type to a quadrature type
"""
default_quadrature(::Type{Tri}) = Tri3
default_quadrature(::Type{Tet}) = Tet4
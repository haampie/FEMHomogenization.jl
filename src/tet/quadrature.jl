abstract type Tet4 <: QuadRule end
abstract type Tet5 <: QuadRule end

default_quadrature(::Type{Tet}) = Tet4

function quadrature_rule(::Type{Tet4})
    weights = (1/24, 1/24, 1/24, 1/24)
    a, b = 0.5854101966249685, 0.1381966011250105
    points = (Coord{3}(a,b,b), Coord{3}(b,a,b), 
              Coord{3}(b,b,a), Coord{3}(b,b,b))

    return weights, points
end

function quadrature_rule(::Type{Tet5})
    weights = (-2/15, 3/40, 3/40, 3/40, 3/40)
    points = (Coord{3}(1/4,1/4,1/4), Coord{3}(1/2,1/6,1/6), 
              Coord{3}(1/6,1/6,1/6), Coord{3}(1/6,1/6,1/2), 
              Coord{3}(1/6,1/2,1/6))
    
    return weights, points
end

function affine_map(m::Mesh{Tet,Tv,Ti}, el::SVector{4,Ti}) where {Tv,Ti}
    p1, p2, p3, p4 = m.nodes[el[1]], m.nodes[el[2]], m.nodes[el[3]], m.nodes[el[4]]
    return [p2 - p1 p3 - p1 p4 - p1], p1
end
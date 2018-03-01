using StaticArrays

"""
Starting out with a list of nodes and triangles, we recursively refine things
and at the bottom level we integrate.

Potentially easily parallellizable as well.
"""
function matrix_free_2d(lvl)
    triangle = ((SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.0, 1.0)), 
                (SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(1.0, 1.0)))
    points = Vector{SVector{2,Float64}}()

    for t in triangle
        refine(t, lvl, points)
    end

    return points
end

function refine(t::NTuple{3}, level::Int, points)
    if level == 0
        push!(points, t[1], t[2], t[3])
        return points
    end


    n1, n2, n3 = t
    c1 = (n1 + n2) / 2
    c2 = (n1 + n3) / 2
    c3 = (n2 + n3) / 2

    refine((n1, c1, c2), level - 1, points)
    refine((n2, c1, c3), level - 1, points)
    refine((n3, c2, c3), level - 1, points)

    return points
end
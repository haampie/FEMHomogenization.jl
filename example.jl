using StaticArrays

import Base.show

const Coord = SVector{2, Float64}

struct Triangle
    n1::Int
    n2::Int
    n3::Int
end

struct Mesh
    nodes::Vector{Coord}
    triangles::Vector{Triangle}
end

show(io::IO, t::Triangle) = print(io, "(", t.n1, ", ", t.n2, ", ", t.n3, ")")
show(io::IO, c::Coord) = print(io, "(", c[1], ", ", c[2], ")")

# """
# Creates a standard uniform mesh of the domain [0,1]
# with triangular elements
# """
# function uniform_mesh(n = 16)
#     xs = linspace(0, 1, n + 1)
#     p = Vector{Coord}((n + 1)^2)
#     t = Vector{Triangle}(2n^2)

#     # Nodes
#     for i = 1 : n + 1, j = 1 : n + 1
#         idx = (i - 1) * (n + 1) + j
#         p[idx] = Coord(xs[j], xs[i])
#     end

#     # Triangles
#     triangle = 1
#     for i = 1 : n, j = 1 : n
#         idx = (i - 1) * (n + 1) + j
#         # (Top left, top right, bottom left)
#         t[triangle] = Triangle(idx, idx + 1, idx + n + 1)
#         triangle += 1

#         # (Top right, bottom left, bottom right)
#         t[triangle] = Triangle(idx + 1, idx + n + 1, idx + n + 2)
#         triangle += 1
#     end

#     return Mesh(p, t)
# end

function load(nodes_file::String, triangle_file::String)
    nmat = readdlm(nodes_file, ' ', Float64)
    tmat = readdlm(triangle_file, ' ', Int64)
    nodes = size(nmat, 1)
    triangles = size(tmat, 1)

    p = Vector{Coord}(nodes)
    t = Vector{Triangle}(triangles)

    for i = 1 : nodes
        p[i] = Coord(nmat[i, 1], nmat[i, 2])
    end

    for i = 1 : triangles
        t[i] = Triangle(tmat[i, 1], tmat[i, 2], tmat[i, 3])
    end

    return Mesh(p, t)
end

integrate(g) = (g(0.5, 0) + g(0.5, 0.5) + g(0, 0.5)) / 6

"""
Assembles the coefficient matrix A
"""
function assemble()
    mesh = load("p.txt", "t.txt")

    # Loop over all triangles & compute local system matrix
    for (idx, triangle) in enumerate(mesh.triangles)
        p1 = mesh.nodes[triangle.n1]
        p2 = mesh.nodes[triangle.n2]
        p3 = mesh.nodes[triangle.n3]
        
        Bk = [p2 - p1 p3 - p1]
        detBk = abs(det(Bk))
    end

    mesh
end
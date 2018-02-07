using StaticArrays

import Base.show

const Coord = SVector{2, Float64}
const Triangle = SVector{3, Int}

struct Mesh
    nodes::Vector{Coord}
    triangles::Vector{Triangle}
end

@inline triangle_coords(m::Mesh, t::Triangle) = m.nodes[t[1]], m.nodes[t[2]], m.nodes[t[3]]

show(io::IO, t::Triangle) = print(io, "(", t[1], ", ", t[2], ", ", t[3], ")")
show(io::IO, c::Coord) = print(io, "(", c[1], ", ", c[2], ")")

"""
Creates a standard uniform mesh of the domain [0,1]
with triangular elements
"""
function uniform_mesh(n::Int = 16)
    xs = linspace(0, 1, n + 1)
    p = Vector{Coord}((n + 1)^2)
    t = Vector{Triangle}(2n^2)

    # Nodes
    for i = 1 : n + 1, j = 1 : n + 1
        idx = (i - 1) * (n + 1) + j
        p[idx] = Coord(xs[j], xs[i])
    end

    # Triangles
    triangle = 1
    for i = 1 : n, j = 1 : n
        idx = (i - 1) * (n + 1) + j
        
        # (Top left, top right, bottom left)
        t[triangle] = Triangle(idx, idx + 1, idx + n + 1)
        triangle += 1

        # (Top right, bottom left, bottom right)
        t[triangle] = Triangle(idx + 1, idx + n + 1, idx + n + 2)
        triangle += 1
    end

    return Mesh(p, t)
end

function build_linear_shape_funcs()
    # Build prototype shape functions (evaluated in grid quadrature points)
    ϕ_f = @SVector [
        (x, y) -> 1.0 - x - y, 
        (x, y) -> x, 
        (x, y) -> y
    ]

    ∇ϕ_f = @SVector [
        (x, y) -> SVector{2, Float64}(-1.0, -1.0),
        (x, y) -> SVector{2, Float64}(1.0, 0.0),
        (x, y) -> SVector{2, Float64}(0.0, 1.0)
    ]

    quad_points = ((0.0, 0.5), (0.5, 0.0), (0.5, 0.5))

    # ϕs[i, j] is the 
    ϕs = @SMatrix [ϕ(x[1], x[2]) for ϕ = ϕ_f, x = quad_points]

    # ∇ϕs[i, j]
    ∇ϕs = @SMatrix [∇ϕ(x[1], x[2]) for ∇ϕ = ∇ϕ_f, x = quad_points]

    return ϕs, ∇ϕs
end

integrate(g) = (g(0.5, 0) + g(0.5, 0.5) + g(0, 0.5)) / 6

"""
Assembles the coefficient matrix A
"""
function assemble(n::Int = 16)
    # Get a mesh
    mesh = uniform_mesh(n)
    ϕs, ∇ϕs = build_linear_shape_funcs()

    # Loop over all triangles & compute local system matrix
    for triangle in mesh.triangles
        p1, p2, p3 = triangle_coords(mesh, triangle)
        coord_transform = [p2 - p1 p3 - p1]
        invBk = inv(coord_transform')

        # Local system matrix
        A_local = zeros(SMatrix{3,3})

        # Compute a(ϕ_i, ϕ_j) for all combinations of i and j in the triangle
        for i = 1:3, j = 1:3
            ∇ϕ_i = invBk * ∇ϕs[i]
            ∇ϕ_j = invBk * ∇ϕs[j]
            
            for k = 1:3
                A_local[i,j] += ∇ϕ_i[k]' * ∇ϕ_j[k]
            end
        end

        A_local .*= abs(det(coord_transform))
            
        # # Put Ae and be into A and b 
        # for n = 1:3
        #     for m = 1:3
        #         A(th(n), th(m)) = A(th(n), th(m)) + Ae(n,m);
                
        #     end
        #     b(th(n)) = b(th(n)) + be(n);
        # end
    end

    mesh
end
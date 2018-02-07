using StaticArrays

import Base.show

const Coord = SVector{2, Float64}
const Triangle = SVector{3, Int}

struct Mesh
    n::Int
    nodes::Vector{Coord}
    triangles::Vector{Triangle}
end

@inline triangle_coords(m::Mesh, t::Triangle) = m.nodes[t[1]], m.nodes[t[2]], m.nodes[t[3]]

struct GraphBuilder
    edges::Vector{Vector{Int}}
end

struct Graph
    n_nodes::Int
    n_edges::Int
    edges::Vector{Vector{Int}}
end

function add_edge!(c::GraphBuilder, i::Int, j::Int)
    if i == j
        push!(c.edges[i], i)
    else
        push!(c.edges[i], j)
        push!(c.edges[j], i)
    end
end

"""
Creates the connectivity graph from the triangles
"""
function mesh_to_graph(m::Mesh)
    c = GraphBuilder([[] for i = 1 : m.n])

    # Self loops for each node
    for i = 1 : m.n
        add_edge!(c, i, i)
    end

    # Edges of all triangles
    for t in m.triangles
        add_edge!(c, t[1], t[2])
        add_edge!(c, t[2], t[3])
        add_edge!(c, t[1], t[3])
    end

    # Remove duplicates
    n_edges = 0
    for i = 1 : m.n
        c.edges[i] = sort!(unique(c.edges[i]))
        n_edges += length(c.edges[i])
    end

    return Graph(m.n, n_edges, c.edges)
end

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

    return Mesh(length(p), p, t)
end

function build_linear_shape_funcs()
    # This guy is constant, so don't bother evaluating functions etc.
    ∇ϕs = (SVector(-1.0, -1.0), SVector(1.0, 0.0), SVector(0.0, 1.0))

    ϕs = @SMatrix [ϕ(x[1], x[2]) for ϕ in (
            (x, y) -> 1.0 - x - y, 
            (x, y) -> x, 
            (x, y) -> y
        ), x = (
            (0.0, 0.5),
            (0.5, 0.0),
            (0.5, 0.5)
        )
    ]

    return ϕs, ∇ϕs
end

integrate(g) = (g(0.5, 0) + g(0.5, 0.5) + g(0, 0.5)) / 6

function coefficient_matrix_factory(g::Graph)
    nzval = zeros(g.n_edges)
    rowval = Vector{Int}(g.n_edges)
    colptr = Vector{Int}(g.n_nodes + 1)
    colptr[1] = 1

    edge_num = 1
    for i = 1 : g.n_nodes
        colptr[i + 1] = colptr[i] + length(g.edges[i])
        rowval[colptr[i] : colptr[i + 1] - 1] .= g.edges[i]
    end

    return SparseMatrixCSC(g.n_nodes, g.n_nodes, colptr, rowval, nzval)
end

"""
Assembles the coefficient matrix A
"""
function assemble(n::Int = 16)
    # Get a mesh
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    ϕs, ∇ϕs = build_linear_shape_funcs()

    # Slow variant
    # A = spzeros(mesh.n, mesh.n)

    # Fast variant
    A = coefficient_matrix_factory(graph)

    A_local = zeros(3, 3)

    # Loop over all triangles & compute local system matrix
    for triangle in mesh.triangles
        p1, p2, p3 = triangle_coords(mesh, triangle)
        coord_transform = [p2 - p1 p3 - p1]
        invBk = inv(coord_transform')

        # Local system matrix
        fill!(A_local, 0.0)

        # Transform the gradients
        ∇ϕ = @SVector [invBk * ∇ϕs[i] for i = 1 : 3]

        # Compute a(ϕ_i, ϕ_j) with the quadrature scheme 
        # for all combinations of i and j in the support of the triangle
        for i = 1:3, j = 1:3, k = 1:3
            A_local[i,j] += dot(∇ϕ[i], ∇ϕ[j]) + ϕs[i,k] * ϕs[j,k]
        end

        A_local .*= abs(det(coord_transform))

        # Put A_local into A
        for n = 1:3, m = 1:3
            tn, tm = triangle[n], triangle[m]

            # Slow variant
            # A[tn, tm] += A_local[n,m]

            # Fast variant
            offset = searchsortedfirst(graph.edges[tn], tm)
            A.nzval[A.colptr[tn] + offset - 1] += A_local[n,m]
        end
    end

    A, mesh, graph
end
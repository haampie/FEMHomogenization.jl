import Base.show

"""
Create three shape functions on each
vertex of a reference triangle
"""
function build_linear_shape_funcs(points::NTuple{n,Coord}) where {n}
    ϕ1 = create_basis(points, :(1 - x - y))
    ϕ2 = create_basis(points, :x)
    ϕ3 = create_basis(points, :y)
    return (ϕ1, ϕ2, ϕ3)
end

"""
Builds the SparseMatrixCSC structure from the graph
of the mesh. Initializes all `nzval`s with zero.
The `nzval`s are set in the assembly phase.
"""
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
Map an edge (from, to) to the index of the value in the
sparse matrix
"""
@inline function edge_to_idx(A::SparseMatrixCSC, g::Graph, from, to)
    offset = searchsortedfirst(g.edges[from], to)
    return A.colptr[from] + offset - 1
end

"""
Assembles the coefficient matrix A
"""
function assemble(f::Function, a11::Function, a22::Function, n::Int = 16)
    # Quadrature scheme
    points = (Coord(0.0, 0.5), Coord(0.5, 0.0), Coord(0.5, 0.5))
    weights = (1/6, 1/6, 1/6)

    # Get a mesh
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    basis = build_linear_shape_funcs(points)
    A = coefficient_matrix_factory(graph)
    b = zeros(mesh.n)

    A_local = zeros(3, 3)
    b_local = zeros(3)

    # Loop over all triangles & compute local system matrix
    for triangle in mesh.triangles
        p1, p2, p3 = triangle_coords(mesh, triangle)
        coord_transform = [p2 - p1 p3 - p1]
        invBk = inv(coord_transform') 

        # Reset local matrix and vector
        fill!(A_local, 0.0)
        fill!(b_local, 0.0)

        # Compute a(ϕ_i, ϕ_j) with the quadrature scheme in all points k
        # for all combinations of i and j in the support of the triangle
        for i = 1:3 
            for j = 1:3, k = 1:3
                x = coord_transform * points[k] + p1
                ∇ϕi = invBk * basis[i].∇ϕ[k]
                ∇ϕj = invBk * basis[j].∇ϕ[k]
                ϕi = basis[i].ϕ[k]
                ϕj = basis[j].ϕ[k]
                g = a11(x) * ∇ϕi[1] * ∇ϕj[1] + a22(x) * ∇ϕi[2] * ∇ϕj[2]
                A_local[i,j] += weights[k] * g
            end

            for k = 1:3
                x = coord_transform * points[k] + p1
                b_local[i] += weights[k] * f(x) * basis[i].ϕ[k]
            end
        end

        Bk_det = abs(det(coord_transform))
        A_local .*= Bk_det
        b_local .*= Bk_det

        # Put A_local into A
        for n = 1:3, m = 1:3
            i, j = triangle[n], triangle[m]
            idx = edge_to_idx(A, graph, i, j)
            A.nzval[idx] += A_local[n,m]
        end

        # Put b_local in b
        for n = 1:3
            b[triangle[n]] += b_local[n]
        end
    end

    # Build the matrix for interior connections only
    Ai = A[mesh.interior, mesh.interior]
    bi = b[mesh.interior]

    A, b, Ai, bi, mesh, graph
end

function checkerboard(m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m+1, m+1))

    # Given a coordinate, return the value
    return (x::Coord) -> begin
        x_idx = 1 + floor(Int, x[1] * m)
        y_idx = 1 + floor(Int, x[2] * m)
        return A[y_idx, x_idx]
    end
end

function example(n = 16, c = 50)
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    f = (x::Coord) -> x[1] * x[2]

    A, b, Ai, bi, mesh, graph = assemble(f, a11, a22, n)

    x = zeros(mesh.n)
    x[mesh.interior] .= Ai \ bi

    x, f.(mesh.nodes)
end
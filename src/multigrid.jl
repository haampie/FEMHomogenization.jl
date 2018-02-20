
"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.
"""
function refine(m::Mesh{Tri,Ti,Tv}, graph::Graph{Ti}) where {Tv,Ti}
    Nn = length(m.nodes)
    Nt = length(m.triangles)
    Ne = graph.total[end] - 1

    # Each edge is split 2, so Nn + Ne is the number of nodes
    nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    triangles = Vector{SVector{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(nodes, m.nodes)
    
    # Add the new ones
    idx = Nn + 1
    for (from, edges) in enumerate(graph.edges), to in edges
        nodes[idx] = (m.nodes[from] + m.nodes[to]) / 2
        idx += 1
    end

    # Split each triangle in four smaller ones
    for (i, t) in enumerate(m.triangles)

        # Index of the nodes on the new edges
        a = edge_index(graph, t[1], t[2]) + Nn
        b = edge_index(graph, t[2], t[3]) + Nn
        c = edge_index(graph, t[3], t[1]) + Nn

        # Split the triangle in 4 pieces
        idx = 4i - 3
        triangles[idx + 0] = SVector(t[1], a, c)
        triangles[idx + 1] = SVector(t[2], a, b)
        triangles[idx + 2] = SVector(t[3], b, c)
        triangles[idx + 3] = SVector(a   , b, c)
    end

    return Mesh(Tri, nodes, triangles)
end

"""
Return the interpolation operator
"""
function interpolation_operator(mesh::Mesh{Te,Ti,Tv}, graph::Graph{Ti}) where {Te,Ti,Tv}
    # Interpolation operator
    Nn = length(graph.edges)
    Ne = graph.total[end] - 1

    nzval = Vector{Tv}(Nn + 2Ne)
    colptr = Vector{Ti}(Nn + Ne + 1)
    rowval = Vector{Ti}(Nn + 2Ne)

    # Nonzero values
    for i = 1 : Nn
        nzval[i] = 1.0
    end

    for i = Nn + 1 : Nn + 2Ne
        nzval[i] = 0.5
    end

    # Column pointer
    for i = 1 : Nn + 1
        colptr[i] = i
    end

    for i = Nn + 2 : Nn + Ne + 1
        colptr[i] = 2 + colptr[i - 1]
    end

    # Row values
    for i = 1 : Nn
        rowval[i] = i
    end

    idx = Nn + 1
    for (from, edges) in enumerate(graph.edges), to in edges
        rowval[idx] = from
        rowval[idx + 1] = to
        idx += 2
    end

    return SparseMatrixCSC(Nn, Nn + Ne, colptr, rowval, nzval)
end

"""
A geometric level of the grid
"""
struct Level{Te,Tv,Ti}
    mesh::Mesh{Te,Ti,Tv}
    graph::Graph{Ti}
    boundary::Vector{Ti}
    interior::Vector{Ti}
end

"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.
"""
function refine(m::Mesh{Tri,Ti,Tv}, graph::FastGraph{Ti}) where {Tv,Ti}
    Nn = length(m.nodes)
    Nt = length(m.triangles)
    Ne = graph.ptr[end] - 1

    # Each edge is split 2, so Nn + Ne is the number of nodes
    nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    triangles = Vector{SVector{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(nodes, m.nodes)
    
    # Add the new ones
    idx = Nn + 1
    @inbounds for i = 1 : length(graph.ptr) - 1
        for j = graph.ptr[i] : graph.ptr[i + 1] - 1
            nodes[idx] = (m.nodes[i] + m.nodes[graph.adj[j]]) / 2
            idx += 1
        end
    end

    # Split each triangle in four smaller ones
    @inbounds for (i, t) in enumerate(m.triangles)

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
function interpolation_operator(mesh::Mesh{Te,Ti,Tv}, graph::FastGraph{Ti}) where {Te,Ti,Tv}
    # Interpolation operator
    Nn = length(mesh.nodes)
    Ne = length(graph.adj)

    nzval = Vector{Tv}(Nn + 2Ne)
    colptr = Vector{Ti}(Nn + Ne + 1)
    rowval = Vector{Ti}(Nn + 2Ne)

    # Nonzero values
    @inbounds for i = 1 : Nn
        nzval[i] = 1.0
    end

    @inbounds for i = Nn + 1 : Nn + 2Ne
        nzval[i] = 0.5
    end

    # Column pointer
    @inbounds for i = 1 : Nn + 1
        colptr[i] = i
    end

    @inbounds for i = Nn + 2 : Nn + Ne + 1
        colptr[i] = 2 + colptr[i - 1]
    end

    # Row values
    @inbounds for i = 1 : Nn
        rowval[i] = i
    end

    idx = Nn + 1
    for i = 1 : Nn
        for j = graph.ptr[i] : graph.ptr[i + 1] - 1
            rowval[idx] = i
            rowval[idx + 1] = graph.adj[j]
            idx += 2
        end
    end

    # Note the transpose
    return SparseMatrixCSC(Nn, Nn + Ne, colptr, rowval, nzval)'
end

"""
A geometric level of the grid
"""
struct Level{Te,Tv,Ti}
    mesh::Mesh{Te,Ti,Tv}
    graph::FastGraph{Ti}
    interior::Vector{Ti}
end
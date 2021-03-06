"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.
"""
function refine(mesh::Mesh{Tri,Tv,Ti}, graph::Graph{Ti}) where {Tv,Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    Ne = length(graph.adj)

    # Each edge is split 2, so Nn + Ne is the number of nodes
    nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    triangles = Vector{SVector{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(nodes, mesh.nodes)
    
    # Add the new ones
    idx = Nn + 1
    @inbounds for i = 1 : length(graph.ptr) - 1, j = graph.ptr[i] : graph.ptr[i + 1] - 1
        nodes[idx] = (mesh.nodes[i] + mesh.nodes[graph.adj[j]]) / 2
        idx += 1
    end

    # Split each triangle in four smaller ones
    @inbounds for (i, t) in enumerate(mesh.elements)

        # Index of the nodes on the new edges
        a = edge_index(graph, t[1], t[2]) + Nn
        b = edge_index(graph, t[2], t[3]) + Nn
        c = edge_index(graph, t[3], t[1]) + Nn

        # Split the triangle in 4 pieces
        idx = 4i - 3
        triangles[idx + 0] = SVector(t[1], a, c)
        triangles[idx + 1] = SVector(t[2], b, a)
        triangles[idx + 2] = SVector(t[3], c, b)
        triangles[idx + 3] = SVector(a   , b, c)
    end

    return Mesh(Tri, nodes, triangles)
end

"""
If you aren't interested in the graph, this is helpful
"""
function refine(mesh::Mesh{Tri,Tv,Ti}, refinements::Int) where {Tv,Ti}
    for i = 1 : refinements
        mesh = refine(mesh, remove_duplicates!(to_graph(mesh)))
    end

    return mesh
end

function refine_with_operator(mesh::Mesh{Tri})
    graph = remove_duplicates!(to_graph(mesh))
    return refine(mesh, graph), interpolation_operator(mesh, graph)
end
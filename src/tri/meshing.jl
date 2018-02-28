"""
Constructor for triangles
"""
function Mesh(Te::Type{Tri}, nodes::Vector{SVector{2,Tv}}, tris::Vector{SVector{3,Ti}}) where {Tv,Ti}
    Mesh{Tri,Tv,Ti,2,3}(nodes, tris)
end

"""
Construct the adjacency list much like the colptr and rowval arrays in the 
SparseMatrixCSC type
"""
function to_graph(mesh::Mesh{Tri})
    Nn = length(mesh.nodes)
    ptr = zeros(Int, Nn + 1)

    # Count edges per node
    @inbounds for triangle in mesh.elements
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            idx = triangle[a] < triangle[b] ? triangle[a] : triangle[b]
            ptr[idx + 1] += 1
        end
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{Int}(last(ptr) - 1)
    indices = copy(ptr)

    @inbounds for triangle in mesh.elements
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            from, to = sort(triangle[a], triangle[b])
            adj[indices[from]] = to
            indices[from] += 1
        end
    end

    sort_edges!(Graph(ptr, adj))
end

"""
Build the graph and at the same time find the boundary & interior nodes
"""
function construct_graph_and_find_interior_nodes(mesh::Mesh{Tri})
    graph = to_graph(mesh)
    boundary = find_boundary_nodes(graph)
    interior = complement(boundary, length(mesh.nodes))
    remove_duplicates!(graph)

    graph, boundary, interior
end

"""
Divide the unit square into a mesh of triangles
"""
function unit_square(refinements::Int = 4)
    nodes = SVector{2,Float64}[(0, 0), (1, 0), (1, 1), (0, 1)]
    triangles = SVector{3,Int64}[(1, 2, 3), (1, 4, 3)]
    mesh = Mesh(Tri, nodes, triangles)

    graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)

    for i = 1 : refinements
        mesh = refine(mesh, graph)
        graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
    end

    return mesh, graph, interior
end
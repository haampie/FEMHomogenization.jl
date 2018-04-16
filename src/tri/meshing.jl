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

function generic_square(refinements::Int = 4, x = 1.0, y = 1.0)
    nodes = SVector{2,Float64}[(0, 0), (x, 0), (x, y), (0, y)]
    triangles = SVector{3,Int64}[(1, 2, 3), (1, 4, 3)]
    mesh = Mesh(Tri, nodes, triangles)

    graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)

    for i = 1 : refinements
        mesh = refine(mesh, graph)
        graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
    end

    return mesh, graph, interior
end

"""
Divide the domain [0, width] × [0, height] into a mesh of (n + 1) × (m + 1) grid
points / n × m cells.
"""
function rectangle(m, n, width, height)
    nodes = Vector{SVector{2,Float64}}((n+1) * (m+1))
    elements = Vector{SVector{3,Int64}}(2 * n * m)
    interior = Vector{Int64}((n - 1) * (m - 1))

    Δx = width / m
    Δy = height / n

    node_idx = 1

    # Nodes
    @inbounds for j = 1 : n + 1, i = 1 : m + 1
        nodes[node_idx] = ((i - 1) * Δx, (j - 1) * Δy)
        node_idx += 1
    end

    node_idx = 1
    el_idx = 1

    # Elements
    @inbounds for j = 1 : n
        for i = 1 : m
            elements[el_idx + 0] = (node_idx, node_idx + m + 1, node_idx + m + 2)
            elements[el_idx + 1] = (node_idx, node_idx + 1, node_idx + m + 2)
            el_idx += 2
            node_idx += 1
        end
        node_idx += 1
    end

    # Interior
    node_idx = 1
    @inbounds for j = 2 : n, i = 2 : m
        interior[node_idx] = (j - 1) * (m + 1) + i
        node_idx += 1
    end

    return Mesh(Tri, nodes, elements), interior
end
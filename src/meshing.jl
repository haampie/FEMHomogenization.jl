"""
Mesh is just a bunch of nodes and triangles
"""
struct Mesh{Te<:MeshElement,Ti,Tv}
    nodes::Vector{SVector{2,Tv}}
    triangles::Vector{SVector{3,Ti}}
end

function Mesh(Te::Type{<:MeshElement}, nodes::Vector{SVector{2,Tv}}, triangles::Vector{SVector{3,Ti}}) where {Tv,Ti}
    Mesh{Te,Ti,Tv}(nodes, triangles)
end

"""
Adjacency list with data structure like SparseMatrixCSC's colptr and rowval.
"""
struct FastGraph{Ti}
    ptr::Vector{Ti}
    adj::Vector{Ti}
end

"""
Given an edge between nodes (n1, n2), return
the natural index of the edge.

Costs are O(log b) where b is the connectivity
"""
function edge_index(graph::FastGraph{Ti}, n1::Ti, n2::Ti) where {Ti}
    n1, n2 = sort((n1, n2))
    return searchsortedfirst(graph.adj, n2, graph.ptr[n1], graph.ptr[n1 + 1] - 1, Base.Order.Forward)
end

"""
Sort the nodes in the adjacency list
"""
function sort_edges!(g::FastGraph)
    @inbounds for i = 1 : length(g.ptr) - 1
        sort!(g.adj, g.ptr[i], g.ptr[i + 1] - 1, QuickSort, Base.Order.Forward)
    end

    return g
end

"""
Returns a sorted list of nodes on the boundary of the domain
Complexity is O(E + B log B) where E is the number of edges and B the number
of nodes on the boundary.
"""
function collect_boundary_nodes!(g::FastGraph{Ti}, boundary_nodes::Vector{Ti}) where {Ti}
    idx = 1

    # Loop over all the nodes
    @inbounds for node = 1 : length(g.ptr) - 1
        last = g.ptr[node + 1]

        # Detect whether this nodes belongs to at least one boundary edge
        node_belongs_to_boundary_edge = false

        # Loop over pairs of outgoing edges from this node:
        # (node -> adj[idx], node -> adj[idx + 1]) form the pair.
        # If adj[idx] != adj[idx + 1], then the edge node -> adj[idx] 
        # occurs only once and therefore belongs on the boundary
        # If adj[idx] == adj[idx + 1], then, the edge node -> adj[idx]
        # occurs twice and therefore must be inbetween two triangles and hence
        # in the interior.
        while idx + 1 < last
            if g.adj[idx] == g.adj[idx + 1]
                idx += 2
            else
                node_belongs_to_boundary_edge = true
                push!(boundary_nodes, g.adj[idx])
                idx += 1
            end
        end

        # If there is still one edge left, it occurs alone, and is therefore
        # part of the boundary.
        if idx < last
            node_belongs_to_boundary_edge = true
            push!(boundary_nodes, g.adj[idx])
        end

        # Finally push the current node as well if it is part of a boundary edge
        if node_belongs_to_boundary_edge
            push!(boundary_nodes, node)
        end
    end

    return remove_duplicates!(sort!(boundary_nodes))
end

"""
Returns a sorted list of nodes in the interior of the domain
Complexity is O(N) where N is the number of nodes
"""
function to_interior(boundary_nodes::Vector{Ti}, n::Integer) where {Ti}
    interior_nodes = Vector{Ti}(n - length(boundary_nodes))
    num = 1
    idx = 1

    @inbounds for i in boundary_nodes
        while num < i
            interior_nodes[idx] = num
            num += 1
            idx += 1
        end
        num += 1
    end

    @inbounds for i = num : n
        interior_nodes[idx] = i
        idx += 1
    end

    return interior_nodes
end


"""
Remove duplicate edges from an adjacency list with sorted edges
"""
function remove_duplicates!(g::FastGraph)
    Nn = length(g.ptr) - 1
    slow = 0
    fast = 1

    @inbounds for next = 2 : Nn + 1
        last = g.ptr[next]
        
        # If there is an edge going out from `node` copy the first one to the 
        # `slow` position and copy the remaining unique edges after it
        if fast < last

            # Copy the first 'slow' item
            slow += 1
            g.adj[slow] = g.adj[fast]
            fast += 1

            # From then on only copy distinct values
            while fast < last
                if g.adj[fast] != g.adj[slow]
                    slow += 1
                    g.adj[slow] = g.adj[fast]
                end
                fast += 1
            end
        end

        g.ptr[next] = slow + 1
    end

    # Finally we resize the adjacency list
    resize!(g.adj, slow)

    return g
end

"""
Remove duplicate entries from a vector.
Resizes / shrinks the vector as well.
"""
function remove_duplicates!(vec::Vector)
    length(vec) ≤ 1 && return vec

    j = 1
    @inbounds for i = 2 : length(vec)
        if vec[i] != vec[j]
            j += 1
            vec[j] = vec[i]
        end
    end

    resize!(vec, j)

    vec
end

"""
Construct the adjacency list much like the
colptr and rowval arrays in the SparseMatrixCSC
type
"""
function to_graph(mesh::Mesh{Tri})
    Nn = length(mesh.nodes)
    ptr = zeros(Int, Nn + 1)

    # Count edges per node
    @inbounds for triangle in mesh.triangles
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

    @inbounds for triangle in mesh.triangles
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            from, to = sort(triangle[a], triangle[b])
            adj[indices[from]] = to
            indices[from] += 1
        end
    end

    FastGraph(ptr, adj)
end

"""
Build the graph and at the same time find the boundary & interior nodes
"""
function construct_graph_and_find_interior_nodes(mesh::Mesh)
    graph = to_graph(mesh)
    sort_edges!(graph)
    boundary = collect_boundary_nodes!(graph, Int[])
    interior = to_interior(boundary, length(mesh.nodes))
    remove_duplicates!(graph)

    graph, boundary, interior
end

"""
Divide the unit square into a mesh of triangles
"""
function uniform_square(refinements::Int = 4)
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
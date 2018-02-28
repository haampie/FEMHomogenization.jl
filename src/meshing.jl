"""
Mesh is a collection of nodes and elements connecting the nodes
Right now the only supported element type is triangles.
"""
struct Mesh{Te<:MeshElement,Tv,Ti,d,c}
    nodes::Vector{SVector{d,Tv}}
    elements::Vector{SVector{c,Ti}}
end

"""
Adjacency list with data structure like SparseMatrixCSC's colptr and rowval.
"""
struct Graph{Ti}
    ptr::Vector{Ti}
    adj::Vector{Ti}
end

"""
Given an edge between nodes (n1, n2), return the natural index of the edge.
Costs are O(log b) where b is the connectivity
"""
function edge_index(graph::Graph{Ti}, n1::Ti, n2::Ti) where {Ti <: Integer}
    n1, n2 = sort((n1, n2))
    return binary_search(graph.adj, n2, graph.ptr[n1], graph.ptr[n1 + 1] - one(Ti))
end

"""
Sort the nodes in the adjacency list
"""
function sort_edges!(g::Graph)
    @inbounds for i = 1 : length(g.ptr) - 1
        sort!(g.adj, Int(g.ptr[i]), g.ptr[i + 1] - 1, QuickSort, Base.Order.Forward)
    end

    return g
end

"""
Returns a sorted list of nodes on the boundary of the domain. 
Complexity is O(E + B log B) where E is the number of edges and B the number
of nodes on the boundary.
"""
function find_boundary_nodes(g::Graph{Ti}) where {Ti}
    idx = 1
    boundary_nodes = Vector{Ti}()

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
Remove duplicate edges from an adjacency list with sorted edges
"""
function remove_duplicates!(g::Graph)
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

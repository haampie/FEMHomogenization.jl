"""
Adjacency list
"""
struct Graph{Ti}
    edges::Vector{Vector{Ti}}
    total::Vector{Ti}
end

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
Given an edge between nodes (n1, n2), return
the natural index of the edge.

Costs are O(log b) where b is the connectivity
"""
function edge_index(graph::Graph{Ti}, n1::Ti, n2::Ti) where {Ti}
    n1, n2 = sort((n1, n2))
    offset = searchsortedfirst(graph.edges[n1], n2)
    graph.total[n1] + offset - 1
end


"""
Add a new edge to a graph (this is slow / allocating)
"""
function add_edge!(g::Graph{Ti}, from::Ti, to::Ti) where {Ti}
    from, to = sort((from, to))
    push!(g.edges[from], to)
end

"""
Sort the nodes in the adjacency list
"""
function sort_edges!(g::Graph)
    for edges in g.edges
        sort!(edges)
    end
end

"""
Find all edges that appear only once in the adjacency list,
because that edge belongs to the boundary
"""
function collect_boundary_nodes!(g::Graph{Ti}, boundary_points::Vector{Ti}) where {Ti}
    for (idx, edges) in enumerate(g.edges)
        if collect_boundary_nodes!(edges, boundary_points)
            push!(boundary_points, idx)
        end
    end
end

"""
Find all edges that appear only once in the adjacency list,
because that edge belongs to the boundary
"""
function collect_boundary_nodes!(vec::Vector{Ti}, boundary_points::Vector{Ti}) where {Ti}
    Ne = length(vec)

    if Ne == 0
        return false
    end

    if Ne == 1
        push!(boundary_points, vec[1])
        return true
    end

    return_value = false

    j = 1
    @inbounds while j + 1 ≤ Ne
        if vec[j] == vec[j + 1]
            j += 2
        else
            return_value = true
            push!(boundary_points, vec[j])
            j += 1
        end
    end

    if j == Ne
        push!(boundary_points, vec[j])
        return_value = true
    end

    return return_value
end

"""
Remove all duplicate edges
"""
function remove_duplicates!(g::Graph)
    for adj in g.edges
        remove_duplicates!(adj)
    end

    g
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
Convert a mesh of nodes + triangles to a graph
"""
function to_graph(m::Mesh{Te,Ti,Tv}) where {Te,Tv,Ti}
    Nn = length(m.nodes)
    edges = [sizehint!(Ti[], 5) for i = 1 : Nn]
    total = ones(Ti, Nn + 1)
    g = Graph(edges, total)
    
    for triangle in m.triangles
        add_edge!(g, triangle[1], triangle[2])
        add_edge!(g, triangle[2], triangle[3])
        add_edge!(g, triangle[3], triangle[1])
    end

    # Collect the boundary nodes
    boundary_points = Vector{Ti}();
    sort_edges!(g)
    collect_boundary_nodes!(g, boundary_points)
    remove_duplicates!(g)
    sort!(boundary_points)
    remove_duplicates!(boundary_points)

    # TODO: refactor this
    interior_points = Vector{Ti}(Nn - length(boundary_points))
    num, idx = 1, 1
    @inbounds for i in boundary_points
        while num < i
            interior_points[idx] = num
            num += 1
            idx += 1
        end
        num += 1
    end

    @inbounds for i = num : Nn
        interior_points[idx] = i
        idx += 1
    end

    @inbounds for i = 1 : Nn
        g.total[i + 1] = g.total[i] + length(g.edges[i])
    end

    return g, boundary_points, interior_points
end
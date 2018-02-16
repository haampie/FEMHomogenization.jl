module Refinement

using StaticArrays
using WriteVTK

import Base.sort, Base.isless

"""
Adjacency list
"""
struct MyGraph{Ti}
    edges::Vector{Vector{Ti}}
    total::Vector{Ti}
end

sort(t::NTuple{2,T}) where {T} = t[1] < t[2] ? (t[1], t[2]) : (t[2], t[1])

@inline isless(a::SVector, b::SVector) = a.data < b.data

function init_tri(nodes::Vector{SVector{2,Tv}}, elements::Vector{SVector{3,Ti}}) where {Tv,Ti}
    Nt = length(elements)

    edges = Vector{SVector{2,Ti}}(3Nt)

    # Put all edges in a list
    idx = 1
    for (from, to) in ((1, 2), (2, 3), (1, 3)), element in elements
        edges[idx] = sort((element[from], element[to]))
        idx += 1
    end

    # Sort the edges and stuff
    ordering = sort!([Pair(idx, edge) for (idx, edge) in enumerate(edges)], by = x -> x.second)
    inv_perm = invperm([pair.first for pair in ordering])
    sorted_edges = [pair.second for pair in ordering]

    # Remove the duplicates

    return inv_perm, sorted_edges, edges
end

"""
Given an edge between nodes (n1, n2), return
the natural index of the edge.

Costs are O(log b) where b is the connectivity
"""
function edge_index(graph::MyGraph{Ti}, n1::Ti, n2::Ti) where {Ti}
    n1, n2 = sort((n1, n2))
    offset = searchsortedfirst(graph.edges[n1], n2)
    graph.total[n1] + offset - 1
end

"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.

TODO: pre-allocate a bunch of stuff!
"""
function refine(nodes::Vector{SVector{2,Tv}}, triangles::Vector{SVector{3,Ti}}, graph::MyGraph{Ti}) where {Tv,Ti}
    Nn = length(nodes)
    Nt = length(triangles)
    Ne = graph.total[end] - 1

    # Each edge is split 2, so Nn + Ne is the number of nodes
    fine_nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    fine_triangles = Vector{SVector{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(fine_nodes, nodes)
    
    # Add the new ones
    idx = Nn + 1
    for (from, edges) in enumerate(graph.edges), to in edges
        fine_nodes[idx] = (nodes[from] + nodes[to]) / 2
        idx += 1
    end

    # Split each triangle in four smaller ones
    for (i, t) in enumerate(triangles)

        # Index of the nodes on the new edges
        a = edge_index(graph, t[1], t[2]) + Nn
        b = edge_index(graph, t[2], t[3]) + Nn
        c = edge_index(graph, t[3], t[1]) + Nn

        # Split the triangle in 4 pieces
        idx = 4i - 3
        fine_triangles[idx    ] = SVector(t[1], a, c)
        fine_triangles[idx + 1] = SVector(t[2], a, b)
        fine_triangles[idx + 2] = SVector(t[3], b, c)
        fine_triangles[idx + 3] = SVector(a   , b, c)
    end

    fine_nodes, fine_triangles
end

"""
Remove all duplicate edges
"""
function remove_duplicates!(g::MyGraph)
    for (idx, adj) in enumerate(g.edges)
        remove_duplicates!(sort!(adj))
    end

    g
end

"""
Add a new edge to a graph (this is slow / allocating)
"""
function add_edge!(g::MyGraph{Ti}, from::Ti, to::Ti) where {Ti}
    from, to = sort((from, to))
    push!(g.edges[from], to)
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
function to_graph(nodes::Vector{SVector{2,Tv}}, triangles::Vector{SVector{3,Ti}}) where {Tv,Ti}
    n = length(nodes)
    edges = [sizehint!(Ti[], 5)  for i = 1 : n]
    total = ones(Ti, n + 1)
    g = MyGraph(edges, total)
    
    for triangle in triangles
        add_edge!(g, triangle[1], triangle[2])
        add_edge!(g, triangle[2], triangle[3])
        add_edge!(g, triangle[3], triangle[1])
    end

    remove_duplicates!(g)

    for i = 1 : n
        g.total[i + 1] = g.total[i] + length(g.edges[i])
    end

    return g
end

"""
Refine a grid a few times uniformly
"""
function example(refinements = 9, ::Type{Ti} = Int32, ::Type{Tv} = Float64) where {Ti,Tv}
    # nodes = SVector{2,Tv}[(0.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    # triangles = SVector{3,Ti}[(1, 2, 3), (1, 4, 3)]

    nodes = SVector{2,Tv}[(1,0), (3,2), (1,2), (2,4), (0,4)]
    triangles = SVector{3,Ti}[(1,2,3), (2,3,4), (3,4,5)]
    
    graph = to_graph(nodes, triangles)

    for i = 1 : refinements
        @time begin
            nodes, triangles = refine(nodes, triangles, graph)
            graph = to_graph(nodes, triangles)
        end
    end

    nodes, triangles, graph
end

function save_file(name::String, nodes, triangles, f)
    node_matrix = [x[i] for i = 1:2, x in nodes]
    triangle_stuff = [MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in triangles]
    vtkfile = vtk_grid(name, node_matrix, triangle_stuff)
    vtk_point_data(vtkfile, f.(nodes), "f")
    vtk_save(vtkfile)
end

end
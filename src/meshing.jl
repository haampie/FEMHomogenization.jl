const Triangle = SVector{3,Float64}

"""
Creates a standard uniform mesh of the domain [0,1]
with triangular elements
"""
function uniform_mesh(n::Int = 16)
    xs = linspace(0, 1, n + 1)
    
    total_nodes = (n + 1)^2
    total_triangles = 2n^2
    total_boundary = 4n
    total_interior = total_nodes - total_boundary

    nodes = Vector{Coord{2}}(total_nodes)
    triangles = Vector{Triangle}(total_triangles)
    boundary_nodes = Vector{Int}(total_boundary)
    interior_nodes = Vector{Int}(total_interior)

    # Nodes
    idx_ext, idx_int = 1, 1
    for i = 1 : n + 1, j = 1 : n + 1
        idx = (i - 1) * (n + 1) + j
        nodes[idx] = Coord{2}(xs[j], xs[i])

        # On the edge?
        if i == 1 || i == n + 1 || j == 1 || j == n + 1
            boundary_nodes[idx_ext] = idx
            idx_ext += 1
        else
            interior_nodes[idx_int] = idx
            idx_int += 1
        end
    end

    # Triangles
    triangle = 1
    for i = 1 : n, j = 1 : n
        idx = (i - 1) * (n + 1) + j
        
        # (Top left, top right, bottom left)
        triangles[triangle] = Triangle(idx, idx + 1, idx + n + 1)
        triangle += 1

        # (Top right, bottom left, bottom right)
        triangles[triangle] = Triangle(idx + 1, idx + n + 1, idx + n + 2)
        triangle += 1
    end

    return Mesh{Tri,2,3}(total_nodes, nodes, triangles, boundary_nodes, interior_nodes)
end

struct GraphBuilder
    edges::Vector{Vector{Int}}
end

function remove_duplicates!(vec::Vector)
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

function add_edge!(c::GraphBuilder, i::Int, j::Int)
    if i == j
        push!(c.edges[i], i)
    else
        push!(c.edges[i], j)
        push!(c.edges[j], i)
    end
end

"""
Creates the connectivity graph from the triangles
"""
function mesh_to_graph(m::Mesh)
    c = GraphBuilder([Int[] for i = 1 : m.n])

    for node in c.edges
        sizehint!(node, 7)
    end

    # Self loops for each node
    for i = 1 : m.n
        add_edge!(c, i, i)
    end

    # Edges of all triangles
    for t in m.elements
        add_edge!(c, t[1], t[2])
        add_edge!(c, t[2], t[3])
        add_edge!(c, t[1], t[3])
    end

    # Remove duplicates
    n_edges = 0
    for i = 1 : m.n
        remove_duplicates!(sort!(c.edges[i]))
        n_edges += length(c.edges[i])
    end

    return Graph(m.n, n_edges, c.edges)
end
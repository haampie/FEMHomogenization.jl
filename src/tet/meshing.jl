"""
Constructor for tetrahedrons
"""
function Mesh(Te::Type{Tet}, nodes::Vector{SVector{3,Tv}}, tets::Vector{SVector{4,Ti}}) where {Tv,Ti}
    Mesh{Tet,Tv,Ti,3,4}(nodes, tets)
end

"""
Construct an edge graph for the tetrahedron mesh
"""
function to_graph(mesh::Mesh{Tet,Tv,UInt32}) where {Tv}
    Nn = length(mesh.nodes)
    ptr = zeros(UInt32, Nn + 1)

    # Count edges per node
    @inbounds for tet in mesh.elements, i = 1 : 4, j = i + 1 : 4
        idx = tet[i] < tet[j] ? tet[i] : tet[j]
        ptr[idx + one(UInt32)] += one(UInt32)
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{UInt32}(ptr[end] - 1)
    indices = copy(ptr)

    @inbounds for tet in mesh.elements, i = 1 : 4, j = i + 1 : 4
        if tet[i] < tet[j]
            from = tet[i]
            to = tet[j]
        else
            from = tet[j]
            to = tet[i]
        end

        adj[indices[from]] = to
        indices[from] += 1
    end

    remove_duplicates!(sort_edges!(Graph(ptr, adj)))
end

"""
Construct a mesh on the unit cube
"""
function unit_cube(refinements::Int = 3, ::Type{Tv} = Float64) where {Tv}
    nodes = SVector{3,Tv}[(0,0,0), (1,0,0), (0,1,0), (1,1,0), 
                          (0,0,1), (1,0,1), (0,1,1), (1,1,1)]

    tets = SVector{4,UInt32}[(1,2,3,5), (2,3,4,8), (2,5,6,8), 
                             (2,3,5,8), (3,5,7,8)]

    mesh = Mesh(Tet, nodes, tets)

    for i = 1 : refinements
        mesh = refine(mesh, to_graph(mesh))
    end

    return mesh, find_interior_nodes(mesh)
end

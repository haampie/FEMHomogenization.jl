"""
Two tricks: counting sort + packing of two UInt32's into one UInt64
"""
function list_faces(mesh::Mesh{Tet,Tv,UInt32}) where {Tv}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    ptr = zeros(UInt32, Nn + 1)
    const ONE = one(UInt32)
    
    # Count things.
    @inbounds for tet in mesh.elements, (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
        ptr[min(tet[a], tet[b], tet[c]) + ONE] += ONE
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{UInt64}(ptr[end] - 1)
    indices = copy(ptr)

    # To construct the graph we have to sort things.
    sorted_tet = Vector{UInt32}(4)

    @inbounds for tet in mesh.elements
        copy!(sorted_tet, tet)
        sort!(sorted_tet, 1, 4, InsertionSort, Base.Order.Forward)
        for (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
            from = sorted_tet[a]
            adj[indices[from]] = pack(sorted_tet[b], sorted_tet[c])
            indices[from] += ONE
        end
    end

    return sort_faces!(ptr, adj)
end

function find_boundary_nodes(ptr::Vector{UInt32}, adj::Vector{UInt64})
    idx = one(UInt32)
    boundary_nodes = Vector{UInt32}()

    const ONE = one(UInt32)
    const TWO = ONE + ONE

    # Loop over all the nodes
    @inbounds for node = 1 : length(ptr) - 1
        last = ptr[node + 1]

        # Detect whether this nodes belongs to at least one boundary edge
        node_belongs_to_boundary_edge = false

        while idx + ONE < last
            if adj[idx] == adj[idx + ONE]
                idx += TWO
            else
                node_belongs_to_boundary_edge = true
                push!(boundary_nodes, unpack(adj[idx])...)
                idx += ONE
            end
        end

        # If there is still one edge left, it occurs alone, and is therefore
        # part of the boundary.
        if idx < last
            node_belongs_to_boundary_edge = true
            push!(boundary_nodes, unpack(adj[idx])...)
            idx += ONE
        end

        # Finally push the current node as well if it is part of a boundary edge
        if node_belongs_to_boundary_edge
            push!(boundary_nodes, node)
        end
    end

    return remove_duplicates!(sort!(boundary_nodes))
end

function sort_faces!(ptr, adj)
    @inbounds for i = 1 : length(ptr) - 1
        sort!(adj, Int(ptr[i]), ptr[i + 1] - 1, QuickSort, Base.Order.Forward)
    end

    return ptr, adj
end

"""
Find interior nodes of a tetrahedron mesh
"""
function find_interior_nodes(mesh::Mesh{Tet,Tv,UInt32}) where {Tv}
    boundary_nodes = find_boundary_nodes(list_faces(mesh)...)
    return complement(boundary_nodes, length(mesh.nodes))
end

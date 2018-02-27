"""
Pack two UInt32's into a UInt64
"""
function pack(a::UInt32, b::UInt32)
    return UInt64(a) << 32 + UInt64(b)
end

"""
Unpack a UInt64 into two UInt32's
"""
function unpack(a::UInt64)
    return UInt32(a >> 32), UInt32(a & 0x00000000ffffffff)
end

function find_unique_faces(mesh::Mesh{Tet,Tv,Ti}) where {Tv,Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    ptr = zeros(Ti, Nn + 1)
    
    # Count things.
    @inbounds for tet in mesh.elements, (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
        ptr[min(tet[a], tet[b], tet[c]) + 1] += 1
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
    sorted_tet = Vector{Ti}(4)
    @inbounds for tet in mesh.elements
        copy!(sorted_tet, tet)
        sort!(sorted_tet, 1, 4, InsertionSort, Base.Order.Forward)
        for (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
            from = sorted_tet[a]
            adj[indices[from]] = pack(UInt32(sorted_tet[b]), UInt32(sorted_tet[c]))
            indices[from] += 1
        end
    end

    return ptr, adj
end

function collecting_of_boundary_stuff!(ptr, adj, boundary_nodes::Vector{Ti}) where {Ti}
    idx = 1

    # Loop over all the nodes
    @inbounds for node = 1 : length(ptr) - 1
        last = ptr[node + 1]

        # Detect whether this nodes belongs to at least one boundary edge
        node_belongs_to_boundary_edge = false

        while idx + 1 < last
            if adj[idx] == adj[idx + 1]
                idx += 2
            else
                node_belongs_to_boundary_edge = true
                push!(boundary_nodes, unpack(adj[idx])...)
                idx += 1
            end
        end

        # If there is still one edge left, it occurs alone, and is therefore
        # part of the boundary.
        if idx < last
            node_belongs_to_boundary_edge = true
            push!(boundary_nodes, unpack(adj[idx])...)
            idx += 1
        end

        # Finally push the current node as well if it is part of a boundary edge
        if node_belongs_to_boundary_edge
            push!(boundary_nodes, node)
        end
    end

    return remove_duplicates!(sort!(boundary_nodes))
end

function hash_bench_test(refinements::Int = 6)
    # Set up a problem.
    mesh = tetra_division(refinements)
    ptr, adj = find_unique_faces(mesh)
    sort_faces_and_stuff!(ptr, adj)
    boundary_nodes = collecting_of_boundary_stuff!(ptr, adj, Int32[])
    interior_nodes = to_interior(boundary_nodes, length(mesh.nodes))

    return mesh, ptr, adj, interior_nodes
end
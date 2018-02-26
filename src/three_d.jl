function to_graph(mesh::Mesh{Tet,Tv,Ti}) where {Tv, Ti}
    Nn = length(mesh.nodes)
    ptr = zeros(Ti, Nn + 1)

    # Count edges per node
    @inbounds for tet in mesh.elements, i = 1 : 4, j = i + 1 : 4
        idx = tet[i] < tet[j] ? tet[i] : tet[j]
        ptr[idx + 1] += 1
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{Ti}(ptr[end] - 1)
    indices = copy(ptr)

    @inbounds for tet in mesh.elements, i = 1 : 4, j = i + 1 : 4
        from, to = sort(tet[i], tet[j])
        adj[indices[from]] = to
        indices[from] += 1
    end

    FastGraph(ptr, adj)
end

function refine(mesh::Mesh{Tet,Tv,Ti}) where {Tv, Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)

    # Collect all edges
    graph = to_graph(mesh)
    remove_duplicates!(sort_edges!(graph))

    Ne = length(graph.adj)

    ### Refine the grid.
    new_nodes = Vector{SVector{3,Tv}}(Nn + Ne)
    copy!(new_nodes, mesh.nodes)

    ## Split the edges
    @inbounds begin
        idx = Nn + 1
        for from = 1 : Nn, to = graph.ptr[from] : graph.ptr[from + 1] - 1
            new_nodes[idx] = (mesh.nodes[from] + mesh.nodes[graph.adj[to]]) / 2
            idx += 1
        end
    end

    ## Next, build new tetrahedrons...
    new_tets = Vector{SVector{4,Ti}}(8Nt)
    edge_nodes = MVector{10,Ti}()

    tet_idx = 1
    @inbounds for tet in mesh.elements

        # Collect the nodes
        edge_nodes[1] = tet[1]
        edge_nodes[2] = tet[2]
        edge_nodes[3] = tet[3]
        edge_nodes[4] = tet[4]

        # Find the mid-points (6 of them)
        idx = 5
        for i = 1 : 4, j = i + 1 : 4
            edge_nodes[idx] = edge_index(graph, tet[i], tet[j]) + Nn
            idx += 1
        end

        # Generate new tets!
        for (a,b,c,d) in ((1,5,6,7), (2,5,8,9), (3,6,8,10), (4,7,9,10), (5,6,8,9), (6,8,9,10), (6,7,9,10), (5,6,7,9))
            new_tets[tet_idx] = (edge_nodes[a], edge_nodes[b], edge_nodes[c], edge_nodes[d])
            tet_idx += 1
        end
    end

    return Mesh(Tet, new_nodes, new_tets)
end

"""
Detect the interior stuff
"""
function do_things_with_faces(mesh::Mesh{Tet,Tv,Ti}) where {Tv,Ti}
    Nn = length(mesh.nodes)
    Nt = length(mesh.elements)
    ptr = zeros(Ti, Nn + 1)
    
    for tet in mesh.elements, (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
        # assume that @assert tet[a] < tet[b] < tet[c] holds
        ptr[tet[a] + 1] += 1
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{SVector{2,Ti}}(ptr[end] - 1)
    indices = copy(ptr)

    @inbounds for tet in mesh.elements
        for (a,b,c) in ((1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4))
            from = tet[a]
            adj[indices[from]] = SVector{2,Ti}(tet[b], tet[c])
            indices[from] += 1
        end
    end

    return ptr, adj
end

function sort_faces_and_stuff!(ptr, adj)
    @inbounds for i = 1 : length(ptr) - 1
        sort!(adj, ptr[i], ptr[i + 1] - 1, QuickSort, Base.Order.Forward)
    end

    return ptr, adj
end

function collect_boundary_nodes!(ptr, adj, boundary_nodes::Vector{Ti}) where {Ti}
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
                push!(boundary_nodes, adj[idx][1], adj[idx][2])
                idx += 1
            end
        end

        # If there is still one edge left, it occurs alone, and is therefore
        # part of the boundary.
        if idx < last
            node_belongs_to_boundary_edge = true
            push!(boundary_nodes, adj[idx][1], adj[idx][2])
        end

        # Finally push the current node as well if it is part of a boundary edge
        if node_belongs_to_boundary_edge
            push!(boundary_nodes, node)
        end
    end

    return remove_duplicates!(sort!(boundary_nodes))
end

function tetra_division(refinements::Int = 3, ::Type{Tv} = Float64, ::Type{Ti} = Int) where {Tv,Ti}
    # nodes = SVector{3,Tv}[(0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (1.0,1.0,0.0), (0.0,0.0,1.0), (1.0,0.0,1.0), (0.0,1.0,1.0), (1.0,1.0,1.0)]
    # tets = SVector{4,Ti}[(1,2,3,5), (2,3,4,8), (2,5,6,8), (2,3,5,8), (3,5,7,8)]
    nodes = SVector{3,Tv}[(0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
    tets = SVector{4,Ti}[(1,2,3,4)]
    mesh = Mesh(Tet, nodes, tets)

    for i = 1 : refinements
        mesh = refine(mesh)
    end

    return mesh
end

function put_it_together(refinements::Int)
    mesh = tetra_division(refinements)
    ptr, adj = do_things_with_faces(mesh)
    sort_faces_and_stuff!(ptr, adj)
    result = collect_boundary_nodes!(ptr, adj, Int[])

    return mesh, result
end

function cube_stuff()
    # (x, y, z)
    nodes = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    
    # This stuff fills a cube
    tetras = ((1,2,3,5), (2,3,4,8), (2,5,6,8), (2,3,5,8), (3,5,7,8))

    node_matrix = [x[i] for i = 1:3, x in nodes]
    tetra_list = [MeshCell(VTKCellTypes.VTK_TETRA, [t...]) for t in tetras]
    vtkfile = vtk_grid("tetra_test", node_matrix, tetra_list)
    # vtk_point_data(vtkfile, data, "f")
    vtk_save(vtkfile)
end
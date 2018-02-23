function to_graph(nodes::Vector{SVector{3,Tv}}, tets::Vector{SVector{4,Ti}}) where {Tv, Ti}
    Nn = length(nodes)
    ptr = zeros(Ti, Nn + 1)

    # Count edges per node
    @inbounds for tet in tets, i = 1 : 4, j = i + 1 : 4
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

    @inbounds for tet in tets, i = 1 : 4, j = i + 1 : 4
        from, to = sort(tet[i], tet[j])
        adj[indices[from]] = to
        indices[from] += 1
    end

    FastGraph(ptr, adj)
end

function refine(nodes::Vector{SVector{3,Tv}}, tets::Vector{SVector{4,Ti}}) where {Tv, Ti}
    Nn = length(nodes)
    Nt = length(tets)

    # Collect all edges
    graph = to_graph(nodes, tets)
    remove_duplicates!(sort_edges!(graph))

    Ne = length(graph.adj)

    ### Refine the grid.
    new_nodes = Vector{SVector{3,Tv}}(Nn + Ne)
    copy!(new_nodes, nodes)

    ## Split the edges
    @inbounds begin
        idx = Nn + 1
        for from = 1 : Nn, to = graph.ptr[from] : graph.ptr[from + 1] - 1
            new_nodes[idx] = (nodes[from] + nodes[graph.adj[to]]) / 2
            idx += 1
        end
    end

    ## Next, build new tetrahedrons...
    new_tets = Vector{SVector{4,Ti}}(8Nt)
    edge_nodes = MVector{10,Ti}()

    tet_idx = 1
    @inbounds for tet in tets

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
        for (a,b,c,d) in ((1,5,6,7), (2,5,8,9), (3,6,8,10), (4,7,9,10), (5,7,8,9), (6,7,8,10), (7,8,9,10), (5,6,7,8))
            new_tets[tet_idx] = (edge_nodes[a], edge_nodes[b], edge_nodes[c], edge_nodes[d])
            tet_idx += 1
        end
    end

    return new_nodes, new_tets
end

function tetra_division(refinements::Int = 3, ::Type{Tv} = Float64, ::Type{Ti} = Int) where {Tv,Ti}
    nodes = SVector{3,Tv}[
        (0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (1.0,1.0,0.0),
        (0.0,0.0,1.0), (1.0,0.0,1.0), (0.0,1.0,1.0), (1.0,1.0,1.0)
    ]

    tets = SVector{4,Ti}[(1,2,3,5), (2,3,4,8), (2,5,6,8), (2,3,5,8), (3,5,7,8)]

    for i = 1 : refinements
        nodes, tets = refine(nodes, tets)
    end

    @show length(tets)
    
    node_matrix = [x[i] for i = 1:3, x in nodes]
    tetra_list = MeshCell[MeshCell(VTKCellTypes.VTK_TETRA, [t...]) for t in tets]
    vtkfile = vtk_grid("tetra_division", node_matrix, tetra_list)
    vtk_cell_data(vtkfile, rand(length(tets)), "cells")
    vtk_save(vtkfile)

    return nodes, tets
end

function cube_stuff()
    # (x, y, z)
    nodes = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0),
              (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    
    # This stuff fills a cube
    tetras = ((1,2,3,5), (2,3,4,8), (2,5,6,8), (2,3,5,8), (3,5,7,8))

    node_matrix = [x[i] for i = 1:3, x in nodes]
    tetra_list = MeshCell[MeshCell(VTKCellTypes.VTK_TETRA, [t...]) for t in tetras]
    vtkfile = vtk_grid("tetra_test", node_matrix, tetra_list)
    # vtk_point_data(vtkfile, data, "f")
    vtk_save(vtkfile)
end
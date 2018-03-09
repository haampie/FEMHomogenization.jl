function assemble_without_quad()
    
end

function constant_coefficients(refinements = 3)
    nodes = [SVector(0.0, 0.0), SVector(1.0,0.0), SVector(0.0,1.0)]
    triangles = [SVector(1,2,3)]
    mesh = Mesh(Tri, nodes, triangles)
    graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)

    for i = 1 : refinements
        mesh = refine(mesh, graph)
        graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
    end

    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    A = assemble_matrix(mesh, bilinear_form)

    return A
end
using BenchmarkTools

function matrix_free_2d(refinements)
    ps = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    ts = SVector{3,Int64}[(1, 2, 3)]

    mesh = Mesh(Tri, ps, ts)
    graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)

    println(length(boundary) / length(mesh.nodes), "\t", length(mesh.nodes))

    for i = 1 : refinements
        mesh = refine(mesh, graph)
        graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
        println(length(boundary) / length(mesh.nodes), "\t", length(mesh.nodes))
    end
end

function my_assembly(m::Mesh{Te,Tv,Ti}, bilinear_form, quad::Type{<:QuadRule} = default_quadrature(Te)) where {Te,Tv,Ti}
    # Quadrature scheme
    ϕs, ∇ϕs = get_basis_funcs(Te)
    ws, xs = quadrature_rule(quad)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nt = length(m.elements)
    Nn = length(m.nodes)
    Nq = length(xs)
    
    # This is for now hard-coded...
    const dof = length(m.elements[1])
    
    # The local system matrix
    A_local = zeros(dof, dof)

    idx = 1

    # Loop over all elements & compute the local system matrix
    for element in m.elements
        jac, shift = affine_map(m, element)
        J = inv(jac')
        detJ = abs(det(J))

        # Reset local matrix
        fill!(A_local, zero(Tv))

        # For each quad point
        @inbounds for k = 1 : Nq
            w = ws[k]
            for j = 1:dof
                v = basis[k][j]
                for i = 1:dof
                    u = basis[k][i]
                    A_local[i,j] += w * (dot(J * u.grad, J * v.grad) + u.ϕ * v.ϕ)
                end
            end
        end

        @inbounds for j = 1:dof, i = 1:dof
            A_local[i,j] *= detJ
        end
    end

    A_local
end

function profile_stuff(ref)
    mesh, graph, interior = unit_square(ref)
    bf = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ
    @profile my_assembly(mesh, bf)
end

function bench_stuff(ref)
    mesh, graph, interior = unit_square(ref)
    bf = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ
    @benchmark my_assembly($mesh, $bf)
end

function run_stuff(ref)
    @time mesh, graph, interior = unit_square(ref)
    @time my_assembly(mesh, identity)

    return mesh
end

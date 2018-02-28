function example_3d_assembly(refinements::Int = 5)
    mesh, int = unit_cube(refinements)
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ
    load = x -> sqrt(x[1] * x[2] * x[3])

    A = assemble_matrix(mesh, bilinear_form)
    b = assemble_rhs(mesh, load)
    x = zeros(b)

    A_int = A[int, int]
    b_int = b[int]

    @inbounds x[int] .= A_int \ b_int

    return save_file("results", mesh, Dict(
        "x" => x,
        "f" => load.(mesh.nodes),
    ))

    return x
end

function three_d_multigrid(steps = 2)
    mesh, int = unit_cube(3)

    As = Vector{SparseMatrixCSC{Float64,Int}}(steps + 1)
    Ps = Vector{SparseMatrixCSC{Float64,Int}}(steps)

    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)

    As[1] = assemble_matrix(mesh, bilinear_form)

    for i = 1 : steps
        graph = to_graph(mesh)
        Ps[i] = interpolation_operator(mesh, graph)
        mesh = refine(mesh, graph)
        As[i + 1] = assemble_matrix(mesh, bilinear_form)
    end

    return As, Ps
end
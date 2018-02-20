function example4(refinements::Int = 6, shift::Float64 = 1.0)
    # Initial grid
    mesh, graph, interior = uniform_square(refinements)

    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + shift * u.ϕ * v.ϕ
    load = x -> sqrt(x[1] * x[2])

    A = assemble_matrix(mesh, bilinear_form)
    b = assemble_rhs(mesh, load)
    x = zeros(b)

    A_int = A[interior, interior]
    b_int = b[interior]

    x[interior] .= A_int \ b_int

    return mesh, x
end

function example_multigrid(refinements::Int = 10, ::Type{Ti} = Int64, ::Type{Tv} = Float64) where {Ti,Tv}
    interpolation = Vector{SparseMatrixCSC{Tv,Ti}}(refinements - 1)
    levels = Vector{Level{Tri,Tv,Ti}}(refinements)

    # Initial mesh is 2 triangles in a square
    nodes = SVector{2,Tv}[(0, 0), (1, 0), (1, 1), (0, 1)]
    triangles = SVector{3,Ti}[(1, 2, 3), (1, 4, 3)]
    mesh = Mesh(Tri, nodes, triangles)
    graph, boundary, interior = to_graph(mesh)
    levels[1] = Level(mesh, graph, boundary, interior)

    # Then we refine a couple times
    for i = 1 : refinements - 1
        interpolation[i] = interpolation_operator(graph)
        mesh = refine(mesh, graph)
        graph, boundary, interior = to_graph(mesh)
        levels[i + 1] = Level(mesh, graph, boundary, interior)
    end

    # Start with random values on the coarsest grid
    # and interpolate them to the finer grids.
    vals = rand(length(levels[1].mesh.nodes))
    save_file("grid_01", levels[1].mesh, vals)

    for i = 2 : refinements
        vals = interpolation[i - 1]' * vals
        name = @sprintf "grid_%02d" i
        save_file(name, levels[i].mesh, vals)
    end
end
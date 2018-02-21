function example_assembly(refinements::Int = 10)
    mesh, graph, interior = uniform_square(refinements)
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    A = assemble_matrix(mesh, bilinear_form)

    return A, interior, mesh
end

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

function vcycle(level::Int, A::SparseMatrixCSC, r::Vector, Ps::Vector)
    if size(A, 1) ≤ 64 || level == 1
        return A \ r
    end

    e = zeros(size(r))

    for i = 1 : 5
        e += 0.2 * (r - A * e)
    end

    e += Ps[level]' * vcycle(level - 1, dropzeros!(Ps[level] * (A * Ps[level]')), Ps[level] * (r - A * e), Ps)

    for i = 1 : 5
        e += 0.2 * (r - A * e)
    end

    return e
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
        interpolation[i] = interpolation_operator(mesh, graph)
        mesh = refine(mesh, graph)
        graph, boundary, interior = to_graph(mesh)
        levels[i + 1] = Level(mesh, graph, boundary, interior)
    end

    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + norm(x)
    load = x -> 1.0

    # Large matrix A and large rhs b
    A = assemble_matrix(last(levels).mesh, bilinear_form)
    b = assemble_rhs(last(levels).mesh, load)
    x = rand(size(b))
    ω = 0.2

    @time exact = A \ b

    # Coarse grid:
    @time for i = 1 : 4
        x += vcycle(length(interpolation), A, b - A * x, interpolation)
        @show norm(exact - x)
    end
end
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
    graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
    levels[1] = Level(mesh, graph, boundary, interior)

    # Then we refine a couple times
    for i = 1 : refinements - 1
        interpolation[i] = interpolation_operator(mesh, graph)
        mesh = refine(mesh, graph)
        graph, boundary, interior = construct_graph_and_find_interior_nodes(mesh)
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

"""
Refine a given mesh `mesh` a total of `levels` times and store each mesh in an
array. Also store the interpolation operator from each coarse grid to each finer
grid.
"""
function build_multigrid_meshes(mesh::Mesh{Te,Ti,Tv}, levels::Int) where {Te,Ti,Tv}
    Ps = Vector{SparseMatrixCSC{Tv,Ti}}(levels - 1)
    grids = Vector{Level{Tri,Tv,Ti}}(levels)

    graph, _, interior = construct_graph_and_find_interior_nodes(mesh)
    grids[1] = Level(mesh, graph, interior)

    # Then we refine a couple times
    for i = 2 : levels
        Ps[i - 1] = interpolation_operator(mesh, graph)
        mesh = refine(mesh, graph)
        graph, _, interior = construct_graph_and_find_interior_nodes(mesh)
        grids[i] = Level(mesh, graph, interior)
    end

    return grids, Ps
end

function assemble_multigrid_matrices(grids::Vector{Level{Te,Tv,Ti}}, Ps, bilinear_form) where {Te,Tv,Ti}
    k = length(grids)
    As = Vector{SparseMatrixCSC{Tv,Ti}}(k)
    As_int = Vector{SparseMatrixCSC{Tv,Ti}}(k)
    
    # Assemble the finest grid
    is = grids[k].interior
    As[k] = assemble_matrix(grids[k].mesh, bilinear_form)
    As_int[k] = As[k][is, is]

    # And then build the Galerkin projections P' * A * P
    for j = k - 1 : -1 : 1
        is = grids[j].interior
        As[j] = Ps[j]' * (As[j + 1] * Ps[j])
        As_int[j] = As[j][is, is]
    end

    return As, As_int
end

"""
This example refines a grid with 49 unknowns a few times
to a grid of 261121 unknowns with an integrand dot(∇u, ∇v) and
a load function of f(x) = 1
"""
function example_multigrid_stuff()
    # The (integrand of) the bilinear form and the load function
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    load = x -> 1.0

    # Build an initial grid
    mesh, graph, interior = uniform_square(3)

    # Refine the grid a couple times
    grids, Ps = build_multigrid_meshes(mesh, 7)

    # Build the coefficient matrices
    As, As_int = assemble_multigrid_matrices(grids, Ps, bilinear_form)

    # Build a right-hand side on the finest grid
    b = assemble_rhs(grids[end].mesh, load)
    b_int = b[grids[end].interior]

    # Allocate a solution vector
    x = zeros(length(grids[end].mesh.nodes))

    # Solve the problem via a direct method
    x[grids[end].interior] .= As_int[end] \ b_int

    return x
end
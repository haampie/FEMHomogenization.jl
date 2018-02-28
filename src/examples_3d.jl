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

"""
Refine a given mesh `mesh` a total of `levels` times and store each mesh in an
array.
"""
function build_multigrid_meshes(mesh::Mesh{Tet,Tv,Ti}, levels::Int) where {Ti,Tv}
    grids = Vector{Grid{Tet,Tv,Ti}}(levels)
    graph = to_graph(mesh)
    interior = find_interior_nodes(mesh)
    grids[1] = Grid(mesh, graph, interior)

    # Then we refine a couple times
    for i = 2 : levels
        mesh = refine(mesh, graph)
        graph = to_graph(mesh)
        interior = find_interior_nodes(mesh)
        grids[i] = Grid(mesh, graph, interior)
    end

    return grids
end

"""
Solve the problem
-Δu = 1.0 in Ω
  u = 0   on ∂Ω
using multigrid.
"""
function three_d_multigrid(;start = 3, refinements = 4, steps = 20, smooth = 5, ωs = [3.7, 3.7, 3.7])
    @assert length(ωs) == refinements - 1

    # The (integrand of) the bilinear form and the load function
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    load = x -> 1.0

    # Build an initial grid
    mesh, _ = unit_cube(start)

    # Refine the grid a couple times
    grids = build_multigrid_meshes(mesh, refinements)

    # Get the interpolation operators
    @time Ps = build_interpolation_operators(grids)

    # Build the coefficient matrices
    @time As = assemble_multigrid_matrices(grids, Ps, bilinear_form)

    @show size(As[end]) size(As[1])

    # Build a right-hand side on the finest grid
    b = assemble_rhs(grids[end].mesh, load)
    b_int = b[grids[end].interior]
    
    @time mg = initialize_multigrid(grids, Ps, As, ωs)

    # Do the multigrid solve
    x1 = zeros(b);
    x2 = zeros(b);
    @time x1[grids[end].interior] .= solve(mg, b_int, steps = steps, smooth = smooth)


    return x1, grids[end].mesh, mg
end
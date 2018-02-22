"""
Create a uniformly refined grid on the unit square
"""
function example_assembly(refinements::Int = 10)
    mesh, graph, interior = uniform_square(refinements)
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    A = assemble_matrix(mesh, bilinear_form)

    return A, interior, mesh
end

"""
Solve a simple problem with a direct method
"""
function example_solve(refinements::Int = 6)
    mesh, graph, interior = uniform_square(refinements)
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ
    load = x -> sqrt(x[1] * x[2])

    A = assemble_matrix(mesh, bilinear_form)
    b = assemble_rhs(mesh, load)
    x = zeros(b)

    A_int = A[interior, interior]
    b_int = b[interior]

    @inbounds x[interior] .= A_int \ b_int

    return mesh, x
end

"""
A geometric level of the grid
"""
struct Grid{Te,Tv,Ti}
    mesh::Mesh{Te,Ti,Tv}
    graph::FastGraph{Ti}
    interior::Vector{Ti}
end

"""
The full matrix A_all and the A_all[interior, interior] matrix.
"""
struct FEMMatrix{Tv,Ti}
    A_all::SparseMatrixCSC{Tv,Ti}
    A_int::SparseMatrixCSC{Tv,Ti}
end

"""
Putting it all together
"""
struct Multigrid{Te,Tv,Ti}
    grids::Vector{Grid{Te,Tv,Ti}}
    As::Vector{FEMMatrix{Tv,Ti}}
    Ps::Vector{SparseMatrixCSC{Tv,Ti}}
end

"""
Refine a given mesh `mesh` a total of `levels` times and store each mesh in an
array.
"""
function build_multigrid_meshes(mesh::Mesh{Te,Ti,Tv}, levels::Int) where {Te,Ti,Tv}
    grids = Vector{Grid{Tri,Tv,Ti}}(levels)
    graph, _, interior = construct_graph_and_find_interior_nodes(mesh)
    grids[1] = Grid(mesh, graph, interior)

    # Then we refine a couple times
    for i = 2 : levels
        mesh = refine(mesh, graph)
        graph, _, interior = construct_graph_and_find_interior_nodes(mesh)
        grids[i] = Grid(mesh, graph, interior)
    end

    return grids
end

"""
Returns an array of interpolation operators from coarse grids to fine grids
"""
function build_interpolation_operators(grids::Vector{Grid{Te,Tv,Ti}}) where {Te,Tv,Ti}
    levels = length(grids)
    Ps = Vector{SparseMatrixCSC{Tv,Ti}}(levels - 1)

    for i = 1 : levels - 1
        Ps[i] = interpolation_operator(grids[i].mesh, grids[i].graph)
    end

    return Ps    
end

"""
We build the FEM matrix on the finest grid and then construct the Galerkin
projections to construct the matrices on the coarser grids.
"""
function assemble_multigrid_matrices(grids::Vector{Grid{Te,Tv,Ti}}, Ps, bilinear_form) where {Te,Tv,Ti}
    k = length(grids)
    As = Vector{FEMMatrix{Tv,Ti}}(k)
    
    # Assemble the finest grid
    begin
        is = grids[k].interior
        A_all = assemble_matrix(grids[k].mesh, bilinear_form)
        A_int = A_all[is, is]
        As[k] = FEMMatrix(A_all, A_int)
    end

    # And then build the Galerkin projections P' * A * P
    for j = k - 1 : -1 : 1
        is = grids[j].interior
        A_all = Ps[j]' * (As[j + 1].A_all * Ps[j])
        A_int = A_all[is, is]
        As[j] = FEMMatrix(A_all, A_int)
    end

    return As
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
    grids = build_multigrid_meshes(mesh, 8)

    @show length(grids[end].interior)

    # Get the interpolation operators
    Ps = build_interpolation_operators(grids)

    # Build the coefficient matrices
    As = assemble_multigrid_matrices(grids, Ps, bilinear_form)

    # Build a right-hand side on the finest grid
    b = assemble_rhs(grids[end].mesh, load)

    # Just a bag of all the stuff we need.
    mg = Multigrid(grids, As, Ps)

    # Do the multigrid solve
    @time x = solve(mg, b)

    @time As[end].A_int \ b[grids[end].interior]

    return x
end

"""
Basically the W-cycle of multigrid
"""
function solve(mg::Multigrid, b, steps::Int = 20, smooth::Int = 3)
    levels = length(mg.grids)

    x = rand(size(b))
    r = b - mg.As[end].A_all * x
    @show norm(r)

    # Do the W-cycle
    for i = 1 : steps
        x += vcycle(mg, r, levels, smooth)
        r = b - mg.As[end].A_all * x
        @show norm(view(r, mg.grids[end].interior))
    end

    return x
end

"""
A single, recursive V-cycle pass of multigrid
"""
function vcycle(mg::Multigrid, b, level::Int, smooth::Int)
    b_int = b[mg.grids[level].interior]

    # Direct solve on coarsest grid
    if level == 1
        x = mg.As[1].A_int \ b_int
    else
        # Otherwise pre-smooth, recurse, and post-smooth
        x = zeros(b_int)

        # Pre-smooth
        for i = 1 : smooth
            x += 0.11 * (b_int - mg.As[level].A_int * x)
        end

        # Coarse grid correction
        r = zeros(size(mg.As[level].A_all, 1))
        r[mg.grids[level].interior] .= b_int - mg.As[level].A_int * x
        update = mg.Ps[level - 1] * vcycle(mg, mg.Ps[level - 1]' * r, level - 1, smooth)
        x .+= update[mg.grids[level].interior]

        # Post-smooth
        for i = 1 : smooth
            x += 0.11 * (b_int - mg.As[level].A_int * x)
        end
    end

    correction = zeros(b)
    correction[mg.grids[level].interior] .= x

    return correction
end
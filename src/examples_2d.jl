"""
Create a uniformly refined grid on the unit square
"""
function example_assembly(refinements::Int = 10)
    mesh, graph, interior = unit_square(refinements)
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    A = assemble_matrix(mesh, bilinear_form)

    return A, interior, mesh
end

"""
Solve a simple problem with a direct method
"""
function example_solve(refinements::Int = 6)
    mesh, graph, interior = unit_square(refinements)
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
    mesh::Mesh{Te,Tv,Ti}
    graph::Graph{Ti}
    interior::Vector{Ti}
end

"""
Some pre-allocated vectors for each level
"""
struct Level{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    P::SparseMatrixCSC{Tv,Ti}
    x::Vector{Tv}
    r::Vector{Tv}
    b::Vector{Tv}
    tmp::Vector{Tv}
end

struct Multigrid{Tf,Tv,Ti}
    levels::Vector{Level{Tv,Ti}}
    A_coarse::Tf
    b_coarse::Vector{Tv}
    ωs::Vector{Tv}
end

"""
Refine a given mesh `mesh` a total of `levels` times and store each mesh in an
array.
"""
function build_multigrid_meshes(mesh::Mesh{Te,Tv,Ti}, levels::Int) where {Te,Ti,Tv}
    grids = Vector{Grid{Te,Tv,Ti}}(levels)
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
function build_interpolation_operators(grids::Vector{Grid{Te,Tv,Ti}}; interior::Bool = false) where {Te,Tv,Ti}
    levels = length(grids)
    Ps = Vector{SparseMatrixCSC{Tv,Int}}(levels - 1)

    for i = 1 : levels - 1
        P = interpolation_operator(grids[i].mesh, grids[i].graph)
        Ps[i] = P[grids[i + 1].interior, grids[i].interior]
    end

    return Ps    
end

"""
We build the FEM matrix on the finest grid and then construct the Galerkin
projections to construct the matrices on the coarser grids.
"""
function assemble_multigrid_matrices(grids::Vector{Grid{Te,Tv,Ti}}, Ps, bilinear_form) where {Te,Tv,Ti}
    k = length(grids)
    As = Vector{SparseMatrixCSC{Tv,Int}}(k)
    
    # Assemble the finest grid
    A_all = assemble_matrix(grids[k].mesh, bilinear_form)

    # Only keep the 'interior'
    As[k] = A_all[grids[k].interior, grids[k].interior]

    # And then build the Galerkin projections P' * A * P
    for j = k - 1 : -1 : 1
        As[j] = dropzeros!(Ps[j]' * (As[j + 1] * Ps[j]))
    end

    return As
end

"""
This is the stuff we need for the multigrid procedure
"""
function initialize_multigrid(grids::Vector{Grid{Te,Tv,Ti}}, Ps::Vector{SparseMatrixCSC{Tv,Tii}}, As, ωs = []) where {Te,Tv,Ti,Tii}
    k = length(Ps)
    levels = Vector{Level{Tv,Tii}}(k)

    # Finer grids
    for i = 1 : k
        # Move this to a proper constructor
        n = size(As[i + 1], 1)
        x = Vector{Tv}(n)
        r = Vector{Tv}(n)
        b = Vector{Tv}(n)
        tmp = Vector{Tv}(n)
        levels[i] = Level(As[i + 1], Ps[i], x, r, b, tmp)
    end

    # Coarsest grid
    A_coarse = factorize(As[1])
    b_coarse = Vector{Tv}(size(As[1], 1))

    return Multigrid(levels, A_coarse, b_coarse, ωs)
end

"""
Solve the problem
-Δu = 1.0 in Ω
  u = 0   on ∂Ω
using multigrid.
"""
function example_multigrid_stuff()
    # The (integrand of) the bilinear form and the load function
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)
    load = x -> 1.0

    # Build an initial grid
    mesh, graph, interior = unit_square(3)

    # Refine the grid a couple times
    grids = build_multigrid_meshes(mesh, 8)

    # Get the interpolation operators
    @time Ps = build_interpolation_operators(grids)

    # Build the coefficient matrices
    @time As = assemble_multigrid_matrices(grids, Ps, bilinear_form)

    @show size(As[end]) size(As[1])

    # Build a right-hand side on the finest grid
    b = assemble_rhs(grids[end].mesh, load)
    b_int = b[grids[end].interior]
    
    @time mg = initialize_multigrid(grids, Ps, As)

    # Do the multigrid solve
    x1 = zeros(b);
    x2 = zeros(b);
    @time x1[grids[end].interior] .= solve(mg, b_int)
    @time x2[grids[end].interior] .= As[end] \ b_int

    @show norm(x1 - x2)

    return x1, x2, mg
end

"""
Basically the W-cycle of multigrid
"""
function solve(mg::Multigrid, b::Vector; rtol = 1e-6, steps::Int = 20, smooth::Int = 3)
    finest = mg.levels[end]
    x = zeros(finest.x)

    # Initialize with zeros
    fill!(finest.x, 0.0)
    copy!(finest.b, b)
    
    # Norm of the initial residual
    r0 = norm(finest.b)

    # Do the W-cycle
    for i = 1 : steps
        # Do a V-cycle
        vcycle!(mg, length(mg.levels), smooth)

        # Update our x
        x .+= mg.levels[end].x

        # Compute the new residual
        A_mul_B!(finest.tmp, finest.A, x)
        finest.b .= b .- finest.tmp

        # New residual norm
        rel_resnorm = norm(finest.b) / r0

        @show (i, rel_resnorm)

        if rel_resnorm < rtol
            break
        end
    end

    return x
end

"""
A single, recursive V-cycle pass of multigrid
"""
function vcycle!(mg::Multigrid, level::Int, smooth::Int)
    lvl = mg.levels[level]

    # Initial x = 0 and r = b
    fill!(lvl.x, 0.0)
    copy!(lvl.r, lvl.b)

    # Pre-smooth
    for i = 1 : smooth
        # axpy, but just use Julia broadcasting
        lvl.x .+= mg.ωs[level] .* lvl.r

        # r = b - A *x
        A_mul_B!(lvl.tmp, lvl.A, lvl.x)
        lvl.r .= lvl.b .- lvl.tmp
    end

    # Either solve directly or recurse
    if level == 1
        Ac_mul_B!(mg.b_coarse, lvl.P, lvl.r)
        A_mul_B!(lvl.tmp, lvl.P, mg.A_coarse \ mg.b_coarse)
    else
        # Restrict
        Ac_mul_B!(mg.levels[level - 1].b, lvl.P, lvl.r)

        # Solve approximately on the coarse grid
        vcycle!(mg, level - 1, smooth)

        # Interpolate
        A_mul_B!(lvl.tmp, lvl.P, mg.levels[level - 1].x)
    end

    # Coarse grid correction
    lvl.x .+= lvl.tmp

    # Post-smooth
    for i = 1 : smooth
        # r = b - A *x
        A_mul_B!(lvl.tmp, lvl.A, lvl.x)
        lvl.r .= lvl.b .- lvl.tmp
        lvl.x .+= mg.ωs[level] .* lvl.r
    end

    return nothing
end
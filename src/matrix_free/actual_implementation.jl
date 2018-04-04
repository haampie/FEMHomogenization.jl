using BenchmarkTools

import Base: A_mul_B!, At_mul_B!

struct Adjoint{T}
    parent::T
end

struct LocalIntegralParts{Tv,Ti}
    A11::SparseMatrixCSC{Tv,Ti}
    A12::SparseMatrixCSC{Tv,Ti}
    A21::SparseMatrixCSC{Tv,Ti}
    A22::SparseMatrixCSC{Tv,Ti}
end

struct Edge{Ti}
    from::Ti
    to::Ti
end

struct Connectivity{Ti}
    node_to_elements::Dict{Ti,Vector{Ti}}
    edge_to_elements::Dict{Edge{Ti},Vector{Ti}}
end

struct BoundaryEdges{Ti}
    edges::Vector{Vector{Ti}}
end

"""
Fancy Edge constructor
"""
a → b = Edge(a, b)

"""
Store an edge uniquely
"""
↔(a, b) = a < b ? a → b : b → a

struct FullLinearOperator
    coarse
    fine
    ops
    edge_indices
    connectivity
end

struct Interpolation{T}
    P::T
end

A_mul_B!(y, P::Interpolation, x) = A_mul_B!(y, P.P, x)
At_mul_B!(y, P::Interpolation, x) = At_mul_B!(y, P.P, x)
A_mul_B!(α, P::Interpolation, x, β, y) = A_mul_B!(α, P.P, x, β, y)
At_mul_B!(α, P::Interpolation, x, β, y) = At_mul_B!(α, P.P, x, β, y)

"""
We do local refinement in each element
"""
function A_mul_B!(y, A::FullLinearOperator, x)
    Threads.@threads for i = 1 : size(y, 2)
    # for i = 1 : size(y, 2)
        @inbounds begin
            # Compute the coarse affine transformation.
            J, shift = affine_map(A.coarse, A.coarse.elements[i])
            invJ = inv(J)
            P = (invJ * invJ') * det(J)
            
            x_local = view(x, :, i)
            y_local = view(y, :, i)
            
            # We set y_local to 0 here implicitly
            # And then we add to it!
            A_mul_B!(P[1,1], A.ops.A11, x_local, 0.0, y_local)
            A_mul_B!(P[2,1], A.ops.A21, x_local, 1.0, y_local)
            A_mul_B!(P[1,2], A.ops.A12, x_local, 1.0, y_local)
            A_mul_B!(P[2,2], A.ops.A22, x_local, 1.0, y_local)
        end
    end
    
    # Combine the values on the coarse grid edges
    combine_edges!(y, A.connectivity.edge_to_elements, A.coarse.elements, A.edge_indices)
    
    # Combine the values on the coarse grid vertices
    combine_vertices!(y, A.connectivity.node_to_elements, A.coarse.elements)
    
    y
end

"""
Assemble the load vector locally in each coarse element
and then sum along the boundary of the coarse elements.
"""
function assemble_const_load_vector!(b, coarse_mesh, fine_mesh, connectivity, edge_indices)
    Threads.@threads for i = 1 : size(b, 2)
    # for i = 1 : size(b, 2)
        b_local = view(b, :, i)
        fill!(b_local, 0.0)
        
        # Compute the coarse affine transformation.
        J, shift = affine_map(coarse_mesh, coarse_mesh.elements[i])
        detJac = abs(det(J))

        # Loop over the fine elements
        for fine_element in fine_mesh.elements
            J_local, shift = affine_map(fine_mesh, fine_element)
            total_det = abs(det(J_local)) * detJac
            @inbounds for node in fine_element
                b_local[node] += total_det
            end
        end
    end

    # Combine the values on the coarse grid edges
    combine_edges!(b, connectivity.edge_to_elements, coarse_mesh.elements, edge_indices)
    
    # Combine the values on the coarse grid vertices
    combine_vertices!(b, connectivity.node_to_elements, coarse_mesh.elements)
end

"""
We construct matrices that compute the value (∇u[i], ∇u[j]) under the integral;
later we need this again.
"""
function fine_grid_operators(mesh::Mesh{Tri})
    A11 = assemble_matrix(mesh, (u, v, x) -> u.∇ϕ[1] * v.∇ϕ[1])
    A12 = assemble_matrix(mesh, (u, v, x) -> u.∇ϕ[1] * v.∇ϕ[2])
    A21 = assemble_matrix(mesh, (u, v, x) -> u.∇ϕ[2] * v.∇ϕ[1])
    A22 = assemble_matrix(mesh, (u, v, x) -> u.∇ϕ[2] * v.∇ϕ[2])
    
    LocalIntegralParts(A11, A12, A21, A22)
end

"""
Find the nodes on the interior of the boundary of a reference triangle
"""
function fine_grid_edge_nodes(mesh::Mesh{Tri})
    # South
    e1 = find(x -> x[2] ≈ 0.0 && 0.0 < x[1] < 1.0, mesh.nodes)
    
    # North-east
    e2 = find(x -> x[1] + x[2] ≈ 1.0 && 0.0 < x[1] < 1.0, mesh.nodes)
    
    # West
    e3 = find(x -> x[1] ≈ 0.0 && 0.0 < x[2] < 1.0, mesh.nodes)
    
    BoundaryEdges([e1, e2, e3])
end

function refined_reference_triangle(refinements = 3)
    # Refine a reference triangle 3 times
    nodes = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    elements = SVector{3,Int64}[(1, 2, 3)]
    return refine(Mesh(Tri, nodes, elements), refinements)
end

"""
For a given *directed* edge we return the edge index and the orientation of
the edge as a boolean; true = counter-clockwise, false = clockwise. Exact
orientation does not really matter, we only must make sure two edges have the
same directionality.
"""
function edge_number_and_orientation(element, edge)
    from_idx = findfirst(x -> x == edge.from, element)
    to_idx = findfirst(x -> x == edge.to, element)
    
    # ccw if edge is 1→2, 2→3 or 3→1.
    if from_idx + 1 == to_idx || from_idx == 3 && to_idx == 1
        return from_idx, true
    else
        return to_idx, false
    end
end

"""
Combine the values along the edges
"""
function combine_edges!(y, edge_to_elements, coarse_elements, edge_indices)
    @inbounds for (edge, elements) in edge_to_elements
        # Skip boundary edges
        length(elements) != 2 && continue
        
        e1 = elements[1]
        e2 = elements[2]
        
        # Assume the edge is in the direction a → a+1
        fst_local_edge, fst_dir = edge_number_and_orientation(coarse_elements[e1], edge)
        snd_local_edge, snd_dir = edge_number_and_orientation(coarse_elements[e2], edge)
        fst_edge = edge_indices.edges[fst_local_edge]
        snd_edge = edge_indices.edges[snd_local_edge]
        
        interior_nodes = length(fst_edge)
        range_one = 1 : interior_nodes
        range_two = fst_dir == snd_dir ? StepRange(range_one) : reverse(range_one)
        
        # Todo: tidy this to one loop.
        for i = range_one
            idx_one = fst_edge[i]
            idx_two = snd_edge[range_two[i]]
            sum = y[idx_one, e1] + y[idx_two, e2]
            y[idx_one, e1] = sum
            y[idx_two, e2] = sum
        end
    end
    
    y
end

"""
Combine the values along the vertices
"""
function combine_vertices!(y, node_to_elements, coarse_elements)
    @inbounds for (node, elements) in node_to_elements
        # Skip isolated nodes
        length(elements) < 2 && continue
        
        sum = 0.0
        
        # Reduce
        for idx in elements
            local_idx = findfirst(x -> x == node, coarse_elements[idx])
            sum += y[local_idx, idx]
        end
        
        # Store
        for idx in elements
            local_idx = findfirst(x -> x == node, coarse_elements[idx])
            y[local_idx, idx] = sum
        end
    end
    
    y
end

"""
push!(dict, k, v)

Equivalent to `push!(dict[k], v)` when `dict[k]` exists or `dict[k] = [v]` when
`dict[k]` does not exist.
"""
function push_or_create!(dict::Dict{Tk,Vector{Tv}}, k::Tk, v::Tv) where {Tk,Tv}
    push!(get!(() -> Tv[], dict, k), v)
end

"""
For a given coarse grid we should figure out a couple things:
- Given a vertex, which elements is it connected to?
- Given an edge of an element, what other element does it share it with?

Note that we do not pre-allocate here, so we might be slow.
"""
function inspect_coarse_grid(mesh::Mesh{Tri,Tv,Ti}) where {Tv,Ti}
    
    n2e = Dict{Ti,Vector{Ti}}()
    e2e = Dict{Edge{Ti},Vector{Ti}}()
    
    # Find the connectivity of the vertices and edges
    # by looping over all elements
    for (i, e) in enumerate(mesh.elements)
        
        # Push the nodes (1, 2, 3)
        push_or_create!(n2e, e[1], i)
        push_or_create!(n2e, e[2], i)
        push_or_create!(n2e, e[3], i)
        
        # Push the edges (1 ↔ 2, 2 ↔ 3, 3 ↔ 1)
        push_or_create!(e2e, e[1] ↔ e[2], i)
        push_or_create!(e2e, e[2] ↔ e[3], i)
        push_or_create!(e2e, e[1] ↔ e[3], i)
    end 
    
    Connectivity(n2e, e2e)
end

function benchmark_local_refinement_vs_global_operator(refs)
    # Build a coarse mesh
    nodes = SVector{2,Float64}[(0, 0), (1, 3), (3, 3), (2, 1), (4, 1)]
    elements = SVector{3,Int64}[(1, 2, 4), (2, 3, 4), (3, 4, 5)]
    coarse = refine(Mesh(Tri, nodes, elements), 5)
    
    # Build a single fine reference mesh
    fine = refined_reference_triangle(refs)
    
    # Collect nodes on the edges, but not the vertices
    edge_indices = fine_grid_edge_nodes(fine)
    
    # Build the full fine mesh as well for comparison
    full = refine(coarse, refs)
    
    # Build the fine operators {A11, A12, A21, A22}
    operators = fine_grid_operators(fine)
    
    # Find how the edges an vertices of the coarse grid are connected
    connectivity = inspect_coarse_grid(coarse)
    
    A_full = assemble_matrix(full, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ))
    
    # Initialize an x to do multiplication with (and store it in y)
    x1 = ones(length(coarse.elements) * length(fine.nodes))
    y1 = rand!(similar(x1))
    
    x2 = ones(size(A_full, 1))
    y2 = rand!(similar(x2))
    
    @show sizeof(x1) sizeof(x2)
    
    A = FullLinearOperator(coarse, fine, operators, edge_indices, connectivity)
    
    fst = @benchmark A_mul_B!($y1, $A, $x1)
    snd = @benchmark A_mul_B!($y2, $A_full, $x2)
    
    @show norm(y1) norm(y2)
    
    return fst, snd
end

function how_far_can_we_go(coarse_ref, fine_ref)
    # Build a coarse mesh
    nodes = SVector{2,Float64}[(0, 0), (1, 3), (3, 3), (2, 1), (4, 1), (5, 3)]
    elements = SVector{3,Int64}[(1, 2, 4), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
    coarse = refine(Mesh(Tri, nodes, elements), coarse_ref)
    
    # Build a single fine reference mesh
    fine = refined_reference_triangle(fine_ref)
    
    # Collect nodes on the edges, but not the vertices
    edge_indices = fine_grid_edge_nodes(fine)
    
    # Build the fine operators {A11, A12, A21, A22}
    operators = fine_grid_operators(fine)
    
    # Find how the edges an vertices of the coarse grid are connected
    connectivity = inspect_coarse_grid(coarse)
    
    # Initialize a random x to do multiplication with (and store it in y)
    x = ones(length(coarse.elements) * length(fine.nodes))
    y = rand!(similar(x))
    
    @show length(x)
    
    A = FullLinearOperator(coarse, fine, operators, edge_indices, connectivity)
    
    A_mul_B!(y, A, x)
    
    @show norm(y)
    
    nothing
end

build_levels(As) = map(As) do A
    return FineLevel(
        A,
        zeros(length(A.fine.nodes), length(A.coarse.elements)),
        zeros(length(A.fine.nodes), length(A.coarse.elements)),
        zeros(length(A.fine.nodes), length(A.coarse.elements)),
        0.8
    )
end

function local_multigrid(coarse_ref = 6, fine_ref = 6)
    # Build a coarse mesh
    nodes = SVector{2,Float64}[(0, 0), (1, 3), (3, 3), (2, 1), (4, 1)]
    elements = SVector{3,Int64}[(1, 2, 4), (2, 3, 4), (3, 4, 5)]
    coarse = refine(Mesh(Tri, nodes, elements), coarse_ref)
    connectivity = inspect_coarse_grid(coarse)

    # Build the factorized coarse grid operator
    A_coarse = factorize(assemble_matrix(coarse, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ)))

    # Build a local fine mesh that's gonna be refined a couple times.
    fine = refined_reference_triangle(0)
    edge_indices = fine_grid_edge_nodes(fine)
    operators = fine_grid_operators(fine)
    
    # Just build the first one 'by hand' so we can reuse the element type
    A_fst_fine = FullLinearOperator(coarse, fine, operators, edge_indices, connectivity)
    As = Vector{typeof(A_fst_fine)}(fine_ref)
    Ps = Vector{SparseMatrixCSC{Float64,Int}}(fine_ref - 1)
    As[1] = A_fst_fine

    for i = 1 : fine_ref - 1
        fine, P = refine_with_operator(fine)
        
        # Collect nodes on the edges, but not the vertices
        edge_indices = fine_grid_edge_nodes(fine)
        
        # Build the fine operators {A11, A12, A21, A22}
        operators = fine_grid_operators(fine)
        
        As[i + 1] = FullLinearOperator(coarse, fine, operators, edge_indices, connectivity)
        Ps[i] = P
    end
    
    # Allocate the state vecs for each level
    lvls = build_levels(As)

    fill!(lvls[end].x, 0.0)
    assemble_const_load_vector!(lvls[end].b, coarse, fine, connectivity, lvls[end].A.edge_indices)

    @show length(lvls[end].x)

    # Finally build the multigrid struct
    mg = Multigrid(lvls, Ps, A_coarse)
    
    @time my_vcycle!(mg, fine_ref, 2) 

    return mg, coarse
end

struct FineLevel{To,Tv}
    A::To # Our linear operator
    x::Tv # Approximate solution to Ax=b
    b::Tv # rhs
    r::Tv # residual r = Ax - b
    ω
end

struct Multigrid{Tv,To,Tp}
    fine::Vector{FineLevel{To,Tv}}
    Ps::Vector{Tp}
    A_coarse
end

function residual!(lvl::FineLevel)
    # r = Ax - b
    A_mul_B!(lvl.r, lvl.A, lvl.x)
    lvl.r .-= lvl.b
end

"""
Do `steps` steps of Richardson iteration on the given level
"""
function smooth!(lvl::FineLevel, steps::Int)
    for i = 1 : steps
        # r = Ax - b
        residual!(lvl)

        # x = x - ω * r
        lvl.x .-= lvl.ω .* lvl.r
    end
    
    nothing
end

"""
Run a single v-cycle of multigrid.
"""
function my_vcycle!(mg::Multigrid, idx::Int, steps::Int)
    if idx == 1
        println("TODO coarse grid stuff!")
        solve_low_dimensional_system!(mg)
    else
        # Current level
        curr = mg.fine[idx]

        # Coarser level
        next = mg.fine[idx - 1]

        println("Level ", idx)

        # Interpolation operator from next -> curr
        P = Interpolation(mg.Ps[idx - 1])
        
        # Smoothing steps with Richardson iteration
        smooth!(curr, steps)

        # Compute the residual r = Ax - b
        residual!(curr)
        
        # Restrict
        At_mul_B!(next.b, P, curr.r)

        # Solve on the next level
        my_vcycle!(mg, idx - 1, steps)

        println("Level ", idx)

        # Interpolate (x -= P * (P'AP) \ (P' b))
        A_mul_B!(-1.0, P, next.x, 1.0, curr.x)
        
        # Smoothing steps with Richardson iteration
        smooth!(curr, steps)
    end
end

function solve_low_dimensional_system!(mg::Multigrid)
    # Copy things and compute A \ b.

    # First a sanity check.

    lvl = mg.fine[1]

    xs = fill(NaN, length(lvl.A.coarse.nodes))

    for (j, elements) in enumerate(lvl.A.coarse.elements)
        for (i, node) in enumerate(elements)

            if xs[node] === NaN
                xs[node] = lvl.b[i, j]
            else
                if xs[node] != lvl.b[i, j]
                    println(xs[node], ' ', lvl.b[i,j])
                end
            end
        end
    end

end
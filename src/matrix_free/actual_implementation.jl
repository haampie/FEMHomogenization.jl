using BenchmarkTools

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
    e1::Vector{Ti}
    e2::Vector{Ti}
    e3::Vector{Ti}
end

"""
Fancy Edge constructor
"""
a → b = Edge(a, b)

"""
Store an edge uniquely
"""
↔(a, b) = a < b ? a → b : b → a

"""
We do local refinement in each element
"""
function mul!(y::AbstractVector, coarse::Mesh{Tri,Tv,Ti}, fine_mesh, ops::LocalIntegralParts{Tv,Ti}, x::AbstractVector) where {Tv,Ti}
    total = length(coarse.elements)

    Threads.@threads for i = 1 : total
        @inbounds begin
        offset = (i - 1) * length(fine_mesh.nodes) + 1
        range = offset : offset + length(fine_mesh.nodes) - 1
        x_local = view(x, range)
        y_local = view(y, range)
        
        # Compute the coarse affine transformation.
        J, shift = affine_map(coarse, coarse.elements[i])
        invJ = inv(J)
        P = (invJ * invJ') * det(J)

        # We set y_local to 0 here implicitly
        # And then we add to it!
        A_mul_B!(P[1,1], ops.A11, x_local, 0.0, y_local)
        A_mul_B!(P[2,1], ops.A21, x_local, 1.0, y_local)
        A_mul_B!(P[1,2], ops.A12, x_local, 1.0, y_local)
        A_mul_B!(P[2,2], ops.A22, x_local, 1.0, y_local)
        end
    end

    y
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

    BoundaryEdges(e1, e2, e3)
end

function refined_reference_triangle(refinements = 3)
    # Refine a reference triangle 3 times
    nodes = SVector{2,Float64}[(0, 0), (1, 0), (0, 1)]
    elements = SVector{3,Int64}[(1, 2, 3)]
    return refine(Mesh(Tri, nodes, elements), refinements)
end

function coarse_grid_with_local_refinement()
    # Some simple mesh
    nodes = SVector{2,Float64}[(0, 0), (1, 3), (2, 1), (3, 3), (4, 1)]
    elements = SVector{3,Int64}[(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    coarse_mesh = refine(Mesh(Tri, nodes, elements), 5)
    fine_mesh = refined_reference_triangle(6)
    full_mesh = refine(coarse_mesh, 6)
    fine_ops = fine_grid_operators(fine_mesh)

    A = assemble_matrix(full_mesh, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ))

    # Now for each coarse element we store the unknowns:
    # So we store nodes on the edges of coarse_mesh twice
    x = rand(length(coarse_mesh.elements) * length(fine_mesh.nodes))
    y = similar(x)
    
    x2 = rand(size(A, 1))
    y2 = similar(x2)
    
    println("Total DOF = ", size(A, 1))
    println("With repeated unknowns = ", length(x))
    println("nnz(A_full) = ", nnz(A))
    println("nnz(A_fine) = ", nnz(fine_ops.A11))
    println("# coarse grid elements = ", length(coarse_mesh.elements))
    println("# fine grid elements = ", length(fine_mesh.elements))

    println("Starting!")
    fst = @benchmark mul!($y, $coarse_mesh, $fine_mesh, $fine_ops, $x)
    snd = @benchmark A_mul_B!($y2, $A, $x2)

    fst, snd
end

"""
Combine the values along the edges
"""
function combine_edges!(y, tmp::AbstractVector, coarse::Mesh{Tri}, fine::Mesh{Tri}, edge_indices::BoundaryEdges, connectivity::Connectivity)
    @show length(edge_indices.e1)
    for (edge, elements) ∈ connectivity.edge_to_elements
        # Skip boundary edges
        length(elements) != 2 && continue

        # First

        # Second
        println("Element ", elements[1], " and ", elements[2], " through ", edge)
    end

    y
end

"""
Combine the values along the vertices
"""
function combine_vertices!(y, coarse::Mesh{Tri}, fine::Mesh{Tri}, connectivity::Connectivity)
    for (node, elements) ∈ connectivity.node_to_elements
        # Skip isolated nodes
        length(elements) < 2 && continue

        sum = 0.0

        # Reduce
        for idx in elements
            local_index = (idx - 1) * length(fine.nodes) + findfirst(x -> x == node, coarse.elements[idx])
            sum += y[local_index]
        end

        # Store
        for idx in elements
            local_index = (idx - 1) * length(fine.nodes) + findfirst(x -> x == node, coarse.elements[idx])
            y[local_index] = sum
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

function example(refs)
    # Build a coarse mesh
    nodes = SVector{2,Float64}[(0, 0), (1, 3), (2, 1), (3, 3), (4, 1)]
    elements = SVector{3,Int64}[(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    coarse = refine(Mesh(Tri, nodes, elements), 0)

    # Build a single fine reference mesh
    fine = refined_reference_triangle(refs)

    # Collect nodes on the edges, but not the vertices
    edge_indices = fine_grid_edge_nodes(fine)

    # Build the full fine mesh as well for comparison
    full = refine(coarse, 6)

    # Build the fine operators {A11, A12, A21, A22}
    operators = fine_grid_operators(fine)

    # Find how the edges an vertices of the coarse grid are connected
    connectivity = inspect_coarse_grid(coarse)

    # Initialize a random x to do multiplication with (and store it in y)
    x = rand(length(coarse.elements) * length(fine.nodes))
    y = similar(x)

    # Do the local multiplication
    mul!(y, coarse, fine, operators, x)

    tmp = zeros(2^refs - 1)

    # Combine the values on the coarse grid edges
    combine_edges!(y, tmp, coarse, fine, edge_indices, connectivity)
    
    # Combine the values on the coarse grid vertices
    combine_vertices!(y, coarse, fine, connectivity)
end
import Base.show

"""
Builds the SparseMatrixCSC structure from the graph
of the mesh. Initializes all `nzval`s with zero.
The `nzval`s are set in the assembly phase.
"""
function coefficient_matrix_factory(g::Graph)
    nzval = zeros(g.n_edges)
    rowval = Vector{Int}(g.n_edges)
    colptr = Vector{Int}(g.n_nodes + 1)
    colptr[1] = 1

    edge_num = 1
    for i = 1 : g.n_nodes
        @inbounds colptr[i + 1] = colptr[i] + length(g.edges[i])
        @inbounds rowval[colptr[i] : colptr[i + 1] - 1] .= g.edges[i]
    end

    return SparseMatrixCSC(g.n_nodes, g.n_nodes, colptr, rowval, nzval)
end

"""
Map an edge (from, to) to the index of the value in the
sparse matrix
"""
@inline function edge_to_idx(A::SparseMatrixCSC, g::Graph, from, to)
    offset = searchsortedfirst(g.edges[from], to)
    return A.colptr[from] + offset - 1
end

struct BasisFunction{d,T}
    ϕ::T
    grad::SVector{d,T}
    ∇ϕ::MVector{d,T}

    BasisFunction{d,T}(ϕ, grad, ∇ϕ) where {d,T} = new(ϕ, grad, ∇ϕ)
end

BasisFunction(ϕ::T, grad::SVector{d,T}) where {d,T} = BasisFunction{d,T}(ϕ, grad, zeros(MVector{d,T}))

function affine_map(p1, p2, p3)
    jac = [p2 - p1 p3 - p1]
    return jac, p1
end

"""
Evaluate ϕs and ∇ϕs in all quadrature points xs.
"""
function evaluate_basis_funcs(ϕs, ∇ϕs, xs)
    d = 2
    n = length(xs)
    basis = Vector{Vector{BasisFunction{d,Float64}}}(n)

    # Go over each quad point x
    for (i, x) in enumerate(xs)
        inner = Vector{BasisFunction{d,Float64}}(n)

        # Evaluate ϕ and ∇ϕ in x
        for (j, (ϕ, ∇ϕ)) in enumerate(zip(ϕs, ∇ϕs))
            inner[j] = BasisFunction(ϕ(x), SVector(∇ϕ(x)))
        end

        basis[i] = inner
    end

    return basis
end

function build_matrix(mesh::Mesh, graph::Graph, bilinear_form)
    # Quadrature scheme
    ws, xs = quadrature_rule(Tri3)
    ϕs, ∇ϕs = get_basis_funcs(Tri)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)
    
    # Nodes in each element
    ns = length(first(mesh.elements))

    A = coefficient_matrix_factory(graph)
    A_local = zeros(MMatrix{ns,ns})

    # Loop over all elements & compute local system matrix
    for element in mesh.elements
        jac, shift = affine_map(nodes(mesh, element)...)
        invJac = inv(jac')

        # Reset local matrix
        fill!(A_local, 0.0)

        # Transform the gradient
        for point in basis, f in point
            A_mul_B!(f.∇ϕ, invJac, f.grad)
        end

        # For each quad point
        @inbounds for k = 1 : length(xs)
            x = jac * xs[k] + shift

            for i = 1:ns, j = 1:ns
                A_local[i,j] += ws[k] * bilinear_form(basis[k][i], basis[k][j], x)
            end
        end

        A_local .*= abs(det(jac))

        # Put A_local into A
        @inbounds for i = 1:ns, j = 1:ns
            idx = edge_to_idx(A, graph, element[i], element[j])
            A.nzval[idx] += A_local[i,j]
        end
    end

    # Build the matrix for interior connections only
    return A[mesh.interior, mesh.interior]
end

function build_rhs(mesh::Mesh, f)
    # Quadrature scheme
    ws, xs = quadrature_rule(Tri3)
    ϕs, ∇ϕs = get_basis_funcs(Tri)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)
    
    # Nodes in each element
    ns = length(first(mesh.elements))

    b = zeros(mesh.n)
    b_local = zeros(MVector{ns})

    # Loop over all elements & compute local system matrix
    for element in mesh.elements
        jac, shift = affine_map(nodes(mesh, element)...)

        # Reset local matrix
        fill!(b_local, 0.0)

        # For each quad point
        @inbounds for k = 1 : length(xs)
            x = jac * xs[k] + shift

            for i = 1:ns
                b_local[i] += ws[k] * f(x) * basis[k][i].ϕ
            end
        end

        b_local .*= abs(det(jac))

        # Put b_local into b
        @inbounds for i = 1:ns
            b[element[i]] += b_local[i]
        end
    end

    # Build the matrix for interior connections only
    return b[mesh.interior]
end

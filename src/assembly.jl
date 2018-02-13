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

function jacobian(p1, p2, p3)
    jac = [p2 - p1 p3 - p1]
    return jac, inv(jac'), p1
end

"""
Assembles the coefficient matrix A
"""
function build_matrix(mesh::Mesh, graph::Graph, bilinear_form::Function)

    # Quadrature scheme
    weights, quad_points = quadrature_rule(Tri3)

    # Reference basis functions
    basis = element_basis(Tri, quad_points)

    # This one will hold the transformed stuff
    transformed_basis = deepcopy(basis)

    A = coefficient_matrix_factory(graph)
    A_local = zeros(3, 3)

    # Loop over all elements & compute local system matrix
    for element in mesh.elements
        jac, jacInv, shift = jacobian(nodes(mesh, element)...)

        # Reset local matrix
        fill!(A_local, 0.0)

        # Transform the gradient
        for i = 1 : length(basis)
            A_mul_B!(transformed_basis[i].∇ϕ, jacInv, basis[i].∇ϕ)
        end

        for (k, point) = enumerate(quad_points)
            x = jac * point + shift

            for i = 1:3, j = 1:3
                A_local[i,j] += weights[k] * bilinear_form(transformed_basis[i], transformed_basis[j], k, x)
            end
        end

        A_local .*= abs(det(jac))

        # Put A_local into A
        for n = 1:3, m = 1:3
            i, j = element[n], element[m]
            idx = edge_to_idx(A, graph, i, j)
            A.nzval[idx] += A_local[n,m]
        end
    end

    # Build the matrix for interior connections only
    Ai = A[mesh.interior, mesh.interior]

    A, Ai
end

# function build_rhs(mesh::Mesh, graph::Graph, f::Function)
#     # Quadrature scheme
#     points = (Coord(0.0, 0.5), Coord(0.5, 0.0), Coord(0.5, 0.5))
#     weights = (1/6, 1/6, 1/6)
#     basis = build_linear_shape_funcs(points)
#     b_local = zeros(3)
#     b = zeros(mesh.n)

#     # Loop over all triangles & compute local rhs
#     for element in mesh.elements
#         p1, p2, p3 = triangle_coords(mesh, element)
#         coord_transform = [p2 - p1 p3 - p1]
#         invBk = inv(coord_transform')

#         # Reset local vector
#         fill!(b_local, 0.0)

#         for i = 1:3, k = 1:3
#             x = coord_transform * points[k] + p1
#             b_local[i] += weights[k] * f(x) * basis[i].ϕ[k]
#         end

#         b_local .*= abs(det(coord_transform))

#         # Put b_local in b
#         for n = 1:3
#             b[triangle[n]] += b_local[n]
#         end
#     end

#     # Build the matrix for interior connections only
#     bi = b[mesh.interior]

#     b, bi
# end

"""
    checkerboard(m::Int)

Returns a lambda function that maps a coordinate
Coord(x, y) to a coefficient
"""
function checkerboard(m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m+1, m+1))

    # Given a coordinate, return the value
    return (x::Coord) -> begin
        x_idx = 1 + floor(Int, x[1] * m)
        y_idx = 1 + floor(Int, x[2] * m)
        return A[y_idx, x_idx]
    end
end

"""
For a fixed dimensionality of the problem, see 
how λ changes the contraction factor.
"""
function example1(n = 16, c = 50)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)

    _, A = build_matrix(mesh, graph, checkerboard(c), checkerboard(c))
    _, Ā = build_matrix(mesh, graph, x::Coord -> 3.0, x::Coord -> 3.0)
    _, b = build_rhs(mesh, graph, x::Coord -> x[1] * x[2])
    
    exact = A \ b

    all_errors = []
    λs = linspace(0.1, 1.0, 10)

    for λ in λs
        println(λ)

        # Construct the shifted matrices
        A_shift = A + λ^2 * I
        Ā_shift = Ā + λ^2 * I

        errors = Float64[]

        # Start with a random v the size of b
        v = rand(length(b))

        for i = 1 : 3
        
            # Compute the residual
            r = b - A * v

            # Solve (λ² + A)u₀ = r
            u₀ = A_shift \ r

            # Solve (λ² + A)u₁ = λ²u₀
            u₁ = A_shift \ (λ^2 .* u₀)

            # Solve (λ² + Ā)ū₁ = λ²u₀
            ū₁ = Ā_shift \ (λ^2 .* u₀)
    
            # Solve Āū = λ²ū₁
            ū = Ā \ (λ^2 .* ū₁)

            # Solve (λ² + A)ũ = (λ² + Ā)ū
            ũ = A_shift \ (Ā_shift * ū)

            v .+= u₀ .+ u₁ .+ ũ

            push!(errors, norm(exact - v))
        end

        push!(all_errors, errors)
    end

    return all_errors, λs
end

"""
For a fixed λ and a fixed dimension of the problem
find the contraction factor as the number of checkerboard
cells increases.
"""
function example2(cs = 10 : 10 : 100, n = 255, λ = 0.25)
    ρs = []

    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    _, Ā = build_matrix(mesh, graph, x::Coord -> 3.0, x::Coord -> 3.0)
    _, b = build_rhs(mesh, graph, x::Coord -> x[1] * x[2])
    Ā_shift = Ā + λ^2 * I

    for c = cs
        @show c

        _, A = build_matrix(mesh, graph, checkerboard(c), checkerboard(c))
        exact = A \ b

        A_shift = A + λ^2 * I

        errors = Float64[]

        # Start with a random v the size of b
        v = rand(length(b))

        for i = 1 : 10
            @show i
        
            # Compute the residual
            r = b - A * v

            # Solve (λ² + A)u₀ = r
            u₀ = A_shift \ r

            # Solve (λ² + A)u₁ = λ²u₀
            u₁ = A_shift \ (λ^2 .* u₀)

            # Solve (λ² + Ā)ū₁ = λ²u₀
            ū₁ = Ā_shift \ (λ^2 .* u₀)

            # Solve Āū = λ²ū₁
            ū = Ā \ (λ^2 .* ū₁)

            # Solve (λ² + A)ũ = (λ² + Ā)ū
            ũ = A_shift \ (Ā_shift * ū)

            v .+= u₀ .+ u₁ .+ ũ

            push!(errors, norm(exact - v))
        end

        contractions = errors[2 : end] ./ errors[1 : end - 1]

        @show contractions

        push!(ρs, errors)
    end
    
    cs, ρs
end


function example3(n::Int = 512)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    bilinear_form = (u, v, k, x) -> dot(u.∇ϕ[:,k], v.∇ϕ[:,k])
    _, A = build_matrix(mesh, graph, bilinear_form)
end
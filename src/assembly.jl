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
    return jac, convert(Matrix, inv(jac')), p1
end

"""
Assembles the coefficient matrix A
"""
function build_matrix(mesh::Mesh, graph::Graph, bilinear_form::Function)
    # Quadrature scheme
    weights, quad_points = quadrature_rule(Tri3)

    # Reference basis functions
    reference_bases = element_basis(Tri, quad_points)

    # This one will hold the transformed stuff
    transformed_bases = element_basis(Tri, quad_points)

    # Number of nodes per element
    nodes_per_element = length(reference_bases)

    A = coefficient_matrix_factory(graph)
    A_local = zeros(nodes_per_element, nodes_per_element)

    # Loop over all elements & compute local system matrix
    for element in mesh.elements
        jac, jacInv, shift = jacobian(nodes(mesh, element)...)

        # Reset local matrix
        fill!(A_local, 0.0)

        # Transform the gradient
        for i = 1:nodes_per_element
            A_mul_B!(transformed_bases[i].∇ϕ, jacInv, reference_bases[i].∇ϕ)
        end

        for (k, point) = enumerate(quad_points)
            x = jac * point + shift

            for i = 1:nodes_per_element, j = 1:nodes_per_element
                u = transformed_bases[i]
                v = transformed_bases[j]
                A_local[i,j] += weights[k] * bilinear_form(u, v, k, x)
            end
        end

        A_local .*= abs(det(jac))

        # Put A_local into A
        for n = 1:nodes_per_element, m = 1:nodes_per_element
            i, j = element[n], element[m]
            idx = edge_to_idx(A, graph, i, j)
            A.nzval[idx] += A_local[n,m]
        end
    end

    # Build the matrix for interior connections only
    return A[mesh.interior, mesh.interior]
end

function build_rhs(mesh::Mesh, graph::Graph, f::Function)
    # Quadrature scheme
    weights, quad_points = quadrature_rule(Tri3)

    # Reference basis functions
    bases = element_basis(Tri, quad_points)

    # Number of nodes per element
    nodes_per_element = length(bases)

    b = zeros(mesh.n)
    b_local = zeros(nodes_per_element)

    # Loop over all elements & compute local system matrix
    for element in mesh.elements
        jac, _, shift = jacobian(nodes(mesh, element)...)

        # Reset local matrix
        fill!(b_local, 0.0)

        for (k, point) = enumerate(quad_points)
            x = jac * point + shift

            for i = 1:nodes_per_element
                b_local[i] += weights[k] * f(x) * bases[i].ϕ[k]
            end
        end

        b_local .*= abs(det(jac))

        # Put b_local into b
        for n = 1:nodes_per_element
            b[element[n]] += b_local[n]
        end
    end

    # Build the matrix for interior connections only
    return b[mesh.interior]
end

"""
    checkerboard(m::Int)

Returns a lambda function that maps a coordinate
Coord(x, y) to a coefficient
"""
function checkerboard(m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m+1, m+1))

    # Given a coordinate, return the value
    return (x::Coord{2}) -> begin
        x_idx::Int = 1 + floor(Int, x[1] * m)
        y_idx::Int = 1 + floor(Int, x[2] * m)
        return A[y_idx, x_idx]::Float64
    end
end

"""
For a fixed λ and a fixed dimension of the problem
find the contraction factor as the size of the domain increases
"""
function example2(cells = 10 : 10 : 100, n = 513, λ = 0.25)
    ρs = []
    
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)

    ā = (u, v, k, x) -> 3.0 * (u.∇ϕ[1,k] * v.∇ϕ[1,k] + u.∇ϕ[2,k] * v.∇ϕ[2,k])
    m = (u, v, k, x) -> u.ϕ[k] * v.ϕ[k]

    # Mass matrix
    M = build_matrix(mesh, graph, m)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, ā)

    for c = cells

        # Size of the domain
        r = Float64(c)

        @show r

        # Checkerboard pattern
        a11 = checkerboard(c)
        a22 = checkerboard(c)

        # Bilinear form (notice the 1 / r^2 bit)
        a = (u, v, k, x) -> a11(x) * u.∇ϕ[1,k] * v.∇ϕ[1,k] + a22(x) * u.∇ϕ[2,k] * v.∇ϕ[2,k]

        # Discretized differential operator
        A = build_matrix(mesh, graph, a)
        
        # Some right-hand side f(x,y) = 1 / xy
        b = build_rhs(mesh, graph, x -> exp(-r^2*(x[1]^2 + x[2]^2)))

        # Effective lambda parameter
        rλ_squared = (r * λ)^2

        # The scaled mass matrix
        M_scaled = rλ_squared * M

        # Shifted homogenized operator
        Ā_shift = M_scaled + Ā

        # Shifted differential operator
        A_shift = M_scaled + A

        # Compute the exact solution
        exact = A \ b

        errors = Float64[]

        # Start with a random v the size of b
        v = rand(length(b))

        for i = 1 : 4
            @show i

            # Solve (r²λ² + L)u₀ = residual
            u₀ = A_shift \ (b - A * v)

            # We need this twice
            Mu₀ = M_scaled * u₀

            # Solve (r²λ² + L)u₁ = r²λ²u₀
            u₁ = A_shift \ Mu₀

            # Solve (r²λ² + L̄)ū₁ = r²λ²u₀
            ū₁ = Ā_shift \ Mu₀

            # Solve L̄ū = r²λ²ū₁
            ū = Ā \ (M_scaled * ū₁)

            # Solve (r²λ² + L)ũ = (r²λ² + L̄)ū
            ũ = A_shift \ (Ā_shift * ū)

            v .+= u₀ .+ u₁ .+ ũ

            error = norm(exact - v)

            @show error

            push!(errors, error)
        end

        contractions = errors[2 : end] ./ errors[1 : end - 1]

        @show contractions
        @show errors

        push!(ρs, errors)
    end
    
    cells, ρs
end

function show_updates(c = 40, n = 513, λ = 0.25)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)

    # Size of the domain
    r = Float64(c)

    # Bilinear forms
    ā = (u, v, k, x) -> 3.0 * (u.∇ϕ[1,k] * v.∇ϕ[1,k] + u.∇ϕ[2,k] * v.∇ϕ[2,k])
    m = (u, v, k, x) -> u.ϕ[k] * v.ϕ[k]
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    a = (u, v, k, x) -> a11(x) * u.∇ϕ[1,k] * v.∇ϕ[1,k] + a22(x) * u.∇ϕ[2,k] * v.∇ϕ[2,k]

    # Mass matrix
    M = build_matrix(mesh, graph, m)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, ā)

    # Discretized differential operator
    A = build_matrix(mesh, graph, a)
    
    # Some right-hand side f(x,y) = 1
    b = build_rhs(mesh, graph, x -> 1.0)

    # Effective lambda parameter
    rλ_squared = (r * λ)^2

    # The scaled mass matrix
    M_scaled = rλ_squared * M

    # Shifted homogenized operator
    Ā_shift = M_scaled + Ā

    # Shifted differential operator
    A_shift = M_scaled + A

    # Compute the exact solution
    exact = A \ b

    # Start with a random v the size of b
    v = rand(length(b))

    steps = []

    for i = 1 : 4
        @show i

        # Solve (r²λ² + L)u₀ = residual
        u₀ = A_shift \ (b - A * v)

        # We need this twice
        Mu₀ = M_scaled * u₀

        # Solve (r²λ² + L)u₁ = r²λ²u₀
        u₁ = A_shift \ Mu₀

        # Solve (r²λ² + L̄)ū₁ = r²λ²u₀
        ū₁ = Ā_shift \ Mu₀

        # Solve L̄ū = r²λ²ū₁
        ū = Ā \ (M_scaled * ū₁)

        # Solve (r²λ² + L)ũ = (r²λ² + L̄)ū
        ũ = A_shift \ (Ā_shift * ū)

        push!(steps, (copy(v), copy(u₀), copy(u₁), copy(ũ)))

        v .+= u₀ .+ u₁ .+ ũ

        @show norm(exact - v)
    end

    # Show the error reduction of each update
    for step in steps
        v = zeros(exact)
        for i = 1 : 4
            v .+= step[i]
            print(i, ": ", norm(v - exact), ", ")
        end
        println()
    end

    return exact, steps, v
end

function example3(n::Int = 512)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    bilinear_form = (u, v, k, x) -> u.ϕ[k] * v.ϕ[k]
    return build_matrix(mesh, graph, bilinear_form)
end

function example4(n::Int = 512, c::Int = 10)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    λ = 0.25

    B1 = (u, v, k, x) -> a11(x) * u.∇ϕ[1,k] * v.∇ϕ[1,k] + a22(x) * u.∇ϕ[2,k] * v.∇ϕ[2,k]
    B2 = (u, v, k, x) -> 3.0 * (u.∇ϕ[1,k] * v.∇ϕ[1,k] + u.∇ϕ[2,k] * v.∇ϕ[2,k])
    B3 = (u, v, k, x) -> u.ϕ[k] * v.ϕ[k]
    f = x -> x[1] * x[2]
    
    # Differential operator
    A = build_matrix(mesh, graph, B1)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, B2)

    # Mass matrix
    M = build_matrix(mesh, graph, B3)

    # Rhs
    b = build_rhs(mesh, graph, f)

    return A \ b, Ā \ b
end

function example5(n::Int = 512, shift::Float64 = 1.0)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    
    B = (u, v, k, x) -> u.∇ϕ[1,k] * v.∇ϕ[1,k] + u.∇ϕ[2,k] * v.∇ϕ[2,k] + shift * u.ϕ[k] * v.ϕ[k]
    f = x -> x[1] * x[2]
    
    # Differential operator
    A = build_matrix(mesh, graph, B)

    # Rhs
    b = build_rhs(mesh, graph, f)

    return A \ b
end
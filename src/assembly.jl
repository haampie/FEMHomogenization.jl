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

function jacobian(p1, p2, p3)
    jac = [p2 - p1 p3 - p1]
    return jac, inv(jac'), p1
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
        jac, jacInv, shift = jacobian(nodes(mesh, element)...)

        # Reset local matrix
        fill!(A_local, 0.0)

        # Transform the gradient
        for point in basis, f in point
            A_mul_B!(f.∇ϕ, jacInv, f.grad)
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
        jac, _, shift = jacobian(nodes(mesh, element)...)

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

    ā = (u, v, x) -> 3.0 * dot(u.∇ϕ, v.∇ϕ)
    m = (u, v, x) -> u.ϕ * v.ϕ

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
        a = (u, v, x) -> a11(x) * u.∇ϕ[1] * v.∇ϕ[1] + a22(x) * u.∇ϕ[2] * v.∇ϕ[2]

        # Discretized differential operator
        A = build_matrix(mesh, graph, a)
        
        # Some right-hand side f(x,y) = 1 / xy
        b = build_rhs(mesh, x -> exp(-r^2*(x[1]^2 + x[2]^2)))

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
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    ā = (u, v, x) -> 3.0 * dot(u.∇ϕ, v.∇ϕ)
    m = (u, v, x) -> u.ϕ * v.ϕ
    a = (u, v, x) -> a11(x) * u.∇ϕ[1] * v.∇ϕ[1] + a22(x) * u.∇ϕ[2] * v.∇ϕ[2]

    # Mass matrix
    M = build_matrix(mesh, graph, m)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, ā)

    # Discretized differential operator
    A = build_matrix(mesh, graph, a)
    
    # Some right-hand side f(x,y) = 1
    b = build_rhs(mesh, x -> 1.0)

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
    bilinear_form = (u, v, x) -> u.ϕ * v.ϕ
    return build_matrix(mesh, graph, bilinear_form)
end

function example4(n::Int = 512, c::Int = 10)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    λ = 0.25

    B1 = (u, v, x) -> a11(x) * u.∇ϕ[1] * v.∇ϕ[1] + a22(x) * u.∇ϕ[2] * v.∇ϕ[2]
    B2 = (u, v, x) -> 3.0 * dot(u.∇ϕ, v.∇ϕ)
    B3 = (u, v, x) -> u.ϕ * v.ϕ
    f = x -> x[1] * x[2]
    
    # Differential operator
    A = build_matrix(mesh, graph, B1)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, B2)

    # Mass matrix
    M = build_matrix(mesh, graph, B3)

    # Rhs
    b = build_rhs(mesh, f)

    return A \ b, Ā \ b
end

function example5(n::Int = 512, shift::Float64 = 1.0)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)
    
    B = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + shift * u.ϕ * v.ϕ
    f = x -> sqrt(x[1] * x[2])
    
    # Differential operator
    A = build_matrix(mesh, graph, B)

    # Rhs
    b = build_rhs(mesh, f)

    return A \ b
end

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
function example1(cells = 10 : 10 : 100, n = 513, λ = 0.25)
    ρs = []

    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)

    ā = (u, v, x) -> 3.0 * dot(u.∇ϕ, v.∇ϕ)
    m = (u, v, x) -> u.ϕ * v.ϕ

    # Mass matrix
    M = build_matrix(mesh, graph, Tri3, m)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, Tri3, ā)

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
        A = build_matrix(mesh, graph, Tri3, a)
        
        # Some right-hand side f(x,y) = 1 / xy
        b = build_rhs(mesh, Tri3, x -> exp(-r^2*(x[1]^2 + x[2]^2)))

        # Effective lambda parameter
        rλ_squared = (r * λ)^2

        # The scaled mass matrix
        M_scaled = rλ_squared * M

        # Shifted homogenized operator
        Ā_shift = M_scaled + Ā

        # Shifted differential operator
        A_shift = M_scaled + A

        # Factorize it
        A_shift_fact = cholfact(A_shift)
        Ā_shift_fact = cholfact(Ā_shift)
        Ā_fact = cholfact(Ā)

        # Compute the exact solution
        exact = A \ b

        errors = Float64[]

        # Start with a random v the size of b
        v = rand(length(b))

        for i = 1 : 10
            @show i

            # Solve (r²λ² + L)u₀ = residual
            u₀ = A_shift_fact \ (b - A * v)

            # We need this twice
            Mu₀ = M_scaled * u₀

            # Solve (r²λ² + L)u₁ = r²λ²u₀
            u₁ = A_shift_fact \ Mu₀

            # Solve (r²λ² + L̄)ū₁ = r²λ²u₀
            ū₁ = Ā_shift_fact \ Mu₀

            # Solve L̄ū = r²λ²ū₁
            ū = Ā_fact \ (M_scaled * ū₁)

            # Solve (r²λ² + L)ũ = (r²λ² + L̄)ū
            ũ = A_shift_fact \ (Ā_shift * ū)

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

function example2(c = 40, n = 513, λ = 0.25)
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
    f = x -> 1.0

    # Mass matrix
    M = build_matrix(mesh, graph, Tri3, m)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, Tri3, ā)

    # Discretized differential operator
    A = build_matrix(mesh, graph, Tri3, a)

    # Some right-hand side f(x,y) = 1
    b = build_rhs(mesh, Tri3, f)

    # Effective lambda parameter
    rλ_squared = (r * λ)^2

    # The scaled mass matrix
    M_scaled = rλ_squared * M

    # Shifted homogenized operator
    Ā_shift = M_scaled + Ā

    # Shifted differential operator
    A_shift = M_scaled + A

    # Factorize it
    A_shift_fact = cholfact(A_shift)
    Ā_shift_fact = cholfact(Ā_shift)
    Ā_fact = cholfact(Ā)

    # Compute the exact solution
    exact = A \ b

    # Start with a random v the size of b
    v = rand(length(b))

    steps = []

    for i = 1 : 4
        @show i

        # Solve (r²λ² + L)u₀ = residual
        u₀ = A_shift_fact \ (b - A * v)

        # We need this twice
        Mu₀ = M_scaled * u₀

        # Solve (r²λ² + L)u₁ = r²λ²u₀
        u₁ = A_shift_fact \ Mu₀

        # Solve (r²λ² + L̄)ū₁ = r²λ²u₀
        ū₁ = Ā_shift_fact \ Mu₀

        # Solve L̄ū = r²λ²ū₁
        ū = Ā_fact \ (M_scaled * ū₁)

        # Solve (r²λ² + L)ũ = (r²λ² + L̄)ū
        ũ = A_shift_fact \ (Ā_shift * ū)

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
    bilinear_form = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ
    return build_matrix(mesh, graph, Tri3, bilinear_form)
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
    A = build_matrix(mesh, graph, Tri3, B1)

    # Homogenized operator
    Ā = build_matrix(mesh, graph, Tri3, B2)

    # Mass matrix
    M = build_matrix(mesh, graph, Tri3, B3)

    # Rhs
    b = build_rhs(mesh, Tri3, f)

    return A \ b, Ā \ b
end

function example5(n::Int = 512, shift::Float64 = 1.0)
    mesh = uniform_mesh(n)
    graph = mesh_to_graph(mesh)

    B = (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + shift * u.ϕ * v.ϕ
    f = x -> sqrt(x[1] * x[2])

    # Differential operator
    A = build_matrix(mesh, graph, Tri3, B)

    # Rhs
    b = build_rhs(mesh, Tri3, f)

    return A \ b
end

"""
Refine a grid a few times uniformly
"""
function refinement_example(refinements = 10, ::Type{Ti} = Int32, ::Type{Tv} = Float64) where {Ti,Tv}
    nodes = SVector{2,Tv}[(0, 0), (1, 0), (1, 1), (0, 1)]
    triangles = SVector{3,Ti}[(1, 2, 3), (1, 4, 3)]
    graph, boundary, interior = to_graph(nodes, triangles)

    for i = 1 : refinements
        nodes, triangles = refine(nodes, triangles, graph)
        graph, boundary, interior = to_graph(nodes, triangles)
    end

    A = assemble_matrix(nodes, triangles, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ))
    b = assemble_rhs(nodes, triangles, x -> 1.0)

    return A, b, interior
end

"""
A geometric level of the grid
"""
struct Level{Tv,Ti}
    nodes::Vector{SVector{2,Tv}}
    triangles::Vector{SVector{3,Ti}}
    graph::MyGraph{Ti}
    boundary::Vector{Ti}
    interior::Vector{Ti}
end

function build_hierarchy(refinements::Int = 10, ::Type{Ti} = Int64, ::Type{Tv} = Float64) where {Ti,Tv}
    interpolation = Vector{SparseMatrixCSC{Tv,Ti}}(refinements - 1)
    levels = Vector{Level{Tv,Ti}}(refinements)

    # Initial mesh is 2 triangles in a square
    nodes = SVector{2,Tv}[(0, 0), (1, 0), (1, 1), (0, 1)]
    triangles = SVector{3,Ti}[(1, 2, 3), (1, 4, 3)]
    graph, boundary, interior = to_graph(nodes, triangles)
    levels[1] = Level(nodes, triangles, graph, boundary, interior)

    # Then we refine a couple times
    for i = 1 : refinements - 1
        interpolation[i] = interpolation_operator(nodes, graph)
        nodes, triangles = refine(nodes, triangles, graph)
        graph, boundary, interior = to_graph(nodes, triangles)
        levels[i + 1] = Level(nodes, triangles, graph, boundary, interior)
    end

    # Start with random values on the coarsest grid
    # and interpolate them to the finer grids.
    vals = rand(length(levels[1].nodes))
    save_file("grid_01", levels[1].nodes, levels[1].triangles, vals)

    for i = 2 : refinements
        vals = interpolation[i - 1]' * vals
        name = @sprintf "grid_%02d" i
        save_file(name, levels[i].nodes, levels[i].triangles, vals)
    end
end
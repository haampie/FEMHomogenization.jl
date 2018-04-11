"""
checkerboard(m::Int)

Returns a lambda function that maps a coordinate
Coord(x, y) to a coefficient
"""
function checkerboard(m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m+1, m+1))

    # Given a coordinate, return the value
    return (x::Coord{2}) -> begin
        x_idx = floor(Int, x[1]) + 1
        y_idx = floor(Int, x[2]) + 1
        return A[y_idx, x_idx]
    end
end

"""
Evaluated a(x) for each element number
"""
function checkerboard_elements(mesh::Mesh, m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m, m))

    as = zeros(length(mesh.elements))

    for (idx, element) in enumerate(mesh.elements)
        # Midpoint of triangle
        coord = mapreduce(i -> mesh.nodes[i], +, element) / length(element)

        # Indices in the A matrix
        x_idx = floor(Int, coord[1]) + 1
        y_idx = floor(Int, coord[2]) + 1

        as[idx] = A[y_idx, x_idx]
    end

    return (idx::Int) -> as[idx]
end
"""
This example solves a laplace-like problem with an
oscillating conductivity (checkerboard pattern).
It also solves the homogenized problem and saves
the results to a vtk file.
"""
function example3(refinements::Int = 6, c::Int = 10)
    mesh, graph, interior = unit_square(refinements)
    
    a11 = checkerboard(c)
    a22 = checkerboard(c)
    λ = 0.25

    B1 = (u, v, x) -> a11(x) * u.∇ϕ[1] * v.∇ϕ[1] + a22(x) * u.∇ϕ[2] * v.∇ϕ[2]
    B2 = (u, v, x) -> 3.0 * dot(u.∇ϕ, v.∇ϕ)
    B3 = (u, v, x) -> u.ϕ * v.ϕ
    f = x -> x[1] * x[2]

    # Differential and homogenized operator
    A = assemble_matrix(mesh, B1)
    Ā = assemble_matrix(mesh, B2)

    # Rhs
    b = assemble_rhs(mesh, f)
    b_int = b[interior]

    x = zeros(b)
    x̄ = zeros(b)

    x[interior] = A[interior,interior] \ b_int
    x̄[interior] = Ā[interior,interior] \ b_int

    save_file("results", mesh, Dict(
        "x"     => x, 
        "x_bar" => x̄, 
        "a11"   => a11.(mesh.nodes), 
        "a22"   => a22.(mesh.nodes)
    ))
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

function interior_subdomain(mesh, boundary_layer_size, domain_width)
    center = @SVector [domain_width / 2, domain_width / 2]
    R = domain_width / 2 - boundary_layer_size
    
    # Find the elements in the R circle
    elements_in_circle = find(el -> begin
        midpoint = mapreduce(i -> mesh.nodes[i], +, el) / 3
        norm(center - midpoint, Inf) < R
    end, mesh.elements)

    # Collect the nodes of these elements
    nodes_in_circle = Vector{Int}(3 * length(elements_in_circle))
    idx = 1
    for el_idx in elements_in_circle
        element = mesh.elements[el_idx]
        nodes_in_circle[idx + 0] = element[1]
        nodes_in_circle[idx + 1] = element[2]
        nodes_in_circle[idx + 2] = element[3]
        idx += 3
    end

    return unique(nodes_in_circle), elements_in_circle
end

"""
    ens(sr, cr, steps)

We refine a square `sr` times as a coarse grid on which we build the checkerboard pattern.
Then we refine the cells `cr` times to construct a fine FEM mesh.
"""
function ens(square_refine = 6, cell_refine = 2, steps = 2)
    width = 2^square_refine
    mesh, graph, interior = generic_square(square_refine + cell_refine, width, width)
    a11, a22 = checkerboard_elements(mesh, width), checkerboard_elements(mesh, width)

    println("Total nodes: ", length(mesh.nodes))
    println("Ω = [0, ", width, "] × [0, ", width, "]")

    bf_mass = (u, v, x) -> u.ϕ * v.ϕ
    bf_oscillating = (u, v, idx) -> a11(idx) * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx) * u.∇ϕ[2] * v.∇ϕ[2]
    M = assemble_matrix(mesh, bf_mass)
    A = assemble_matrix_elementwise(mesh, bf_oscillating)

    M_int = M[interior, interior]
    A_int = A[interior, interior]

    # Build the initial rhs with ξ = [1 0]
    ξ = (1.0, 0.0)

    # Partial integration of the term appearing in the paper
    load = (u, idx) -> -(a11(idx) * ξ[1] * u.∇ϕ[1] + a22(idx) * ξ[2] * u.∇ϕ[2])

    # Construct rhs
    b = assemble_rhs_with_gradient(mesh, load)
    
    λ = 1.0
    v_int = (λ .* M_int .+ A_int) \ b[interior]
    v = zeros(length(mesh.nodes))
    v[interior] .= v_int

    vs = Vector{Float64}[copy(v)]

    for i = 1 : steps
        println("Step = ", i)
        λ /= 2
        fill!(v, 0.0)
        @time myA = λ .* M_int .+ A_int
        @time myRhs = λ .* M_int * v_int
        @time v_int .= myA \ myRhs
        v[interior] .= v_int
        push!(vs, copy(v))
    end

    # We average over a shrinking domain Br, so we allocate some vector that will be zeroed
    # out on Ω \ Br.
    v_prev_masked = zeros(length(mesh.nodes))
    v_curr_masked = zeros(length(mesh.nodes))
    ones_masked = zeros(length(mesh.nodes))
    smaller_domain = zeros(length(mesh.elements))

    # The boundary layer is initially 3 cells big.
    boundary_layer = 3.0
    masked_elements = Vector{Float64}[]

    σ = 0.0
    λ = 1.0

    for k = 1 : steps + 1

        mask, small_domain_elements = interior_subdomain(mesh, boundary_layer, width)

        fill!(v_prev_masked, 0.0)
        fill!(v_curr_masked, 0.0)
        fill!(ones_masked, 0.0)
        fill!(smaller_domain, 0.0)

        smaller_domain[small_domain_elements] .= 1.0
        v_curr_masked[mask] .= vs[k][mask]
        ones_masked[mask] .= 1.0
        area = dot(ones_masked, M * ones_masked)

        push!(masked_elements, copy(smaller_domain))

        # The first k is special: use partial integration again.
        if k == 1
            v_prev_masked[mask] .= b[mask]
            δσ = λ * (dot(v_prev_masked, v_curr_masked) + dot(v_curr_masked, M * v_curr_masked)) / area
        else
            v_prev_masked[mask] .= vs[k - 1][mask]
            δσ = λ * (dot(v_prev_masked, M * v_curr_masked) + dot(v_curr_masked, M * v_curr_masked)) / area
        end

        λ *= 2
        σ += δσ
        @show δσ

        # Double the boundary layer size
        boundary_layer *= 2
    end

    @show 5.0 - σ

    save_to_vtk("results", mesh, Dict(
        "v1"    => vs[1],
        "v2"    => vs[2],
        "v3"    => vs[3]
    ), Dict(
        "a11"   => a11.(1 : length(mesh.elements)),
        "a22"   => a22.(1 : length(mesh.elements)),
        "mask1" => masked_elements[1],
        "mask2" => masked_elements[2],
        "mask3" => masked_elements[3]
    ))
end


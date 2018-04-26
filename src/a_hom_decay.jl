"""
Evaluated a(x) for each element number
"""
function checkerboard_elements(mesh::Mesh{Tri}, m::Int)
    A = map(x -> x ? 9.0 : 1.0, rand(Bool, m, m))

    as = zeros(length(mesh.elements))

    for idx = 1 : length(mesh.elements)
        element = mesh.elements[idx]

        # Midpoint of triangle
        coord = mapreduce(i -> mesh.nodes[i], +, element) / length(element)

        # Indices in the A matrix
        x_idx = floor(Int, coord[1]) + 1
        y_idx = floor(Int, coord[2]) + 1

        as[idx] = A[y_idx, x_idx]
    end

    return idx::Int -> begin
        @inbounds val = as[idx]
        return val
    end
end

"""
Tests whether each element belongs to the interior of the domain.
Returns a list of nodes and a sorted list of element ids
"""
function interior_subdomain(mesh::Mesh{Tri}, total_width::Int, interior_width::Int)
    center = @SVector [total_width / 2, total_width / 2]
    
    elements_in_interior = find(el -> begin
        midpoint = mapreduce(i -> mesh.nodes[i], +, el) / 3
        norm(center - midpoint, Inf) < interior_width / 2
    end, mesh.elements)

    # Collect the nodes of these elements
    nodes_in_interior = Vector{Int}(3 * length(elements_in_interior))
    idx = 1
    for el_idx in elements_in_interior
        element = mesh.elements[el_idx]
        nodes_in_interior[idx + 0] = element[1]
        nodes_in_interior[idx + 1] = element[2]
        nodes_in_interior[idx + 2] = element[3]
        idx += 3
    end

    return unique(nodes_in_interior), elements_in_interior
end

# Returns the integer size boundary layer
@inline interior_domain_width(n::Int, k::Int) = floor(Int, 2.0 ^ (n - k/2))

function create_mask(r1::Float64, r2::Float64, total_width::Int)
    width = r2 - r1
    mid = Coord{2}(total_width / 2, total_width / 2)

    # more or less a mollified indicator function
    return x::Coord{2} -> begin
        dist = norm(x - mid)
        dist ≤ r1 && return 1.0
        dist ≥ r2 && return 0.0
        exp(1.0 - 1.0 / (1.0 - (dist - r1) / width))
    end
end

function run(n::Int, refinements::Int = 2, ξ::NTuple{2,Float64} = (1.0, 0.0), verbose::Bool = true)
    # Some stuff we wish to plot
    if verbose
        plot_nodes = Dict{String,Vector{Float64}}()
        plot_elements = Dict{String,Vector{Float64}}()
    end
    
    # Store intermediate σ²s
    σ²s = Vector{Float64}(n)

    # Total number of `coarse cells'
    interior_width = interior_domain_width(n, 0)

    # ∂ is our initial boundary layer
    ∂ = 10

    # Total width of the domain including the boundary layer
    total_width = interior_width + 2∂

    # Our FEM mesh will be refined `refinements` times (one refinement splits a triangle into 4 triangles)
    grid_cells = total_width * 2^refinements

    # We generate the FEM mesh up to the boundary and we get a list of interior nodes as well.
    mesh, interior = rectangle(grid_cells, grid_cells, total_width, total_width)

    # Some bookkeeping for FEM nodes & elements
    total_nodes = length(mesh.nodes)
    total_elements = length(mesh.elements)

    @show total_nodes total_elements

    # We build the checkerboard pattern on each cell [n, n+1] x [m, m+1]
    a11 = checkerboard_elements(mesh, total_width)
    a22 = checkerboard_elements(mesh, total_width)

    # The integrand of the bilinear forms (idx is the element number)
    bf_oscillating = (u, v, idx) -> a11(idx)::Float64 * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx)::Float64 * u.∇ϕ[2] * v.∇ϕ[2]

    # We construct the mass matrix over the whole domain [0, total_width]^2
    M = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ)
    A = assemble_matrix_elementwise(mesh, bf_oscillating)

    # But we need only the interior part cause of the Dirichlet b.c.
    M_int = M[interior, interior]
    A_int = A[interior, interior]

    # First rhs via partial integration (note the minus sign)
    load = (u, idx, x) -> -(a11(idx)::Float64 * ξ[1] * u.∇ϕ[1] + a22(idx)::Float64 * ξ[2] * u.∇ϕ[2])
    aξ∇v₋₁ = assemble_rhs_with_gradient(mesh, load)

    # Solve (μ - ∇⋅a∇)v₀ = v₋₁ ↔ ∫(μ v₀ϕ + (a∇v₀)⋅∇ϕ)dx = -∫aξ⋅∇ϕ dx ∀ϕ
    # No boundary integral here because of our artificial Dirichlet b.c.
    v = zeros(total_nodes)
    v_int = (M_int .+ A_int) \ aξ∇v₋₁[interior]
    v[interior] .= v_int    

    # Also define the previous v right now.
    v_prev = zeros(v)

    ##
    ## Compute the first term of the sum
    ##

    # We determine the interior subdomain
    interior_nodes, interior_elements = interior_subdomain_circle(mesh, total_width, interior_width)

    # Build the mass matrix & load by integrating just over the interior domain.
    # Compute the boundary integral for v₀
    initial_mask = create_mask(interior_width / 4, interior_width / 2, total_width)

    M_decay = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ * initial_mask(x)::Float64)
    load_decay = (u, idx, x) -> -(a11(idx)::Float64 * ξ[1] * u.∇ϕ[1] + a22(idx)::Float64 * ξ[2] * u.∇ϕ[2]) * initial_mask(x)::Float64
    aξ∇v₋₁_decay = assemble_rhs_with_gradient(mesh, load_decay)
    @show sum(M_decay)
    σ² = (dot(aξ∇v₋₁_decay, v) + dot(v, M_decay * v)) / sum(M_decay)
    σ²s[1] = σ²

    # Plot some stuff
    if verbose
        tmp = zeros(total_elements)
        tmp[interior_elements] .= 1.0
        plot_elements["Interior 0"] = tmp
        plot_elements["a₁₁"] = a11.(1 : total_elements)
        plot_elements["a₂₂"] = a22.(1 : total_elements)
        plot_nodes["aξ∇v₋₁"] = aξ∇v₋₁
        plot_nodes["aξ∇v₋₁_decay"] = aξ∇v₋₁_decay
        plot_nodes["mask 0"] = initial_mask.(mesh.nodes)
        plot_nodes["v0"] = copy(v)
    end

    ##
    ## Compute v₁, v₂, ... and compute the other terms of σ²
    ##

    λ = 1.0

    @inbounds for k = 1 : n-1
        verbose && @show k

        # This is just scaling λ down
        M_int .*= 0.5
        λ *= 2
        
        # Store the previous v
        copy!(v_prev, v)

        # Compute the new v and restore the b.c.
        @time copy!(v_int, (M_int + A_int) \ (M_int * v_int))
        v[interior] .= v_int

        interior_width = interior_domain_width(n, k)
        interior_nodes, interior_elements = interior_subdomain_circle(mesh, total_width, interior_width)

        mask = create_mask(interior_width / 4, interior_width / 2, total_width)

        M_decay = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ * mask(x)::Float64)

        σ² += λ * (dot(v_prev, M_decay * v) + dot(v, M_decay * v)) / sum(M_decay)

        σ²s[k + 1] = σ²

        # Plot some things
        if verbose
            tmp = zeros(total_elements)
            tmp[interior_elements] .= 1.0
            plot_elements["Interior $k"] = tmp
            plot_nodes["v$k"] = copy(v)
            plot_nodes["mask $k"] = mask.(mesh.nodes)
        end
    end

    if verbose
        println("Saving")
        save_to_vtk("results", mesh, plot_nodes, plot_elements)
    end

    return σ²s
end

function repeat(;times = 10, ref_coarse = 6, ref_fine = 3, θ = 1.0, file = "/mathwork/stoppeh1/results.txt")
    xi = (cos(θ), sin(θ))

    sigmas = zeros(times, ref_coarse)

    for i = 1 : times
        sigmas[i, :] .= run(ref_coarse, ref_fine, xi, false)

        @show sigmas[i, :] mean(sigmas[1 : i, end])
    end

    writedlm(file, sigmas)

    return sigmas
end

function effects_of_h_refinement(times = 100, n = 3, refs = 1:5)
    results = Vector{Matrix{Float64}}(length(refs))
    θ = 1.0
    xi = (cos(θ), sin(θ))

    for (idx, refinements) in enumerate(refs)

        println("Running with ", refinements, " refinements.")

        # Generate the same `a`
        srand(1)

        sigmas = zeros(times, n)

        for i = 1 : times
            sigmas[i, :] .= run(n, refinements, xi, false)
        end

        results[idx] = sigmas
    end

    return results
end

function test_fem(from = 4, to = 11)
    hs = Float64[]
    values = Float64[]

    total_width = 2 ^ from

    for m = from : to
        srand(1)
        mesh, interior = rectangle(2^m, 2^m, total_width, total_width)
        @show length(mesh.nodes)
        a11 = checkerboard_elements(mesh, total_width)
        a22 = checkerboard_elements(mesh, total_width)
        bf_oscillating = (u, v, idx::Int) -> a11(idx)::Float64 * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx)::Float64 * u.∇ϕ[2] * v.∇ϕ[2]
        A = assemble_matrix_elementwise(mesh, bf_oscillating)
        b = assemble_rhs(mesh, x -> 1.0)
        A_int = A[interior,interior]
        b_int = b[interior]
        x = zeros(length(mesh.nodes))
        x[interior] .= A_int \ b_int

        push!(hs, total_width / 2^m)
        push!(values, sqrt(dot(x, A * x)))

        # save_to_vtk("simple_fem_$(lpad(m, 2, 0))", mesh, Dict("x" => copy(x)), Dict("a11" => a11.(1:length(mesh.elements))))
    end

    return hs, values
end

function interior_subdomain_circle(mesh::Mesh{Tri}, total_width::Int, interior_width::Int)
    center = @SVector [total_width / 2, total_width / 2]
    
    elements_in_interior = find(el -> begin
        midpoint = mapreduce(i -> mesh.nodes[i], +, el) / 3
        norm(center - midpoint) < interior_width / 2
    end, mesh.elements)

    # Collect the nodes of these elements
    nodes_in_interior = Vector{Int}(3 * length(elements_in_interior))
    idx = 1
    for el_idx in elements_in_interior
        element = mesh.elements[el_idx]
        nodes_in_interior[idx + 0] = element[1]
        nodes_in_interior[idx + 1] = element[2]
        nodes_in_interior[idx + 2] = element[3]
        idx += 3
    end

    return unique(nodes_in_interior), elements_in_interior
end



"""
Solve the problem

  (λ + ∇⋅a∇)u = λ on Ω 
            u = 0 on ∂Ω

to inspect the boundary layer size; u ≡ 1 on the center of the domain, u = 0 on the boundary
"""
function show_boundary_size(;square_refine = 6, cell_refine = 2, λ = 1.0, filename = "boundary_layer")
    width = 2^square_refine
    mesh, graph, interior = generic_square(square_refine + cell_refine, width, width)

    # Make a cut at y = width / 2
    middle_nodes = sort!(find(node -> abs(node[2] - width/2) ≤ 10eps(), mesh.nodes), by = node -> mesh.nodes[node][1])
    x_coords = map(idx -> mesh.nodes[idx][1], middle_nodes)

    srand(1)
    a11 = checkerboard_elements(mesh, width)
    a22 = checkerboard_elements(mesh, width)

    bf_mass = (u, v, x) -> u.ϕ * v.ϕ
    bf_oscillating = (u, v, idx) -> a11(idx) * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx) * u.∇ϕ[2] * v.∇ϕ[2]
    
    M = assemble_matrix(mesh, bf_mass)
    A = assemble_matrix_elementwise(mesh, bf_oscillating)
    b = assemble_rhs(mesh, x -> λ)

    M_int = M[interior, interior]
    A_int = A[interior, interior]
    b_int = b[interior]
    x = zeros(length(mesh.nodes))

    x[interior] .= (λ .* M_int .+ A_int) \ b_int

    x_line = x[middle_nodes]
    inds_above = extrema(find(x -> x > 0.90, x_line))

    x_above = (x_coords[inds_above[1]],x_coords[inds_above[2]])
    @show inds_above x_above

    # return x_coords, x_line, 

    save_to_vtk(filename, mesh, Dict(
        "x" => x
    ), Dict(
        "a11" => a11.(1 : length(mesh.elements)),
        "a22" => a22.(1 : length(mesh.elements))
    ))
end
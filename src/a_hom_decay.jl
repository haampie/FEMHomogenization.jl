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
@inline interior_domain_width(n::Int, k::Int) = unsafe_trunc(Int, 2.0 ^ (Float64(n) - k / 2))

@inline function create_mask(r1::Float64, r2::Float64, total_width::Int)
    width = r2 - r1
    mid = Coord{2}(total_width / 2, total_width / 2)

    # more or less a mollified indicator function
    return x::Coord{2} -> begin
        dist = norm(x - mid)
        interp = 4.0 - 8.0 * (dist - r1) / width
        dist ≤ r1 && return 1.0
        dist ≥ r2 && return 0.0
        return (tanh(interp) + 1) / 2
    end
end

function run(n::Int, nodes_per_cell::Int = 16, ξ::NTuple{2,Float64} = (1.0, 0.0), verbose = true, ε = 0.3)
    # Some stuff we wish to plot
    # if verbose
    #     plot_nodes = Dict{String,Vector{Float64}}()
    #     plot_elements = Dict{String,Vector{Float64}}()
    # end
    
    # Store intermediate σ²s
    σ²s = Vector{Float64}(n)

    # Total number of `coarse cells'
    interior_width = interior_domain_width(n, 0)

    # ∂ is our initial boundary layer
    ∂ = 10

    # Total width of the domain including the boundary layer
    total_width = interior_width + 2∂

    # We generate the FEM mesh up to the boundary and we get a list of interior nodes as well.
    # mesh, interior = rectangle(grid_cells, grid_cells, total_width, total_width)
    mesh, interior = generate_stretched_grid(total_width, -1.8, nodes_per_cell)

    # Some bookkeeping for FEM nodes & elements
    total_nodes = length(mesh.nodes)
    total_elements = length(mesh.elements)

    # @show total_nodes total_elementsnodes_per_cell

    # We build the checkerboard pattern on each cell [n, n+1] x [m, m+1]
    a11 = construct_checkerboard(total_width + 2)
    a22 = construct_checkerboard(total_width + 2)
    moll = Mollifier{2}(ε)

    # The integrand of the bilinear forms (idx is the element number)
    # bf_checkerboard = (u, v, x::Coord{2}) -> mollify(a11, x, moll) * u.∇ϕ[1] * v.∇ϕ[1] + 
    #                                          mollify(a22, x, moll) * u.∇ϕ[2] * v.∇ϕ[2]
    bf_checkerboard = (u, v, x::Coord{2}) -> a11(x) * u.∇ϕ[1] * v.∇ϕ[1] + 
                                             a22(x) * u.∇ϕ[2] * v.∇ϕ[2]

    # We construct the mass matrix over the whole domain [0, total_width]^2
    # verbose && println("Assembling M and A")
    M = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ)
    A = assemble_matrix(mesh, bf_checkerboard)
    # verbose && println("Done")

    # But we need only the interior part cause of the Dirichlet b.c.
    M_int = M[interior, interior]
    A_int = A[interior, interior]

    # First rhs via partial integration (note the minus sign)
    # load = (u, idx, x::Coord{2}) -> -(mollify(a11, x, moll) * ξ[1] * u.∇ϕ[1] + 
    #                                   mollify(a22, x, moll) * ξ[2] * u.∇ϕ[2])
    load = (u, idx, x::Coord{2}) -> -(a11(x) * ξ[1] * u.∇ϕ[1] + 
                                      a22(x) * ξ[2] * u.∇ϕ[2])
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

    # Build the mass matrix & load by integrating just over the interior domain.
    # Compute the boundary integral for v₀
    initial_mask = create_mask(interior_width / 4, interior_width / 2, total_width)
    M_decay = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ * initial_mask(x)::Float64)
    # load_decay = (u, idx, x) -> -(mollify(a11, x, moll) * ξ[1] * u.∇ϕ[1] + mollify(a22, x, moll) * ξ[2] * u.∇ϕ[2]) * initial_mask(x)::Float64
    load_decay = (u, idx, x) -> -(a11(x) * ξ[1] * u.∇ϕ[1] + a22(x) * ξ[2] * u.∇ϕ[2]) * initial_mask(x)::Float64
    aξ∇v₋₁_decay = assemble_rhs_with_gradient(mesh, load_decay)
    # @show sum(M_decay)
    σ² = (dot(aξ∇v₋₁_decay, v) + dot(v, M_decay * v)) / sum(M_decay)
    σ²s[1] = σ²

    # Plot some stuff
    # if verbose
    #     interior_nodes, interior_elements = interior_subdomain_circle(mesh, total_width, interior_width)
    #     midpoints = map(mesh.elements) do idxs
    #         mean(mesh.nodes[idx] for idx in idxs)
    #     end
    #     tmp = zeros(total_elements)
    #     tmp[interior_elements] .= 1.0
    #     plot_elements["Interior 0"] = tmp
    #     plot_elements["a₁₁"] = a11.(midpoints)
    #     plot_elements["a₂₂"] = a22.(midpoints)
    #     # plot_nodes["a₁₁ mollified"] = mollify.(a11, mesh.nodes, moll)
    #     # plot_nodes["a₂₂ mollified"] = mollify.(a22, mesh.nodes, moll)
    #     plot_nodes["aξ∇v₋₁"] = aξ∇v₋₁
    #     plot_nodes["aξ∇v₋₁_decay"] = aξ∇v₋₁_decay
    #     plot_nodes["mask 0"] = initial_mask.(mesh.nodes)
    #     plot_nodes["v0"] = copy(v)
    # end

    ##
    ## Compute v₁, v₂, ... and compute the other terms of σ²
    ##

    λ = 1.0

    @inbounds for k = 1 : n-1
        # verbose && @show k

        # This is just scaling λ down
        M_int .*= 0.5
        λ *= 2
        
        # Store the previous v
        copy!(v_prev, v)

        # Compute the new v and restore the b.c.
        copy!(v_int, (M_int + A_int) \ (M_int * v_int))
        v[interior] .= v_int

        interior_width = interior_domain_width(n, k)

        let mask = create_mask(interior_width / 4, interior_width / 2, total_width)
            M_decay = assemble_matrix(mesh, (u, v, x::Coord{2}) -> u.ϕ * v.ϕ * mask(x)::Float64)
            σ² += λ * (dot(v_prev, M_decay * v) + dot(v, M_decay * v)) / sum(M_decay)

            # if verbose 
            #     plot_nodes["mask $k"] = mask.(mesh.nodes)
            # end
        end

        σ²s[k + 1] = σ²

        # Plot some things
        # if verbose
        #     interior_nodes, interior_elements = interior_subdomain_circle(mesh, total_width, interior_width)
        #     tmp = zeros(total_elements)
        #     tmp[interior_elements] .= 1.0
        #     plot_elements["Interior $k"] = tmp
        #     plot_nodes["v$k"] = copy(v)
        # end
    end

    # if verbose
    #     println("Saving")
    #     save_to_vtk("results", mesh, plot_nodes, plot_elements)
    # end

    return σ²s
end

function repeat(;times = 10, ref_coarse = 6, nodes_per_cell = 13, θ = 1.0, file = "/mathwork/stoppeh1/results.txt", ε = 0.21)
    xi = (cos(θ), sin(θ))

    sigmas = zeros(times, ref_coarse)

    for i = 1 : times
        sigmas[i, :] .= run(ref_coarse, nodes_per_cell, xi, false, ε)
        @show sigmas[i, :] mean(sigmas[1 : i, end])
    end

    writedlm(file, sigmas)

    return sigmas
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

    @show length(mesh.nodes) length(mesh.elements)

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

    @show dot(x, λ .* (M * x) .+ (A * x)) / 2 - dot(x, b)

    save_to_vtk(filename, mesh, Dict(
        "x_uni" => x
    ), Dict(
        "a11" => a11.(1 : length(mesh.elements)),
        "a22" => a22.(1 : length(mesh.elements))
    ))
end


function test_stretched_grid_boundary_layer(n = 32, α = -1.5, k = 10, λ = 1.0)
    mesh, interior = generate_stretched_grid(n, α, k)

    @show length(mesh.nodes) length(mesh.elements)

    srand(1)
    a11 = checkerboard_elements(mesh, n)
    a22 = checkerboard_elements(mesh, n)

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

    @show dot(x, λ .* (M * x) .+ (A * x)) / 2 - dot(x, b)

    save_to_vtk("stretched", mesh, Dict(
        "x_stretched" => x
    ), Dict(
        "a11" => a11.(1 : length(mesh.elements)),
        "a22" => a22.(1 : length(mesh.elements))
    ))
end

function test_stretched_grid_v1_term(n = 32, α = -1.5, k = 10)
    mesh, interior = generate_stretched_grid(n, α, k)

    ξ = (1 / sqrt(2), 1 / sqrt(2))

    @show length(mesh.nodes) length(mesh.elements)

    srand(1)
    a11 = checkerboard_elements(mesh, n)
    a22 = checkerboard_elements(mesh, n)

    bf_mass = (u, v, x) -> u.ϕ * v.ϕ
    bf_oscillating = (u, v, idx) -> a11(idx) * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx) * u.∇ϕ[2] * v.∇ϕ[2]
    
    M = assemble_matrix(mesh, bf_mass)
    M_int = M[interior, interior]

    A = assemble_matrix_elementwise(mesh, bf_oscillating)
    A_int = A[interior, interior]

    # First rhs via partial integration (note the minus sign)
    load = (u, idx, x) -> -(a11(idx) * ξ[1] * u.∇ϕ[1] + a22(idx) * ξ[2] * u.∇ϕ[2])
    aξ∇v₋₁ = assemble_rhs_with_gradient(mesh, load)

    # Solve (μ - ∇⋅a∇)v₀ = v₋₁ ↔ ∫(μ v₀ϕ + (a∇v₀)⋅∇ϕ)dx = -∫aξ⋅∇ϕ dx ∀ϕ
    # No boundary integral here because of our artificial Dirichlet b.c.
    v = zeros(length(mesh.nodes))
    v_int = (M_int .+ A_int) \ aξ∇v₋₁[interior]
    v[interior] .= v_int

    save_to_vtk("stretched", mesh, Dict(
        "v_stretched" => v,
        "aξ∇v₋₁" => aξ∇v₋₁
    ), Dict(
        "a11" => a11.(1 : length(mesh.elements)),
        "a22" => a22.(1 : length(mesh.elements))
    ))
end
"""
Evaluated a(x) for each element number
"""
function checkerboard_elements(mesh::Mesh{Tri}, m::Int)
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

    return idx::Int -> as[idx]
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

"""
Insert a value into the dictionary if the key does not exist, or remove a
key when the key does exist.
"""
function insert_or_delete!(dict::Dict, key, value)
    if haskey(dict, key)
        delete!(dict, key)
    else
        dict[key] = value
    end

    dict
end

"""
    get_boundary(mesh, elements) -> Dict{Edge,element_id}

Given a subset of element id's, we detect the boundary by iterating of the all
edges of each element. If an edge a → b is new, we add it to the dictionary
edge_to_element[a → b] = element_id. If it is already stored, we delete the
entry from the dictionary.
"""
function get_boundary(mesh::Mesh{Tri}, elements::Vector{Ti}) where {Ti}
    edge_to_element = Dict{Edge{Ti},Ti}()

    @inbounds for idx in elements
        e = mesh.elements[idx]
        insert_or_delete!(edge_to_element, e[1] ↔ e[2], idx)
        insert_or_delete!(edge_to_element, e[1] ↔ e[3], idx)
        insert_or_delete!(edge_to_element, e[2] ↔ e[3], idx)
    end

    edge_to_element
end

"""
Compute the outward pointing unit normal for an edge of a triangle,
Gram-Schmid on the vec from midpoint -> node & node -> node.
"""
function unit_normal(mesh::Mesh{Tri}, edge::Edge, el_idx)
    @inbounds begin
        e = mesh.elements[el_idx]
        mass = (mesh.nodes[e[1]] + mesh.nodes[e[2]] + mesh.nodes[e[3]]) / 3
        n1 = mesh.nodes[edge.to] - mesh.nodes[edge.from]
        n1 /= norm(n1)
        n2 = mesh.nodes[edge.to] - mass
        n2 /= norm(n2)
        n2 = n2 - dot(n1, n2) * n1
        n2 /= norm(n2)
        return n2
    end
end


# Returns the integer size boundary layer
interior_domain_width(n::Int, k::Int) = floor(Int, 2.0 ^ (n - k/2))

"""
Compute the boundary integral for the v₀ term with a simple trapezoidal rule
"""
function compute_boundary_integral(mesh::Mesh{Tri}, boundary, v₀, a11, a22, ξ)
    boundary_integral = 0.0
    @inbounds for (edge, el_idx) in boundary
        from = edge.from
        to = edge.to
        h = norm(mesh.nodes[to] - mesh.nodes[from])
        n = unit_normal(mesh, edge, el_idx)
        constant::Float64 = a11(el_idx) * ξ[1] * n[1] + a22(el_idx) * ξ[2] * n[2]
        boundary_integral += (h/2) * (v₀[from] + v₀[to]) * constant
    end

    return boundary_integral
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
    interior_width = 2^n

    # ∂ is our initial boundary layer
    ∂ = 5

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
    load = (u, idx) -> -(a11(idx) * ξ[1] * u.∇ϕ[1] + a22(idx) * ξ[2] * u.∇ϕ[2])
    aξ∇v₋₁ = assemble_rhs_with_gradient(mesh, load)

    # Solve (μ - ∇⋅a∇)v₀ = v₋₁ ↔ ∫(μ v₀ϕ + (a∇v₀)⋅∇ϕ)dx = -∫aξ⋅∇ϕ dx ∀ϕ
    # No boundary integral here because of our artificial Dirichlet b.c.
    v = zeros(total_nodes)
    v_int = (M_int .+ A_int) \ aξ∇v₋₁[interior]
    v[interior] .= v_int

    # Also define the previous v right now.
    v_prev = zeros(v)
    v_prev_int = zeros(v_int)

    ##
    ## Compute the first term of the sum
    ##

    # We determine the interior subdomain
    interior_nodes, interior_elements = interior_subdomain(mesh, total_width, interior_width)

    # Build the mass matrix & load by integrating just over the interior domain.
    # Compute the boundary integral for v₀
    M_small = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ, interior_elements)
    aξ∇v₋₁_small = assemble_rhs_with_gradient(mesh, load, interior_elements)
    boundary_term::Float64 = compute_boundary_integral(mesh, get_boundary(mesh, interior_elements), v, a11, a22, ξ)

    σ² = (boundary_term + dot(aξ∇v₋₁_small, v) + dot(v, M_small * v)) / interior_width^2

    @show boundary_term

    ###
    # return the middle node value
    ###

    node = findfirst(x -> x[1] == total_width / 2 && x[2] == total_width / 2, mesh.nodes)
    return v[node]

    σ²s[1] = σ²

    # Plot some stuff
    if verbose
        tmp = zeros(total_elements)
        tmp[interior_elements] .= 1.0
        plot_elements["Interior 0"] = tmp
        plot_elements["a₁₁"] = a11.(1 : total_elements)
        plot_elements["a₂₂"] = a22.(1 : total_elements)
        plot_nodes["aξ∇v₋₁"] = aξ∇v₋₁
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
        copy!(v_prev_int, v_int)

        # Compute the new v and restore the b.c.
        copy!(v_int, (M_int + A_int) \ (M_int * v_int))
        v[interior] .= v_int

        interior_width = interior_domain_width(n, k)
        interior_nodes, interior_elements = interior_subdomain(mesh, total_width, interior_width)

        M_small = assemble_matrix(mesh, (u, v, x) -> u.ϕ * v.ϕ, interior_elements)
        σ² += λ * (dot(v_prev, M_small * v) + dot(v, M_small * v)) / interior_width^2

        σ²s[k + 1] = σ²

        # Plot some things
        if verbose
            tmp = zeros(total_elements)
            tmp[interior_elements] .= 1.0
            plot_elements["Interior $k"] = tmp
            plot_nodes["v$k"] = copy(v)
        end
    end

    verbose && save_to_vtk("results", mesh, plot_nodes, plot_elements)

    return σ²s
end

function repeat(times = 10, n = 6, refinements = 3, θ = 0.0)
    srand(1)
    xi = (cos(θ), sin(θ))

    sigmas = zeros(times, n)

    for i = 1 : times
        sigmas[i, :] .= run(n, refinements, xi, false)

        @show sigmas[i, :]
    end

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

function effects_of_h_refinement_2(times = 100, n = 3, refs = 1:5)
    results = []
    θ = 1.0
    xi = (cos(θ), sin(θ))

    for (idx, refinements) in enumerate(refs)

        println("Running with ", refinements, " refinements.")

        # Generate the same `a`
        srand(1)

        vs = zeros(times)

        for i = 1 : times
            vs[i] = run(n, refinements, xi, false)
        end

        push!(results, vs)
    end

    return results
end

function test_fem(n = 4)
    values = Float64[]

    total_width = 2 ^ n

    for m = n : 10
        srand(1)
        mesh, interior = rectangle(2^m, 2^m, total_width, total_width)
        @show length(mesh.nodes)
        a11 = checkerboard_elements(mesh, total_width)
        a22 = checkerboard_elements(mesh, total_width)
        bf_oscillating = (u, v, idx::Int) -> a11(idx)::Float64 * u.∇ϕ[1] * v.∇ϕ[1] + a22(idx)::Float64 * u.∇ϕ[2] * v.∇ϕ[2]
        A = assemble_matrix_elementwise(mesh, bf_oscillating)[interior,interior]
        b = assemble_rhs(mesh, x -> 1.0)[interior]
        x = zeros(length(mesh.nodes))
        x[interior] .= A \ b

        push!(values, x[findfirst(z -> z[1] == total_width/2 && z[2] == total_width/2, mesh.nodes)])

        save_to_vtk("simple_fem_$(lpad(m, 2, 0))", mesh, Dict("x" => copy(x)), Dict("a11" => a11.(1:length(mesh.elements))))
    end

    values

end

function analyze_conv_linf()
    for n = 1 : 5
        results = test_fem(n)
        err = abs.(results[1:end-1] .- results[end])

        @show err[1:end-1] ./ err[2:end]
    end
end
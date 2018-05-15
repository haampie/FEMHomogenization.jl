"""
    generate_stretched_grid(n, α, k)

Stretched grid of domain [0,n]^2 with stretched subgrids at [0,1]^2 with k+1
nodes per dimension per cell. In the inner cell we have nodes at 
    α * x * (x - 0.5) * (x - 1) + x
where x = linspace(0, 1, k)
"""
function generate_stretched_grid(n, α, k)
    xs = linspace(0, 1, k + 1)
    ys = α .* xs .* (xs .- 0.5) .* (xs .- 1.0) .+ xs

    cells_per_dim = k * n
    total_nodes = (cells_per_dim + 1) ^ 2
    total_elements = 2 * cells_per_dim ^ 2
    total_nodes_interior = (cells_per_dim - 1) ^ 2

    nodes = Vector{Coord{2}}(total_nodes)
    elements = Vector{SVector{3,Int}}(total_elements)

    # Nodes
    let node_idx = 1
        @inbounds for j = 1 : cells_per_dim + 1, i = 1 : cells_per_dim + 1
            local_x = float(div(i - 1, k)) + ys[mod(i - 1, k) + 1]
            local_y = float(div(j - 1, k)) + ys[mod(j - 1, k) + 1]
            nodes[node_idx] = (local_x, local_y)
            node_idx += 1
        end
    end

    # Generate the elements    
    let el_idx = 1, node_idx = 1
        @inbounds for j = 1 : cells_per_dim
            for i = 1 : cells_per_dim
                elements[el_idx + 0] = (node_idx, node_idx + cells_per_dim + 1, node_idx + cells_per_dim + 2)
                elements[el_idx + 1] = (node_idx, node_idx + 1, node_idx + cells_per_dim + 2)
                el_idx += 2
                node_idx += 1
            end
            node_idx += 1
        end
    end

    @views interior = collect(reshape(reshape(1 : total_nodes, (cells_per_dim + 1, cells_per_dim + 1))[2 : end - 1, 2 : end - 1], :))

    return Mesh(Tri, nodes, elements), interior
end

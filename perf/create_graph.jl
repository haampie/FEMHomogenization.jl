using BenchmarkTools

import FEM: uniform_square, Mesh
import Base: sort

@inline function sort(a::T, b::T) where {T}
    return a < b ? (a, b) : (b, a)
end

function create_a_graph_1(mesh)
    Nn = length(mesh.nodes)
    edges = [Int[] for i = 1 : Nn]

    @inbounds for triangle in mesh.triangles
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            from, to = sort(triangle[a], triangle[b])
            push!(edges[from], to)
        end
    end

    return edges
end

function create_a_graph_2(mesh)
    Nn = length(mesh.nodes)
    ptr = zeros(Int, Nn + 1)

    # Count edges per node
    @inbounds for triangle in mesh.triangles
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            idx = triangle[a] < triangle[b] ? triangle[a] : triangle[b]
            ptr[idx + 1] += 1
        end
    end

    # Accumulate
    ptr[1] = 1
    @inbounds for i = 1 : Nn
        ptr[i + 1] += ptr[i]
    end

    # Build adjacency list
    adj = Vector{Int}(last(ptr) - 1)
    indices = copy(ptr)

    @inbounds for triangle in mesh.triangles
        for (a, b) in ((1, 2), (1, 3), (2, 3))
            from, to = sort(triangle[a], triangle[b])
            adj[indices[from]] = to
            indices[from] += 1
        end
    end

    ptr, adj
end

function sort_1!(adj::Vector{Vector{Ti}}) where {Ti}
    for edge in adj
        sort!(edge)
    end
end

function sort_2!(ptr::Vector{Ti}, adj::Vector{Ti}) where {Ti}
    @inbounds for i = 1 : length(ptr) - 1
        sort!(view(adj, ptr[i] : ptr[i + 1] - 1))
    end
end

function run()
    mesh, _ = FEM.uniform_square(4)

    adj_vec = create_a_graph_1(mesh)
    ptr, adj = create_a_graph_2(mesh)

    sort_1!(adj_vec)
    sort_2!(ptr, adj)

    idx = 1
    for edge in adj_vec
        for node in edge
            if node != adj[idx]
                throw("Incorrect")
            end
            idx += 1
        end
    end
end

function bench(n::Int = 9)
    mesh, _ = FEM.uniform_square(n)

    b = @benchmark sort_2!(create_a_graph_2($mesh)...)
    a = @benchmark sort_1!(create_a_graph_1($mesh))

    return a, b
end
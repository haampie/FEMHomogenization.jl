"""
    triangle_rotation(nodes_per_side) -> Vector{Int}

Returns the permutation p of one rotation of a triangle with n nodes per side
15                  1
13 14               2  6
10 11 12        →   3  7 10
 6  7  8  9         4  8 11 13
 1  2  3  4  5      5  9 12 14 15

It satisfies the group properties:

    I = 1:sum(1:n)
    p = triangle_rotation(n)
    p != I
    p[p] != I
    p[p][p] == I
"""
function triangle_rotation(nodes_per_side::Int)
    permutation = Vector{Int}(sum(1 : nodes_per_side))

    idx = 1
    @inbounds for layer = 0 : nodes_per_side - 1
        perm = nodes_per_side - layer
        for step = (nodes_per_side - 1 : -1 : layer)
            permutation[idx] = perm
            perm += step
            idx += 1
        end
    end

    permutation
end

"""
    triangle_mirror(nodes_per_side) -> Vector{Int}

Returns the permutation p of mirroring a triangle with n nodes per side
15                 15
13 14              14 13
10 11 12        →  12 11 10
 6  7  8  9         9  8  7  6
 1  2  3  4  5      5  4  3  2  1

It satisfies the group properties:

    I = 1:sum(1:n)
    p = triangle_rotation(n)
    p != I
    p[p] == I
"""
function triangle_mirror(nodes_per_side::Int)
    permutation = Vector{Int}(sum(1 : nodes_per_side))

    idx = 1
    offset = 0
    @inbounds for i = 1 : nodes_per_side
        for j = nodes_per_side : -1 : i
            permutation[idx] = offset + j
            idx += 1
        end
        offset += nodes_per_side - i
    end

    permutation
end
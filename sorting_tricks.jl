"""
Find an efficient way to sort n vectors of size b
"""
function sorting_tricks(n::Int = 1_000, b::Int = 5)
    edges = [[rand(1 : n) for i = 1 : b] for j = 1 : n]

    for neighbours in edges
        sort!(neighbours)
    end

    searchsortedfirst(edges[4], 30)

    edges
end
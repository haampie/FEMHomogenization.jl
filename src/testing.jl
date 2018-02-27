using Base: Order
using StaticArrays

function binary_search_worse(v::AbstractVector, x, lo::Int, hi::Int, o::Ordering)
    lo -= 1
    hi += 1
    @inbounds while lo < hi-1
        m = (lo+hi)>>>1
        if lt(o, v[m], x)
            lo = m
        else
            hi = m
        end
    end
    return hi
end

function binary_search_better(v::AbstractVector, x, lo::Ti, hi::Ti, o::Ordering) where {Ti <: Integer}
    lo -= one(Ti)
    hi += one(Ti)
    @inbounds while lo < hi - one(Ti)
        m = (lo + hi) >>> 1
        if lt(o, v[m], x)
            lo = m
        else
            hi = m
        end
    end
    return hi
end

function run_stuff(n, tets::Vector{SVector{4,Ti}}) where {Ti<:Integer}
    count = zeros(Ti, n)

    @inbounds for tet in tets, i = 1 : 4, j = i + 1 : 4
        from, to = tet[i] < tet[j] ? (tet[i], tet[j]) : (tet[j], tet[i])
        count[from] += to
    end

    return count
end

function run_stuff_2(n, tets::Vector{NTuple{4,Ti}}) where {Ti<:Integer}
    count = zeros(Ti, n)

    @inbounds for tet in tets, i = 1 : 4, j = i + 1 : 4
        if tet[i] < tet[j]
            from = tet[i]
            to = tet[j]
        else
            from = tet[j]
            to = tet[i]
        end

        count[from] += to
    end

    return count
end

using StaticArrays

struct Mesh{Tv,Ti}
    nodes::Vector{SVector{2,Tv}}
    elements::Vector{SVector{3,Ti}}
    element_to_edges::Vector{SVector{3,Ti}}
    edge_to_elements::Vector{SVector{2,Ti}}
    edges::Vector{SVector{2,Ti}}
end

function init_tri(nodes::Vector{SVector{2,Tv}}, elements::Vector{SVector{3,Ti}}) where {Tv,Ti}
    3 * length(elements)
    for element in elements, (from, to) in ((1, 2), (2, 3), (3, 1))
        println(element[from], " to ", element[to])
    end
end

function example()
    nodes = SVector{2,Float64}[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    elements = SVector{3,Int32}[(1, 2, 3), (2, 3, 4)]

    init_tri(nodes, elements)
end

function refine(t1::SVector{2,T}, t2::SVector{2,T}, t3::SVector{2,T}) where {T}

    # New nodes
    n1 = (t1 + t2) / 2
    n2 = (t2 + t3) / 2
    n3 = (t3 + t1) / 2

    # 

    # New edges?
    return n1, n2, n3
end
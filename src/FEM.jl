module FEM
using Calculus
using StaticArrays

export assemble

const Coord = SVector{2, Float64}
const Triangle = SVector{3, Int}

struct Mesh
    n::Int
    nodes::Vector{Coord}
    triangles::Vector{Triangle}
    boundary::Vector{Int}
    interior::Vector{Int}
end

struct Graph
    n_nodes::Int
    n_edges::Int
    edges::Vector{Vector{Int}}
end

include("meshing.jl")
include("basis_functions.jl")
include("assembly.jl")

end
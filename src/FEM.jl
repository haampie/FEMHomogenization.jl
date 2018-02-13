module FEM
using Calculus
using StaticArrays

export assemble

const Coord{d} = SVector{d,Float64}

struct Mesh{d,e}
    n::Int
    nodes::Vector{SVector{d,Float64}}
    elements::Vector{SVector{e,Int}}
    boundary::Vector{Int}
    interior::Vector{Int}
end

struct Graph
    n_nodes::Int
    n_edges::Int
    edges::Vector{Vector{Int}}
end

# include("symbolic_basis_functions.jl")
include("quadrature.jl")
include("elements.jl")
include("meshing.jl")
include("assembly.jl")

end
module FEMHomogenization

using StaticArrays
using WriteVTK

const Coord{d} = SVector{d,Float64}

include("elements.jl")
include("quadrature.jl")
include("meshing.jl")
include("assembly.jl")
include("multigrid.jl")
include("utils.jl")
include("examples.jl")
include("example_homogenization.jl")
include("three_d.jl")
include("simple_hash.jl")

end
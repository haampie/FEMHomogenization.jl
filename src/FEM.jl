module FEM

import Base.sort, Base.isless

using Calculus
using StaticArrays
using WriteVTK

const Coord{d} = SVector{d,Float64}

include("quadrature.jl")
include("elements.jl")
include("meshing.jl")
include("assembly.jl")
include("multigrid.jl")
include("utils.jl")
include("examples.jl")
include("example_homogenization.jl")

end
module FEMHomogenization

using StaticArrays
using WriteVTK

const Coord{d} = SVector{d,Float64}

include("elements.jl")
include("quadrature.jl")
include("meshing.jl")
include("assembly.jl")
include("refinement.jl")

include("tet/elements.jl")
include("tet/quadrature.jl")
include("tet/meshing.jl")
include("tet/boundary.jl")
include("tet/refinement.jl")

include("tri/elements.jl")
include("tri/quadrature.jl")
include("tri/meshing.jl")
include("tri/refinement.jl")

include("utils.jl")
include("examples_2d.jl")
include("examples_3d.jl")
include("example_homogenization.jl")


end

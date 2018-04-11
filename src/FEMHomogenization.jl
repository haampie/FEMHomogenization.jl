module FEMHomogenization

using StaticArrays
using WriteVTK

import Base: size

const Coord{d} = SVector{d,Float64}

include("elements.jl")
include("quadrature.jl")
include("meshing.jl")
include("assembly.jl")
include("assembly_elementwise.jl")
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
# include("examples_2d.jl")
# include("examples_3d.jl")
# include("example_homogenization.jl")
include("matrix_free/actual_implementation.jl")
# include("matrix_free/matrix_free_2d.jl")
# include("matrix_free/generate_stuff.jl")
# include("block_jacobi/block_jacobi.jl")
# include("matrix_free/constant_coefficients.jl")

end

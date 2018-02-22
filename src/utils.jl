import Base.sort

sort(t::NTuple{2,T}) where {T} = t[1] < t[2] ? (t[1], t[2]) : (t[2], t[1])

@inline function sort(a::T, b::T) where {T}
    return a < b ? (a, b) : (b, a)
end

"""
Save a mesh with nodal values as a vtk file that can be used in Paraview.
"""
function save_file(name::String, m::Mesh{Tri}, values::Dict{String,T}) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.triangles]
    vtkfile = vtk_grid(name, node_matrix, triangle_list)

    for (v_name, data) in values
        vtk_point_data(vtkfile, data, v_name)
    end
    
    vtk_save(vtkfile)
end

function save_file(name::String, m::Mesh{Tri}, data::T) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.triangles]
    vtkfile = vtk_grid(name, node_matrix, triangle_list, compress=false)
    vtk_point_data(vtkfile, data, "f")
    vtk_save(vtkfile)
end
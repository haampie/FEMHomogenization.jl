import Base: sort, isless

function isless(a::SVector{2,T}, b::SVector{2,T}) where {T}
    if a[1] != b[1]
        return a[1] < b[1]
    end

    return a[2] < b[2]
end

sort(t::NTuple{2,T}) where {T} = t[1] < t[2] ? (t[1], t[2]) : (t[2], t[1])

@inline function sort(a::T, b::T) where {T}
    return a < b ? (a, b) : (b, a)
end

function binary_search(v::AbstractVector, x, lo::Ti, hi::Ti) where {Ti <: Integer}
    lo -= one(Ti)
    hi += one(Ti)
    @inbounds while lo < hi - one(Ti)
        m = (lo + hi) >>> 1
        if v[m] < x
            lo = m
        else
            hi = m
        end
    end
    return hi
end

"""
Save a mesh with nodal values as a vtk file that can be used in Paraview.
"""
function save_file(name::String, m::Mesh{Tri}, values::Dict{String,T}) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list)

    for (v_name, data) in values
        vtk_point_data(vtkfile, data, v_name)
    end
    
    vtk_save(vtkfile)
end

function save_file(name::String, m::Mesh{Tri}, data::T) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list, compress=false)
    vtk_point_data(vtkfile, data, "f")
    vtk_save(vtkfile)
end

function save_file(name::String, m::Mesh{Tet}, values::Dict{String,T}) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:3, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TETRA, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list, compress=false)
    
    for (v_name, data) in values
        vtk_point_data(vtkfile, data, v_name)
    end

    vtk_save(vtkfile)
end

function save_file(name::String, m::Mesh{Tet}, data::T) where {T <: AbstractArray}
    node_matrix = [x[i] for i = 1:3, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TETRA, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list, compress=false)
    vtk_point_data(vtkfile, data, "f")
    vtk_save(vtkfile)
end

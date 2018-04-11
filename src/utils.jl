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

"""
    binary_search(v, x, lo, hi)

Return the index of the first occurence of x in v[lo:hi]
"""
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

# f(n, σ) = exp.(-σ .* (linspace(-1, 1, n) .^ 2 .+ linspace(-1, 1, n)' .^ 2))
# contourf(conv2(randn(500, 500), f(20, 1.0)))

"""
    complement(sorted_vec, n)

Returns a sorted vector of the numbers 1:n \ sorted_vec. Useful to identify
the interior nodes if the boundary nodes are known.
"""
function complement(boundary_nodes::Vector{Ti}, n::Integer) where {Ti}
    interior_nodes = Vector{Ti}(n - length(boundary_nodes))
    num = 1
    idx = 1

    @inbounds for i in boundary_nodes
        while num < i
            interior_nodes[idx] = num
            num += 1
            idx += 1
        end
        num += 1
    end

    @inbounds for i = num : n
        interior_nodes[idx] = i
        idx += 1
    end

    return interior_nodes
end

"""
Remove duplicate entries from a sorted vector. Resizes the vector as well.
"""
function remove_duplicates!(vec::Vector)
    n = length(vec)

    # Can only be unique
    n ≤ 1 && return vec

    # Discard repeated entries
    slow = 1
    @inbounds for fast = 2 : n
        vec[slow] == vec[fast] && continue
        slow += 1
        vec[slow] = vec[fast]
    end

    # Return the resized vector with unique elements
    return resize!(vec, slow)
end

"""
Pack two UInt32's into a UInt64
"""
@inline pack(a::UInt32, b::UInt32) = (UInt64(a) << 32) + UInt64(b)

"""
Unpack a UInt64 into two UInt32's
"""
@inline unpack(a::UInt64) = UInt32(a >> 32), UInt32(a & 0x00000000ffffffff)


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

function save_to_vtk(name::String, m::Mesh{Tri}, pointdata::Dict{String,S}, celldata::Dict{String,T}) where {S,T}
    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list, compress=false)

    for (v_name, data) in pointdata
        vtk_point_data(vtkfile, data, v_name)
    end

    for (v_name, data) in celldata
        vtk_cell_data(vtkfile, data, v_name)
    end

    vtk_save(vtkfile)
end
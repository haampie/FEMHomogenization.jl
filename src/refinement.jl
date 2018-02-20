import Base.sort, Base.isless

"""
Adjacency list
"""
struct MyGraph{Ti}
    edges::Vector{Vector{Ti}}
    total::Vector{Ti}
end

sort(t::NTuple{2,T}) where {T} = t[1] < t[2] ? (t[1], t[2]) : (t[2], t[1])

@inline function isless(a::SVector{2,T}, b::SVector{2,T}) where {T}
    if a.data[1] < b.data[1]
        return true
    elseif a.data[1] > b.data[1]
        return false
    else
        return a.data[2] < b.data[2]
    end
end

"""
Given an edge between nodes (n1, n2), return
the natural index of the edge.

Costs are O(log b) where b is the connectivity
"""
function edge_index(graph::MyGraph{Ti}, n1::Ti, n2::Ti) where {Ti}
    n1, n2 = sort((n1, n2))
    offset = searchsortedfirst(graph.edges[n1], n2)
    graph.total[n1] + offset - 1
end

"""
Uniformly refine a mesh of triangles: each triangle
is split into four new triangles.
"""
function refine(nodes::Vector{SVector{2,Tv}}, triangles::Vector{SVector{3,Ti}}, graph::MyGraph{Ti}) where {Tv,Ti}
    Nn = length(nodes)
    Nt = length(triangles)
    Ne = graph.total[end] - 1

    # Each edge is split 2, so Nn + Ne is the number of nodes
    fine_nodes = Vector{SVector{2,Tv}}(Nn + Ne)

    # Each triangle is split in 4, so 4Nt triangles
    fine_triangles = Vector{SVector{3,Ti}}(4Nt)

    # Keep the old nodes in place
    copy!(fine_nodes, nodes)
    
    # Add the new ones
    idx = Nn + 1
    for (from, edges) in enumerate(graph.edges), to in edges
        fine_nodes[idx] = (nodes[from] + nodes[to]) / 2
        idx += 1
    end

    # Split each triangle in four smaller ones
    for (i, t) in enumerate(triangles)

        # Index of the nodes on the new edges
        a = edge_index(graph, t[1], t[2]) + Nn
        b = edge_index(graph, t[2], t[3]) + Nn
        c = edge_index(graph, t[3], t[1]) + Nn

        # Split the triangle in 4 pieces
        idx = 4i - 3
        fine_triangles[idx + 0] = SVector(t[1], a, c)
        fine_triangles[idx + 1] = SVector(t[2], a, b)
        fine_triangles[idx + 2] = SVector(t[3], b, c)
        fine_triangles[idx + 3] = SVector(a   , b, c)
    end

    fine_nodes, fine_triangles
end

function contains_sorted(a::Vector{T}, x::T) where {T}
    return searchsortedfirst(a, x) != length(a) + 1
end

"""
Return the interpolation operator
"""
function interpolation_operator(nodes::Vector{SVector{2,Tv}}, graph::MyGraph{Ti}) where {Tv,Ti}
    # Interpolation operator
    Nn = length(graph.edges)
    Ne = graph.total[end] - 1

    nzval = Vector{Tv}(Nn + 2Ne)
    colptr = Vector{Ti}(Nn + Ne + 1)
    rowval = Vector{Ti}(Nn + 2Ne)

    # Nonzero values
    for i = 1 : Nn
        nzval[i] = 1.0
    end

    for i = Nn + 1 : Nn + 2Ne
        nzval[i] = 0.5
    end

    # Column pointer
    for i = 1 : Nn + 1
        colptr[i] = i
    end

    for i = Nn + 2 : Nn + Ne + 1
        colptr[i] = 2 + colptr[i - 1]
    end

    # Row values
    for i = 1 : Nn
        rowval[i] = i
    end

    idx = Nn + 1
    for (from, edges) in enumerate(graph.edges), to in edges
        rowval[idx] = from
        rowval[idx + 1] = to
        idx += 2
    end

    return SparseMatrixCSC(Nn, Nn + Ne, colptr, rowval, nzval)
end

"""
Add a new edge to a graph (this is slow / allocating)
"""
function add_edge!(g::MyGraph{Ti}, from::Ti, to::Ti) where {Ti}
    from, to = sort((from, to))
    push!(g.edges[from], to)
end

"""
Sort the nodes in the adjacency list
"""
function sort_edges!(g::MyGraph)
    for edges in g.edges
        sort!(edges)
    end
end

"""
Find all edges that appear only once in the adjacency list,
because that edge belongs to the boundary
"""
function collect_boundary_nodes!(g::MyGraph{Ti}, boundary_points::Vector{Ti}) where {Ti}
    for (idx, edges) in enumerate(g.edges)
        if collect_boundary_nodes!(edges, boundary_points)
            push!(boundary_points, idx)
        end
    end
end

"""
Find all edges that appear only once in the adjacency list,
because that edge belongs to the boundary
"""
function collect_boundary_nodes!(vec::Vector{Ti}, boundary_points::Vector{Ti}) where {Ti}
    Ne = length(vec)

    if Ne == 0
        return false
    end

    if Ne == 1
        push!(boundary_points, vec[1])
        return true
    end

    return_value = false

    j = 1
    @inbounds while j + 1 ≤ Ne
        if vec[j] == vec[j + 1]
            j += 2
        else
            return_value = true
            push!(boundary_points, vec[j])
            j += 1
        end
    end

    if j == Ne
        push!(boundary_points, vec[j])
        return_value = true
    end

    return return_value
end

"""
Remove all duplicate edges
"""
function remove_duplicates!(g::MyGraph)
    for adj in g.edges
        remove_duplicates!(adj)
    end

    g
end

"""
Remove duplicate entries from a vector.
Resizes / shrinks the vector as well.
"""
function remove_duplicates!(vec::Vector)
    length(vec) ≤ 1 && return vec

    j = 1
    @inbounds for i = 2 : length(vec)
        if vec[i] != vec[j]
            j += 1
            vec[j] = vec[i]
        end
    end

    resize!(vec, j)

    vec
end

"""
Convert a mesh of nodes + triangles to a graph
"""
function to_graph(nodes::Vector{SVector{2,Tv}}, triangles::Vector{SVector{3,Ti}}) where {Tv,Ti}
    n = length(nodes)
    edges = [sizehint!(Ti[], 5) for i = 1 : n]
    total = ones(Ti, n + 1)
    g = MyGraph(edges, total)
    
    for triangle in triangles
        add_edge!(g, triangle[1], triangle[2])
        add_edge!(g, triangle[2], triangle[3])
        add_edge!(g, triangle[3], triangle[1])
    end

    # Collect the boundary nodes
    boundary_points = Vector{Ti}();
    sort_edges!(g)
    collect_boundary_nodes!(g, boundary_points)
    remove_duplicates!(g)
    sort!(boundary_points)
    remove_duplicates!(boundary_points)

    # TODO: refactor this
    interior_points = Vector{Ti}(n - length(boundary_points))
    num, idx = 1, 1
    @inbounds for i in boundary_points
        while num < i
            interior_points[idx] = num
            num += 1
            idx += 1
        end
        num += 1
    end

    @inbounds for i = num : n
        interior_points[idx] = i
        idx += 1
    end

    @inbounds for i = 1 : n
        g.total[i + 1] = g.total[i] + length(g.edges[i])
    end

    return g, boundary_points, interior_points
end

"""
Returns the affine map from the blueprint triangle to the given
triangle.

TODO: dispatch on element type (Triangle)
"""
function affine_map(nodes::Vector{SVector{2,Tv}}, el::SVector{3,Ti}) where {Tv,Ti}
    p1, p2, p3 = nodes[el[1]], nodes[el[2]], nodes[el[3]]
    return [p2 - p1 p3 - p1], p1
end

"""
Build a sparse coefficient matrix for a given mesh and bilinear form
"""
function assemble_matrix(nodes::Vector{SVector{2,Tv}}, elements::Vector{SVector{3,Ti}}, bilinear_form) where {Ti,Tv}
    # Quadrature scheme
    ws, xs = quadrature_rule(Tri3)
    ϕs, ∇ϕs = get_basis_funcs(Tri)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nt = length(elements)
    Nn = length(nodes)
    Nq = length(xs)
    dof = 3
    
    # Some stuff
    is = Vector{Ti}(dof * dof * Nt)
    js = Vector{Ti}(dof * dof * Nt)
    vs = Vector{Tv}(dof * dof * Nt)

    A_local = zeros(dof, dof)

    idx = 1

    # Loop over all elements & compute local system matrix
    for element in elements
        jac, shift = affine_map(nodes, element)
        invJac = inv(jac')
        detJac = abs(det(jac))

        # Transform the gradient
        @inbounds for point in basis, f in point
            A_mul_B!(f.∇ϕ, invJac, f.grad)
        end

        # Reset local matrix
        fill!(A_local, zero(Tv))

        # For each quad point
        @inbounds for k = 1 : Nq
            x = jac * xs[k] + shift

            for i = 1:dof, j = 1:dof
                A_local[i,j] += ws[k] * bilinear_form(basis[k][i], basis[k][j], x)
            end
        end

        # Copy stuff
        @inbounds for i = 1:dof, j = 1:dof
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * detJac
            idx += 1
        end
    end

    return dropzeros!(sparse(is, js, vs, Nn, Nn))
end

"""
Build a right-hand side
"""
function assemble_rhs(nodes::Vector{SVector{2,Tv}}, elements::Vector{SVector{3,Ti}}, f) where {Ti,Tv}
    # Quadrature scheme
    ws, xs = quadrature_rule(Tri3)
    ϕs, ∇ϕs = get_basis_funcs(Tri)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nn = length(nodes)
    Nq = length(xs)
    dof = 3
    
    # Some stuff
    b = zeros(Nn)
    b_local = zeros(dof)

    # Loop over all elements & compute local system matrix
    for element in elements
        jac, shift = affine_map(nodes, element)
        invJac = inv(jac')
        detJac = abs(det(jac))

        # Reset local matrix
        fill!(b_local, zero(Tv))

        # For each quad point
        @inbounds for k = 1 : Nq
            x = jac * xs[k] + shift

            for i = 1:dof
                b_local[i] += ws[k] * f(x) * basis[k][i].ϕ
            end
        end

        # Copy stuff
        @inbounds for i = 1:dof
            b[element[i]] += b_local[i] * detJac
        end
    end

    return b
end

"""
Save a mesh with nodal values as a vtk file that can be used in Paraview.
"""
function save_file(name::String, nodes, triangles, values)
    node_matrix = [x[i] for i = 1:2, x in nodes]
    triangle_stuff = [MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in triangles]
    vtkfile = vtk_grid(name, node_matrix, triangle_stuff)
    vtk_point_data(vtkfile, values, "f")
    vtk_save(vtkfile)
end
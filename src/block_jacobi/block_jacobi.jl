using IterativeSolvers

function block_jaboci(name::String, ref::Int)
    m, _, int = unit_square(ref)
    bilinear_form = (u, v, x) -> 8 * u.ϕ * v.ϕ + (1.0 + sqrt(x[1])) * u.∇ϕ[1] * v.∇ϕ[1] + 
                                                 (1.0 + sqrt(x[2])) * u.∇ϕ[2] * v.∇ϕ[2]
    
    A = assemble_matrix(m, bilinear_form)
    b = assemble_rhs(m, x -> 1.0)
    
    # Solve the problem with a direct method
    A_int = A[int,int]
    x = zeros(b)
    b_int = b[int]
    x_int = A_int \ b_int
    x[int] .= x_int

    ###
    # Do standard jacobi
    myx = rand(length(int))
    my_iterable = IterativeSolvers.jacobi_iterable(myx, A_int, b_int, maxiter = 50)
    @time for item in my_iterable
        @show norm(x_int - myx)
    end

    ###

    # Split the domain in 2 parts
    half = div(length(m.elements), 2)

    # Find the A11 block
    interior_set = IntSet(int)
    fst = IntSet(reshape([element[i] for element in m.elements[1:half], i = 1:3], :))
    fst = intersect!(fst, interior_set)
    snd = setdiff(interior_set, fst)
    fst = collect(fst)
    snd = collect(snd)

    Ã = [A[fst,fst] A[fst,snd]; A[snd,fst] A[snd,snd]]
    D = blkdiag(A[fst,fst], A[snd,snd])
    R = Ã - D
    b̃ = [b[fst];b[snd]]
    
    x_ref = [x[fst];x[snd]]
    x̃ = rand(size(b̃))

    @time for i = 1 : 50
        @show norm(x_ref - x̃)
        x̃ = D \ (b̃ - R * x̃)
    end

    other_x = zeros(b)
    other_x[fst] = x̃[1:length(fst)]
    other_x[snd] = x̃[length(fst)+1:end]

    node_matrix = [x[i] for i = 1:2, x in m.nodes]
    triangle_list = MeshCell[MeshCell(VTKCellTypes.VTK_TRIANGLE, Vector(t)) for t in m.elements]
    vtkfile = vtk_grid(name, node_matrix, triangle_list)
    # vtk_cell_data(vtkfile, data, "cells")
    vtk_point_data(vtkfile, abs.(x - other_x), "e")
    vtk_save(vtkfile)
end
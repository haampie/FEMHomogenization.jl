function refined_reference_tet(refs::Int = 4)
    nodes = SVector{3,Float64}[(0, 0, 0),(1, 0, 0),(0, 1, 0),(0, 0, 1)]
    tets = SVector{4,UInt32}[(1,2,3,4)]
    mesh = Mesh(Tet, nodes, tets)
    
    for i = 1 : refs
        mesh = refine(mesh, to_graph(mesh))
    end

    validate_orientation(mesh)

    # Define an ordering of the nodes on the faces with identical orientation
    faces = ((1, 2, 4), (3, 1, 4), (2, 3, 4), (2, 1, 3))

    face_nodes = map(faces) do face
        i, j, k = face
        # Compute the normal of the face
        origin = nodes[i]
        side_a = nodes[j] - origin
        side_b = nodes[k] - origin
        n = cross(side_a, side_b)
        n /= norm(n)

        # Find all the nodes on the face
        node_per_face = find(x -> abs(dot(n, x - origin)) < .001, mesh.nodes)

        fst_side_unit = side_a / norm(side_a)
        snd_side_unit = side_b / norm(side_b)

        # Sort the nodes first along b then along a (probably watch out for rounding errors)
        sort!(node_per_face, lt = (i1, i2) -> begin
            n1 = mesh.nodes[i1] - origin
            n2 = mesh.nodes[i2] - origin
            res = cmp(abs(dot(n1, snd_side_unit)), abs(dot(n2, snd_side_unit)))
            res == -1 && return true
            res == 1 && return false
            return abs(dot(n1, fst_side_unit)) < abs(dot(n2, fst_side_unit))
        end)

        node_per_face
    end

    @show length.(face_nodes)

    x1 = zeros(length(mesh.nodes)); x1[face_nodes[1]] .= (1 : length(face_nodes[1]));
    x2 = zeros(length(mesh.nodes)); x2[face_nodes[2]] .= (1 : length(face_nodes[2]));
    x3 = zeros(length(mesh.nodes)); x3[face_nodes[3]] .= (1 : length(face_nodes[3]));
    x4 = zeros(length(mesh.nodes)); x4[face_nodes[4]] .= (1 : length(face_nodes[4]));

    save_to_vtk("refinement_orientation", mesh, Dict(
        "x1" => x1,
        "x2" => x2,
        "x3" => x3,
        "x4" => x4
    ), Dict(
        "cells" => Vector{Float64}(1 : length(mesh.elements))
    ))
end

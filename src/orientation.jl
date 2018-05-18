function validate_orientation(mesh)
    for element in mesh.elements
        jac, shift = affine_map(mesh, element)
        invJac = inv(jac')
        detJac = det(jac)
        if detJac ≤ 0
            @show element
        end
    end
end

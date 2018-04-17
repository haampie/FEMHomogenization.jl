function assemble_matrix_elementwise(m::Mesh{Te,Tv,Ti}, bilinear_form; quad::Type{<:QuadRule} = default_quadrature(Te)) where {Te,Tv,Ti}
    # Quadrature scheme
    ϕs, ∇ϕs = get_basis_funcs(Te)
    ws, xs = quadrature_rule(quad)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nt = length(m.elements)
    Nn = length(m.nodes)
    Nq = length(xs)
    
    # This is for now hard-coded...
    dof = length(m.elements[1])
    
    # We'll pre-allocate the triples (is, js, vs) that are used to
    # construct the sparse matrix A
    is = Vector{Int64}(dof * dof * Nt)
    js = Vector{Int64}(dof * dof * Nt)
    vs = Vector{Tv}(dof * dof * Nt)
    
    # The local system matrix
    A_local = zeros(dof, dof)

    idx = 1

    # Loop over all elements & compute the local system matrix
    for (el_number, element) in enumerate(m.elements)
        jac, shift = affine_map(m, element)
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
                A_local[i,j] += ws[k] * bilinear_form(basis[k][i], basis[k][j], el_number)
            end
        end

        # Copy the local matrix over to the global one
        @inbounds for i = 1:dof, j = 1:dof
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * detJac
            idx += 1
        end
    end

    # Build the sparse matrix
    return dropzeros!(sparse(is, js, vs, Nn, Nn))
end

function assemble_rhs_with_gradient(m::Mesh{Te,Tv,Ti}, f; quad::Type{<:QuadRule} = default_quadrature(Te)) where {Te,Tv,Ti}
    # Quadrature scheme
    ϕs, ∇ϕs = get_basis_funcs(Te)
    ws, xs = quadrature_rule(quad)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nn = length(m.nodes)
    Nq = length(xs)
    
    # This is for now hard-coded...
    dof = length(m.elements[1])
    
    # Global rhs
    b = zeros(Nn)

    # Local rhs
    b_local = zeros(dof)

    # Loop over all elements & compute the local rhs
    for (e_idx, element) in enumerate(m.elements)
        jac, shift = affine_map(m, element)
        invJac = inv(jac')
        detJac = abs(det(jac))

        # Reset local rhs
        fill!(b_local, zero(Tv))

        @inbounds for point in basis, p in point
            A_mul_B!(p.∇ϕ, invJac, p.grad)
        end

        # For each quad point
        @inbounds for k = 1 : Nq
            x = jac * xs[k] + shift

            for i = 1:dof
                b_local[i] += ws[k] * f(basis[k][i], e_idx)
            end
        end

        # Copy the local rhs over to the global one
        @inbounds for i = 1:dof
            b[element[i]] += b_local[i] * detJac
        end
    end

    return b
end

"""
Assemble the operator just on provided elements (but retain the numbering)
"""
function assemble_matrix(m::Mesh{Te,Tv,Ti}, bilinear_form, elements::Vector{Ti}; quad::Type{<:QuadRule} = default_quadrature(Te)) where {Te,Tv,Ti}
    # Quadrature scheme
    ϕs, ∇ϕs = get_basis_funcs(Te)
    ws, xs = quadrature_rule(quad)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nt = length(elements)
    Nn = length(m.nodes)
    Nq = length(xs)
    
    # This is for now hard-coded...
    dof = length(m.elements[1])
    
    # We'll pre-allocate the triples (is, js, vs) that are used to
    # construct the sparse matrix A
    is = Vector{Int64}(dof * dof * Nt)
    js = Vector{Int64}(dof * dof * Nt)
    vs = Vector{Tv}(dof * dof * Nt)
    
    # The local system matrix
    A_local = zeros(dof, dof)

    idx = 1

    # Loop over all elements & compute the local system matrix
    for (el_idx, element) in enumerate(m.elements)

        # If the current element is not listed, skip this guy.
        if isempty(searchsorted(elements, el_idx))
            continue
        end

        jac, shift = affine_map(m, element)
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

        # Copy the local matrix over to the global one
        @inbounds for i = 1:dof, j = 1:dof
            is[idx] = element[i]
            js[idx] = element[j]
            vs[idx] = A_local[i,j] * detJac
            idx += 1
        end
    end

    # Build the sparse matrix
    return dropzeros!(sparse(view(is, 1:idx-1), view(js, 1:idx-1), view(vs, 1:idx-1), Nn, Nn))
end

function assemble_rhs_with_gradient(m::Mesh{Te,Tv,Ti}, f, elements; quad::Type{<:QuadRule} = default_quadrature(Te)) where {Te,Tv,Ti}
    # Quadrature scheme
    ϕs, ∇ϕs = get_basis_funcs(Te)
    ws, xs = quadrature_rule(quad)
    basis = evaluate_basis_funcs(ϕs, ∇ϕs, xs)

    Nn = length(m.nodes)
    Nq = length(xs)
    
    # This is for now hard-coded...
    dof = length(m.elements[1])
    
    # Global rhs
    b = zeros(Nn)

    # Local rhs
    b_local = zeros(dof)

    # Loop over all elements & compute the local rhs
    for (e_idx, element) in enumerate(m.elements)

        # If the current element is not listed, skip this guy.
        if isempty(searchsorted(elements, e_idx))
            continue
        end

        jac, shift = affine_map(m, element)
        invJac = inv(jac')
        detJac = abs(det(jac))

        # Reset local rhs
        fill!(b_local, zero(Tv))

        @inbounds for point in basis, p in point
            A_mul_B!(p.∇ϕ, invJac, p.grad)
        end

        # For each quad point
        @inbounds for k = 1 : Nq
            x = jac * xs[k] + shift

            for i = 1:dof
                b_local[i] += ws[k] * f(basis[k][i], e_idx)
            end
        end

        # Copy the local rhs over to the global one
        @inbounds for i = 1:dof
            b[element[i]] += b_local[i] * detJac
        end
    end

    return b
end

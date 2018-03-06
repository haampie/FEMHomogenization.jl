struct MyQuad{A,B} end

function generate_quad_stuff(::Type{MyQuad{Tri,Tri3}})
    ϕs, ∇ϕs = get_basis_funcs(Tri)
    ws, xs = quadrature_rule(Tri3)

    exprs = Expr[]
    A = :(SMatrix{3,3,Float64,9}())
    values = :(SVector{6,Float64}())

    # Quadrature stuff
    for j = 1 : 3, i = j : 3
        sum = Expr(:call, :+)
        for (w, x) in zip(ws, xs)
            u = ϕs[i](x)
            v = ϕs[j](x)
            ∇u = ∇ϕs[i](x)
            ∇v = ∇ϕs[j](x)
            push!(sum.args, :($w * (dot(J * SVector{2,Float64}($(∇u[1]), $(∇u[2])), J * SVector{2,Float64}($(∇v[1]), $(∇v[2]))) + $u * $v)))
        end
        push!(values.args, sum)
    end

    push!(exprs, :(values = $values))

    indices = zeros(Int, 3, 3)
    idx = 1
    for i = 1 : 3, j = i : 3
        indices[i,j] = idx
        indices[j,i] = idx
        idx += 1
    end

    for i = 1 : 3, j = 1 : 3
        push!(A.args, :(values[$(indices[i,j])]))
    end

    return quote
        @inbounds begin
            $(exprs...)

            A = $A
            
            xs = SVector{3,Float64}(x[element[1]], x[element[2]], x[element[3]])
            ys = A * xs
            y[element[1]] += ys[1] * detJ
            y[element[2]] += ys[2] * detJ
            y[element[3]] += ys[3] * detJ
        end
    end
end

@generated function update_stuff(m::Mesh{Tri,Tv,Ti}, element, quad::MyQuad{A,B}, y::AbstractVector, x::AbstractVector) where {Tv,Ti,A,B}
    quad_stuff = generate_quad_stuff(quad)

    return quote
        $(Expr(:meta, :inline))

        jac, shift = affine_map(m, element)
        J = inv(jac')
        detJ = abs(det(jac))

        @inbounds begin
            $quad_stuff
        end
    end
end

function do_the_assembly(m::Mesh{Tri,Tv,Ti}, quad::MyQuad{A,B}, y::AbstractVector, x::AbstractVector) where {Tv,Ti,A,B}
    # Quadrature scheme
    Nt = length(m.elements)
    Nn = length(m.nodes)
    Nq = 3
        
    fill!(y, zero(Tv))

    # Loop over all elements & compute the local system matrix
    Threads.@threads for element in m.elements
        update_stuff(m, element, quad, y, x)
    end

    y
end

function generate_the_func(ref = 6)
    mesh, graph, interior = unit_square(ref)

    x = ones(length(mesh.nodes))
    y = similar(x)

    do_the_assembly(mesh, MyQuad{Tri,Tri3}(), y, x)

    @time do_the_assembly(mesh, MyQuad{Tri,Tri3}(), y, x)
end

function bench_implementations(refinements)
    mesh, _, _ = unit_square(refinements)
    x = ones(length(mesh.nodes))
    y = similar(x)

    A = assemble_matrix(mesh, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ) + u.ϕ * v.ϕ)

    a = A * x
    b = do_the_assembly(mesh, MyQuad{Tri,Tri3}(), similar(x), x)
    c = multiply_mass_matrix(mesh, identity, similar(x), x)

    @show a ≈ b b ≈ c a ≈ c

    thd = @benchmark A_mul_B!($y, $A, $x)
    fst = @benchmark do_the_assembly($mesh, MyQuad{Tri,Tri3}(), $y, $x)
    snd = @benchmark multiply_mass_matrix($mesh, identity, $y, $x)

    fst, snd, thd
end

function profile_generated_thing(refinements)
    mesh, _, _ = unit_square(refinements)
    x = ones(length(mesh.nodes))
    y = similar(x)

    do_the_assembly(mesh, MyQuad{Tri,Tri3}(), y, x)
    @profile do_the_assembly(mesh, MyQuad{Tri,Tri3}(), y, x)
end

function bench_generated_thing(refinements)
    mesh, _, _ = unit_square(refinements)
    x = ones(length(mesh.nodes))
    y = similar(x)

    do_the_assembly(mesh, MyQuad{Tri,Tri3}(), y, x)
    @benchmark do_the_assembly($mesh, MyQuad{Tri,Tri3}(), $y, $x)
end
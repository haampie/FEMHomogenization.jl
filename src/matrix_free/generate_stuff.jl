struct MyQuad{E<:MeshElement,B<:QuadRule,T<:Number} end

function generate_quad_stuff(::Type{MyQuad{E,Q,T}}) where {E,Q,T}
    ϕs, ∇ϕs = get_basis_funcs(E)
    ws, xs = quadrature_rule(Q)

    d = dim(E)
    n = size(E)

    A = :(SMatrix{$n,$n,$T,$(n*n)}())

    # Quadrature stuff
    for j = 1 : n, i = 1 : n
        sum = Expr(:call, :+)
        for (w, x) in zip(ws, xs)
            ∇u = :(SVector{$d,$T}($(∇ϕs[i](x)...)))
            ∇v = :(SVector{$d,$T}($(∇ϕs[j](x)...)))
            push!(sum.args, :($w * dot(J * $∇u, J * $∇v)))
        end
        push!(A.args, sum)
    end

    # Fix the hard coded constants here.
    return quote
        @inbounds begin
            A = $A
            xs = SVector{$n,$T}(x[element[1]], x[element[2]], x[element[3]])
            ys = A * xs
            y[element[1]] += ys[1] * detJ
            y[element[2]] += ys[2] * detJ
            y[element[3]] += ys[3] * detJ
        end
    end
end

@generated function update_stuff(m::Mesh{Tri,Tv,Ti}, element, quad::MyQuad{A,B}, y::AbstractVector, x::AbstractVector) where {Tv,Ti,A,B}
    quad_stuff = generate_quad_stuff(quad)

    code = quote
        $(Expr(:meta, :inline))

        jac, shift = affine_map(m, element)
        J = inv(jac')
        detJ = abs(det(jac))

        @inbounds begin
            $quad_stuff
        end
    end

    println(code)

    return code
end

function do_the_assembly(m::Mesh{Tri,Tv,Ti}, quad::MyQuad{A,B}, y::AbstractVector, x::AbstractVector) where {Tv,Ti,A,B}
    # Quadrature scheme
    Nt = length(m.elements)
    Nn = length(m.nodes)
    Nq = 3
        
    fill!(y, zero(Tv))

    # Loop over all elements & compute the local system matrix
    for element in m.elements
        update_stuff(m, element, quad, y, x)
    end

    y
end

function instance_of_assembly(m::Mesh{Tri,Float64,Ti}, y, x) where {Ti}
    Nt = length(m.elements)
    Nn = length(m.nodes)
    Nq = 3
        
    fill!(y, 0.0)

    # Loop over all elements & compute the local system matrix
    @inbounds for element in m.elements
        e1 = element[1]
        e2 = element[2]
        e3 = element[3]
        p1 = m.nodes[e1]
        p2 = m.nodes[e2]
        p3 = m.nodes[e3]
        jac = [p2 - p1 p3 - p1]
        J = inv(jac')
        detJ = abs(det(jac))
        A = SMatrix{3, 3, Float64, 9}(+(1 * dot(J * SVector{2, Float64}(-1.0, -1.0), J * SVector{2, Float64}(-1.0, -1.0))), +(1 * dot(J * SVector{2, Float64}(1.0, 0.0), J * SVector{2, Float64}(-1.0, -1.0))), +(1 * dot(J * SVector{2, Float64}(0.0, 1.0), J * SVector{2, Float64}(-1.0, -1.0))), +(1 * dot(J * SVector{2, Float64}(-1.0, -1.0), J * SVector{2, Float64}(1.0, 0.0))), +(1 * dot(J * SVector{2, Float64}(1.0, 0.0), J * SVector{2, Float64}(1.0, 0.0))), +(1 * dot(J * SVector{2, Float64}(0.0, 1.0), J * SVector{2, Float64}(1.0, 0.0))), +(1 * dot(J * SVector{2, Float64}(-1.0, -1.0), J * SVector{2, Float64}(0.0, 1.0))), +(1 * dot(J * SVector{2, Float64}(1.0, 0.0), J * SVector{2, Float64}(0.0, 1.0))), +(1 * dot(J * SVector{2, Float64}(0.0, 1.0), J * SVector{2, Float64}(0.0, 1.0))))
        xs = SVector{3, Float64}(x[e1], x[e2], x[e3])
        ys = detJ * (A * xs)

        y[e1] += ys[1]
        y[e2] += ys[2]
        y[e3] += ys[3]
    end

    y
end

function generate_the_func(ref = 6, Q::Type{<:QuadRule} = Tri1)
    mesh, graph, interior = unit_square(ref)

    x = ones(length(mesh.nodes))
    y = similar(x)

    do_the_assembly(mesh, MyQuad{Tri,Q,Float64}(), y, x)

    @time do_the_assembly(mesh, MyQuad{Tri,Q,Float64}(), y, x)
end

function bench_implementations(refinements)
    mesh, _, _ = unit_square(refinements)
    x = ones(length(mesh.nodes))
    y = similar(x)

    A = assemble_matrix(mesh, (u, v, x) -> dot(u.∇ϕ, v.∇ϕ))

    a = A * x
    b = do_the_assembly(mesh, MyQuad{Tri,Tri1,Float64}(), similar(x), x)
    c = multiply_mass_matrix(mesh, identity, similar(x), x)

    @show a ≈ b b ≈ c a ≈ c

    thd = @benchmark A_mul_B!($y, $A, $x)
    fst = @benchmark do_the_assembly($mesh, MyQuad{Tri,Tri1,Float64}(), $y, $x)
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

function test_stuff(refinements = 3)
    mesh, _, _ = unit_square(refinements)
    x = ones(length(mesh.nodes))
    y = similar(x)
    @profile instance_of_assembly(mesh, y, x)
end
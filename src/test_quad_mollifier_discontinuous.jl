a(x) = x > 1.0 ? 9.0 : 1.0

function integrate(f, lo, hi, qxs, qws)
    c = (hi - lo) / 2
    d = (hi + lo) / 2
    total = 0.0
    @inbounds for i = 1 : length(qxs)
        total += qws[i] * f(c * qxs[i] + d)
    end
    total * c / 2
end

function integrate_fenced(f, lo, hi, qxs, qws, fences)
    nodes = linspace(lo, hi, fences + 1)
    total = 0.0
    for i = 1 : fences
        total += integrate(f, nodes[i], nodes[i+1], qxs, qws)
    end
    total
end

function test_quad_mollifier_discontinuous_a(x)
    # Try a naive integration rule
    qws = gauss_legendre_weights(Val{16})
    qxs = gauss_legendre_coords(Val{16}) # (-1, 1)

    return integrate(a, x - 0.5, x + 0.5, qxs, qws)
end

function test_quad_mollifier_discontinuous_b(x)
    # Try a naive integration rule
    qws = gauss_legendre_weights(Val{4})
    qxs = gauss_legendre_coords(Val{4})

    if abs(x - 1.0) < 0.5
        return integrate(a, x - 0.5, 1.0, qxs, qws) + integrate(a, 1.0, x + 0.5, qxs, qws)
    else
        return integrate(a, x - 0.5, x + 0.5, qxs, qws)
    end
end

function integrate_mollifier_1d(fences = 10, rule::Type = Val{4})
    # Compute ∫ϕdx to high accuracy.
    qws = gauss_legendre_weights(rule)
    qxs = gauss_legendre_coords(rule)
    return 2*integrate_fenced(x -> mollifier_value(Coord{1}(x)), 0.0, 1.0, qxs, qws, fences)
end

# function mollify_something(x, rule::Type = Val{4}, ε = 1/2, fences = 4)
#     # Compute ∫ϕdx to high accuracy.
#     qws = gauss_legendre_weights(rule)
#     qxs = gauss_legendre_coords(rule)
#     I = integrate_mollifier_1d()
#     from, to = x - ε, x + ε
#     f(y) = a(y) * mollifier_value(Coord{1}((y - x) / ε)) / ε

#     nodes = linspace(from, to, fences + 1)
#     for i = 1 : fences
#         integrate_fenced(f, from, 1.0, qxs, qws, div(fences, 2)) + 
#         integrate_fenced(f, 1.0, to, qxs, qws, div(fences, 2))
#     else
#         integrate_fenced(f, from, to, qxs, qws, fences)
#     end

#     return value / I
# end
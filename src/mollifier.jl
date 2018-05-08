gauss_legendre_weights(::Type{Val{16}}) = (
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541
)

gauss_legendre_coords(::Type{Val{16}}) = (
	-0.9894009349916499,
	-0.9445750230732326,
	-0.8656312023878318,
	-0.7554044083550030,
	-0.6178762444026438,
	-0.4580167776572274,
	-0.2816035507792589,
    -0.0950125098376374,
	0.0950125098376374,
	0.2816035507792589,
	0.4580167776572274,
	0.6178762444026438,
	0.7554044083550030,
	0.8656312023878318,
	0.9445750230732326,
	0.9894009349916499
)

gauss_legendre_weights(::Type{Val{10}}) = (
    0.0666713443086881,
    0.1494513491505806,
    0.2190863625159820,
    0.2692667193099963,
    0.2955242247147529,
    0.2955242247147529,
    0.2692667193099963,
    0.2190863625159820,
    0.1494513491505806,
    0.0666713443086881
)

gauss_legendre_coords(::Type{Val{10}}) = (
	-0.9739065285171717,
    -0.8650633666889845,
    -0.6794095682990244,
    -0.4333953941292472,
    -0.1488743389816312,
    0.1488743389816312,
    0.4333953941292472,
    0.6794095682990244,
    0.8650633666889845,
    0.9739065285171717
)

"""
Evaluate the mollifier in `x`
"""
function mollifier_value(x::SVector)
    dist = dot(x, x)
    dist ≥ 1 ? 0.0 : exp(-1.0 / (1.0 - dist))
end

struct Mollifier{dim,X}
    f_local::Array{Float64,dim}
    mask::Array{Float64,dim}
    xs::X
    ε::Float64
end

"""
    Mollifier{2}(ε = 0.5, quad = Val{10})

Construct a 2D mollifier of size [-ε, ε]². We evaluate the mask wᵢ * φᵢ in each
Gauss-Legendre quad point in advance, and pre-allocate a vector where we can
store evaluated values of a function.
"""
function Mollifier{T}(ε = 0.5, quad::Type = Val{10}) where {T}
    xs = gauss_legendre_coords(quad)
    ws = SVector(gauss_legendre_weights(quad))
    W = ws * ws'
    mask = [mollifier_value(Coord{2}(xs[i], xs[j])) for i = 1 : length(xs), j = 1 : length(xs)]
    mask .*= W
    mask ./= sum(mask)
    Mollifier{2,typeof(xs)}(similar(mask), mask, xs, ε)
end

"""
    mollify(f, x, m) -> Float64

Given a function `f`, a coordinate `x` and a mollifier `m`, find the mollified
value f_ε(x) = ∫f(y)ϕ_ε(y)dy.
"""
@inline function mollify(f, x::Coord{2}, m::Mollifier{2})
    @inbounds for j = 1 : length(m.xs)
        y_val = x[2] + m.ε * m.xs[j]
        for i = 1 : length(m.xs)
            x_val = x[1] + m.ε * m.xs[i]
            m.f_local[j, i] = f(Coord{2}(x_val, y_val))
        end
    end

    vecdot(m.f_local, m.mask)
end

function construct_checkerboard(coarse_cells)
    coarse = coarse_cells + 2
    A = [rand(Bool) ? 1.0 : 9.0 for i = 1 : coarse, j = 1 : coarse]

    return function a(x::Coord{2})
        Base.@_propagate_inbounds_meta
        x_int = unsafe_trunc(Int, x[1]) + 2
        y_int = unsafe_trunc(Int, x[2]) + 2
        A[y_int, x_int]
    end
end


# gauss_legendre_weights() = (
#     0.0666713443086881,
#     0.1494513491505806,
#     0.2190863625159820,
#     0.2692667193099963,
#     0.2955242247147529,
#     0.2955242247147529,
#     0.2692667193099963,
#     0.2190863625159820,
#     0.1494513491505806,
#     0.0666713443086881
# )

# gauss_legendre_coords() = (
#     -0.9739065285171717,
#     -0.8650633666889845,
#     -0.6794095682990244,
#     -0.4333953941292472,
#     -0.1488743389816312,
#      0.1488743389816312,
#      0.4333953941292472,
#      0.6794095682990244,
#      0.8650633666889845,
#      0.9739065285171717
# )

gauss_legendre_weights() = (
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

gauss_legendre_coords() = (
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

function mollifier(x::Coord{2})
    dist = x[1] * x[1] + x[2] * x[2]
    dist ≥ 1 ? 0.0 : exp(-1.0 / (1.0 - dist))
end

"""
Evaluates wᵢ * φᵢ in each 2D Gauss-Legendre quad point in [-1, 1]²
"""
function construct_mask()
    xs = gauss_legendre_coords()
    ws = gauss_legendre_weights()
    W = [ws...] * [ws...]'
    X = [mollifier(Coord{2}(xs[i], xs[j])) for i = 1 : length(xs), j = 1 : length(xs)]

    mask = X .* W
    mask ./= sum(mask)

    mask
end

function construct_checkerboard(coarse_cells)
    coarse = coarse_cells + 2
    A = [rand(Bool) ? 1.0 : 9.0 for i = 1 : coarse, j = 1 : coarse]

    return function a(x::Coord{2})
        x_int = floor(Int, x[1]) + 2
        y_int = floor(Int, x[2]) + 2
        A[y_int, x_int]
    end
end

function evaluate_mollified_checkerboard(a, nodes::Vector{Coord{2}}, ε = 0.5)
    Nn = length(nodes)
    a_ε = Vector{Float64}(Nn)
    mask = construct_mask()
    a_local = similar(mask)
    xs = gauss_legendre_coords()
    Nx = length(xs)

    @inbounds for node_idx = 1 : Nn
        node = nodes[node_idx]
        for j = 1 : Nx, i = 1 : Nx
            offset = @SVector [xs[i], xs[j]]
            a_local[j, i] = a(node + ε * offset)
        end
        
        a_ε[node_idx] = vecdot(a_local, mask)
    end

    return a_ε
end

function mollify_things(coarse = 64, refs = 1, ε = 0.5)
    a = construct_checkerboard(coarse)
    mesh, interior = rectangle(coarse * 2^refs, coarse * 2^refs, coarse, coarse)

    midpoints = map(mesh.elements) do e
        (mesh.nodes[e[1]] + mesh.nodes[e[2]] + mesh.nodes[e[3]]) / 3
    end

    save_to_vtk("mollifier", mesh, Dict(
        "a_eps" => evaluate_mollified_checkerboard(a, mesh.nodes, ε)
    ), Dict(
        "a" => a.(midpoints)
    ))
end

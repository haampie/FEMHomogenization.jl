function load(nodes_file::String, triangle_file::String)
    nmat = readdlm(nodes_file, ' ', Float64)
    tmat = readdlm(triangle_file, ' ', Int64)
    nodes = size(nmat, 1)
    triangles = size(tmat, 1)

    p = Vector{Coord}(nodes)
    t = Vector{Triangle}(triangles)

    for i = 1 : nodes
        p[i] = Coord(nmat[i, 1], nmat[i, 2])
    end

    for i = 1 : triangles
        t[i] = Triangle(tmat[i, 1], tmat[i, 2], tmat[i, 3])
    end

    return Mesh(p, t)
end
